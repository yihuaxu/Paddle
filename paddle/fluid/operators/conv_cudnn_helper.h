/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <algorithm>
#include <array>
#include <memory>
#include <vector>
#include "paddle/fluid/framework/operator_kernel_configs.h"
#include "paddle/fluid/operators/conv_cudnn_op_cache.h"
#include "paddle/fluid/platform/cudnn_desc.h"
namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DataLayout = platform::DataLayout;
template <typename T>
using ScalingParamType = typename platform::CudnnDataType<T>::ScalingParamType;
using framework::AlgorithmsCache;
static inline void GetNCDHW(const framework::DDim& dims,
                            const DataLayout& layout, int* N, int* C, int* D,
                            int* H, int* W) {
  *N = dims[0];
  *C = layout == DataLayout::kNCHW ? dims[1] : dims[dims.size() - 1];
  int i = layout == DataLayout::kNCHW ? 0 : 1;
  if (dims.size() == 5) {
    *D = dims[2 - i];
    *H = dims[3 - i];
    *W = dims[4 - i];
  } else {
    *D = 1;
    *H = dims[2 - i];
    *W = dims[3 - i];
  }
}

template <typename DeviceContext, typename T, size_t D>
static void RemovePaddingSlice(const framework::ExecutionContext& context,
                               const Tensor* input, Tensor* out,
                               const std::vector<int>& starts,
                               const std::vector<int>& axes) {
  auto& place =
      *context.template device_context<DeviceContext>().eigen_device();
  auto in_dims = input->dims();
  auto new_out_dims = out->dims();
  auto offsets = Eigen::array<int, D>();
  auto extents = Eigen::array<int, D>();
  for (size_t i = 0; i < D; ++i) {
    offsets[i] = 0;
    extents[i] = new_out_dims[i];
  }

  int start;
  for (size_t i = 0; i < axes.size(); ++i) {
    start = starts[i];
    if (start < 0) {
      start = (start + in_dims[axes[i]]);
    }
    start = std::max(start, 0);
    offsets[axes[i]] = start;
  }
  auto in_t =
      framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
          *input);

  auto out_t =
      framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
          *out, new_out_dims);
  out_t.device(place) = in_t.slice(offsets, extents);
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
  out << "[";
  for (auto const& tmp : v) out << tmp << ",";
  out << "]";
  return out;
}

using framework::AlgorithmsCache;

struct ConvArgs {
  cudnnHandle_t handle;
  platform::TensorDescriptor idesc, odesc;
  platform::FilterDescriptor wdesc;
  platform::ConvolutionDescriptor cdesc;
  const framework::Tensor *x, *w, *o;

  // strides
  std::vector<int> s;
  // paddings
  std::vector<int> p;
  // dilations
  std::vector<int> d;

  ConvArgs(const framework::Tensor* x, const framework::Tensor* w,
           const framework::Tensor* o, const std::vector<int> s,
           const std::vector<int> p, const std::vector<int> d)
      : x(x), w(w), o(o), s(s), p(p), d(d) {}
};

template <typename perf_t>
struct SearchAlgorithm {};

template <>
struct SearchAlgorithm<cudnnConvolutionFwdAlgoPerf_t> {
  using perf_t = cudnnConvolutionFwdAlgoPerf_t;
  using algo_t = cudnnConvolutionFwdAlgo_t;

  template <typename T>
  static algo_t Find(const ConvArgs& args, bool exhaustive_search,
                     bool deterministic, int algo_cache_id,
                     const framework::ExecutionContext& ctx) {
    auto dtype = platform::CudnnDataType<T>::type;
    bool has_got_workspace_size = true;
    bool exhaustive = (exhaustive_search) & (dtype != CUDNN_DATA_HALF);
    size_t workspace_size_limit = FLAGS_conv_workspace_size_limit * 1024 * 1024;
    size_t workspace_size = 0;
    algo_t algo;

#if CUDA_VERSION >= 9000 && CUDNN_VERSION_MIN(7, 0, 1)
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    if (dev_ctx.GetComputeCapability() >= 70 && dtype == CUDNN_DATA_HALF) {
      CUDNN_ENFORCE(platform::dynload::cudnnSetConvolutionMathType(
          args.cdesc.desc(), CUDNN_TENSOR_OP_MATH));
      VLOG(5) << "use cudnn_tensor_op_math";
    } else {
      CUDNN_ENFORCE(platform::dynload::cudnnSetConvolutionMathType(
          args.cdesc.desc(), CUDNN_DEFAULT_MATH));
      VLOG(5) << "NOT use cudnn_tensor_op_math";
    }
#endif

    if (!exhaustive) {
#if CUDNN_VERSION >= 7001
      int perf_count;
      int best_algo_idx = 0;
      std::unique_ptr<perf_t[]> perf_results(new perf_t[kNUM_CUDNN_FWD_ALGS]);
      CUDNN_ENFORCE(platform::dynload::cudnnGetConvolutionForwardAlgorithm_v7(
          args.handle, args.idesc.desc(), args.wdesc.desc(), args.cdesc.desc(),
          args.odesc.desc(), kNUM_CUDNN_FWD_ALGS, &perf_count,
          perf_results.get()));
      algo = (perf_results.get())[best_algo_idx].algo;
      workspace_size = GetWorkspaceSize(args, algo);

      if (workspace_size > workspace_size_limit) {
        has_got_workspace_size = false;
        VLOG(1) << "Fallback to non-v7 method to find conv algorithm becasue "
                   "the workspace size request("
                << workspace_size << ") exceeds the limit("
                << workspace_size_limit << ")";
      }
      if (!has_got_workspace_size) {
        CUDNN_ENFORCE(platform::dynload::cudnnGetConvolutionForwardAlgorithm(
            args.handle, args.idesc.desc(), args.wdesc.desc(),
            args.cdesc.desc(), args.odesc.desc(),
            CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, workspace_size_limit,
            &algo));
      }
#else
      CUDNN_ENFORCE(platform::dynload::cudnnGetConvolutionForwardAlgorithm(
          args.handle, args.idesc.desc(), args.wdesc.desc(), args.cdesc.desc(),
          args.odesc.desc(), CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
          workspace_size_limit, &algo));
#endif
      VLOG(3) << "choose algo " << algo;
    } else {
      AlgorithmsCache<algo_t>& algo_cache =
          ctx.GetKernelConfig<AlgorithmsCache<algo_t>>(algo_cache_id);
      auto& dev_ctx =
          ctx.template device_context<platform::CUDADeviceContext>();
      auto workspace_handle = dev_ctx.cudnn_workspace_handle();

      auto x_dims = framework::vectorize(args.x->dims());
      auto w_dims = framework::vectorize(args.w->dims());

      VLOG(10) << "cudnnConvolutionFwdAlgoPerf_t algo_cache_id:"
               << algo_cache_id << ", x_dims:" << x_dims
               << ", w_dims:" << w_dims << ", args.s" << args.s << ", args.p"
               << args.p << ", args.d" << args.d;

      algo = algo_cache.GetAlgorithm(
          x_dims, w_dims, args.s, args.p, args.d, 0, [&]() {
            int returned_algo_count;
            std::array<perf_t, kNUM_CUDNN_FWD_ALGS> perf_stat;

            auto cudnn_find_func = [&](void* cudnn_workspace_ptr) {
              CUDNN_ENFORCE(
                  platform::dynload::cudnnFindConvolutionForwardAlgorithmEx(
                      args.handle, args.idesc.desc(), args.x->data<T>(),
                      args.wdesc.desc(), args.w->data<T>(), args.cdesc.desc(),
                      args.odesc.desc(), const_cast<T*>(args.o->data<T>()),
                      kNUM_CUDNN_FWD_ALGS, &returned_algo_count,
                      perf_stat.data(), cudnn_workspace_ptr,
                      workspace_size_limit));
            };
            workspace_handle.RunFuncSync(cudnn_find_func, workspace_size_limit);

            VLOG(3) << "FwdAlgo Perf result: (algo: stat, time, memory)";
            for (int i = 0; i < returned_algo_count; ++i) {
              const auto& stat = perf_stat[i];
              VLOG(3) << stat.algo << ": " << stat.status << " " << stat.time
                      << " " << stat.memory;
            }
            return perf_stat[0].algo;
          });
    }
    VLOG(3) << "choose algo " << algo;
    return algo;
  }

  static size_t GetWorkspaceSize(const ConvArgs& args, algo_t algo) {
    size_t workspace_size = 0;
    CUDNN_ENFORCE(platform::dynload::cudnnGetConvolutionForwardWorkspaceSize(
        args.handle, args.idesc.desc(), args.wdesc.desc(), args.cdesc.desc(),
        args.odesc.desc(), algo, &workspace_size));
    return workspace_size;
  }
};

template <>
struct SearchAlgorithm<cudnnConvolutionBwdDataAlgoPerf_t> {
  using perf_t = cudnnConvolutionBwdDataAlgoPerf_t;
  using algo_t = cudnnConvolutionBwdDataAlgo_t;

  template <typename T>
  static algo_t Find(const ConvArgs& args, bool exhaustive_search,
                     bool deterministic, int algo_cache_id,
                     const framework::ExecutionContext& ctx) {
    auto dtype = platform::CudnnDataType<T>::type;
    bool exhaustive = (exhaustive_search) & (dtype != CUDNN_DATA_HALF);
    size_t workspace_size_limit = FLAGS_conv_workspace_size_limit * 1024 * 1024;
    size_t workspace_size = 0;
    bool has_got_workspace_size = true;
    algo_t algo;

#if CUDA_VERSION >= 9000 && CUDNN_VERSION_MIN(7, 0, 1)
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    if (dev_ctx.GetComputeCapability() >= 70 && dtype == CUDNN_DATA_HALF) {
      CUDNN_ENFORCE(platform::dynload::cudnnSetConvolutionMathType(
          args.cdesc.desc(), CUDNN_TENSOR_OP_MATH));
      VLOG(5) << "use cudnn_tensor_op_math";
    } else {
      CUDNN_ENFORCE(platform::dynload::cudnnSetConvolutionMathType(
          args.cdesc.desc(), CUDNN_DEFAULT_MATH));
      VLOG(5) << "NOT use cudnn_tensor_op_math";
    }
#endif

    if (!exhaustive && !deterministic) {
#if CUDNN_VERSION >= 7001
      int perf_count;
      int best_algo_idx = 0;
      std::unique_ptr<perf_t[]> perf_results(
          new perf_t[kNUM_CUDNN_BWD_DATA_ALGS]);
      CUDNN_ENFORCE(
          platform::dynload::cudnnGetConvolutionBackwardDataAlgorithm_v7(
              args.handle, args.wdesc.desc(), args.odesc.desc(),
              args.cdesc.desc(), args.idesc.desc(), kNUM_CUDNN_BWD_DATA_ALGS,
              &perf_count, perf_results.get()));
      algo = (perf_results.get())[best_algo_idx].algo;

#if CUDNN_VERSION < 7500
      int stride_dim = args.x->dims().size() - 2;
      bool blacklist = std::any_of(args.s.begin(), args.s.begin() + stride_dim,
                                   [=](int n) { return n != 1; });
      if (blacklist && (static_cast<cudnnConvolutionBwdDataAlgo_t>(
                            perf_results[best_algo_idx].algo) ==
                            CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING ||
                        static_cast<cudnnConvolutionBwdDataAlgo_t>(
                            perf_results[best_algo_idx].algo) ==
                            CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT)) {
        algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
      }
#endif
      workspace_size = GetWorkspaceSize(args, algo);
      if (workspace_size > workspace_size_limit) {
        has_got_workspace_size = false;
        VLOG(1) << "Fallback to non-v7 method to find conv algorithm becasue "
                   "the workspace size request("
                << workspace_size << ") exceeds the limit("
                << workspace_size_limit << ")";
      }
      if (!has_got_workspace_size) {
        CUDNN_ENFORCE(
            platform::dynload::cudnnGetConvolutionBackwardDataAlgorithm(
                args.handle, args.wdesc.desc(), args.odesc.desc(),
                args.cdesc.desc(), args.idesc.desc(),
                CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
                workspace_size_limit, &algo));
      }
#else
      CUDNN_ENFORCE(platform::dynload::cudnnGetConvolutionBackwardDataAlgorithm(
          args.handle, args.wdesc.desc(), args.odesc.desc(), args.cdesc.desc(),
          args.idesc.desc(), CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
          workspace_size_limit, &algo));
#endif
    } else if (deterministic) {
      return CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    } else {
      AlgorithmsCache<algo_t>& algo_cache =
          ctx.GetKernelConfig<AlgorithmsCache<algo_t>>(algo_cache_id);
      auto& dev_ctx =
          ctx.template device_context<platform::CUDADeviceContext>();
      auto workspace_handle = dev_ctx.cudnn_workspace_handle();

      auto x_dims = framework::vectorize(args.x->dims());
      auto w_dims = framework::vectorize(args.w->dims());

      VLOG(10) << "cudnnConvolutionFwdAlgoPerf_t algo_cache_id:"
               << algo_cache_id << ", x_dims:" << x_dims
               << ", w_dims:" << w_dims << ", args.s" << args.s << ", args.p"
               << args.p << ", args.d" << args.d;

      algo = algo_cache.GetAlgorithm(
          x_dims, w_dims, args.s, args.p, args.d, 0, [&]() {
            int returned_algo_count;
            std::array<perf_t, kNUM_CUDNN_FWD_ALGS> perf_stat;

            auto cudnn_find_func = [&](void* cudnn_workspace_ptr) {
              CUDNN_ENFORCE(
                  platform::dynload::
                      cudnnFindConvolutionBackwardDataAlgorithmEx(
                          args.handle, args.wdesc.desc(), args.w->data<T>(),
                          args.odesc.desc(), args.o->data<T>(),
                          args.cdesc.desc(), args.idesc.desc(),
                          const_cast<T*>(args.x->data<T>()),
                          kNUM_CUDNN_BWD_DATA_ALGS, &returned_algo_count,
                          perf_stat.data(), cudnn_workspace_ptr,
                          workspace_size_limit));
            };
            workspace_handle.RunFuncSync(cudnn_find_func, workspace_size_limit);

            VLOG(3) << "BwdDataAlgo Perf result: (algo: stat, time, memory)";
            for (int i = 0; i < returned_algo_count; ++i) {
              const auto& stat = perf_stat[i];
              VLOG(3) << stat.algo << ": " << stat.status << " " << stat.time
                      << " " << stat.memory;
            }

            return perf_stat[0].algo;
          });
    }
    VLOG(3) << "choose algo " << algo;
    return algo;
  }

  static size_t GetWorkspaceSize(const ConvArgs& args, algo_t algo) {
    size_t workspace_size = 0;
    CUDNN_ENFORCE(
        platform::dynload::cudnnGetConvolutionBackwardDataWorkspaceSize(
            args.handle, args.wdesc.desc(), args.odesc.desc(),
            args.cdesc.desc(), args.idesc.desc(), algo, &workspace_size));
    return workspace_size;
  }
};

template <>
struct SearchAlgorithm<cudnnConvolutionBwdFilterAlgoPerf_t> {
  using perf_t = cudnnConvolutionBwdFilterAlgoPerf_t;
  using algo_t = cudnnConvolutionBwdFilterAlgo_t;

  template <typename T>
  static algo_t Find(const ConvArgs& args, bool exhaustive_search,
                     bool deterministic, int algo_cache_id,
                     const framework::ExecutionContext& ctx) {
    auto dtype = platform::CudnnDataType<T>::type;
    bool exhaustive = (exhaustive_search) & (dtype != CUDNN_DATA_HALF);
    size_t workspace_size_limit = FLAGS_conv_workspace_size_limit * 1024 * 1024;
    size_t workspace_size = 0;
    bool has_got_workspace_size = true;

#if CUDA_VERSION >= 9000 && CUDNN_VERSION_MIN(7, 0, 1)
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    if (dev_ctx.GetComputeCapability() >= 70 && dtype == CUDNN_DATA_HALF) {
      CUDNN_ENFORCE(platform::dynload::cudnnSetConvolutionMathType(
          args.cdesc.desc(), CUDNN_TENSOR_OP_MATH));
      VLOG(5) << "use cudnn_tensor_op_math";
    } else {
      CUDNN_ENFORCE(platform::dynload::cudnnSetConvolutionMathType(
          args.cdesc.desc(), CUDNN_DEFAULT_MATH));
      VLOG(5) << "NOT use cudnn_tensor_op_math";
    }
#endif

    algo_t algo;
    if (!exhaustive && !deterministic) {
#if CUDNN_VERSION >= 7001
      using perf_t = cudnnConvolutionBwdFilterAlgoPerf_t;
      int perf_count;
      int best_algo_idx = 0;
      std::unique_ptr<perf_t[]> perf_results(
          new perf_t[kNUM_CUDNN_BWD_FILTER_ALGS]);
      CUDNN_ENFORCE(
          platform::dynload::cudnnGetConvolutionBackwardFilterAlgorithm_v7(
              args.handle, args.idesc.desc(), args.odesc.desc(),
              args.cdesc.desc(), args.wdesc.desc(), kNUM_CUDNN_BWD_FILTER_ALGS,
              &perf_count, perf_results.get()));
      algo = (perf_results.get())[best_algo_idx].algo;
      workspace_size = GetWorkspaceSize(args, algo);
      if (workspace_size > workspace_size_limit) {
        has_got_workspace_size = false;
        VLOG(1) << "Fallback to non-v7 method to find conv algorithm becasue "
                   "the workspace size request("
                << workspace_size << ") exceeds the limit("
                << workspace_size_limit << ")";
      }
      if (!has_got_workspace_size) {
        CUDNN_ENFORCE(
            platform::dynload::cudnnGetConvolutionBackwardFilterAlgorithm(
                args.handle, args.idesc.desc(), args.odesc.desc(),
                args.cdesc.desc(), args.wdesc.desc(),
                CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
                workspace_size_limit, &algo));
      }
#else
      CUDNN_ENFORCE(
          platform::dynload::cudnnGetConvolutionBackwardFilterAlgorithm(
              args.handle, args.idesc.desc(), args.odesc.desc(),
              args.cdesc.desc(), args.wdesc.desc(),
              CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
              workspace_size_limit, &algo));
#endif
    } else if (deterministic) {
      return CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
    } else {
      AlgorithmsCache<algo_t>& algo_cache =
          ctx.GetKernelConfig<AlgorithmsCache<algo_t>>(algo_cache_id);
      auto& dev_ctx =
          ctx.template device_context<platform::CUDADeviceContext>();
      auto workspace_handle = dev_ctx.cudnn_workspace_handle();

      auto x_dims = framework::vectorize(args.x->dims());
      auto w_dims = framework::vectorize(args.w->dims());

      VLOG(10) << "cudnnConvolutionFwdAlgoPerf_t algo_cache_id:"
               << algo_cache_id << ", x_dims:" << x_dims
               << ", w_dims:" << w_dims << ", args.s" << args.s << ", args.p"
               << args.p << ", args.d" << args.d;

      algo = algo_cache.GetAlgorithm(
          x_dims, w_dims, args.s, args.p, args.d, 0, [&]() {
            int returned_algo_count;
            std::array<perf_t, kNUM_CUDNN_FWD_ALGS> perf_stat;
            auto cudnn_find_func = [&](void* cudnn_workspace_ptr) {
              CUDNN_ENFORCE(
                  platform::dynload::
                      cudnnFindConvolutionBackwardFilterAlgorithmEx(
                          args.handle, args.idesc.desc(), args.x->data<T>(),
                          args.odesc.desc(), args.o->data<T>(),
                          args.cdesc.desc(), args.wdesc.desc(),
                          const_cast<T*>(args.w->data<T>()),
                          kNUM_CUDNN_BWD_FILTER_ALGS, &returned_algo_count,
                          perf_stat.data(), cudnn_workspace_ptr,
                          workspace_size_limit));
            };
            workspace_handle.RunFuncSync(cudnn_find_func, workspace_size_limit);

            VLOG(3) << "BwdFilterAlgo Perf result: (algo: stat, time, memory)";
            for (int i = 0; i < returned_algo_count; ++i) {
              const auto& stat = perf_stat[i];
              VLOG(3) << stat.algo << ": " << stat.status << " " << stat.time
                      << " " << stat.memory;
            }
            return perf_stat[0].algo;
          });
    }
    VLOG(3) << "choose algo " << algo;
    return algo;
  }

  static size_t GetWorkspaceSize(const ConvArgs& args, algo_t algo) {
    size_t workspace_size = 0;
    CUDNN_ENFORCE(
        platform::dynload::cudnnGetConvolutionBackwardFilterWorkspaceSize(
            args.handle, args.idesc.desc(), args.odesc.desc(),
            args.cdesc.desc(), args.wdesc.desc(), algo, &workspace_size));
    return workspace_size;
  }
};

}  // namespace operators
}  // namespace paddle
