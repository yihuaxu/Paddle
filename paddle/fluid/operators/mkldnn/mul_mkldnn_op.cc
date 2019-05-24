/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>
#include <vector>
#include "paddle/fluid/operators/mul_op.h"
#include "paddle/fluid/platform/mkldnn_helper.h"

namespace paddle {
namespace operators {

using framework::DataLayout;
using Tensor = framework::Tensor;
using framework::DataLayout;
using framework::Tensor;
using framework::DDim;
using framework::ExecutionContext;
using platform::MKLDNNDeviceContext;
using platform::to_void_cast;
using platform::GetMKLDNNFormat;
using mkldnn::memory;
using mkldnn::inner_product_forward;
using mkldnn::primitive;
using mkldnn::stream;
using mkldnn::prop_kind;

template <typename T, typename K>
class MulPrimitiveFactory {
 public:
  explicit MulPrimitiveFactory(const mkldnn::engine& engine)
      : engine_(engine) {}

  virtual ~MulPrimitiveFactory() {}

  virtual inner_product_forward CreateMulPrimitive(
      const Tensor* input, const Tensor* weights, Tensor* output,
      const ExecutionContext& ctx) {
    if (fc_) {
      UpdateDataPointers(ctx, output, input);
      return *fc_;
    }
    auto src_desc = CreateMemDescriptor(input, mkldnn::memory::format::nc);
    input_ = CreateMemory(src_desc, input);

    weights_ = TransposeWeights(weights);
    auto dst_desc = CreateMemDescriptor(output, memory::format::any);

    fc_ = CreateFcPrimitive(*input_, *weights_, dst_desc, output, ctx);
    return *fc_;
  }

  virtual inner_product_forward CreateMulPrimitive(
      const Tensor* input, const Tensor* weights, Tensor* output,
      const ExecutionContext& ctx, const std::vector<float>& scale_weight) {
    return CreateMulPrimitive(input, weights, output, ctx);
  }

 protected:
  void UpdateDataPointers(const ExecutionContext& ctx, Tensor* out,
                          const Tensor* in) {
    input_->set_data_handle(const_cast<T*>(in->data<T>()));
    output_->set_data_handle(out->mutable_data<T>(ctx.GetPlace()));
    if (out->format() == memory::format::format_undef) {
      auto output_format = output_->get_primitive_desc().desc().data.format;
      out->set_format((memory::format)output_format);
    }
  }

  virtual mkldnn::memory Reorder(const memory::desc& src_desc,
                                 const memory::desc& dst_desc,
                                 const void* src_data) {
    auto src_mem = memory({src_desc, engine_}, const_cast<void*>(src_data));
    auto dst_mem = memory({dst_desc, engine_});

    auto reorder = mkldnn::reorder(src_mem, dst_mem);
    stream(stream::kind::eager).submit({reorder}).wait();

    return dst_mem;
  }

  mkldnn::memory::desc CreateMemDescriptor(const std::vector<int>& dims,
                                           memory::format format) {
    return platform::MKLDNNMemDesc(dims, platform::MKLDNNGetDataType<T>(),
                                   format);
  }

  mkldnn::memory::desc CreateMemDescriptor(const std::vector<int>& dims,
                                           memory::data_type type,
                                           memory::format format) {
    return platform::MKLDNNMemDesc(dims, type, format);
  }

  mkldnn::memory::desc CreateMemDescriptor(const Tensor* tensor,
                                           memory::format format) {
    auto dims = framework::vectorize2int(tensor->dims());
    return CreateMemDescriptor(dims, format);
  }

  mkldnn::memory::desc CreateMemDescriptor(const Tensor* tensor,
                                           memory::data_type type,
                                           memory::format format) {
    auto dims = framework::vectorize2int(tensor->dims());
    return CreateMemDescriptor(dims, type, format);
  }

  mkldnn::memory CreateMemory(const mkldnn::memory::desc& desc,
                              const Tensor* tensor) {
    return CreateMemory(desc, tensor->data<T>());
  }

  mkldnn::memory CreateMemory(const mkldnn::memory::desc& desc,
                              const void* data) {
    return memory({desc, engine_}, const_cast<void*>(data));
  }

  mkldnn::memory TransposeWeights(const Tensor* weights) {
    auto dims = framework::vectorize2int(weights->dims());
    std::swap(dims[0], dims[1]);  // Correct output dimensions
    auto src_desc = CreateMemDescriptor(dims, memory::format::io);
    auto dst_desc = CreateMemDescriptor(dims, memory::format::oi);
    return Reorder(src_desc, dst_desc, weights->data<T>());
  }

  inner_product_forward CreateFcPrimitive(const memory& src_memory,
                                          const memory& weights_memory,
                                          const memory::desc& dst_desc,
                                          Tensor* output,
                                          const ExecutionContext& ctx) {
    const auto weights_desc = weights_memory.get_primitive_desc().desc();
    const auto src_desc = src_memory.get_primitive_desc().desc();

    auto fc_prim_desc = CreateFcPrimDesc(src_desc, weights_desc, dst_desc);

    output_ = CreateDstMemory(fc_prim_desc, ctx, output);

    return inner_product_forward(fc_prim_desc, src_memory, weights_memory,
                                 *output_);
  }

  mkldnn::inner_product_forward::primitive_desc CreateFcPrimDesc(
      const mkldnn::memory::desc& input_desc,
      const mkldnn::memory::desc& weights_desc,
      const mkldnn::memory::desc& bias_desc,
      const mkldnn::memory::desc& dst_desc) {
    auto fc_desc =
        inner_product_forward::desc(prop_kind::forward_scoring, input_desc,
                                    weights_desc, bias_desc, dst_desc);

    return inner_product_forward::primitive_desc(fc_desc, engine_);
  }

  mkldnn::inner_product_forward::primitive_desc CreateFcPrimDesc(
      const mkldnn::memory::desc& input_desc,
      const mkldnn::memory::desc& weights_desc,
      const mkldnn::memory::desc& dst_desc) {
    auto fc_desc = inner_product_forward::desc(prop_kind::forward, input_desc,
                                               weights_desc, dst_desc);

    return inner_product_forward::primitive_desc(fc_desc, engine_);
  }

  mkldnn::memory CreateDstMemory(
      const mkldnn::inner_product_forward::primitive_desc& fc_prim_desc,
      const ExecutionContext& ctx, Tensor* output) {
    auto dst_prim_desc = fc_prim_desc.dst_primitive_desc();
    auto buffer_size = dst_prim_desc.get_size();
    T* output_data = output->mutable_data<T>(
        ctx.GetPlace(), ::paddle::memory::Allocator::kDefault, buffer_size);
    output->set_format((memory::format)dst_prim_desc.desc().data.format);
    return memory(dst_prim_desc, to_void_cast<T>(output_data));
  }

 protected:
  const mkldnn::engine& engine_;
  boost::optional<memory> bias_;
  boost::optional<memory> input_;
  boost::optional<memory> output_;
  boost::optional<memory> weights_;
  boost::optional<inner_product_forward> fc_;
};

static std::string GetHash(const Tensor* input, const Tensor* weights,
                           const std::string& suffix) {
  auto dim2str = [](const DDim& operand_dims) {
    std::string str = "";
    for (int i = 0; i < operand_dims.size(); ++i) {
      str += std::to_string(operand_dims[i]) + "-";
    }
    return str;
  };
  return std::to_string((unsigned)input->format()) + dim2str(weights->dims()) +
         suffix;
}

template <typename T, typename K>
class QuantMulPrimitiveFactory : public MulPrimitiveFactory<T, K> {
 public:
  using MulPrimitiveFactory<T, K>::MulPrimitiveFactory;
  virtual inner_product_forward CreateMulPrimitive(
      const Tensor* input, const Tensor* weights, Tensor* output,
      const ExecutionContext& ctx, const std::vector<float>& scale_weight) {
    if (this->fc_) {
      this->UpdateDataPointers(ctx, output, input);
      return *(this->fc_);
    }
    auto src_dt = platform::MKLDNNGetDataType<T>();
    auto src_desc =
        this->CreateMemDescriptor(input, src_dt, mkldnn::memory::format::nc);
    this->input_ = this->CreateMemory(src_desc, input);

    if (std::is_same<K, float>::value) {
      auto weights_dt = platform::MKLDNNGetDataType<K>();
      auto trans_weights_ = this->TransposeWeights(weights, weights_dt);
      this->weights_ = this->UpdateWeights(trans_weights_, scale_weight);
    } else if (std::is_same<K, int8_t>::value) {
      this->weights_ =
          this->TransposeWeights(weights, platform::MKLDNNGetDataType<K>());
    }
    auto dst_dt = platform::MKLDNNGetDataType<int>();
    auto dst_desc =
        this->CreateMemDescriptor(output, dst_dt, memory::format::any);

    this->fc_ = this->CreateFcPrimitive(*(this->input_), *(this->weights_),
                                        dst_desc, output, ctx);
    return *(this->fc_);
  }

  virtual mkldnn::memory Reorder(const memory::desc& src_desc,
                                 const memory::desc& dst_desc,
                                 const void* src_data,
                                 const std::vector<float>& scale = {1.0f}) {
    auto mask = scale.size() > 1 ? 1 : 0;
    mkldnn::primitive_attr attr;
    attr.set_output_scales(mask, scale);
    auto src_mem =
        memory({src_desc, this->engine_}, const_cast<void*>(src_data));
    auto dst_mem = memory({dst_desc, this->engine_});
    auto reorder_pd = mkldnn::reorder::primitive_desc(
        src_mem.get_primitive_desc(), dst_mem.get_primitive_desc(), attr);

    auto reorder = mkldnn::reorder(reorder_pd, src_mem, dst_mem);
    stream(stream::kind::eager).submit({reorder}).wait();

    return dst_mem;
  }

  mkldnn::memory TransposeWeights(const Tensor* weights,
                                  mkldnn::memory::data_type type) {
    auto dims = framework::vectorize2int(weights->dims());
    std::swap(dims[0], dims[1]);  // Correct output dimensions
    auto src_desc = this->CreateMemDescriptor(dims, type, memory::format::io);
    auto dst_desc = this->CreateMemDescriptor(dims, type, memory::format::oi);
    return Reorder(src_desc, dst_desc, weights->data<T>());
  }

  mkldnn::memory UpdateWeights(mkldnn::memory weights,
                               const std::vector<float>& scale_weight) {
    auto dims = weights.get_primitive_desc().desc().data.dims;
    auto ndims = weights.get_primitive_desc().desc().data.ndims;
    auto weights_dims = std::vector<int>(dims, dims + ndims);
    auto weights_dt = platform::MKLDNNGetDataType<K>();
    auto weights_desc = this->CreateMemDescriptor(
        weights_dims, memory::data_type::s8, memory::format::oi);
    auto user_weights_desc =
        this->CreateMemDescriptor(weights_dims, weights_dt, memory::format::oi);
    return Reorder(user_weights_desc, weights_desc, weights.get_data_handle(),
                   scale_weight);
  }
};

template <typename T, typename K>
std::shared_ptr<MulPrimitiveFactory<T, K>> GetPrimitiveFactory(
    const MKLDNNDeviceContext& dev_ctx, const ExecutionContext& ctx,
    const Tensor* input, const Tensor* weights,
    const mkldnn::engine& mkldnn_engine) {
  const std::string key = GetHash(input, weights, ctx.op().Output("Out"));

  auto prim_creator =
      std::static_pointer_cast<MulPrimitiveFactory<T, K>>(dev_ctx.GetBlob(key));
  if (prim_creator == nullptr) {
    bool enable_quant =
        std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value;
    if (enable_quant) {
      prim_creator =
          std::make_shared<QuantMulPrimitiveFactory<T, K>>(mkldnn_engine);
    } else {
      prim_creator = std::make_shared<MulPrimitiveFactory<T, K>>(mkldnn_engine);
    }
    dev_ctx.SetBlob(key, prim_creator);
  }

  return prim_creator;
}

template <typename T, typename K>
class MulMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");

    auto& dev_ctx =
        ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    const Tensor* x = ctx.Input<Tensor>("X");
    const Tensor* y = ctx.Input<Tensor>("Y");
    Tensor* z = ctx.Output<Tensor>("Out");
    const Tensor x_matrix =
        x->dims().size() > 2 ? framework::ReshapeToMatrix(
                                   *x, ctx.template Attr<int>("x_num_col_dims"))
                             : *x;
    const Tensor y_matrix =
        y->dims().size() > 2 ? framework::ReshapeToMatrix(
                                   *y, ctx.template Attr<int>("y_num_col_dims"))
                             : *y;

    auto z_dim = z->dims();
    if (z_dim.size() != 2) {
      z->Resize({x_matrix.dims()[0], y_matrix.dims()[1]});
    }

    auto prim_creator = GetPrimitiveFactory<T, K>(dev_ctx, ctx, &x_matrix,
                                                  &y_matrix, mkldnn_engine);
    auto scale_weights = ctx.Attr<std::vector<float>>("scale_y");
    auto fc = prim_creator->CreateMulPrimitive(&x_matrix, &y_matrix, z, ctx,
                                               scale_weights);
    stream(stream::kind::eager).submit({fc}).wait();

    z->set_layout(DataLayout::kMKLDNN);
    z->set_format(z_dim.size() != 2
                      ? platform::MKLDNNFormatForSize(z_dim.size(), x->format())
                      : z->format());

    if (z_dim.size() != 2) {
      z->Resize(z_dim);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(mul, MKLDNN, ::paddle::platform::CPUPlace,
                                    U8, ops::kMULMKLDNNINT8,
                                    ops::MulMKLDNNKernel<uint8_t, float>);
REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(mul, MKLDNN, ::paddle::platform::CPUPlace,
                                    S8, ops::kMULMKLDNNINT8,
                                    ops::MulMKLDNNKernel<int8_t, float>);
// REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(mul, MKLDNN,
// ::paddle::platform::CPUPlace,
//                                    FP32, ops::kMULMKLDNNFP32,
//                                    ops::MulMKLDNNKernel<float, float>);
REGISTER_OP_KERNEL(mul, MKLDNN, ::paddle::platform::CPUPlace,
                   paddle::operators::MulMKLDNNKernel<float, float>);
