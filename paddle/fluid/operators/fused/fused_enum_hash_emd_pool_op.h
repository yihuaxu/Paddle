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

#pragma once
#include <algorithm>
#include <string>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/jit/kernels.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using DDim = framework::DDim;

template <typename DeviceContext, typename T>
class FusedEnumHashEmdPoolKernel : public framework::OpKernel<T> {
 public:
  inline void table_compute(const T *x, T *y, T *z, int n) const {
    const size_t block = YMM_FLOAT_BLOCK;
    const size_t num_ = n;
    const size_t rest_ = num_ % block;
    const size_t end_ = num_ - rest_;

    __m256 tmp;
    size_t j;
    if (rest_ != 0) {
      j = num_ - block;
      tmp = _mm256_loadu_ps((const float *)y + j);
    }
    for (j = 0; j < end_; j += block) {
      _mm256_storeu_ps(reinterpret_cast<float *>(z) + j,
                       _mm256_add_ps(_mm256_loadu_ps((const float *)y + j),
                                     _mm256_loadu_ps((const float *)x + j)));
    }
    if (rest_ != 0) {
      j = num_ - block;
      _mm256_storeu_ps(
          reinterpret_cast<float *>(z) + j,
          _mm256_add_ps(tmp, _mm256_loadu_ps((const float *)x + j)));
    }
  }
  void Compute(const framework::ExecutionContext &context) const override {
    auto win_size = context.Attr<std::vector<int>>("win_size");
    int pad_value = context.Attr<int>("pad_value");
    int max_win_size = *std::max_element(win_size.begin(), win_size.end());
    std::vector<int> num_hash = context.Attr<std::vector<int>>("num_hash");
    std::vector<int> mod_by = context.Attr<std::vector<int>>("mod_by");
    PADDLE_ENFORCE(
        num_hash.size() == mod_by.size() && num_hash.size() == win_size.size(),
        "All attributes's count should be equal!");
    const LoDTensor *table1 = context.Input<LoDTensor>("W1");
    auto table1_dims = table1->dims();
    auto table1_data = table1->data<T>();
    const LoDTensor *table0 = context.Input<LoDTensor>("W0");
    auto table0_dims = table0->dims();
    auto table0_data = table0->data<T>();

    auto *out = context.Output<LoDTensor>("Out");
    auto out_data = out->mutable_data<T>(context.GetPlace());
    auto out_dims = out->dims();

    PADDLE_ENFORCE_EQ(
        static_cast<uint64_t>(out_dims[1]), table0_dims[1],
        "The actual output data's size mismatched with table0's width.");

    // auto table_compute =
    //    jit::Get<jit::kVAdd, jit::XYZNTuples<T>,
    //    platform::CPUPlace>(std::min(table0_dims[1],table1_dims[1]));

    memset(out_data, 0, out->memory_size());

    int64_t *enum_buf;
    int ret = posix_memalign(reinterpret_cast<void **>(&enum_buf), 64,
                             sizeof(int64_t) * max_win_size);
    PADDLE_ENFORCE_EQ(ret, 0, "Failed to allocate the temporary memory!");

    const char input_names[2][10] = {"X0", "X1"};
    for (int i = 0; i < 2; i++) {
      auto *in = context.Input<LoDTensor>(input_names[i]);
      auto in_dims = in->dims();
      auto in_lod = in->lod();

      PADDLE_ENFORCE_EQ(
          static_cast<uint64_t>(in_dims[0]), in_lod[0].back(),
          "The actual input data's size mismatched with LoD information.");

      auto lod0 = in_lod[0];
      auto in_data = in->data<int64_t>();

      PADDLE_ENFORCE_EQ(
          static_cast<uint64_t>(out_dims[0]), lod0.size() - 1,
          "The actual output data's size mismatched with LoD information.");
      for (size_t hash_idx = 0; hash_idx < num_hash.size(); hash_idx++) {
        PADDLE_ENFORCE_EQ(
            static_cast<uint64_t>(out_dims[1]),
            num_hash[hash_idx] * table1_dims[1],
            "The actual output data's size mismatched with table0's width.");
      }

      int out_offset = 0;
      for (size_t lod_idx = 0; lod_idx < lod0.size() - 1; lod_idx++) {
        auto start_pos = lod0[lod_idx];
        auto end_pos = lod0[lod_idx + 1];
        auto hash_len = table0_dims[1];
        for (size_t pos = start_pos; pos < end_pos; pos++) {
          auto hash_idx = in_data[pos];
          auto table_offset = hash_idx * hash_len;
          table_compute(table0_data + table_offset, out_data + out_offset,
                        out_data + out_offset, hash_len);
        }

        hash_len = table1_dims[1];
        for (size_t idx = 0; idx < win_size.size(); idx++) {
          auto last_dim = win_size[idx];
          size_t pos = start_pos;
          if (last_dim < static_cast<int>(end_pos - start_pos)) {
            while (pos <= end_pos - last_dim) {
              for (int ihash = 0; ihash != num_hash[idx]; ++ihash) {
                auto hash_idx =
                    XXH64(in_data + pos, sizeof(int) * last_dim, ihash);
                hash_idx %= mod_by[idx];
                auto hash_offset = ihash * hash_len;
                auto table_offset = hash_idx * hash_len;
                table_compute(table1_data + table_offset,
                              out_data + out_offset + hash_offset,
                              out_data + out_offset + hash_offset, hash_len);
              }
              pos++;
            }
          }
          for (int k = last_dim - (end_pos - pos); k < last_dim; k++) {
            for (int j = 0; j < last_dim; j++) {
              enum_buf[j] = j < last_dim - k
                                ? in_data[end_pos + k - last_dim + j]
                                : pad_value;
            }
            for (int ihash = 0; ihash != num_hash[idx]; ++ihash) {
              auto hash_idx = XXH64(enum_buf, sizeof(int) * last_dim, ihash);
              hash_idx %= mod_by[idx];
              auto hash_offset = ihash * hash_len;
              auto table_offset = hash_idx * hash_len;
              table_compute(table1_data + table_offset,
                            out_data + out_offset + hash_offset,
                            out_data + out_offset + hash_offset, hash_len);
            }
          }
        }
        out_offset += out_dims[1];
      }
    }
    std::free(enum_buf);
  }
};

}  // namespace operators
}  // namespace paddle
