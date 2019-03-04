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
  void Compute(const framework::ExecutionContext &context) const override {
    auto *out = context.Output<LoDTensor>("Out");
    auto out_data = out->mutable_data<T>(context.GetPlace());
    auto out_dims = out->dims();

    const LoDTensor *table0 = context.Input<LoDTensor>("W0");
    auto table0_dims = table0->dims();
    auto table0_data = table0->data<T>();
    const LoDTensor *table1 = context.Input<LoDTensor>("W1");
    auto table1_dims = table1->dims();
    auto table1_data = table1->data<T>();

    PADDLE_ENFORCE_EQ(
        static_cast<uint64_t>(out_dims[1]), table0_dims[1],
        "The actual output data's size mismatched with table0's width.");

    auto win_size = context.Attr<std::vector<int>>("win_size");
    int pad_value = context.Attr<int>("pad_value");
    int max_win_size = *std::max_element(win_size.begin(), win_size.end());
    std::vector<int> num_hash = context.Attr<std::vector<int>>("num_hash");
    std::vector<int> mod_by = context.Attr<std::vector<int>>("mod_by");

    PADDLE_ENFORCE(
        num_hash.size() == mod_by.size() && num_hash.size() == win_size.size(),
        "All attributes's count should be equal!");

    auto table0_compute =
        jit::Get<jit::kVAdd, jit::XYZNTuples<T>, platform::CPUPlace>(
            table0_dims[1]);
    auto table1_compute =
        jit::Get<jit::kVAdd, jit::XYZNTuples<T>, platform::CPUPlace>(
            table1_dims[1]);

    // Zero output meory to prepare for SUM operation in furture.
    std::memset(out_data, 0, out->memory_size());

    int64_t *enum_buf = NULL;
    int enum_buf_len = 0;

    // To definate the input names.
    const char input_names[][10] = {"X0", "X1"};

    for (int i = 0; i < 2; i++) {
      auto *in = context.Input<LoDTensor>(input_names[i]);
      auto in_dims = in->dims();
      auto in_lod = in->lod();
      auto in_len = in->numel();
      auto in_data = in->data<int64_t>();

      auto lod0 = in_lod[0];

      // To check the output, input and table's dimension?
      PADDLE_ENFORCE_EQ(
          static_cast<uint64_t>(out_dims[0]), lod0.size() - 1,
          "The actual output data's size mismatched with LoD information.");

      for (size_t hash_idx = 0; hash_idx < num_hash.size(); hash_idx++) {
        PADDLE_ENFORCE_EQ(
            static_cast<uint64_t>(out_dims[1]),
            num_hash[hash_idx] * table1_dims[1],
            "The actual output data's size mismatched with table0's width.");
      }

      PADDLE_ENFORCE_EQ(
          static_cast<uint64_t>(in_dims[0]), in_lod[0].back(),
          "The actual input data's size mismatched with LoD information.");

      // To check the buffer size and decide to realloc the memory.
      if (enum_buf_len < in_len) {
        enum_buf_len = in_len;

        // If the buffer was allocated, it need be free firstly.
        if (enum_buf != NULL) {
          std::free(enum_buf);
        }

        // To re-allocate the temporary memory for input data's enumerate
        // operation.
        int ret =
            posix_memalign(reinterpret_cast<void **>(&enum_buf), 64,
                           sizeof(int64_t) * (enum_buf_len + max_win_size - 1));
        PADDLE_ENFORCE_EQ(ret, 0, "Failed to allocate the temporary memory!");
      }

      // To prepare enumerate buffer content.
      // For example:
      //  Input: ABCDE
      //  Max_win_size: 4
      //  Pad_value: x
      //  Enum_Buf: ABCDExxx
      std::memcpy(enum_buf, in_data, sizeof(int64_t) * in_len);
      for (auto idx = 0; idx < max_win_size - 1; idx++) {
        enum_buf[in_len + idx] = pad_value;
      }

      int out_offset = 0;
      for (size_t lod_idx = 0; lod_idx < lod0.size() - 1; lod_idx++) {
        auto start_pos = lod0[lod_idx];
        auto end_pos = lod0[lod_idx + 1];

        for (size_t pos = start_pos; pos < end_pos; pos++) {
          // According to table0's hash value, it finish the SUM operation with
          // output's data.
          auto hash_len = table0_dims[1];
          // To calculate the table offset.
          auto hash_idx = enum_buf[pos];
          auto table_offset = hash_idx * hash_len;

          // To summary the hash value with output data.
          table0_compute(table0_data + table_offset, out_data + out_offset,
                         out_data + out_offset, hash_len);

          // To cacluate the hash of input data and summery table1 with output's
          // data.
          hash_len = table1_dims[1];
          for (size_t idx = 0; idx < win_size.size(); idx++) {
            auto last_dim = win_size[idx];
            for (int ihash = 0; ihash != num_hash[idx]; ++ihash) {
              // To calculate the hash offset.
              auto hash_offset = ihash * hash_len;

              // To change the pos can omit the enumerate opeation via the
              // expanding buffer.
              hash_idx = XXH64(enum_buf + pos, sizeof(int) * last_dim, ihash) %
                         mod_by[idx];

              // To calculate the table offset.
              table_offset = hash_idx * hash_len;

              // To summary the hash value with output data.
              table1_compute(table1_data + table_offset,
                             out_data + out_offset + hash_offset,
                             out_data + out_offset + hash_offset, hash_len);
            }
          }
        }
        out_offset += out_dims[1];
      }
    }

    // Free the allocated buffer.
    if (enum_buf != NULL) {
      std::free(enum_buf);
    }
  }
};

}  // namespace operators
}  // namespace paddle
