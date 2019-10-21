// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

/*! \file paddle_api.h
 */

/*! \mainpage Paddle Inference APIs
 * \section intro_sec Introduction
 * The Paddle inference library aims to offer an high performance inference SDK
 * for Paddle users.
 */

#include <cassert>
#include <map>
#include <memory>
#include <string>
#include <vector>

/*! \namespace paddle
 */
namespace paddle {

/** paddle data type.
 */
enum PaddleDType {
  FLOAT32,
  INT64,
  INT32,
  UINT8,
  // TODO(Superjomn) support more data types if needed.
};

/**
 * \brief Memory manager for `PaddleTensor`.
 *
 * The PaddleBuf holds a buffer for data input or output. The memory can be
 * allocated by user or by PaddleBuf itself, but in any case, the PaddleBuf
 * should be reused for better performance.
 *
 * For user allocated memory, the following API can be used:
 * - PaddleBuf(void* data, size_t length) to set an external memory by
 * specifying the memory address and length.
 * - Reset(void* data, size_t length) to reset the PaddleBuf with an external
 *memory.
 * ATTENTION, for user allocated memory, deallocation should be done by users
 *externally after the program finished. The PaddleBuf won't do any allocation
 *or deallocation.
 *
 * To have the PaddleBuf allocate and manage the memory:
 * - PaddleBuf(size_t length) will allocate a memory of size `length`.
 * - Resize(size_t length) resize the memory to no less than `length`, ATTENTION
 *  if the allocated memory is larger than `length`, nothing will done.
 *
 * Usage:
 *
 * Let PaddleBuf manage the memory internally.
 * \code{cpp}
 * const int num_elements = 128;
 * PaddleBuf buf(num_elements * sizeof(float));
 * \endcode
 *
 * Or
 * \code{cpp}
 * PaddleBuf buf;
 * buf.Resize(num_elements * sizeof(float));
 * \endcode
 * Works the exactly the same.
 *
 * One can also make the `PaddleBuf` use the external memory.
 * \code{cpp}
 * PaddleBuf buf;
 * void* external_memory = new float[num_elements];
 * buf.Reset(external_memory, num_elements*sizeof(float));
 * ...
 * delete[] external_memory; // manage the memory lifetime outside.
 * \endcode
 */
class PaddleBuf {
 public:
  /** PaddleBuf allocate memory internally, and manage it.
   */
  explicit PaddleBuf(size_t length)
      : data_(new char[length]), length_(length), memory_owned_(true) {}
  /** Set external memory, the PaddleBuf won't manage it.
   */
  PaddleBuf(void* data, size_t length)
      : data_(data), length_(length), memory_owned_{false} {}
  /** Copy only available when memory is managed externally.
   */
  explicit PaddleBuf(const PaddleBuf&);

  /** Resize the memory.
   */
  void Resize(size_t length);
  /** Reset to external memory, with address and length set.
   */
  void Reset(void* data, size_t length);
  /** Tell whether the buffer is empty.
   */
  bool empty() const { return length_ == 0; }
  /** Get the data's memory address.
   */
  void* data() const { return data_; }
  /** Get the memory length.
   */
  size_t length() const { return length_; }

  ~PaddleBuf() { Free(); }
  PaddleBuf& operator=(const PaddleBuf&);
  PaddleBuf& operator=(PaddleBuf&&);
  PaddleBuf() = default;
  PaddleBuf(PaddleBuf&& other);

 private:
  void Free();
  void* data_{nullptr};  // pointer to the data memory.
  size_t length_{0};     // number of memory bytes.
  bool memory_owned_{true};
};

/** Basic input and output data structure for PaddlePredictor.
 */
struct PaddleTensor {
  PaddleTensor() = default;
  std::string name;  // variable name.
  std::vector<int> shape;
  PaddleBuf data;  // blob of data.
  PaddleDType dtype;
  std::vector<std::vector<size_t>> lod;  // Tensor+LoD equals LoDTensor
};

int PaddleDtypeSize(PaddleDType dtype);

}  // namespace paddle
