// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <unordered_map>
#include <vector>

namespace paddle {
namespace train {

using paddle::framework::proto::VarType;

template <typename T>
constexpr paddle::PaddleDType GetPaddleDType();

template <>
constexpr paddle::PaddleDType GetPaddleDType<int64_t>() {
  return paddle::PaddleDType::INT64;
}

template <>
constexpr paddle::PaddleDType GetPaddleDType<float>() {
  return paddle::PaddleDType::FLOAT32;
}

bool IsPersistable(const paddle::framework::VarDesc* var) {
  if (var->Persistable() &&
      var->GetType() != paddle::framework::proto::VarType::FEED_MINIBATCH &&
      var->GetType() != paddle::framework::proto::VarType::FETCH_LIST &&
      var->GetType() != paddle::framework::proto::VarType::RAW) {
    return true;
  }
  return false;
}

}  // namespace train
}  // namespace paddle
