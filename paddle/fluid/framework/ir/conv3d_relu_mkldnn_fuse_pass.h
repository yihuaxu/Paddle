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

#include "paddle/fluid/framework/ir/conv_relu_mkldnn_fuse_pass.h"

namespace paddle {
namespace framework {
namespace ir {

/*
 * Fuse the CONV3D and ReLU to a Conv3DReLUOp.
 */
class Conv3DReLUFusePass : public ConvReLUFusePass {
 public:
  bool is_conv3d() const override { return true; }
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
