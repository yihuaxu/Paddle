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

#include <string>
#include <unordered_map>
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

namespace paddle {
namespace framework {
namespace ir {

/*
 * Fuse Mult Mul operators to a Mul and Split.
 */
class MultiMulFusePass : public FusePassBase {
 public:
  virtual ~MultiMulFusePass() {}

 protected:
  void GetSpeicalOpNodes(const std::vector<Node*>& nodes,
                         std::string type,  // NOLINT
                         std::vector<Node*>* dst_nodes) const;
  void SortMulOperators(const std::vector<Node*>& nodes,
                        std::unordered_map<int, std::vector<Node*>>&
                            mul_nodes_map) const;  // NOLINT
  void ReplaceMultiMulNodes(const std::unique_ptr<ir::Graph>& graph,
                            Scope* scope,
                            std::unordered_map<int, std::vector<Node*>>&
                                mul_nodes_map) const;  // NOLINT
  bool IsEnableReplace(Scope* scope,
                       std::vector<Node*>& nodes) const;  // NOLINT
  Node* CreateVarNode(const std::unique_ptr<ir::Graph>& graph, Scope* scope,
                      std::string name, DDim dims = make_ddim({1}),
                      bool persistable = false) const;
  Node* CreateMulSplitNode(const std::unique_ptr<ir::Graph>& graph,
                           Scope* scope,
                           std::vector<Node*>& nodes) const;  // NOLINT
  Node* UpdateWeightNode(const std::unique_ptr<ir::Graph>& graph, Scope* scope,
                         std::vector<Node*>& nodes) const;  // NOLINT
  std::unique_ptr<ir::Graph> ApplyImpl(std::unique_ptr<ir::Graph> graph) const;
  const std::string name_scope_{"multi_mul_fuse"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
