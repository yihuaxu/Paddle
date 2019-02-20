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

#include "paddle/fluid/framework/ir/multi_mul_fuse_pass.h"
#include <algorithm>
#include <functional>
#include <string>
#include <vector>
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/operators/math/cpu_vec.h"
#include "paddle/fluid/operators/strided_memcpy.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

void MultiMulFusePass::GetSpeicalOpNodes(const std::vector<Node*>& nodes,
                                         std::string type,
                                         std::vector<Node*>* dst_nodes) const {
  for (auto it = nodes.begin(); it != nodes.end(); ++it) {
    auto node = *it;
    if (node->IsOp() && (!node->Name().compare(type))) {
      dst_nodes->push_back(node);
    }
  }
}

inline int GetMulInputNodeID(const Node* node, std::string type = "X") {
  int id = -1;
  for (auto it = node->inputs.begin(); it != node->inputs.end(); it++) {
    if (0 == node->Op()->Input(type)[0].compare((*it)->Name())) {
      id = (*it)->id();
    }
  }

  return id;
}

inline Node* GetMulInputNode(const Node* node, std::string type = "X") {
  for (auto it = node->inputs.begin(); it != node->inputs.end(); it++) {
    if (0 == node->Op()->Input(type)[0].compare((*it)->Name())) {
      return *it;
    }
  }

  return nullptr;
}

inline Node* GetMulOutNode(const Node* node, std::string type = "Out") {
  for (auto it = node->outputs.begin(); it != node->outputs.end(); it++) {
    if (0 == node->Op()->Output(type)[0].compare((*it)->Name())) {
      return *it;
    }
  }

  return nullptr;
}

inline void BreakNodes(Node* node, bool is_out = true) {
  if (is_out) {
    for (auto it_out = node->outputs.begin(); it_out != node->outputs.end();) {
      for (auto it_in = (*it_out)->inputs.begin();
           it_in != (*it_out)->inputs.end();) {
        if (*it_in == node) {
          it_in = (*it_out)->inputs.erase(it_in);
        } else {
          it_in++;
        }
      }
      it_out = node->outputs.erase(it_out);
    }
  } else {
    for (auto it_in = node->inputs.begin(); it_in != node->inputs.end();) {
      for (auto it_out = (*it_in)->outputs.begin();
           it_out != (*it_in)->outputs.end();) {
        if (*it_out == node) {
          it_out = (*it_in)->outputs.erase(it_in);
        } else {
          it_out++;
        }
      }
      it_in = node->inputs.erase(it_in);
    }
  }
}

void MultiMulFusePass::SortMulOperators(
    const std::vector<Node*>& nodes,
    std::unordered_map<int, std::vector<Node*>>& mul_nodes_map) const {
  std::vector<Node*> mul_nodes;
  GetSpeicalOpNodes(nodes, "mul", &mul_nodes);

  std::sort(mul_nodes.begin(), mul_nodes.end(), [](Node* node1, Node* node2) {
    return GetMulInputNodeID(node1) < GetMulInputNodeID(node2);
  });

  int input_id = -1;
  auto it = mul_nodes.begin();
  auto it_start = it;
  do {
    if (it == mul_nodes.end() || input_id != GetMulInputNodeID(*it)) {
      std::vector<Node*> nodes;
      while (it_start != it) {
        nodes.push_back(*it_start);
        it_start++;
      }
      if (input_id != -1) {
        mul_nodes_map.insert(std::make_pair(input_id, nodes));
      }
      if (it != mul_nodes.end()) {
        input_id = GetMulInputNodeID(*it);
      }
    }
  } while (it++ != mul_nodes.end());
}

// Create the variable Node
Node* MultiMulFusePass::CreateVarNode(const std::unique_ptr<ir::Graph>& graph,
                                      Scope* scope, std::string name, DDim dims,
                                      bool persistable) const {
  Node* node = nullptr;
  VarDesc desc(patterns::PDNodeName(name_scope_, name));
  if (persistable) {
    desc.SetPersistable(true);
  }
  node = graph->CreateVarNode(&desc);

  if (persistable) {
    // If the variable is persistable, then it need be allocate the memory and
    // set up the dimsion.
    auto* tensor = scope->Var(node->Name())->GetMutable<LoDTensor>();
    tensor->Resize(dims);
    tensor->mutable_data<float>(platform::CPUPlace());
  }

  return node;
}

bool MultiMulFusePass::IsEnableReplace(Scope* scope,
                                       std::vector<Node*>& nodes) const {
  if (nodes.size() <= 1) {
    return false;
  }
  auto it = nodes.begin();
  if (it != nodes.end()) {
    auto* init_tensor =
        scope->FindVar((*it)->Op()->Input("Y")[0])->GetMutable<LoDTensor>();
    std::vector<int> init_tz =
        paddle::framework::vectorize2int(init_tensor->dims());

    while (++it != nodes.end()) {
      auto tensor =
          scope->Var((*it)->Op()->Input("Y")[0])->GetMutable<LoDTensor>();
      std::vector<int> tz = paddle::framework::vectorize2int(tensor->dims());

      if (tz != init_tz) {
        return false;
      }
    }
  }

  it = nodes.begin();
  if (it != nodes.end()) {
    int init_x_num_col_dims =
        boost::get<int>((*it)->Op()->GetAttr("x_num_col_dims"));
    int init_y_num_col_dims =
        boost::get<int>((*it)->Op()->GetAttr("y_num_col_dims"));
    while (++it != nodes.end()) {
      int x_num_col_dims =
          boost::get<int>((*it)->Op()->GetAttr("x_num_col_dims"));
      int y_num_col_dims =
          boost::get<int>((*it)->Op()->GetAttr("y_num_col_dims"));

      if ((init_x_num_col_dims != x_num_col_dims) ||
          (init_y_num_col_dims != y_num_col_dims)) {
        return false;
      }
    }
  }

  return true;
}

Node* MultiMulFusePass::UpdateWeightNode(
    const std::unique_ptr<ir::Graph>& graph, Scope* scope,
    std::vector<Node*>& nodes) const {
  Node* out_node = nullptr;
  auto it = nodes.begin();
  Node* weight_node = GetMulInputNode(*it, "Y");
  int axis = boost::get<int>((*it)->Op()->GetAttr("y_num_col_dims"));

  auto weight_tensor =
      scope->FindVar(weight_node->Name())->GetMutable<LoDTensor>();

  // Update the dimsion information
  auto dims = weight_tensor->dims();
  framework::set(dims, axis, framework::get(dims, axis) * nodes.size());
  std::string name = weight_node->Name();
  out_node = CreateVarNode(graph, scope, name, dims, true);
  auto tensor = scope->FindVar(out_node->Name())->GetMutable<LoDTensor>();

  // Update the data, such as weights or biases.
  for (size_t index = 0, out_offset = 0; index < nodes.size(); index++) {
    auto input_tensor =
        scope->Var(nodes[index]->Op()->Input("Y")[0])->GetMutable<LoDTensor>();
    auto out_stride = framework::stride_numel(dims);
    auto in_stride = framework::stride_numel(input_tensor->dims());

    paddle::operators::StridedNumelCopyWithAxis<float>(
        paddle::platform::CPUDeviceContext(), axis,
        tensor->data<float>() + out_offset, out_stride,
        reinterpret_cast<float*>(input_tensor->data<float>()), in_stride,
        in_stride[axis]);
    out_offset += in_stride[axis];
  }

  return out_node;
}

Node* MultiMulFusePass::CreateMulSplitNode(
    const std::unique_ptr<ir::Graph>& graph, Scope* scope,
    std::vector<Node*>& nodes) const {
  Node* out_node = nullptr;
  auto it = nodes.begin();
  Node* old_weight_node = GetMulInputNode(*it, "Y");

  std::string name = (*it)->Op()->Output("Out")[0];
  out_node = CreateVarNode(graph, scope, name);

  auto weight_node = UpdateWeightNode(graph, scope, nodes);

  OpDesc desc;
  int axis = boost::get<int>((*it)->Op()->GetAttr("x_num_col_dims"));
  std::vector<std::string> output_names;
  do {
    output_names.push_back((*it)->Op()->Output("Out")[0]);
    it++;
  } while (it != nodes.end());

  // Configure the Input and output nodes.
  desc.SetInput("X", std::vector<std::string>({out_node->Name()}));
  desc.SetOutput("Out", std::vector<std::string>(output_names));
  desc.SetType("split");
  desc.SetAttr("num", static_cast<int>(nodes.size()));
  desc.SetAttr("axis", static_cast<int>(axis));

  // To create convolution operator node.
  auto split_node = graph->CreateOpNode(&desc);

  // Link variable and operator nodes.
  IR_NODE_LINK_TO(out_node, split_node);
  IR_NODE_LINK_TO(weight_node, split_node);
  for (auto it = nodes.begin(); it != nodes.end(); it++) {
    IR_NODE_LINK_TO(split_node, (*it)->outputs[0]);
  }

  // To remove all remnant nodes.
  std::unordered_set<const Node*> remove_nodes;
  std::for_each(nodes.begin() + 1, nodes.end(), [&](Node* node) {
    std::for_each(node->inputs.begin(), node->inputs.end(),
                  [&](Node* input_node) {
                    if (input_node->Name() == node->Op()->Input("Y")[0]) {
                      remove_nodes.insert(input_node);
                    }
                  });
  });
  remove_nodes.insert(std::make_move_iterator(nodes.begin() + 1),
                      std::make_move_iterator(nodes.end()));

  for (auto it = nodes.begin(); it != nodes.end(); it++) {
    BreakNodes(*it);
  }

  GraphSafeRemoveNodes(graph.get(), {old_weight_node});

  it = nodes.begin();
  (*it)->Op()->SetInput("Y", {weight_node->Name()});
  (*it)->Op()->SetOutput("Out", {out_node->Name()});
  IR_NODE_LINK_TO(weight_node, (*it));
  IR_NODE_LINK_TO((*it), out_node);

  GraphSafeRemoveNodes(graph.get(), remove_nodes);

  return out_node;
}

void MultiMulFusePass::ReplaceMultiMulNodes(
    const std::unique_ptr<ir::Graph>& graph, Scope* scope,
    std::unordered_map<int, std::vector<Node*>>& mul_nodes_map) const {
  for (auto it = mul_nodes_map.begin(); it != mul_nodes_map.end(); it++) {
    if (IsEnableReplace(scope, it->second)) {
      CreateMulSplitNode(graph, scope, it->second);
    }
  }
}

std::unique_ptr<ir::Graph> MultiMulFusePass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  PADDLE_ENFORCE(graph.get());
  FusePassBase::Init(name_scope_, graph.get());

  auto* scope = param_scope();
  PADDLE_ENFORCE(scope);

  // To obtain all operator nodes.
  std::vector<Node*> nodes = TopologySortOperations(*graph);

  std::unordered_map<int, std::vector<Node*>> mul_nodes_map;

  SortMulOperators(nodes, mul_nodes_map);

  ReplaceMultiMulNodes(graph, scope, mul_nodes_map);

  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(multi_mul_fuse_pass, paddle::framework::ir::MultiMulFusePass);
