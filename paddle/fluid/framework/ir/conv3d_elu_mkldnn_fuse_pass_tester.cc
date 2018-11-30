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

#include "paddle/fluid/framework/ir/conv3d_elu_mkldnn_fuse_pass.h"

#include <gtest/gtest.h>
#include "paddle/fluid/framework/op_proto_maker.h"

namespace paddle {
namespace framework {
namespace ir {

void SetOp(ProgramDesc* prog, const std::string& type, const std::string& name,
           const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs, bool use_mkldnn = false) {
  auto* op = prog->MutableBlock(0)->AppendOp();
  op->SetType(type);
  if (type == "conv3d") {
    op->SetAttr("use_mkldnn", use_mkldnn);
    op->SetAttr("name", name);
    op->SetInput("Input", {inputs[0]});
    op->SetInput("Filter", {inputs[1]});
    op->SetInput("Bias", {inputs[2]});
  } else if (type == "elu") {
    op->SetAttr("use_mkldnn", use_mkldnn);
    op->SetInput("X", inputs);
  }
  op->SetOutput("Out", outputs);
  op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
              static_cast<int>(OpRole::kForward));
}

// a->OP0->b
// b->OP1->c
// (c, weights, bias)->conv->f
// (f)->elu->g
ProgramDesc BuildProgramDesc() {
  ProgramDesc prog;
  for (auto& v :
       std::vector<std::string>({"a", "b", "c", "weights", "bias", "f", "g",
                                 "h", "weights2", "bias2", "k", "l"})) {
    auto* var = prog.MutableBlock(0)->Var(v);
    var->SetType(proto::VarType::SELECTED_ROWS);
    if (v == "weights" || v == "bias") {
      var->SetPersistable(true);
    }
  }

  SetOp(&prog, "OP0", "op0", std::vector<std::string>({"a"}),
        std::vector<std::string>({"b"}));
  SetOp(&prog, "OP1", "op1", std::vector<std::string>({"b"}),
        std::vector<std::string>({"c"}));
  // conv+elu, both with MKL-DNN
  SetOp(&prog, "conv3d", "conv1",
        std::vector<std::string>({"c", "weights", "bias"}),
        std::vector<std::string>({"f"}), true);
  SetOp(&prog, "elu", "elu1", std::vector<std::string>({"f"}),
        std::vector<std::string>({"g"}), true);
  SetOp(&prog, "OP3", "op3", std::vector<std::string>({"g"}),
        std::vector<std::string>({"h"}));
  // conv+elu, only one with MKL-DNN
  SetOp(&prog, "conv3d", "conv2",
        std::vector<std::string>({"h", "weights2", "bias2"}),
        std::vector<std::string>({"k"}), true);
  SetOp(&prog, "elu", "elu2", std::vector<std::string>({"k"}),
        std::vector<std::string>({"l"}));

  return prog;
}

TEST(Conv3DeluFusePass, basic) {
  auto prog = BuildProgramDesc();

  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));

  auto pass = PassRegistry::Instance().Get("conv3d_elu_mkldnn_fuse_pass");

  int original_nodes_num = graph->Nodes().size();

  graph = pass->Apply(std::move(graph));

  int current_nodes_num = graph->Nodes().size();

  // Remove 3 Nodes: CONV, elu, conv3d_out
  // Add 1 Node: Convelu
  EXPECT_EQ(original_nodes_num - 2, current_nodes_num);

  // Assert conv3d_elu op in newly generated graph
  int conv3d_elu_count = 0;

  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "conv3d") {
      auto* op = node->Op();
      ASSERT_TRUE(op->HasAttr("use_mkldnn"));
      EXPECT_TRUE(boost::get<bool>(op->GetAttr("use_mkldnn")));
      // check if only "conv1" convolution is fused
      auto op_name = boost::get<std::string>(op->GetAttr("name"));
      if (op_name == "conv1") {
        ASSERT_TRUE(op->HasAttr("fuse_elu"));
        bool fuse_elu = boost::get<bool>(op->GetAttr("fuse_elu"));
        if (fuse_elu) {
          ++conv3d_elu_count;
        }
      } else if (op_name == "conv2") {
        ASSERT_FALSE(op->HasAttr("fuse_elu"));
      }
    }
  }
  EXPECT_EQ(conv3d_elu_count, 1);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(conv3d_elu_mkldnn_fuse_pass);
