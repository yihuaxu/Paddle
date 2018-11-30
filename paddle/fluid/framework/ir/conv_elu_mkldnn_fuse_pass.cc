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

#include "paddle/fluid/framework/ir/conv_elu_mkldnn_fuse_pass.h"
#include <string>
#include <vector>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

std::unique_ptr<ir::Graph> ConvELUFusePass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  PADDLE_ENFORCE(graph.get());
  FusePassBase::Init("conv_elu_mkldnn_fuse", graph.get());

  std::string type = is_conv3d() ? "conv3d" : "conv2d";

  GraphPatternDetector gpd;
  auto* conv_input = gpd.mutable_pattern()
                         ->NewNode("conv_elu_mkldnn_fuse/conv_input")
                         ->AsInput()
                         ->assert_is_op_input(type, "Input");
  patterns::ConvELU conv_elu_pattern(gpd.mutable_pattern(),
                                     "conv_elu_mkldnn_fuse");
  conv_elu_pattern(conv_input, is_conv3d());

  int found_conv_elu_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(40) << "handle ConvELU fuse";
    GET_IR_NODE_FROM_SUBGRAPH(conv_weight, conv_weight,
                              conv_elu_pattern);                      // Filter
    GET_IR_NODE_FROM_SUBGRAPH(conv_out, conv_out, conv_elu_pattern);  // tmp
    GET_IR_NODE_FROM_SUBGRAPH(conv, conv, conv_elu_pattern);          // CONV op
    GET_IR_NODE_FROM_SUBGRAPH(elu_out, elu_out, conv_elu_pattern);    // Out
    GET_IR_NODE_FROM_SUBGRAPH(elu, elu, conv_elu_pattern);            // elu op

    FuseOptions fuse_option = FindFuseOption(*conv, *elu);
    if (fuse_option == DO_NOT_FUSE) {
      VLOG(30) << "do not perform conv+elu fuse";
      return;
    }

    // Transform Conv node into Convelu node.
    OpDesc* desc = conv->Op();
    desc->SetOutput("Output", std::vector<std::string>({elu_out->Name()}));
    desc->SetAttr("fuse_elu", true);
    GraphSafeRemoveNodes(graph.get(), {elu, conv_out});

    PADDLE_ENFORCE(subgraph.count(conv_input));
    IR_NODE_LINK_TO(conv, elu_out);

    found_conv_elu_count++;
  };

  gpd(graph.get(), handler);

  AddStatis(found_conv_elu_count);
  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(conv_elu_mkldnn_fuse_pass,
              paddle::framework::ir::ConvELUFusePass);
