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

template <typename T>
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

    const T* input_data = x_matrix.data<T>();

    std::vector<int> src_tz = paddle::framework::vectorize2int(x_matrix.dims());
    std::vector<int> dst_tz = paddle::framework::vectorize2int(
        {x_matrix.dims()[0], y_matrix.dims()[1]});

    auto transition_dims = y_matrix.dims();
    Tensor weight_transpose;
    paddle::framework::DDim dims(
        paddle::framework::make_dim(transition_dims[1], transition_dims[0]));
    T* weights_data =
        weight_transpose.mutable_data<T>(dims, platform::CPUPlace());

    // Do the transpose for weight tensor
    std::vector<int> axis = {1, 0};
    math::Transpose<platform::CPUDeviceContext, T, 2> trans;
    trans(dev_ctx, y_matrix, &weight_transpose, axis);

    std::vector<int> weights_tz =
        paddle::framework::vectorize2int(weight_transpose.dims());

    auto src_md = platform::MKLDNNMemDesc(
        src_tz, platform::MKLDNNGetDataType<T>(), mkldnn::memory::format::nc);
    auto weights_md =
        platform::MKLDNNMemDesc(weights_tz, platform::MKLDNNGetDataType<T>(),
                                mkldnn::memory::format::nc);
    auto dst_md = platform::MKLDNNMemDesc(
        dst_tz, platform::MKLDNNGetDataType<T>(), mkldnn::memory::format::nc);

    auto desc = mkldnn::inner_product_forward::desc(
        mkldnn::prop_kind::forward_inference, src_md, weights_md, dst_md);

    auto pd =
        mkldnn::inner_product_forward::primitive_desc(desc, mkldnn_engine);

    auto output_data =
        z->mutable_data<T>(ctx.GetPlace(), paddle::memory::Allocator::kDefault,
                           pd.dst_primitive_desc().get_size());

    auto dst_memory =
        mkldnn::memory({dst_md, mkldnn_engine},
                       paddle::platform::to_void_cast<T>(output_data));
    auto src_memory = mkldnn::memory(
        {src_md, mkldnn_engine}, paddle::platform::to_void_cast<T>(input_data));
    auto weights_memory =
        mkldnn::memory({weights_md, mkldnn_engine},
                       paddle::platform::to_void_cast<T>(weights_data));

    auto forward = mkldnn::inner_product_forward(pd, src_memory, weights_memory,
                                                 dst_memory);

    std::vector<mkldnn::primitive> pipeline = {forward};
    mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();

    z->set_layout(DataLayout::kMKLDNN);
    z->set_format(z_dim.size() != 2
                      ? platform::MKLDNNFormatForSize(z_dim.size(), x->format())
                      : (mkldnn::memory::format)pd.dst_primitive_desc()
                            .desc()
                            .data.format);

    if (z_dim.size() != 2) {
      z->Resize(z_dim);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(mul, MKLDNN, ::paddle::platform::CPUPlace,
                   paddle::operators::MulMKLDNNKernel<float>);
