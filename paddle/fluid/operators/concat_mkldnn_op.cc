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

#include "paddle/fluid/operators/concat_op.h"
#include "paddle/fluid/platform/mkldnn_helper.h"

namespace paddle {
namespace operators {

using framework::DataLayout;
using mkldnn::memory;
using mkldnn::concat;
using mkldnn::primitive;
using mkldnn::reorder;
using mkldnn::stream;
using platform::to_void_cast;

template <typename T>
class ConcatMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");

    auto ins = ctx.MultiInput<framework::Tensor>("X");
    framework::Tensor* output = ctx.Output<framework::Tensor>("Out");
    int64_t axis = static_cast<int64_t>(ctx.Attr<int>("axis"));

    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    T* output_data = output->mutable_data<T>(ctx.GetPlace());

    PADDLE_ENFORCE(ins.size() > axis, "axis value should be smaller than \
                   ins's size.");

    memory::format output_format{memory::format::format_undef};

    std::vector<memory::primitive_desc> srcs_pd;
    std::vector<memory> srcs;

    for (auto* in : ins) {

        std::vector<int> src_tz = paddle::framework::vectorize2int(in->dims());

        auto src_md = platform::MKLDNNMemDesc(
          src_tz, platform::MKLDNNGetDataType<T>(), in->format());

        auto src_memory_pd = memory::primitive_desc(src_md, mkldnn_engine);

        const T* input_data = in->data<T>();
        auto src_memory = memory(src_memory_pd, to_void_cast<T>(input_data));

        srcs_pd.push_back(src_memory_pd);
        srcs.push_back(src_memory);

    }

    std::vector<int> dst_tz = paddle::framework::vectorize2int(output->dims());
    /* create memory descriptor for pooling without specified format
     * ('any') which lets a primitive (pooling in this case) choose
     * the memory format preferred for best performance
    */
    auto dst_md = platform::MKLDNNMemDesc(dst_tz, mkldnn::memory::f32,
                                        mkldnn::memory::format::any);

    auto concat_pd = concat::primitive_desc(dst_md,
                                static_cast<int>(axis), srcs_pd);

    auto dst_memory = memory(concat_pd.dst_primitive_desc(), output_data);

    std::vector<primitive::at> inputs;
    for (size_t i = 0; i < srcs.size(); i++) {
        inputs.push_back(srcs[i]);
    }

    auto concat_p = concat(concat_pd, inputs, dst_memory);

    output_format =
          (memory::format)dst_memory.get_primitive_desc().desc().data.format;

    // push primitive to stream and wait until it's executed
    std::vector<mkldnn::primitive> pipeline{concat_p};
    stream(stream::kind::eager).submit(pipeline).wait();

    output->set_layout(DataLayout::kMKLDNN);
    output->set_format(output_format);

  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(concat, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::ConcatMKLDNNKernel<float>);
