//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <time.h>
#include <fstream>
#include <iostream>
#include <vector>

#include "gflags/gflags.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle_api.h"

#include "tester_helper.h"

DEFINE_string(train_data, "", "data file");
DEFINE_int32(batch_size, 1, "batch size");
DECLARE_bool(profile);
DEFINE_int32(iterations, 0, "number of batches to process");
DEFINE_int32(epochs, 0, "number of batches to process");

namespace paddle {
namespace train {

void ReadBinaryFile(const std::string &filename, std::string *contents) {
  std::ifstream fin(filename, std::ios::in | std::ios::binary);
  PADDLE_ENFORCE(static_cast<bool>(fin), "Cannot open file %s", filename);
  fin.seekg(0, std::ios::end);
  contents->clear();
  contents->resize(fin.tellg());
  fin.seekg(0, std::ios::beg);
  fin.read(&(contents->at(0)), contents->size());
  fin.close();
}

std::unique_ptr<paddle::framework::ProgramDesc> Load(
    paddle::framework::Executor *executor, const std::string &model_filename) {
  VLOG(3) << "loading model from " << model_filename;
  std::string program_desc_str;
  ReadBinaryFile(model_filename, &program_desc_str);

  std::unique_ptr<paddle::framework::ProgramDesc> main_program(
      new paddle::framework::ProgramDesc(program_desc_str));
  return main_program;
}

template <typename T>
class TensorReader {
 public:
  TensorReader(std::ifstream &file, size_t beginning_offset,
               std::vector<int> shape, std::string name)
      : file_(file), position(beginning_offset), shape_(shape), name_(name) {
    numel = std::accumulate(shape_.begin(), shape_.end(), size_t{1},
                            std::multiplies<size_t>());
  }

  PaddleTensor NextBatch() {
    PaddleTensor tensor;
    tensor.name = name_;
    tensor.shape = shape_;
    tensor.dtype = GetPaddleDType<T>();
    tensor.data.Resize(numel * sizeof(T));

    file_.seekg(position);
    file_.read(static_cast<char *>(tensor.data.data()), numel * sizeof(T));
    position = file_.tellg();

    if (file_.eof()) LOG(ERROR) << name_ << ": reached end of stream";
    if (file_.fail())
      throw std::runtime_error(name_ + ": failed reading file.");

    return tensor;
  }

 protected:
  std::ifstream &file_;
  size_t position;
  std::vector<int> shape_;
  std::string name_;
  size_t numel;
};

void SetInput(std::vector<std::vector<PaddleTensor>> *inputs,
              int32_t batch_size = FLAGS_batch_size) {
  std::ifstream file(FLAGS_train_data, std::ios::binary);
  if (!file) {
    LOG(ERROR) << "Couldn't open file: " << FLAGS_train_data;
  }
  assert(file != NULL);

  int64_t total_images{0};
  file.read(reinterpret_cast<char *>(&total_images), sizeof(total_images));
  LOG(INFO) << "Total images in file: " << total_images;

  std::vector<int> image_batch_shape{batch_size, 3, 224, 224};
  std::vector<int> label_batch_shape{batch_size, 1};
  auto images_offset_in_file = static_cast<size_t>(file.tellg());
  auto labels_offset_in_file =
      images_offset_in_file + sizeof(float) * total_images * 3 * 224 * 224;

  TensorReader<float> image_reader(file, images_offset_in_file,
                                   image_batch_shape, "image");
  TensorReader<int64_t> label_reader(file, labels_offset_in_file,
                                     label_batch_shape, "label");

  auto iterations_max = total_images / batch_size;
  auto iterations = iterations_max;
  if (FLAGS_iterations > 0 && FLAGS_iterations < iterations_max) {
    iterations = FLAGS_iterations;
  }
  for (auto i = 0; i < iterations; i++) {
    auto images = image_reader.NextBatch();
    auto labels = label_reader.NextBatch();
    inputs->emplace_back(
        std::vector<PaddleTensor>{std::move(images), std::move(labels)});
  }
}

}  // namespace train
}  // namespace paddle

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  paddle::framework::InitDevices(false);

  const auto cpu_place = paddle::platform::CPUPlace();

  // read data from file and prepare batches with test data
  std::vector<std::vector<paddle::PaddleTensor>> inputs;
  paddle::train::SetInput(&inputs);

  paddle::framework::Executor executor(cpu_place);
  paddle::framework::Scope scope;
  auto startup_program = paddle::train::Load(&executor, "startup_program");
  auto train_program = paddle::train::Load(&executor, "main_program");

  std::string loss_name = "";
  for (auto op_desc : train_program->Block(0).AllOps()) {
    if (op_desc->Type() == "mean") {
      loss_name = op_desc->Output("Out")[0];
      break;
    }
  }

  PADDLE_ENFORCE_NE(loss_name, "", "loss not found");

  // init all parameters
  executor.Run(*startup_program, &scope, 0);

  // prepare data
  auto x_var = scope.Var("data");
  auto x_tensor = x_var->GetMutable<paddle::framework::LoDTensor>();
  x_tensor->Resize({FLAGS_batch_size, 3, 224, 224});

  auto x_data = x_tensor->mutable_data<float>(cpu_place);
  auto loss_var = scope.Var(loss_name);

  auto y_var = scope.Var("label");
  auto y_tensor = y_var->GetMutable<paddle::framework::LoDTensor>();
  y_tensor->Resize({FLAGS_batch_size, 1});
  auto y_data = y_tensor->mutable_data<int64_t>(cpu_place);

  constexpr auto clk_coeff = 1000.0 / CLOCKS_PER_SEC;

  paddle::platform::ProfilerState pf_state;
  pf_state = paddle::platform::ProfilerState::kCPU;
  paddle::platform::EnableProfiler(pf_state);

  for (unsigned int epoch = 0; epoch < FLAGS_epochs; epoch++) {
    for (unsigned int iter = 0; iter < FLAGS_iterations; iter++) {
      memcpy(static_cast<void *>(x_data), inputs[iter][0].data.data(),
             inputs[iter][0].data.length());
      memcpy(static_cast<void *>(y_data), inputs[iter][1].data.data(),
             inputs[iter][1].data.length());
      clock_t t1 = clock();
      executor.Run(*train_program, &scope, 0, false, true, {}, true);
      clock_t t2 = clock();
      std::cout
          << "pass: " << epoch << " step: " << iter << " loss: "
          << loss_var->Get<paddle::framework::LoDTensor>().data<float>()[0]
          << " duration: " << (t2 - t1) * clk_coeff << " ms" << std::endl;
    }
  }

  paddle::platform::DisableProfiler(paddle::platform::EventSortingKey::kTotal,
                                    "run_paddle_op_profiler");
  return 0;
}
