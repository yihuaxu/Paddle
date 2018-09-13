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

#include "paddle/fluid/inference/analysis/analyzer.h"
#include <gtest/gtest.h>
#include <thread>  // NOLINT
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/inference/analysis/ut_helper.h"
#include "paddle/fluid/inference/api/analysis_predictor.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_pass.h"
#include "paddle/fluid/platform/profiler.h"

DEFINE_string(infer_model, "", "model path for LAC");
DEFINE_string(infer_data, "", "data file for LAC");
DEFINE_int32(batch_size, 1, "batch size.");
DEFINE_int32(burning, 0, "Burning before repeat.");
DEFINE_int32(repeat, 1, "Running the inference program repeat times.");
DEFINE_bool(test_all_data, false, "Test the all dataset in data file.");
DEFINE_int32(num_threads, 1, "Running the inference program in multi-threads.");

namespace paddle {
namespace inference {
namespace analysis {

struct DataRecord {
  std::vector<int64_t> data;
  std::vector<size_t> lod;
  // for dataset and nextbatch
  size_t batch_iter{0};
  std::vector<std::vector<size_t>> batched_lods;
  std::vector<std::vector<int64_t>> batched_datas;
  std::vector<std::vector<int64_t>> datasets;
  DataRecord() = default;
  explicit DataRecord(const std::string &path, int batch_size = 1) {
    Load(path);
    Prepare(batch_size);
    batch_iter = 0;
  }
  void Load(const std::string &path) {
    std::ifstream file(path);
    std::string line;
    int num_lines = 0;
    datasets.resize(0);
    while (std::getline(file, line)) {
      num_lines++;
      std::vector<std::string> data;
      split(line, ';', &data);
      std::vector<int64_t> words_ids;
      split_to_int64(data[1], ' ', &words_ids);
      datasets.emplace_back(words_ids);
    }
  }
  void Prepare(int bs) {
    if (bs == 1) {
      batched_datas = datasets;
      for (auto one_sentence : datasets) {
        batched_lods.push_back({0, one_sentence.size()});
      }
    } else {
      std::vector<int64_t> one_batch;
      std::vector<size_t> lod{0};
      int bs_id = 0;
      for (auto one_sentence : datasets) {
        bs_id++;
        one_batch.insert(one_batch.end(), one_sentence.begin(),
                         one_sentence.end());
        lod.push_back(lod.back() + one_sentence.size());
        if (bs_id == bs) {
          bs_id = 0;
          batched_datas.push_back(one_batch);
          batched_lods.push_back(lod);
          one_batch.clear();
          one_batch.resize(0);
          lod.clear();
          lod.resize(0);
          lod.push_back(0);
        }
      }
      if (one_batch.size() != 0) {
        batched_datas.push_back(one_batch);
        batched_lods.push_back(lod);
      }
    }
  }
  DataRecord NextBatch() {
    DataRecord data;
    data.data = batched_datas[batch_iter];
    data.lod = batched_lods[batch_iter];
    batch_iter++;
    if (batch_iter >= batched_datas.size()) {
      batch_iter = 0;
    }
    return data;
  }
  DataRecord GetBatch(size_t iter) {
    if (iter >= batched_datas.size()) {
      iter = 0;
    }
    DataRecord data;
    data.data = batched_datas[iter];
    data.lod = batched_lods[iter];
    return data;
  }
};

struct PredictStats {
  int64_t total_samples;
  int64_t total_iters;
  double total_time;
};

void GetOneBatch(std::vector<PaddleTensor> *input_slots, DataRecord *data,
                 int batch_size, size_t iter = -1) {
  auto one_batch =
      iter == (size_t)-1 ? data->NextBatch() : data->GetBatch(iter);
  PaddleTensor input_tensor;
  input_tensor.name = "word";
  input_tensor.shape.assign({static_cast<int>(one_batch.data.size()), 1});
  input_tensor.lod.assign({one_batch.lod});
  input_tensor.dtype = PaddleDType::INT64;
  TensorAssignData<int64_t>(&input_tensor, {one_batch.data});
  PADDLE_ENFORCE_EQ(batch_size, static_cast<int>(one_batch.lod.size() - 1));
  input_slots->assign({input_tensor});
}

const int64_t lac_ref_data[] = {24, 25, 25, 25, 38, 30, 31, 14, 15, 44, 24, 25,
                                25, 25, 25, 25, 44, 24, 25, 25, 25, 36, 42, 43,
                                44, 14, 15, 44, 14, 15, 44, 14, 15, 44, 38, 39,
                                14, 15, 44, 22, 23, 23, 23, 23, 23, 23, 23};

void TestLACPrediction(const std::string &model_path,
                       const std::string &data_file, const int batch_size,
                       const int repeat, bool test_all_data, int num_threads,
                       bool use_analysis = false) {
  NativeConfig config;
  config.model_dir = model_path;
  config.use_gpu = false;
  config.device = 0;
  config.specify_input_name = true;
  AnalysisConfig cfg;
  cfg.model_dir = model_path;
  cfg.use_gpu = false;
  cfg.device = 0;
  cfg.specify_input_name = true;
  cfg.enable_ir_optim = true;
  std::vector<struct PredictStats> stats;
  stats.resize(num_threads);
  DataRecord data(data_file, batch_size);
  std::vector<std::thread> threads;
  std::vector<std::shared_ptr<PaddlePredictor>> predictors;
  std::shared_ptr<PaddlePredictor> predictor;

  for (int tid = 0; tid < num_threads; ++tid) {
    std::vector<PaddleTensor> input_slots, outputs_slots;
    predictors.emplace_back(
        use_analysis
            ? CreatePaddlePredictor<AnalysisConfig,
                                    PaddleEngineKind::kAnalysis>(cfg)
            : CreatePaddlePredictor<NativeConfig, PaddleEngineKind::kNative>(
                  config));
  }

  predictor = predictors[0];

  for (int tid = 0; tid < num_threads; ++tid) {
    threads.emplace_back([&, tid]() {
      std::vector<PaddleTensor> input_slots, outputs_slots;
      GetOneBatch(&input_slots, &data, batch_size, 0);
      for (int i = 0; i < FLAGS_burning; i++) {
        predictor->Run(input_slots, &outputs_slots);
      }
      Timer timer;
      double sum = 0;
      if (test_all_data) {
        LOG(INFO) << "Total number of samples: " << data.datasets.size();
        for (int i = 0; i < repeat; i++) {
          for (size_t bid = 0; bid < data.batched_datas.size(); ++bid) {
            GetOneBatch(&input_slots, &data, batch_size, bid);
            timer.tic();
            predictors[tid]->Run(input_slots, &outputs_slots);
            sum += timer.toc();
          }
        }
        PrintTime(batch_size, repeat, num_threads, tid, sum / repeat);
        LOG(INFO) << "Average latency of each sample: "
                  << sum / repeat / data.datasets.size() << " ms";

        // save stat
        PredictStats &stat = stats[tid];
        stat.total_samples = repeat * data.batched_datas.size() * batch_size;
        stat.total_time = sum;
        stat.total_iters = repeat * data.batched_datas.size();
        return;
      }
      timer.tic();
      for (int i = 0; i < repeat; i++) {
        predictors[tid]->Run(input_slots, &outputs_slots);
      }
      sum += timer.toc();
      PrintTime(batch_size, repeat, num_threads, tid, sum / repeat);

      // save stat
      PredictStats &stat = stats[tid];
      stat.total_samples = repeat * data.batched_datas.size() * batch_size;
      stat.total_time = sum;
      stat.total_iters = repeat * data.batched_datas.size();

      // check result
      EXPECT_EQ(outputs_slots.size(), 1UL);
      auto &out = outputs_slots[0];
      size_t size = std::accumulate(out.shape.begin(), out.shape.end(), 1,
                                    [](int a, int b) { return a * b; });
      size_t batch1_size = sizeof(lac_ref_data) / sizeof(int64_t);
      PADDLE_ENFORCE_GT(size, 0);
      EXPECT_GE(size, batch1_size);
      int64_t *pdata = static_cast<int64_t *>(out.data.data());
      for (size_t i = 0; i < batch1_size; ++i) {
        EXPECT_EQ(pdata[i], lac_ref_data[i]);
      }
    });
  }

  for (int i = 0; i < num_threads; ++i) {
    threads[i].join();
  }

  // collect statistic data
  int64_t total_samples = std::accumulate(
      stats.begin(), stats.end(), 0,
      [](int64_t a, PredictStats &b) { return a + b.total_samples; });
  int64_t total_iters = std::accumulate(
      stats.begin(), stats.end(), 0,
      [](int64_t a, PredictStats &b) { return a + b.total_iters; });
  double total_time = std::accumulate(
      stats.begin(), stats.end(), 0,
      [](double a, PredictStats &b) { return a + b.total_time; });

  LOG(INFO) << "==== Predict with all " << FLAGS_num_threads
            << " threads finished ====";
  LOG(INFO) << "Total samples: " << total_samples
            << ", Total time(ms): " << total_time;
  LOG(INFO) << "Total iterations: " << total_iters
            << ", BatchSize: " << batch_size;
  LOG(INFO) << "Total QPS: "
            << total_samples * 1000 / (total_time / num_threads)
            << ", Aver QPS per thread: " << total_samples * 1000 / total_time;
  LOG(INFO) << "Average latency per iter (ms): " << total_time / total_iters;
  LOG(INFO) << "Average latency per sample (ms): "
            << total_time / total_samples;

  if (use_analysis) {
    // run once for comparion as reference
    std::vector<PaddleTensor> input_slots, outputs_slots;
    GetOneBatch(&input_slots, &data, batch_size, 0);
    predictor->Run(input_slots, &outputs_slots);
    auto &out = outputs_slots[0];
    int64_t *pdata = static_cast<int64_t *>(out.data.data());
    size_t size = std::accumulate(out.shape.begin(), out.shape.end(), 1,
                                  [](int a, int b) { return a * b; });
    auto ref_predictor =
        CreatePaddlePredictor<NativeConfig, PaddleEngineKind::kNative>(config);
    std::vector<PaddleTensor> ref_outputs_slots;
    ref_predictor->Run(input_slots, &ref_outputs_slots);
    EXPECT_EQ(ref_outputs_slots.size(), outputs_slots.size());
    auto &ref_out = ref_outputs_slots[0];
    size_t ref_size =
        std::accumulate(ref_out.shape.begin(), ref_out.shape.end(), 1,
                        [](int a, int b) { return a * b; });
    EXPECT_EQ(size, ref_size);
    int64_t *pdata_ref = static_cast<int64_t *>(ref_out.data.data());
    for (size_t i = 0; i < size; ++i) {
      EXPECT_EQ(pdata_ref[i], pdata[i]);
    }

    AnalysisPredictor *analysis_predictor =
        dynamic_cast<AnalysisPredictor *>(predictor.get());
    auto &fuse_statis = analysis_predictor->analysis_argument()
                            .Get<std::unordered_map<std::string, int>>(
                                framework::ir::kFuseStatisAttr);
    for (auto &item : fuse_statis) {
      LOG(INFO) << "fused " << item.first << " " << item.second;
    }
    int num_ops = 0;
    for (auto &node :
         analysis_predictor->analysis_argument().main_dfg->nodes.nodes()) {
      if (node->IsFunction()) {
        ++num_ops;
      }
    }
    LOG(INFO) << "has num ops: " << num_ops;
    ASSERT_TRUE(fuse_statis.count("fc_fuse"));
    ASSERT_TRUE(fuse_statis.count("fc_gru_fuse"));
    EXPECT_EQ(fuse_statis.at("fc_fuse"), 1);
    EXPECT_EQ(fuse_statis.at("fc_gru_fuse"), 4);
    EXPECT_EQ(num_ops, 11);
  }
}

TEST(Analyzer_LAC, native) {
  LOG(INFO) << "LAC with native";
  TestLACPrediction(FLAGS_infer_model, FLAGS_infer_data, FLAGS_batch_size,
                    FLAGS_repeat, FLAGS_test_all_data, FLAGS_num_threads);
}

TEST(Analyzer_LAC, analysis) {
  LOG(INFO) << "LAC with analysis";
  TestLACPrediction(FLAGS_infer_model, FLAGS_infer_data, FLAGS_batch_size,
                    FLAGS_repeat, FLAGS_test_all_data, 1, true);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
