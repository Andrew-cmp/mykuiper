
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "data/load_data.hpp"
#include "runtime/runtime_ir.hpp"

TEST(test_net, forward_resnet18) {
  using namespace my_infer;
  RuntimeGraph graph("/home/xiaohou/Desktop/myinfer/tmp/resnet/resnet18_batch1.param", "/home/xiaohou/Desktop/myinfer/tmp/resnet/resnet18_batch1.pnnx.bin");
  graph.Build();

  int repeat_number = 2;
  for (int i = 0; i < repeat_number; ++i) {
    std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(3, 224, 224);
    input1->Fill(2.);

    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input1);

    graph.set_inputs("pnnx_input_0", inputs);
    graph.Forward(false);
    std::vector<std::shared_ptr<Tensor<float>>> outputs = graph.get_outputs("pnnx_output_0");
    ASSERT_EQ(outputs.size(), 1);

    const auto& output2 = CSVDataLoader::LoadData<float>("/home/xiaohou/Desktop/myinfer/tmp/resnet/1.csv");
    const auto& output1 = outputs.front()->data().slice(0);
    ASSERT_EQ(output1.size(), output2.size());
    for (uint32_t s = 0; s < output1.size(); ++s) {
      ASSERT_LE(std::abs(output1.at(s) - output2.at(s)), 5e-6);
    }
  }
}

