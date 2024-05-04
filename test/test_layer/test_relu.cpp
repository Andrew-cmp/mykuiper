//调试用gdb -args ./test_kuiper
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../../source/layer/details/relu.hpp"
#include "data/tensor.hpp"

TEST(test_layer, forward_relu1) {
  using namespace my_infer;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(32, 224, 512);
  input->RandN();
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);

  ReluLayer relu_layer;
  const auto status = relu_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, StatusCode::kSuccess);
  for (int i = 0; i < inputs.size(); ++i) {
    std::shared_ptr<Tensor<float>> input_ = inputs.at(i);
    input_->Transform([](const float f) {
      if (f < 0) {
        return 0.f;
      } else {
        return f;
      }
    });
    std::shared_ptr<Tensor<float>> output_ = outputs.at(i);
    CHECK(input_->size() == output_->size());
    uint32_t size = input_->size();
    for (uint32_t j = 0; j < size; ++j) {
      ASSERT_EQ(output_->index(j), input_->index(j));
    }
  }
}

TEST(test_layer, forward_relu2) {
  using namespace my_infer;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 32, 128);
  input->RandN();
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);

  ReluLayer relu_layer;
  const auto status = relu_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, StatusCode::kSuccess);
  for (int i = 0; i < inputs.size(); ++i) {
    std::shared_ptr<Tensor<float>> input_ = inputs.at(i);
    input_->Transform([](const float f) {
      if (f < 0) {
        return 0.f;
      } else {
        return f;
      }
    });
    std::shared_ptr<Tensor<float>> output_ = outputs.at(i);
    CHECK(input_->size() == output_->size());
    uint32_t size = input_->size();
    for (uint32_t j = 0; j < size; ++j) {
      ASSERT_EQ(output_->index(j), input_->index(j));
    }
  }
}

TEST(test_layer, forward_relu3) {
  using namespace my_infer;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 16);
  input->RandN();
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);

  ReluLayer relu_layer;
  const auto status = relu_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, StatusCode::kSuccess);
  for (int i = 0; i < inputs.size(); ++i) {
    std::shared_ptr<Tensor<float>> input_ = inputs.at(i);
    input_->Transform([](const float f) {
      if (f < 0) {
        return 0.f;
      } else {
        return f;
      }
    });
    std::shared_ptr<Tensor<float>> output_ = outputs.at(i);
    CHECK(input_->size() == output_->size());
    uint32_t size = input_->size();
    for (uint32_t j = 0; j < size; ++j) {
      ASSERT_EQ(output_->index(j), input_->index(j));
    }
  }
}

TEST(test_layer, forward_relu4) {
  using namespace my_infer;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 15);
  input->RandN();
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);

  ReluLayer relu_layer;
  const auto status = relu_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, StatusCode::kSuccess);
  for (int i = 0; i < inputs.size(); ++i) {
    std::shared_ptr<Tensor<float>> input_ = inputs.at(i);
    input_->Transform([](const float f) {
      if (f < 0) {
        return 0.f;
      } else {
        return f;
      }
    });
    std::shared_ptr<Tensor<float>> output_ = outputs.at(i);
    CHECK(input_->size() == output_->size());
    uint32_t size = input_->size();
    for (uint32_t j = 0; j < size; ++j) {
      ASSERT_EQ(output_->index(j), input_->index(j));
    }
  }
}

