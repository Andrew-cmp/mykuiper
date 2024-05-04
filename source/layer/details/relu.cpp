#include "relu.hpp"
#include "layer/abstract/layer_factor.hpp"

namespace my_infer{
StatusCode ReluLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                              std::vector<std::shared_ptr<Tensor<float>>>& outputs){
    if(inputs.empty()){
        LOG(ERROR) << "The input tensor array in the relu layer is empty";
        return StatusCode::kInferInputsEmpty;
    }
    if (inputs.size() != outputs.size()) {
        LOG(ERROR) << "The input and output tensor array size of the relu layer do not match";
        return StatusCode::kInferDimMismatch;
    }
    //先对每个batch检查
    const int32_t& batch = inputs.size();
    for(uint32_t i = 0;i < batch;i++){
        const auto& input_data = inputs.at(i);
        auto & output_data = outputs.at(i);
        if (input_data == nullptr || input_data->empty()) {
            LOG(ERROR)
                << "The input tensor array in the relu layer has an empty tensor "
                << i << " th";
            return StatusCode::kInferInputsEmpty;
        }
        if (output_data != nullptr && !output_data->empty()) {
            if (input_data->shapes() != output_data->shapes()) {
                LOG(ERROR) << "The input and output tensor shapes of the relu layer do not match " << i << " th";
                return StatusCode::kInferDimMismatch;

            }
        }
    }
      for (uint32_t i = 0; i < batch; ++i) {
        const std::shared_ptr<Tensor<float>> &input = inputs.at(i);
        CHECK(input == nullptr || !input->empty())
                << "The input tensor array in the relu layer has an empty tensor " << i
                << " th";
        std::shared_ptr<Tensor<float>> output = outputs.at(i);
        if (output == nullptr || output->empty()) {
            LOG(ERROR)
              << "The output tensor array in the relu layer has an empty tensor "
                << i << " th";
            output = std::make_shared<Tensor<float>>(input->shapes());
            outputs.at(i) = output;
        }
        CHECK(output->shapes() == input->shapes())
                << "The input and output tensor shapes of the relu layer do not match "
                << i << " th";
        for (uint32_t j = 0; j < input->size(); ++j) {
            float value = input->index(j);
            output->index(j) = value > 0.f ? value : 0.f;
        }
    }



    return StatusCode::kSuccess;
}

StatusCode ReluLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                     std::shared_ptr<Layer<float>>& relu_layer){
    if(!op){
        LOG(ERROR) << "The relu operator paramter in the layer is null pointer";
        return StatusCode::kParseNullOperator;
    }
    relu_layer = std::make_shared<ReluLayer>();
    return StatusCode::kSuccess;
}
LayerRegistererWrapper kReluCreateInstance(ReluLayer::CreateInstance,"nn.ReLU");
}