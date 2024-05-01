#include "layer/abstract/layer.hpp"


namespace my_infer{

StatusCode Layer<float>::Forward(){
    LOG_IF(FATAL,this->runtime_operator_.expired())<< "Runtime operator is expired or nullptr";

    const auto& runtime_operator = this->runtime_operator_.lock();

    std::vector<std::shared_ptr<Tensor<float>>> layer_input_datas;
    for(const auto& input_operand_data:runtime_operator->input_operands_seq ){
        if(input_operand_data == nullptr){
            return StatusCode::kFunctionNotImplement;
        }
        std::copy(input_operand_data->datas.begin(),input_operand_data->datas.end(),std::back_inserter(layer_input_datas));
    }

    if(layer_input_datas.empty()){
        LOG(ERROR) << runtime_operator->name << " Layer input data is empty";
        return StatusCode::kInferInputsEmpty; 
    }

    for (sftensor layer_input_data : layer_input_datas) {
        if (layer_input_data == nullptr || layer_input_data->empty()) {
            LOG(ERROR) << "Layer input data is empty";
            return StatusCode::kInferInputsEmpty;
        }
    }

    const std::shared_ptr<RuntimeOperand>& output_operand_datas = runtime_operator->output_operands;
    if (output_operand_datas == nullptr || output_operand_datas->datas.empty()) {
        LOG(ERROR) << "Layer output data is empty";
        return StatusCode::kInferOutputsEmpty;
    }
    StatusCode status =
        runtime_operator->layer->Forward(layer_input_datas, output_operand_datas->datas);
    return status;

}

void Layer<float>::set_runtime_operator(const std::shared_ptr<RuntimeOperator>& runtime_operator)
{
    CHECK(runtime_operator != nullptr);
    this->runtime_operator_=runtime_operator;
}
const std::vector<std::shared_ptr<Tensor<float>>>& Layer<float>::weights() const {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet! This is base layer";
}

const std::vector<std::shared_ptr<Tensor<float>>>& Layer<float>::bias() const {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet! This is base layer";
}

void Layer<float>::set_bias(const std::vector<float>& bias) {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet! This is base layer";
}

void Layer<float>::set_bias(const std::vector<std::shared_ptr<Tensor<float>>>& bias) {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet! This is base layer";
}

void Layer<float>::set_weights(const std::vector<float>& weights) {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet! This is base layer";
}

void Layer<float>::set_weights(const std::vector<std::shared_ptr<Tensor<float>>>& weights) {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet! This is base layer";
}
StatusCode Layer<float>::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                 std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet! This is base layer";
  return StatusCode::kFunctionNotImplement;
}

// StatusCode Layer<float>::Forward(){
//     LOG_IF(FATAL, this->runtime_operator_.expired())<<"runtime operator is expired or nullptr";
//     const auto & runtime_operator = this->runtime_operator_.lock();

//     ///直接用引用是不是更好一点？
//     std::vector<std::shared_ptr<Tensor<float>>> layer_input_datas;

//     for(const auto& input_operand_data:runtime_operator_->input_operands_seq){
//         if(input_operand_data == nullptr){
//             return StatusCode::kInferInputsEmpty;
//         }
//         std::copy(input_operand_data->data.begin(),input_operand_data->data.end(),
//                   std::back_inserter(layer_input_datas));
//     }

//     if(layer_input_datas.empty()){
//         LOG(ERROR) << runtime_operator->name << "where is Layer input data?";
//         return StatusCode::kInferInputsEmpty;
//     }

//     for(const auto& layer_input_data:layer_input_datas){
//         if(layer_input_data == nullptr||layer_input_data->empty()){
//             LOG(ERROR) <<"Layer someone input data is empty";
//         }
//     }

//     const std::shared_ptr<RuntimeOperandBase>& output_operands = runtime_operator->output_operands;
//     if(output_operands == nullptr || output_operands->datas.empty()){
        
//         LOG(ERROR) << "Layer output data is empty or Layer output do not shenqingkongjian";
//         return StatusCode::kInferOutputsEmpty;
//     }

//     StatusCode status=
//         runtime_operator->layer->Forward(layer_input_datas,output_operands->datas);

//     return status;
// }
// void Layer<float>::set_runtime_operator(const std::shared_ptr<RuntimeOperator>& runtime_operator){
//     CHECK(runtime_operator != nullptr);
//     this->runtime_operator_=runtime_operator;
// }
}