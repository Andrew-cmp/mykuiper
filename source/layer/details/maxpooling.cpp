#include "maxpooling.hpp"
#include "layer/abstract/layer_factor.hpp"
#include "data/tensor.hpp"
#include "runtime/runtime_ir.hpp"
namespace my_infer{

MaxPoolingLayer::MaxPoolingLayer(uint32_t padding_h, uint32_t padding_w, uint32_t pooling_size_h, uint32_t pooling_size_w, uint32_t stride_h, uint32_t stride_w)
    : NonParamLayer("MaxPooling"),
      padding_h_(padding_h),
      padding_w_(padding_w),
      pooling_size_h_(pooling_size_h),
      pooling_size_w_(pooling_size_w),
      stride_h_(stride_h),
      stride_w_(stride_w) {}


StatusCode MaxPoolingLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs, std::vector<std::shared_ptr<Tensor<float>>> &outputs)
{
    
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the max pooling layer is empty";
    return StatusCode::kInferInputsEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR)
        << "The input and output tensor array size of the max pooling layer "
           "do not match";
    return StatusCode::kInferDimMismatch;
  }

  const uint32_t batch = inputs.size();
  const uint32_t pooling_h = pooling_size_h_;
  const uint32_t pooling_w = pooling_size_w_;
  if (!stride_h_ || !stride_w_) {
    LOG(ERROR) << "The stride parameter is set incorrectly. It must always be "
                  "greater than 0";
    return StatusCode::kInferParameterError;
  }

  for (uint32_t i = 0; i < batch; ++i) {
    const std::shared_ptr<ftensor>& input_data = inputs.at(i);
    if (input_data == nullptr || input_data->empty()) {
      LOG(ERROR) << "The input tensor array in the max pooling layer has an "
                    "empty tensor "
                 << i << "th";
      return StatusCode::kInferInputsEmpty;
    } else {
      uint32_t input_h = input_data->rows();
      uint32_t input_w = input_data->cols();
      uint32_t output_h = uint32_t(std::floor(
          (int(input_h) - int(pooling_h) + 2 * padding_h_) / stride_h_ + 1));
      uint32_t output_w = uint32_t(std::floor(
          (int(input_w) - int(pooling_w) + 2 * padding_w_) / stride_w_ + 1));
      if (!output_w || !output_h) {
        LOG(ERROR) << "The output size of tensor " << i << "th"
                   << " in the max pooling layer is less than zero";
        return StatusCode::kInferOutputsEmpty;
      } else {
        const std::shared_ptr<ftensor>& output_data = outputs.at(i);
        if (output_data != nullptr && !output_data->empty()) {
          if (output_data->rows() != output_h ||
              output_data->cols() != output_w) {
            LOG(ERROR) << "The output tensor array in the max pooling layer "
                          "has an incorrectly sized tensor "
                       << i << "th";
            return StatusCode::kInferOutputsEmpty;
          }
        }
      }
    }
  }

  for (uint32_t i = 0; i < batch; ++i) {
    const std::shared_ptr<Tensor<float>>& input_data = inputs.at(i);
    CHECK(input_data == nullptr || !input_data->empty())
        << "The input tensor array in the max pooling layer has an "
           "empty tensor "
        << i << "th";

    const uint32_t input_h = input_data->rows();
    const uint32_t input_w = input_data->cols();
    const uint32_t input_padded_h = input_data->rows() + 2 * padding_h_;
    const uint32_t input_padded_w = input_data->cols() + 2 * padding_w_;

    const uint32_t input_c = input_data->channels();

    const uint32_t output_h = uint32_t(
        std::floor((int(input_padded_h) - int(pooling_h)) / stride_h_ + 1));
    const uint32_t output_w = uint32_t(
        std::floor((int(input_padded_w) - int(pooling_w)) / stride_w_ + 1));

    std::shared_ptr<Tensor<float>> output_data = outputs.at(i);
    if (output_data == nullptr || output_data->empty()) {
      output_data =
          std::make_shared<Tensor<float>>(input_c, output_h, output_w);
      outputs.at(i) = output_data;
    }

    CHECK(output_data->rows() == output_h && output_data->cols() == output_w &&
          output_data->channels() == input_c)
        << "The output tensor array in the max pooling layer "
           "has an incorrectly sized tensor "
        << i << "th";

    for (uint32_t ic = 0; ic < input_c; ++ic) {
      const arma::fmat& input_channel = input_data->slice(ic);
      arma::fmat& output_channel = output_data->slice(ic);
      for (uint32_t c = 0; c < input_padded_w - pooling_w + 1; c += stride_w_) {
        int output_col = int(c / stride_w_);
        for (uint32_t r = 0; r < input_padded_h - pooling_h + 1;
             r += stride_h_) {
          int output_row = int(r / stride_h_);
          float* output_channel_ptr = output_channel.colptr(output_col);
          //最小的float数，min是最小的正float数
          float max_value = std::numeric_limits<float>::lowest();
          for (uint32_t w = 0; w < pooling_w; ++w) {
            const float* col_ptr = input_channel.colptr(c + w - padding_w_);
            for (uint32_t h = 0; h < pooling_h; ++h) {
              float current_value = 0.f;
              /////这个点的在padding的图上的坐标为（c+w,h+r）,如果这个坐标落在了padding区域内，则current直接赋值为无穷小
              if ((h + r >= padding_h_ && w + c >= padding_w_) &&
                  (h + r < input_h + padding_h_ &&
                   w + c < input_w + padding_w_)) {
                current_value = *(col_ptr + r + h - padding_h_);
                //落在了区域内，则赋值。当然这个值是要映射回未padding的input的坐标
              } else {
                current_value = std::numeric_limits<float>::lowest();
              }
              max_value = max_value > current_value ? max_value : current_value;
            }
          }
          *(output_channel_ptr + output_row) = max_value;
        }
      }
    }
  }
  return StatusCode::kSuccess;
}

StatusCode MaxPoolingLayer::CreateInstance(const std::shared_ptr<RuntimeOperator> &op, std::shared_ptr<Layer<float>> &max_layer)
{
    CHECK(op != nullptr) << "MaxPooling get instance failed, operator is nullptr";
    const std::map<std::string,std::shared_ptr<RuntimeParameter>>& params = op->params;
    ///从op中拿到信息，param的值，传递给maxpooling_layer进行初始化，以forward
    if(params.find("stride") == params.end()){
        LOG(ERROR) << "Can not find the stride parameter";
        return StatusCode::kInferParameterError;
    }
    auto stride = std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("stride"));
    if(!stride){
        LOG(ERROR) << "Can not find the stride parameter";
        return StatusCode::kInferParameterError;
    }


    if(params.find("padding") == params.end()){
        LOG(ERROR) << "Can not find the padding parameter";
        return StatusCode::kInferParameterError;
    }
    auto padding = std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("padding"));
    if(!padding){
        LOG(ERROR) << "Can not find the padding parameter";
        return StatusCode::kInferParameterError;
    }    
    

    if(params.find("kernel_size") == params.end()){
        LOG(ERROR) << "Can not find the kernel_size parameter";
        return StatusCode::kInferParameterError;
    }
    auto kernel_size = std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("kernel_size"));
    if(!kernel_size){
        LOG(ERROR) << "Can not find the kernel_size parameter";
        return StatusCode::kInferParameterError;
    }

    const auto& padding_values = padding->value;
    const auto& stride_values = stride->value;
    const auto& kernel_values = kernel_size->value;


    const uint32_t dims = 2;
    if (padding_values.size() != dims) {
      LOG(ERROR) << "Can not find the right padding parameter";
    return StatusCode::kInferParameterError;
    }

    if (stride_values.size() != dims) {
      LOG(ERROR) << "Can not find the right stride parameter";
    return StatusCode::kInferParameterError;
    }

    if (kernel_values.size() != dims) {
      LOG(ERROR) << "Can not find the right kernel size parameter";
    return StatusCode::kInferParameterError;
    }

    max_layer = std::make_shared<MaxPoolingLayer>(
    padding_values.at(0), padding_values.at(1), kernel_values.at(0),
    kernel_values.at(1), stride_values.at(0), stride_values.at(1));


    return StatusCode::kSuccess;
}
/// 这里传的只是一个算子构造函数的模板，具体的算子（不同padding，stride由函数参数op传递进来）将有GetInstance创建，各不相同。
LayerRegistererWrapper kMaxPoolingGetInstance(MaxPoolingLayer::CreateInstance,"nn.MaxPool2d");

}