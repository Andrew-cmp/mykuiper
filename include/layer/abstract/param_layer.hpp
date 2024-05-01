
#include "layer.hpp"
#ifndef MY_INFER_SOURCE_LAYER_PARAM_LAYER_HPP_
#define MY_INFER_SOURCE_LAYER_PARAM_LAYER_HPP_
namespace my_infer
{
class ParamLayer:public Layer<float>{
    public: 
        explicit ParamLayer(const std::string& layer_name);
        void InitWeightParam(uint32_t param_count, uint32_t param_channel, uint32_t param_height,
                             uint32_t param_width);
        
        void InitBiasParam(uint32_t param_count, uint32_t param_channel, uint32_t Param_height,
                           uint32_t param_width);
        const std::vector<std::shared_ptr<Tensor<float>>>& weights()const override;
        const std::vector<std::shared_ptr<Tensor<float>>>& bias()const override;
        const std::shared_ptr<Tensor<float>> weight(int32_t index) const;
        void set_weights(const std::vector<std::shared_ptr<Tensor<float>>>& weights) override;
        void set_weights(const std::vector<float>& weights) override;
        void set_bias(const std::vector<std::shared_ptr<Tensor<float>>>& bias) override;
        void set_bias(const std::vector<float>& bias) override;
    protected:
        std::vector<std::shared_ptr<Tensor<float>>> weights_;
        std::vector<std::shared_ptr<Tensor<float>>> bias_;
};
} // namespace my_infer

#endif