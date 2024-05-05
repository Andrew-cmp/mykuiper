

#ifndef MY_INFER_SOURCE_LAYER_LINEAR_HPP_
#define MY_INFER_SOURCE_LAYER_LINEAR_HPP_
#include "layer/abstract/layer.hpp"
#include "layer/abstract/param_layer.hpp"

namespace my_infer {
class LinearLayer : public ParamLayer {
 public:
  //  explicit LinearLayer(uint32_t batch, uint32_t in_channel, uint32_t in_dim, uint32_t out_dim,
  //  bool use_bias = true);
  explicit LinearLayer(int32_t in_features, int32_t out_features, bool use_bias);

  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& linear_layer);

 private:
  int32_t in_features_ = 0;
  int32_t out_features_ = 0;
  bool use_bias_ = false;
};
}  // namespace my_infer

#endif  // my_INFER_SOURCE_LAYER_LINEAR_HPP_
