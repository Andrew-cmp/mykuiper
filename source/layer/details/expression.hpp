#ifndef MY_INFER_SOURCE_LAYER_MONOCULAR_EXPRESSION_HPP_
#define MY_INFER_SOURCE_LAYER_MONOCULAR_EXPRESSION_HPP_
#include "layer/abstract/non_param_layer.hpp"
#include "parser/parse_expression.hpp"

namespace my_infer {
class ExpressionLayer : public NonParamLayer {
 public:
  explicit ExpressionLayer(std::string statement);

  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  bool TokenIsOperator(Token token) const;

  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& expression_layer);

 private:
  std::string statement_;
  std::unique_ptr<ExpressionParser> parser_;
};
}  // namespace my_infer
#endif  // MY_INFER_SOURCE_LAYER_MONOCULAR_EXPRESSION_HPP_
