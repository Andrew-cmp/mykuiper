#ifndef my_INFER_INCLUDE_PARSER_RUNTIME_OPERATOR_HPP_
#define my_INFER_INCLUDE_PARSER_RUNTIME_OPERATOR_HPP_
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "runtime/pnnx/ir.h"
#include "runtime_attr.hpp"
#include "runtime_operand.hpp"
#include "runtime_parameter.hpp"
namespace my_infer{
template<typename T>
class Layer;

template<typename T>
struct RuntimeOperatorBase{
      /// Execution order index of this operator
    int32_t start_time = -1;

    int32_t end_time = -1;

    int32_t occur_end_time = -1;

    /// Whether this operator has run in current execution
    bool has_forward = false;

    std::string name;
    std::string type;
    std::shared_ptr<Layer<T>> layer;

    std::vector<std::string> output_names;
    std::shared_ptr<RuntimeOperandBase<T>> output_operands;
    std::map<std::string, std::shared_ptr<RuntimeOperandBase<T>>> input_operands;
    std::vector<std::shared_ptr<RuntimeOperandBase<T>>> input_operands_seq;

    std::map<std::string, std::shared_ptr<RuntimeOperatorBase<T>>> output_operators;

    std::map<std::string, std::shared_ptr<RuntimeParameter>> params;
    std::map<std::string, std::shared_ptr<RuntimeAttribute>> attribute;
};
//看来输入进来的都是float
using RuntimeOperator =RuntimeOperatorBase<float>;
using RuntimeOperatorQuantized = RuntimeOperatorBase<int8_t>;

template<typename T>
class RuntimeOperatorUtils;


template<>
class RuntimeOperatorUtils<float>{
  public:

    static void InitOperatorInput(const std::vector<std::shared_ptr<RuntimeOperator>>& operators);
    static void InitOperatorOutput(const std::vector<pnnx::Operator*>& pnnx_operators,
                                  const std::vector<std::shared_ptr<RuntimeOperator>>& operators);

};

}


#endif
