
#ifndef MY_INFER_INCLUDE_PARSER_RUNTIME_OPERAND_HPP_
#define MY_INFER_INCLUDE_PARSER_RUNTIME_OPERAND_HPP_
#include <memory>
#include <string>
#include <vector>
#include "data/tensor.hpp"
#include "runtime_datatype.hpp"
#include "status_code.hpp"

namespace my_infer{

template <typename T>
struct RuntimeOperandBase{
    explicit RuntimeOperandBase() = default;
    explicit RuntimeOperandBase(std::string name, std::vector<int32_t> shapes,
        std::vector<std::shared_ptr<Tensor<T>>> datas, RuntimeDataType type)
      : name(std::move(name)), shapes(std::move(shapes)), datas(std::move(datas)), type(type) {}

    size_t size()const;


    std::string name;
    RuntimeDataType type = RuntimeDataType::kTypeUnknown;
    std::vector<int32_t> shapes;
    std::vector<std::shared_ptr<Tensor<T>>> datas;

};
template<typename T>
size_t RuntimeOperandBase<T>::size() const{
    if(shapes.empty()){
        return 0;
    }
    size_t size = std::accumulate(shapes.begin(),shapes.end(),1,std::multiplies());
    return size;
}
using RuntimeOperand = RuntimeOperandBase<float>;
using RuntimeOperandQuantized = RuntimeOperandBase<int8_t>;

}




#endif
