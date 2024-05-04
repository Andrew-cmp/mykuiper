#ifndef MY_INFER_RUNTIME_RUNTIME_ATTR_HPP_
#define MY_INFER_RUNTIME_RUNTIME_ATTR_HPP_
#include <glog/logging.h>
#include <vector>
#include "runtime_datatype.hpp"
#include "status_code.hpp"
namespace my_infer
{

struct RuntimeAttribute
{///这里面也能用move定义shape啊，确实，只要能访问这个对象的shape，那个shape就不需要了
    explicit RuntimeAttribute(std::vector<int32_t> shape, RuntimeDataType type,
                            std::vector<char> weight_data):
                            shape(std::move(shape)),type(type),weight_data(std::move(weight_data)){}
    RuntimeAttribute() = default;
    /// 为什么是char?
    std::vector<char> weight_data;
    std::vector<int32_t> shape;
    RuntimeDataType type = RuntimeDataType::kTypeUnknown;
    template<typename T>
    std::vector<T> get(bool need_clear_weight = true); 
};

//只支持float32
template<typename T>
std::vector<T> RuntimeAttribute::get(bool need_clear_weight){
    CHECK(!weight_data.empty());
    CHECK(type != RuntimeDataType::kTypeUnknown);
    const uint16_t elem_size = sizeof(T);
    CHECK_EQ(weight_data.size() % elem_size, 0);
    ///这个好像不太对
    const uint32_t weight_data_size = weight_data.size()/elem_size;
    std::vector<T> weights;

    weights.reserve(weight_data_size);
    switch (type){
        case RuntimeDataType::kTypeFloat32 :{
            static_assert(std::is_same<T,float>::value == true);
            float * weight_data_ptr = reinterpret_cast<float*>(weight_data.data());
            for(uint32_t i = 0;i < weight_data_size;++i){

                float weight = *(weight_data_ptr + i);
                weights.push_back(weight);
            }
            break;
        }
        default:{
            LOG(FATAL) << "unkown weight data type:(weight data type must be float32) "<<int32_t(type);
        }
  
    }
    //need_clear_weight的意思看来是将weight置为空序列
     if(need_clear_weight){
        
        std::vector<char> empty_vec = std::vector<char>();
        this->weight_data.swap(empty_vec);
    }
    return weights;
}


} // namespace my_infer




#endif