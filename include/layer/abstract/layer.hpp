
#ifndef MY_INFER_SOURCE_LAYER_LAYER_HPP_
#define MY_INFER_SOURCE_LAYER_LAYER_HPP_
#include <glog/logging.h>
#include <memory>
#include <string>
#include <vector>
#include "data/tensor.hpp"
#include "runtime/runtime_op.hpp"
#include "status_code.hpp"
namespace my_infer{
template<typename T>
class Layer;

template<>
class Layer<int8_t> {};

template<>
class Layer<float>{
    public:
        explicit Layer(std::string layer_name):layer_name_(std::move(layer_name)){}
        ///这个forward用来表示基类的forward过程，子类不实现这个函数。
        virtual StatusCode Forward();
        //这个forward用来实现子类的forward过程，子类必须实现这个函数，表示算子的计算过程。
        virtual StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                   std::vector<std::shared_ptr<Tensor<float>>>& outputs);
        virtual const std::string& layer_name()const{return this->layer_name_;}
        virtual void set_runtime_operator(const std::shared_ptr<RuntimeOperator>& runtime_operator);


        virtual const std::vector<std::shared_ptr<Tensor<float>>>& weights()const;
        virtual const std::vector<std::shared_ptr<Tensor<float>>>& bias()const;

        virtual void set_weights(const std::vector<std::shared_ptr<Tensor<float>>>& weights);
        ///weight 被pattern的情况
        virtual void set_weights(const std::vector<float>& weights);
        virtual void set_bias(const std::vector<std::shared_ptr<Tensor<float>>>& bias);
        virtual void set_bias(const std::vector<float>& bias);



    protected:
    /// @brief 因为runtime_operator和layer是相互指向相互包含双向包含的关系，所以这里使用weak_ptr来避免循环引用的问题
    //具体可看https://www.luozhiyun.com/archives/762 
        std::weak_ptr<RuntimeOperator> runtime_operator_;
        std::string layer_name_;


};    
}
#endif