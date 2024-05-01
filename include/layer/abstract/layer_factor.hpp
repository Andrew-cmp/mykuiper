#ifndef MY_INFER_SOURCE_LAYER_LAYER_FACTORY_HPP_
#define MY_INFER_SOURCE_LAYER_LAYER_FACTORY_HPP_
#include <map>
#include <memory>
#include <string>
#include "layer.hpp"
#include "runtime/runtime_op.hpp"
namespace my_infer{
class LayerRegisterer{
    private:

        typedef StatusCode (*Creator) (const std::shared_ptr<RuntimeOperator>& op, std::shared_ptr<Layer<float>>& layer);
        typedef std::map<std::string,Creator> CreateRegistry;
        static CreateRegistry* registry_;
    public:
        friend class LayerRegistererWrapper;
        friend class RegistryGarbageCollector;
         /**
         * 向注册表注册算⼦
         * layer_type 算⼦的类型
         * creator 需要注册算⼦的注册表
         */
        static void RegisterCreator(const std::string& layer_type,const Creator& creator);
        static std::shared_ptr<Layer<float>> CreateLayer(const std::shared_ptr<RuntimeOperator>& op);
        //得到全局注册表，作为单例模式中的访问接口，如果没有即创建
        static CreateRegistry* Registry();

        static std::vector<std::string> layer_types();
};
//这是为了更好的注册算子，在注册不同类型的算子的时候，不用再用RegisterCreator了
//而是直接初始化这个类，就调用了RegisterCreator了。

class LayerRegistererWrapper{
    public:
        explicit LayerRegistererWrapper(const LayerRegisterer::Creator& creator,
                                 const std::string& layer_type){
            LayerRegisterer::RegisterCreator(layer_type,creator);
        }

        template<typename... Ts>
        explicit LayerRegistererWrapper(const LayerRegisterer::Creator& creator,
                                 const std::string& layer_type, const Ts&... other_layer_types):
                                 LayerRegistererWrapper(creator,other_layer_types...){
            LayerRegisterer::RegisterCreator(layer_type,creator);
        }
        explicit LayerRegistererWrapper(const LayerRegisterer::Creator& creator) {}

};
class RegistryGarbageCollector{
    public:
        ~RegistryGarbageCollector(){
            if(LayerRegisterer::registry_ != nullptr){
                delete LayerRegisterer::registry_;
                LayerRegisterer::registry_ = nullptr;
            }
        }
    friend class LayerRegisterer;

    private:
        RegistryGarbageCollector() = default;
        RegistryGarbageCollector(const RegistryGarbageCollector&) = default;
};





}




#endif