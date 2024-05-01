#ifndef MY_INFER_INCLUDE_RUNTIME_RUNTIME_IR_HPP_
#define MY_INFER_INCLUDE_RUNTIME_RUNTIME_IR_HPP_
#include <glog/logging.h>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>
#include "layer/abstract/layer.hpp"
#include "runtime/pnnx/ir.h"
#include "runtime/runtime_operand.hpp"
#include "runtime_op.hpp"
namespace my_infer{

class RuntimeGraph{
    public:
        RuntimeGraph(std::string bin_path_,std::string param_path_);


        void set_inputs(const std::string& input_name, const std::vector<sftensor>& inputs);
        std::vector<sftensor> get_outputs(const std::string& output_name)const;

        bool is_input_op(const std::string& op_name)const;

        bool is_output_op(const std::string& op_name) const;
        
        void Build();

        void set_bin_path(const std::string& bin_path);

        void set_param_path(const std::string& param_path);

        const std::string& param_path() const;

        const std::string& bin_path() const;
        
        void Forward(bool debug =false);

    private:

        bool Init();

        void ReverseTopoSort();

        template<typename T>
        void ReverseTopoSortInternal(const std::shared_ptr<RuntimeOperatorBase<T>>& root_op, int32_t& current_forward_idx);

        void CreateNodeRelation();

        template<typename T>
        static void InitGraphOperatorsInput(
            const std::vector<pnnx::Operand*>&inputs,
            const std::shared_ptr<RuntimeOperatorBase<T>>& runtime_operator);
        
        template<typename T>
        static void InitGraphOperatorsOutput(
            const std::vector<pnnx::Operand*>&outputs,
            const std::shared_ptr<RuntimeOperatorBase<T>>& runtime_operator);
        
        template<typename T>
        static void InitGraphAttrs(const std::map<std::string,pnnx::Attribute>& attrs,
                                   const std::shared_ptr<RuntimeOperatorBase<T>>& runtime_operator);
        template<typename T>
        static void InitGraphParams(const std::map<std::string,pnnx::Parameter>& params,
                                   const std::shared_ptr<RuntimeOperatorBase<T>>& runtime_operator);

        template<typename T>
        static std::shared_ptr<Layer<T>> CreateLayer(const std::shared_ptr<RuntimeOperatorBase<T>>& op);


        template <typename T>
        static void PropagateLayerOutputs(
            const std::shared_ptr<RuntimeOperatorBase<T>>& current_op,
            const std::vector<std::shared_ptr<Tensor<T>>>& layer_output_data
        );


    private:
        enum class GraphState{
            NeedInit = -2,
            NeedBuild = -1,
            complete = 0,
        };
    public:
        GraphState graph_state()const;
    private:
        std::string bin_path_;
        std::string param_path_;
        std::unique_ptr<pnnx::Graph> graph_;
        GraphState graph_state_ = GraphState::NeedInit;
        std::vector<std::shared_ptr<RuntimeOperator>> input_ops_;
        std::vector<std::shared_ptr<RuntimeOperator>> output_ops_;
        std::vector<std::shared_ptr<RuntimeOperator>> operators_;
};


}



#endif