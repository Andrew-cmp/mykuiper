
#include <deque>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include "layer/abstract/layer_factor.hpp"
#include "runtime/runtime_ir.hpp"
//#include "utils/time/time_logging.hpp"
namespace my_infer{
RuntimeGraph::RuntimeGraph(std::string param_path,std::string bin_path):
                            param_path_(std::move(param_path)),bin_path_(std::move(bin_path)){}
void RuntimeGraph::set_bin_path(const std::string& bin_path){
    this->bin_path_ = bin_path;
}
void RuntimeGraph::set_param_path(const std::string& param_path){
    this->param_path_ = param_path;
}
const std::string& RuntimeGraph::param_path() const { return this->param_path_; }
const std::string& RuntimeGraph::bin_path() const { return this->bin_path_; }
static bool IsQuantizeOp(const pnnx::Operator* op) { return false; }

bool RuntimeGraph::Init(){
    if(this->bin_path_.empty() || this->param_path_.empty()){
        LOG(ERROR)<<"param or bin is empty";
        return false;
    }
    this->graph_ = std::make_unique<pnnx::Graph>();

    int32_t load_pnnx_result = this->graph_->load(this->bin_path_,this->param_path_);

    if(load_pnnx_result != 0){
        LOG(ERROR) <<"incorrect param path or bin path:"<<param_path_ <<" "<<bin_path_;
        return false;
    }

    this->operators_.clear();
    for(const auto & op: this->graph_->ops){
        if(!op){
            LOG(ERROR) <<"Meet the empty node in the model";
            continue;
        }
        if(!IsQuantizeOp(op)){
            std::shared_ptr<RuntimeOperator> runtime_operator = std::make_shared<RuntimeOperator>();
            //使用读取到的pnnx Graph Operator初始化自己的RuntimeOperator中的数据结构
            runtime_operator->name = op->name;
            runtime_operator->type = op->type;

            InitGraphOperatorsInput(op->inputs, runtime_operator);
            InitGraphOperatorsOutput(op->outputs,runtime_operator);
            InitGraphAttrs(op->attrs,runtime_operator);
            InitGraphParams(op->params,runtime_operator);


            this->operators_.push_back(runtime_operator);

        }
        else{
            LOG(FATAL) << "we do not support quantize right now";
        }


        
    }
    graph_state_ = GraphState::NeedBuild;
    return true;
        ///init做完之后，operatpr的input operand是实打实实例化了的，但这个operand 的datas域size()=0,也就是还没给数据申请空间。
        ///operatpr 的output 就只记录了下名字,甚至没有实例化output operand。
}
void RuntimeGraph::Build(){

    if (graph_state_ == GraphState::complete) {
    LOG(INFO) << "Model has been built already!";
    return ;
    }

    if (graph_state_ == GraphState::NeedInit) {
      bool init_graph = Init();
      LOG_IF(FATAL, !init_graph || graph_state_ == GraphState::NeedInit) << "Init graph failed!";
    }

    CHECK(graph_state_ >= GraphState::NeedBuild)
        << "Graph status error, current state is " << int32_t(graph_state_);
    LOG_IF(FATAL, this->operators_.empty()) << "Graph operators is empty, may be no init";

    CreateNodeRelation();
    ReverseTopoSort();

    //只做check，检查各个shape，检查是否为动态 batch，检查是否有null
    RuntimeOperatorUtils<float>::InitOperatorInput(operators_);
    ///init  output operand of each Operator的data数据域，通过vector.resize(batch)和vector.push(Tensor(row,col,chennel))
    ///涉及到内存复用等
    RuntimeOperatorUtils<float>::InitOperatorOutput(graph_->ops,operators_);

    graph_state_ = GraphState::complete;
    ///不能自动析构吗
    if(graph_ != nullptr){
        graph_.reset();
        graph_=nullptr;
    }

}
StatusCode ExecuteLayer(const std::shared_ptr<Layer<float>>& layer,const std::string& op_name,
                        const std::string& op_type, bool is_debug);
///推理的核心代码和实际执行过程。
///每个operator串行exec
void RuntimeGraph::Forward(bool debug){


    if(this->graph_state_ < GraphState::complete){
        LOG(FATAL) <<"we can do the forward Graph need be build !"
                   <<", current state is "<<int32_t(this->graph_state_);
    }


    // if (debug) {
    //     utils::LayerTimeStatesSingleton::LayerTimeStatesCollectorInit();
    // }
    for(const auto& current_op:this->operators_){
        current_op->has_forward = false;
        CHECK_GT(current_op->start_time ,0);
        //input/output operator是没有计算的，只是将数据放进了operator的output operand的data中。 
        if(is_input_op(current_op->name)|| is_output_op(current_op->name)){
            current_op->has_forward = true;
            continue;
        }
        CHECK(current_op->layer != nullptr)
        << "The layer corresponding to the op " << current_op->name
        << " is empty, indicating that it may not have been created.";

        StatusCode status = ExecuteLayer(current_op->layer,current_op->name, current_op->type,debug);

        CHECK(status == StatusCode::kSuccess)
        << current_op->layer->layer_name()
        << " layer forward failed, error code: " << int32_t(status);

        current_op->has_forward = true;
        PropagateLayerOutputs(current_op,current_op->output_operands->datas);


    }
    //  if (debug) {
    //    utils::LayerTimeLogging::SummaryLogging();
    // }
    for(const auto & op: this->operators_){
        LOG_IF(FATAL, !op->has_forward) << "The operator: " << op->name << " has not been forward yet!";
    }

}

StatusCode ExecuteLayer(const std::shared_ptr<Layer<float>>& layer,const std::string& op_name,
                        const std::string& op_type, bool is_debug)
{
    CHECK(layer!=nullptr);
    StatusCode status;

    if(is_debug){
        //utils::LayerTimeLogging layer_time_logging(op_name, op_type);
        status = layer->Forward();
    }else{
        status = layer->Forward();
    }
    return status;
}

template<typename T>
std::shared_ptr<Layer<T>> RuntimeGraph::CreateLayer(const std::shared_ptr<RuntimeOperatorBase<T>>& op)
{
    LOG_IF(FATAL,!op) << "Operator is empty we can not create layer";
    auto layer = LayerRegisterer::CreateLayer(op);
    LOG_IF(FATAL,!layer)<<"Lay create faild"<<op->type;
    return layer;
}


template<typename T>
void RuntimeGraph::InitGraphOperatorsInput(
    const std::vector<pnnx::Operand*>& inputs,
    const std::shared_ptr<RuntimeOperatorBase<T>>& runtime_operator
){      
    if(inputs.empty()){
        return ;
    }

    CHECK(runtime_operator!= nullptr)<<"The runtime operator is null pointer";

    for(const auto& input:inputs){
        if(!input)continue;
        

        //这里有点奇怪，不能直接用std::copy吗
        std::vector<int32_t> runtime_shapes;
        for(auto shape: input->shape){
            runtime_shapes.push_back(shape);
        }

        ///所有的operand全部使用producer的name，因为一般operator只产生一种输出
        //可能会用到多种输入，所以用producer的name具有唯一性
        const pnnx::Operator* producer = input->producer;
        CHECK(!runtime_shapes.empty());
        std::shared_ptr<RuntimeOperandBase<T>> runtime_operand =
            std::make_shared<RuntimeOperandBase<T>>();
        runtime_operand->name = producer->name;
        runtime_operand->shapes = runtime_shapes;
        runtime_operator->input_operands_seq.push_back(runtime_operand);
        runtime_operator->input_operands.insert({producer->name,runtime_operand});
        switch(input->type){
            case 1:{
                runtime_operand->type = RuntimeDataType::kTypeFloat32;
                break;
            }
            case 7:{
                runtime_operand->type = RuntimeDataType::kTypeInt8;
            }
            default:{
                LOG(FATAL)<<"we only support input of float32 and int8" <<input->type;
            }
        }
    }
}

template<typename T>
void RuntimeGraph::InitGraphOperatorsOutput(
    const std::vector<pnnx::Operand*>& outputs,
    const std::shared_ptr<RuntimeOperatorBase<T>>& runtime_operator
){
    if(outputs.empty())return;
    CHECK(runtime_operator != nullptr) << "The runtime operator is null pointer";
    
    for(const auto & output:outputs){
        if(!output){
            continue;
        }
        ///难道之前的理解都错了？对于一个节点来说，input_operand的name应该是producer
        //output_operand的name应该是consumer？

        //为什么这里没用确认shape？
        //哦因为这里根本没有创建具体的output_operand,
        //std::shared_ptr<RuntimeOperandBase<T>> runtime_operand =
        //    std::make_shared<RuntimeOperandBase<T>>();
        //只是记录的output_operand的名字
        const auto & consumers = output->consumers;
        for(const auto& c:consumers){
            runtime_operator->output_names.push_back(c->name);
        }    
    }
}
template<typename T>
void RuntimeGraph::InitGraphParams(
    const std::map<std::string,pnnx::Parameter>& params,
    const std::shared_ptr<RuntimeOperatorBase<T>>& runtime_operator
){
    if(params.empty())return;
    CHECK(runtime_operator != nullptr) <<"The runtime operator is null pointer";
    for(const auto &[name,param]:params){
        const int32_t type = param.type;
        switch (type){
            case int32_t(RuntimeParameterType::kParameterUnknown):{
                std::shared_ptr<RuntimeParameter> runtime_parameter = 
                std::make_shared<RuntimeParameter>();
                runtime_operator->params.insert({name, runtime_parameter});
                break;
            }
            //还能这么初始化
            case int32_t(RuntimeParameterType::kParameterBool):{
                std::shared_ptr<RuntimeParameterBool> runtime_parameter = 
                std::make_shared<RuntimeParameterBool>(param.b);
                runtime_operator->params.insert({name, runtime_parameter});
                break;
            }
            case int32_t(RuntimeParameterType::kParameterInt):{
                std::shared_ptr<RuntimeParameterInt> runtime_parameter = 
                std::make_shared<RuntimeParameterInt>(param.i);
                runtime_operator->params.insert({name, runtime_parameter});
                break;
            }
            case int32_t(RuntimeParameterType::kParameterFloat): {
              std::shared_ptr<RuntimeParameterFloat> runtime_parameter =
                  std::make_shared<RuntimeParameterFloat>(param.f);
              runtime_operator->params.insert({name, runtime_parameter});
              break;
            }

            case int32_t(RuntimeParameterType::kParameterString): {
              std::shared_ptr<RuntimeParameterString> runtime_parameter =
                  std::make_shared<RuntimeParameterString>(param.s);
              runtime_operator->params.insert({name, runtime_parameter});
              break;
            }
            case int32_t(RuntimeParameterType::kParameterIntArray): {
              std::shared_ptr<RuntimeParameterIntArray> runtime_parameter =
                  std::make_shared<RuntimeParameterIntArray>(param.ai);
              runtime_operator->params.insert({name, runtime_parameter});
              break;
            }

            case int32_t(RuntimeParameterType::kParameterFloatArray): {
              std::shared_ptr<RuntimeParameterFloatArray> runtime_parameter =
                  std::make_shared<RuntimeParameterFloatArray>(param.af);
              runtime_operator->params.insert({name, runtime_parameter});
              break;
            }
            case int32_t(RuntimeParameterType::kParameterStringArray): {
              std::shared_ptr<RuntimeParameterStringArray> runtime_parameter =
                  std::make_shared<RuntimeParameterStringArray>(param.as);
              runtime_operator->params.insert({name, runtime_parameter});
              break;
            }
            default:{
                 LOG(FATAL) << "Unknown parameter type: " << type;
            }
        }
    }
}
///这里把base去掉居然就不行了
template<typename T>
void RuntimeGraph::InitGraphAttrs(
    const std::map<std::string,pnnx::Attribute>& attrs,
    const std::shared_ptr<RuntimeOperatorBase<T>>& runtime_operator
){
    if(attrs.empty()){
        return;
    }
    CHECK(runtime_operator != nullptr) <<"The runtime operator is null pointer";
    for(const auto&[name,attr]:attrs){
        switch (attr.type){
            //咦为什么是1，确实是1，attr的type中 1是float32
            case 1:{
                std::shared_ptr<RuntimeAttribute> runtime_attribute = std::make_shared<RuntimeAttribute>(
                    attr.shape,RuntimeDataType::kTypeFloat32,attr.data
                );
                //哦用shared_ptr是因为，我是在这个函数里创建这个指针的，按理来说随着
                //这个函数调用结束，那么这个指针指向的数据也会被delete
                //但智能指针里面由count记录还有哪个指向这个数据的指针的生命周期还没有结束
                //这片mem就不会被清楚
                //如果用普通指针的话，大概需要为runtime_operator中的attr开辟出一个空间，然后把runtime_attribute的
                //数据移进去
                runtime_operator->attribute.insert({name,runtime_attribute});
            }
                break;

            default:{
                LOG(FATAL) << "Unknown attrobute type"<<attr.type;
            }
        }
    }
}



//所谓的创建节点间的关系，不过是将各个operator的name相连接
//上面init operator的input operand和output operand时候已经做了一些
//下面这些做的更进一步,甚至做的更简单一点，可以直接在init里做完。

void RuntimeGraph::CreateNodeRelation(){

    for(const auto& current_op:this->operators_){
        const std::vector<std::string>& output_names = current_op->output_names;
        for(const auto& output_name:output_names){
            //模式匹配啊，根据current op的output name遍历this->operators_
            for(const auto &op:this->operators_){
                if(op != current_op && op->name == output_name){
                    current_op->output_operators.insert({output_name,op});
                }
            }
        }
        ///TODO:create layer
        ///除了输入输出operator，都创建layer
        if(current_op->type != "pnnx.Input" && current_op->type != "pnnx.Output"){
            auto layer = RuntimeGraph::CreateLayer(current_op);
            ///layer实例创建起来，现在还是孤立无援，下面进行双方赋值，将两方关联起来
            if(layer){
                current_op->layer = layer;
                layer->set_runtime_operator(current_op);
            }else{
                LOG(FATAL) << "Layer" <<current_op<<"create failed";
            }
        }
    }
}
///作用就是根据执行顺序和拓扑排序结果，填充每个operator的start_time，end_time，occur_end_time等
///start_time是operator的开始时间
///end_time 是其子节点的开始时间，如果没有子节点，那么就是start_time+1
///occur_end_time 这时候全都是-1；
void RuntimeGraph::ReverseTopoSort(){
    for(const auto & op:this->operators_){
        if(op != nullptr && op->has_forward == false){
            ///这个地方应该有点问题，如果有多个数据联通图的话，current_forward_idx 应该不一样。
            int32_t current_forward_idex = 0;
            this->ReverseTopoSortInternal(op,current_forward_idex);
        }
    }

    std::sort(this->operators_.begin(),this->operators_.end(),[](const auto&op1,const auto&op2){
        return op1->start_time> op2->start_time;
    });
    int32_t forward_index = 1;
    for(const auto & op:this->operators_){
        op->start_time = forward_index ;
        forward_index += 1;
    }
    for(const auto& op:this->operators_){
        const auto &next_ops =op->output_operators;
        int32_t last_forward_index = -1;
        for(const auto& [_, next_op] : next_ops){
            if(next_op->start_time >= last_forward_index){
                last_forward_index = next_op->start_time;
            }
        }
        ///当他没有子节点的时候，终止time等于starttime+1
        if(last_forward_index = -1){
            op->end_time = op->start_time+1;
        } else {
            ///否则等于所有子节点start_time最大的那个。
            op->end_time = last_forward_index;
        }
        op->occur_end_time = -1;
    }
}

template<typename T>
void RuntimeGraph::ReverseTopoSortInternal(const std::shared_ptr<RuntimeOperatorBase<T>>& root_op,
                                      int32_t& current_forward_idx){
    
    if(root_op == nullptr){
        LOG(INFO) << "Current operator is null ptr";
        return;
    }
    if(root_op->input_operands.empty()&&!root_op->has_forward){
        this->input_ops_.push_back(root_op);
    }
    if(root_op->output_names.empty()&& !root_op->has_forward ){
        this->output_ops_.push_back(root_op);
    }
    ///const auto & [_,next_ops] = root_op->output_operators; 不行？
    const auto& next_ops = root_op->output_operators;

    root_op->has_forward = true;
    for(const auto& [_, next_op] : next_ops){
        if(next_op!=nullptr&&!next_op->has_forward){
            this->ReverseTopoSortInternal(next_op,current_forward_idx);
        }
    }

    for(const auto& [_, next_op] : next_ops){
        CHECK_EQ(next_op->has_forward,true);
    }
    root_op->start_time = current_forward_idx;
    current_forward_idx += 1;
}


RuntimeGraph::GraphState RuntimeGraph::graph_state()const{
    return this->graph_state_;
}

//这里我们将输入input也看作一个节点，并直接给出其 datas
//这里是将输入input的这个节点的输出datas，传递给其子节点operators的输入operand的datas中。
void RuntimeGraph::set_inputs(const std::string& input_name,const std::vector<sftensor>& inputs) {
    CHECK(this->graph_state_==GraphState::complete);

    std::shared_ptr<RuntimeOperator> input_op;
    for(const auto&op : this->input_ops_){
        if(op->name ==input_name){
            input_op = op;
            break;
        }
    }
    CHECK(input_op != nullptr)<<"can not find the input operator: "<<input_name;
    PropagateLayerOutputs(input_op, inputs);

}

///这里是将上一层的输出data传给下一层的input operand data

//注意这里next_input_datas.at(i) == layer_output_data其实只是 传递了指针，
//并且下一层input operand是用vector存储的shared_ptr,vector是在哪进行的扩容呢？
template<typename T>
void RuntimeGraph::PropagateLayerOutputs(
    const std::shared_ptr<RuntimeOperatorBase<T>>& current_op,
    const std::vector<std::shared_ptr<Tensor<T>>>& layer_output_datas
){
    for(const auto &[_,output_op] : current_op->output_operators){

            const auto& next_input_operands = output_op->input_operands;
            /// 找到了layer_output_datas对应的下一个operator的input
            const auto& next_corresponding_input_operands_iter = next_input_operands.find(current_op->name);
            if(next_corresponding_input_operands_iter != next_input_operands.end()){
                //以引用的方式取出来，好修改，改引用就是改源数据
                std::vector<stensor<T>>& next_input_datas = next_corresponding_input_operands_iter->second->datas;
                //这里其实可以直接用赋值了，但是要做类型检查
                for(uint32_t i = 0;i < next_input_datas.size();i++){
                    const stensor<T>& layer_output_data = layer_output_datas.at(i);
                    if(next_input_datas.at(i) != nullptr){
                        //检查一下每个batch里tensor的shape是否一致
                        CHECK(next_input_datas.at(i)->shapes() == layer_output_data->shapes());
                    }
                    next_input_datas.at(i) = layer_output_data;
                }
            }
        }
    }
//和get_input差不多，这个是吧output看作了一个operator，最后的一层节点应该是output节点，所以他只有上层
//operator传到input_operands中的数据，我们把这个数据拿出来就可以了 
std::vector<sftensor> RuntimeGraph::get_outputs(const std::string& output_name)const{
    CHECK(this->graph_state_ == GraphState::complete);
    std::shared_ptr<RuntimeOperator> output_op;
    for(const auto & op:this->output_ops_){
        if(op->name == output_name){
            output_op = op;
        }
    }
    CHECK(output_op!= nullptr);
    //这里获得的也是指针
    std::vector<sftensor> outputs;
    for(const auto&input_operand: output_op->input_operands_seq){
        ///把指针copy过来了。
        std::copy(input_operand->datas.begin(),input_operand->datas.end(),
                  std::back_inserter(outputs));
    
    }
    return outputs;
}
bool RuntimeGraph::is_input_op(const std::string& op_name)const{
    //changed
    for(const auto& op: this->input_ops_){
        CHECK(op!= nullptr);
        if(op->name == op_name)return true;
    }
    return false;
}
bool RuntimeGraph::is_output_op(const std::string& op_name)const{
    for(const auto& op:this->output_ops_){
        CHECK(op!=nullptr);
        if(op->name == op_name)return true;
    }
    return false;
}
}// namespace my_infer