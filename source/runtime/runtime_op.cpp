#include "runtime/runtime_op.hpp"
#include "data/tensor_util.hpp"
namespace my_infer
{
    ///这个地方是做每个operator input_operand的类型检查
void RuntimeOperatorUtils<float>::InitOperatorInput(
    const std::vector<std::shared_ptr<RuntimeOperator>>& operators){
        if(operators.empty()){
            LOG(ERROR)<<"Operators for init input shapes is empty";
            return ;
        }
        for(const auto &op:operators){
            //写错了
            if(op->input_operands.empty()){
                continue;
            }
            else{
                const std::map<std::string,std::shared_ptr<RuntimeOperand>>& input_operands
                =op->input_operands;
                for(const auto&[_,input_operand]:input_operands){
                    ///输入必须都是float32
                    CHECK(input_operand->type == RuntimeDataType::kTypeFloat32);
                    //输入的shape必须不为空
                    CHECK(!input_operand->shapes.empty());
                    ///batch的大小必须大于0
                    CHECK(input_operand->shapes.at(0) > 0 )<<"Dynamic batch size is not supported!";
                    ///输入数据的维度必须是2，3，4
                    CHECK(input_operand->shapes.size() == 2||input_operand->shapes.size() == 3||
                    input_operand->shapes.size() == 4)<<"Unsupported tensor shape sizes:" << input_operand->shapes.size();
                    ///如果inputdata不为空的话，就看看这个存着batch的vector大小是不是和shapes的第0维（存放batch数值）是否相等
                    if(!input_operand->datas.empty()){
                        CHECK_EQ(input_operand->datas.size(),input_operand->shapes.at(0));
                    }
                    //如果数据为空，那么先把这个vector给扩展到batch的大小
                    else{
                        input_operand->datas.resize(input_operand->shapes.at(0));
                    }
                }

            }
        }
}
static void CheckAndReshapeTensor(sftensor& output_tensor,
                                  const std::vector<int32_t>& operand_shapes){
    const std::vector<uint32_t>& tensor_shapes = output_tensor->shapes();
    switch (operand_shapes.size()){
        case 4:{
            if(tensor_shapes[0] != operand_shapes[1] || tensor_shapes[1] != operand_shapes[2] 
             ||tensor_shapes[2] != operand_shapes[3] ){
                output_tensor->Reshape({(uint32_t)operand_shapes[1],(uint32_t)operand_shapes[2],
                                        (uint32_t)operand_shapes[3]});
             }
            break;}
        case 3:{
            //error if(tensor_shapes[0] != operand_shapes[1] || tensor_shapes[1] != operand_shapes[2]){
            if (tensor_shapes[0] != 1 || tensor_shapes[1] != operand_shapes[1] ||
            tensor_shapes[2] != operand_shapes[2]) {
            
                output_tensor->Reshape({(uint32_t)operand_shapes[1],(uint32_t)operand_shapes[2]});
             }
            break;}
        case 2:{
            //error if(tensor_shapes[0] != operand_shapes[1]){
            if (tensor_shapes[0] != 1 || tensor_shapes[1] != operand_shapes[1] || tensor_shapes[2] != 1) {
            output_tensor->Reshape({(uint32_t)operand_shapes[1]});
            }
            break;}
        default:{
            LOG(FATAL) << "Unknown output operand shape length: " << operand_shapes.size();
            break;
            }
    }


}

static sftensor CreateTensor(const std::vector<int32_t>& operand_shapes){
    //operand_shapes[0]是batch_size的大小
    switch (operand_shapes.size()){
        case 2:{
            return TensorCreate<float>(operand_shapes[1]);
        }
        case 3:{
            return TensorCreate<float>(operand_shapes[1],operand_shapes[2]);
        }
        case 4:{
            return TensorCreate<float>(operand_shapes[1],operand_shapes[2],operand_shapes[3]);
        }
        default:{
            LOG(FATAL) << "Unsupport output operand shape length:"<<operand_shapes.size();
        }
    }
}

void RuntimeOperatorUtils<float>::InitOperatorOutput(
    const std::vector<pnnx::Operator*>& pnnx_operators,
    const std::vector<std::shared_ptr<RuntimeOperator>>& operators
){
    CHECK(!pnnx_operators.empty()&&!operators.empty()&&pnnx_operators.size() == operators.size());
    for(uint32_t i = 0;i < pnnx_operators.size();i++){
        const std::vector<pnnx::Operand*> pnnx_operands = pnnx_operators[i]->outputs;
        if(pnnx_operands.empty())continue;
        if(pnnx_operands.size()>1){
            LOG(FATAL) <<"Only support one output operand yet!";
        }
        pnnx::Operand* pnnx_operand = pnnx_operands[0];
        CHECK(pnnx_operand != nullptr && !pnnx_operand->shape.empty());
        CHECK_EQ(pnnx_operand->type,1)<<"the type of pnnx operand is not float32";
        std::vector<int32_t>operand_shapes;
        std::copy_if(pnnx_operand->shape.begin(),pnnx_operand->shape.end(),std::back_inserter(operand_shapes),
                    [](int32_t dim){return dim>0;});
        CHECK(operand_shapes.size() == 2||operand_shapes.size()==3||operand_shapes.size()==4)<<
        "unsupported shape sizes:" << operand_shapes.size();



        const auto& runtime_operator = operators[i];
        auto & runtime_operand = runtime_operator->output_operands;
        size_t operand_size = 
                std::accumulate(operand_shapes.begin(),operand_shapes.end(),1,std::multiplies());
        const int32_t batch = operand_shapes[0];
        //重用内存空间
        if(runtime_operand == nullptr){
            bool have_found_reused = false;
            //对之前所有的operator遍历，看看能不能重用
            for(int j = 0;j < 0;j++){
                //如果已经找到了，那就break跳出
                if(have_found_reused == true)break;
                ///没找到，下面开始验证条件
                const auto & prev_runtime_op = operators.at(j);
                //如果prec_op的outputoperand压根没有，自然不行。occur_end_time代表其operand 空间已经被重用了。
                //并且重用的结束时间为occur_end_time.并且这个是被实时更新的。
                if(prev_runtime_op->output_operands == nullptr||prev_runtime_op->occur_end_time != -1){
                    continue;
                }
                //如果runtime的start time比prev的occur大，说明这时候prev的occur已经可以重用了
                if(runtime_operator->start_time > prev_runtime_op->occur_end_time){
                    prev_runtime_op->occur_end_time = -1;
                }
                ///runtime是在prev结束之后才执行的，所以可以重用他的空间
                if(runtime_operator->start_time>prev_runtime_op->end_time){
                    ///如果两个空间一样大就开始空间重用了。
                    //所以这里还是可以改进，内存池啥的。
                    if(prev_runtime_op->output_operands->size() == operand_size){
                        have_found_reused = true;
                        // runtime_operator->output_operands = std::make_shared<RuntimeOperand>();
                        // runtime_operator->output_operands->type = RuntimeDataType::kTypeFloat32;
                        // runtime_operator->output_operands->name = pnnx_operand->name+"_output";
                        // runtime_operator->output_operands->shapes=operand_shapes;
                        // runtime_operator->output_operands->datas.resize(batch);
                        
                        runtime_operand = std::make_shared<RuntimeOperand>();
                        runtime_operand->type = RuntimeDataType::kTypeFloat32;
                        runtime_operand->name = pnnx_operand->name+"_output";
                        runtime_operand->shapes=operand_shapes;
                        runtime_operand->datas.resize(batch);
                        
                        const auto & prev_runtime_op_tensors = prev_runtime_op->output_operands->datas;
                        ///开始给vector<shared_ptr>batch中的tensor数据复用空间。
                        for(uint32_t b = 0;b < batch;b++){
                            ///使用output_tensor的ptr初始化新的sftensor，但内存区域保持不变
                            ///详情请看make_shared的重载函数
                            sftensor prev_output_tensor = prev_runtime_op_tensors.at(b);
                            runtime_operand->datas[b]
                                =std::make_shared<ftensor>(prev_output_tensor->raw_ptr(),prev_output_tensor->shapes());
                            CheckAndReshapeTensor(runtime_operand->datas[b],operand_shapes);
                        }
                        ///更新prev的occur 为runtime op的endtime，即这个内存会一直重用到runtime 的endtime里。
                        prev_runtime_op->occur_end_time = runtime_operator->end_time;
                    }
                }
            }
            //如果之前的都不能重用，只能申请空间了。
            if(have_found_reused == false){
                std::vector<sftensor> out_operand_datas;
                for(int b = 0 ;b < batch;b++){
                    out_operand_datas.push_back(CreateTensor(operand_shapes));
                }
                runtime_operand = std::make_shared<RuntimeOperand>(pnnx_operand->name+"_output",operand_shapes,
                                                            out_operand_datas,RuntimeDataType::kTypeFloat32);
            }//如果上面找到了，那下面就做一下类型检查   
        }
        else{
            ///runtime_operand->datas只是一个存了各个batch数据 shared_ptr的vector。
            CHECK(runtime_operand->datas.size()==batch);
            CHECK(runtime_operand->type == RuntimeDataType::kTypeFloat32);
            CHECK(runtime_operand->shapes == operand_shapes);
            for(uint32_t b = 0;b< batch ;++b){
                sftensor output_tensor = runtime_operand->datas[b];
                CheckAndReshapeTensor(output_tensor,operand_shapes);
            }
        }
    }
}

} // namespace my_infer
