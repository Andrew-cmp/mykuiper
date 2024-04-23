#ifndef PNNX_IR_H
#define PNNX_IR_H
#include <initializer_list>
#include <map>
#include <set>
#include <string>
#include <vector>
namespace pnnx{
class Parameter{
    public:
        Parameter():type(0){

        }
        Parameter(bool _b):type(1),b(_b){

        }
        Parameter(int _i):type(2),i(_i){

        }
        Parameter(long long _i):type(2),i(_i){

        }

        Parameter(long _i):type(2),i(_i){

        }

        Parameter(float _f):type(3), f(_f){
        }

        Parameter(double _d):type(3), f(_d){
        }

        Parameter(const char* _s):type(4), s(_s){
        }
        Parameter(const std::string& _s):type(4),s(_s){

        }
        Parameter(const std::initializer_list<int>& _ai):type(5), ai(_ai){

        }
        Parameter(const std::initializer_list<int64_t>& _ai):type(5){
            for(const auto& x:_ai){
                ai.push_back((int)x);
            }

        }
        Parameter(const std::vector<int>& _ai):type(5), ai(_ai){

        }
        Parameter(const std::initializer_list<float>& _af):type(6),af(_af){
        }
        Parameter(const std::initializer_list<double>& _af): type(6){
          for (const auto& x : _af)
            af.push_back((float)x);
        }
        Parameter(const std::vector<float>& _af): type(6), af(_af) {
        }
        Parameter(const std::initializer_list<const char*>& _as): type(7) {
          for (const auto& x : _as)
            as.push_back(std::string(x));
        }
        Parameter(const std::initializer_list<std::string>& _as) : type(7), as(_as)    {
        }
        Parameter(const std::vector<std::string>& _as): type(7), as(_as) {
        }
        // range from 0~7 
        //0=null 1=b 2=i 3=f 4=s 5=ai 6=af 7=as 8=others
        int type;
        bool b;
        int i;
        float f;
        std::string s;
        std::vector<int> ai;
        std::vector<float> af;
        std::vector<std::string> as;
        static Parameter parse_from_string(const std::string& value);

};
bool operator==(const Parameter& lhs, const Parameter& rhs);



class Attribute{
    public:
        Attribute(): type(0){
        }
        Attribute(const std::initializer_list<int>& shape, const std::vector<float>& t);

        // range from 0~7 
        //0=null 1=b 2=i 3=f 4=s 5=ai 6=af 7=as 8=others
        int type;
        std::vector<int> shape;
        std::vector<char> data;

};
bool operator==(const Attribute& lhs, const Attribute& rhs);
// concat two attributes along the first axis
Attribute operator+(const Attribute& a, const Attribute& b);



//互相定义的时候，可以先声明一个在前面
class Operator;
class Operand{
    public:
        Operator*producer;
        std::vector<Operator*> consumers;
        // range from 0~7 
        //0=null 1=b 2=i 3=f 4=s 5=ai 6=af 7=as 8=others
        int type;
        std::vector<int> shape;
        std::string name;
        std::map<std::string,Parameter>params;
        void remove_consumer(const Operator* c);
};
class Operator{
    public:
        //用指针而不用Operand是为了节省空间
        std::vector<Operand*>inputs;
        std::vector<Operand*>outputs;
        std::string type;
        std::string name;
        std::vector<std::string>inputnames;
        std::map<std::string, Parameter> params;
        std::map<std::string, Attribute> attrs;
};
class Graph{

    public:
        Graph();
        ~Graph();
        int load(const std::string& parampath, const std::string& binpath);
        int save(const std::string& parampath, const std::string& binpath);
        int python(const std::string& pypath, const std::string& binpath);
        int parse(const std::string& param);
        Operator* new_operator(const std::string& type, const std::string& name);
        Operator* new_operator_before(const std::string& type, const std::string& name, const Operator* cur);
        Operator* new_operator_after(const std::string& type, const std::string& name, const Operator* cur);
        Operand* new_operand(const std::string& name);

        Operand* get_operand(const std::string& name);
        const Operand* get_operand(const std::string& name) const;
        std::vector<Operator*> ops;
        std::vector<Operand*> operands; 

    private:
        Graph(const Graph& rhs);
        Graph& operator=(const Graph& rhs);
};
    
} // namespace pnnx
#endif