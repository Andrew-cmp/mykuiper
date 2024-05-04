
#ifndef MY_INFER_INCLUDE_PARSER_PARSE_EXPRESSION_HPP_
#define MY_INFER_INCLUDE_PARSER_PARSE_EXPRESSION_HPP_

#include<memory>
#include<string>
#include<utility>
#include<vector>
namespace my_infer{
enum class TokenType{
    TokenUnknow =-9,
    TokenInputNumber=-8,
    TokenComma=-7,
    TokenAdd=-6,
    TokenMul=-5,
    TokenLeftBracket=-4,
    TokenRightBracket=-3,    
};
struct Token{
    TokenType token_type = TokenType::TokenUnknow;
    int32_t start_pos = 0;
    int32_t end_pos = 0;
    Token(TokenType token_type,int32_t start_pos,int32_t end_pos):token_type(token_type),start_pos(start_pos),end_pos(end_pos){}
};
struct TokenNode{
    int32_t num_index = -1;

    std::shared_ptr<TokenNode> left = nullptr;
    std::shared_ptr<TokenNode> right = nullptr;

    TokenNode() = default;
    TokenNode(int32_t num_index,std::shared_ptr<TokenNode>left, std::shared_ptr<TokenNode> right);
};
class ExpressionParser{
    public:
        explicit ExpressionParser(std::string statement):statement_(statement){}
        //词法分析 分词
        void Tokenizer(bool retokenize =false);
        //语法分析，产生语法树。
        std::vector<std::shared_ptr<TokenNode>> Generate();

        //得到这个statement的所有tokens
        const std::vector<Token>& tokens() const;
        const std::vector<std::string>& token_str_array()const;

    private:
        std::shared_ptr<TokenNode> Generate_(int32_t& index);
        std::vector<Token> tokens_;
        std::vector<std::string> token_strs_;
        std::string statement_;


};

} // namespace 


#endif