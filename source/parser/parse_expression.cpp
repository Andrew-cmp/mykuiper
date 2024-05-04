#include "parser/parse_expression.hpp"
#include <glog/logging.h>
#include <algorithm>
namespace my_infer{
void ReversePolish(const std::shared_ptr<TokenNode>& root, std::vector<std::shared_ptr<TokenNode>>& reverse_polish);
std::vector<std::shared_ptr<TokenNode>> ExpressionParser::Generate(){

    if(this->tokens_.empty() == true){
        this->Tokenizer(true);
    }
    int32_t index = 0;
    std::shared_ptr<TokenNode> root = Generate_(index);
    CHECK(root != nullptr);
    std::vector<std::shared_ptr<TokenNode>> reverse_polish;
    ReversePolish(root,reverse_polish);
    return reverse_polish;
}
///对于一棵语法树来说，逆波兰表达式就是后序遍历
void ReversePolish(const std::shared_ptr<TokenNode>& root, std::vector<std::shared_ptr<TokenNode>>& reverse_polish){
    if(root != nullptr){
        ReversePolish(root->left, reverse_polish);
        ReversePolish(root->right, reverse_polish);
        reverse_polish.push_back(root);
    }
}
std::shared_ptr<TokenNode> ExpressionParser::Generate_(int32_t& index){
    CHECK(index < this->statement_.size());
    const auto & current_token = this->tokens_.at(index);
    CHECK(current_token.token_type == TokenType::TokenInputNumber||
          current_token.token_type == TokenType::TokenAdd||
          current_token.token_type == TokenType::TokenMul );
    if(current_token.token_type == TokenType::TokenInputNumber){
        uint32_t num_start_pos = current_token.start_pos + 1;
        uint32_t num_end_pos = current_token.end_pos;

        CHECK(num_end_pos>num_start_pos || num_end_pos <= this->statement_.size())
        <<"current token has a wrong length" ;
        const std::string& str_number=
            std::string(this->statement_.begin()+num_start_pos,this->statement_.begin()+num_end_pos);
        return std::make_shared<TokenNode>(std::stoi(str_number),nullptr,nullptr);
    }else if(current_token.token_type == TokenType::TokenAdd || current_token.token_type == TokenType::TokenMul){
        std::shared_ptr<TokenNode> current_node =std::make_shared<TokenNode>();
        index += 1;
        CHECK(index < this->tokens_.size()) << "Missing left bracket!";
        CHECK(this->tokens_.at(index).token_type == TokenType::TokenLeftBracket);
        index += 1;
        CHECK(index < this->tokens_.size()) << "Missing correspond left token!";
        const auto left_token = this->tokens_.at(index);
        if (left_token.token_type == TokenType::TokenInputNumber ||
            left_token.token_type == TokenType::TokenAdd ||
            left_token.token_type == TokenType::TokenMul) {
            current_node->left = Generate_(index);
        } else {
            LOG(FATAL) << "Unknown token type: " << int32_t(left_token.token_type);
        }
        index += 1;
        CHECK(index < this->tokens_.size()) << "Missing comma!";
        CHECK(this->tokens_.at(index).token_type == TokenType::TokenComma);
        index += 1;
        CHECK(index < this->tokens_.size()) << "Missing correspond right token!";
        const auto right_token = this->tokens_.at(index);
        if (right_token.token_type == TokenType::TokenInputNumber ||
            right_token.token_type == TokenType::TokenAdd ||
            right_token.token_type == TokenType::TokenMul) {
            current_node->right = Generate_(index);
        } else {
            LOG(FATAL) << "Unknown token type: " << int32_t(right_token.token_type);
        }
        index += 1;
        CHECK(index < this->tokens_.size()) << "Missing right bracket!";
        CHECK(this->tokens_.at(index).token_type == TokenType::TokenRightBracket);
        return current_node;
    } else {
        LOG(FATAL) << "Unknown token type: " << int32_t(current_token.token_type);
    }
}



void ExpressionParser::Tokenizer(bool retokenize){
    //不做retokenize，而且tokens_是空的。
    if(retokenize == false&&this->tokens_.empty() == false){
        return;
    } 
    CHECK(!this->statement_.empty())<<"The input statement is empty";

    this->statement_.erase(
        std::remove_if(this->statement_.begin(),this->statement_.end(),[](char c){return std::isspace(c);}),
        this->statement_.end()
    );
    for(int i = 0;i < this->statement_.size();){
        char c = statement_.at(i);
        if(c == 'a'){
            CHECK(i+1<this->statement_.size()&&this->statement_.at(i+1) =='d') 
                << "Parse add token failed, illegal character: "<<this->statement_.at(i+1);
            CHECK(i+2<this->statement_.size()&&this->statement_.at(i+2) == 'd')
                << "Parse add token failed, illegal character: "<<this->statement_.at(i+2);
            Token token(TokenType::TokenAdd,i,i+3);
            this->tokens_.push_back(token);
            std::string token_operation = std::string(statement_.begin()+i,statement_.begin()+i+3);
            token_strs_.push_back(token_operation);
            i = i +3;
        }else if(c == 'm'){
            CHECK(i + 1 < statement_.size() && statement_.at(i + 1) == 'u')
                << "Parse multiply token failed, illegal character: " << statement_.at(i + 1);
            CHECK(i + 2 < statement_.size() && statement_.at(i + 2) == 'l')
                << "Parse multiply token failed, illegal character: " << statement_.at(i + 2);
            Token token(TokenType::TokenMul, i, i + 3);
            tokens_.push_back(token);
            std::string token_operation = std::string(statement_.begin() + i, statement_.begin() + i + 3);
            token_strs_.push_back(token_operation);
            i = i + 3;

        }else if(c == ','){
            Token token(TokenType::TokenComma, i, i + 1);
            tokens_.push_back(token);
            std::string token_comma = std::string(statement_.begin() + i, statement_.begin() + i + 1);
            token_strs_.push_back(token_comma);
            i += 1;

        }else if(c == '@'){
            CHECK(i + 1 < statement_.size() && std::isdigit(statement_.at(i + 1)))
                << "Parse number token failed, illegal character: " << statement_.at(i + 1);
            int32_t j = i + 1;
            for (; j < statement_.size(); ++j) {
              if (!std::isdigit(statement_.at(j))) {
                break;
              }
            }
            Token token(TokenType::TokenInputNumber, i, j);
            CHECK(token.start_pos < token.end_pos);
            tokens_.push_back(token);
            std::string token_input_number = std::string(statement_.begin() + i, statement_.begin() + j);
            token_strs_.push_back(token_input_number);
            i = j;
        }else if(c == '('){
            Token token(TokenType::TokenLeftBracket, i, i + 1);
            tokens_.push_back(token);
            std::string token_left_bracket =
                std::string(statement_.begin() + i, statement_.begin() + i + 1);
            token_strs_.push_back(token_left_bracket);
            i += 1;
        }else if(c == ')'){
            Token token(TokenType::TokenRightBracket, i, i + 1);
            tokens_.push_back(token);
            std::string token_right_bracket =
                std::string(statement_.begin() + i, statement_.begin() + i + 1);
            token_strs_.push_back(token_right_bracket);
            i += 1;
        }else {
            LOG(FATAL) << "Unknown  illegal character: " << c;
        }
    }
}

const std::vector<Token>& ExpressionParser::tokens() const { return this->tokens_; }
const std::vector<std::string>& ExpressionParser::token_str_array() const { return this->token_strs_; }
TokenNode::TokenNode(int32_t num_index, std::shared_ptr<TokenNode> left,
                     std::shared_ptr<TokenNode> right)
    : num_index(num_index), left(left), right(right) {}

}