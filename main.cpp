#include "data/load_data.hpp"
#include "data/tensor.hpp"

int main(){
    #ifdef KUIPER_INFER_DATA_BLOB_HPP_
    printf("2");
    #endif
    my_infer::Tensor<float> f1(3, 224, 224);
    f1.Fill(1);
    printf("%d",1);
    
}