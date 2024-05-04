#ifndef KUIPER_INFER_DATA_BLOB_HPP_
#define KUIPER_INFER_DATA_BLOB_HPP_
#include <glog/logging.h>
#include <armadillo>
#include <memory>
#include <numeric>
#include <vector>
namespace my_infer{
template<typename T>
class Tensor{
    public:
        explicit Tensor(T * raw_ptr, uint32_t size);
        explicit Tensor(T * raw_ptr, uint32_t rows, uint32_t cols);
        explicit Tensor(T * raw_ptr, uint32_t channels, uint32_t rows, uint32_t cols);
        explicit Tensor(T * raw_ptr, const std::vector<uint32_t>& shapes);
        explicit Tensor() = default;
        explicit Tensor(uint32_t size);
        explicit Tensor(uint32_t rows, uint32_t cols);
        explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);
        explicit Tensor(const std::vector<uint32_t>& shapes);
    

        uint32_t rows() const;
        uint32_t cols() const;
        uint32_t channels() const;

        size_t size() const;
        //matrix的大小
        size_t plane_size() const;
        

        void set_data(const arma::Cube<T> & data);
        bool empty() const;
        /// 不加const也报错，为什么 


        ///这个地方为什么要用&，不用*？
        T& index(uint32_t offset);
        //两种需求，可是调用者怎么知道返回的是哪种类型呢？
        //const T index(uint32_t offset);这样也不行，也不能重载，因为仅仅返回值不同 
        const T index(uint32_t offset)const; 
        //////////////去掉const
        std::vector<uint32_t> shapes() const;
        //这两什么区别
        const std::vector<uint32_t>& raw_shapes() const;

        //////get_data->data
        //不懂为什么都要搞两种类型
        //不加这两个引用还会出问题，为什么
        arma::Cube<T>& data() ;
        ///arma::Cube<T> get_data() const ;会和下面发生重载冲突冲突。因为仅仅返回值不同是不能进行重载的
        const arma::Cube<T>& data() const;


        //不加这两个引用还会出问题

        arma::Mat<T>& slice(uint32_t channel);
        const  arma::Mat<T>& slice(uint32_t channel)const;



        const T at(uint32_t channel, uint32_t row, uint32_t col)const;
        T& at(uint32_t channel, uint32_t row, uint32_t col);

        void Padding(const std::vector<uint32_t>& pads, T padding_value);

        ////void Transpose(std::vector<uint32_t>&shape);这个东西源文件里没有！！！
        void Fill(T value);
        void Fill(const std::vector<T> & values,bool row_major=true);




        std::vector<T> values(bool row_major = true);
        void Ones();

        void RandN(T mean=0,T var =1);

        void RandU(T min=0,T max = 1);

        ///////void Show() const;删去const
        void Show();

        //这个没加const也报错
        void Reshape(const std::vector<uint32_t> & shapes, bool row_major = false);

        ////没有加默认值
        void Flatten(bool row_major=false);
        void Transform(const std::function<T(T)>& filter);

        T* raw_ptr();

        const T* raw_ptr() const;

        T* raw_ptr(size_t offset);

        const T* raw_ptr(size_t offset) const;
        

        T* matrix_raw_ptr(uint32_t index);

        const T* matrix_raw_ptr(uint32_t index) const;
        
    private:
    std::vector<uint32_t> raw_shapes_;
    void Review(const std::vector<uint32_t>& shapes);
    arma::Cube<T> data_;

};
///用来表示数据类型
template <typename T = float>
using stensor = std::shared_ptr<Tensor<T>>;

using ftensor = Tensor<float>;
using sftensor = std::shared_ptr<Tensor<float>>;

using u1tensor = Tensor<uint8_t>;
using su1tensor = std::shared_ptr<Tensor<uint8_t>>;

//const std::shared_ptr<ftensor>& tensor_ptr = TensorCreate<float>(3, 32, 32);比如这样
}
#endif