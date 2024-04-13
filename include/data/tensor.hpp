


#ifndef KUIPER_INFER_DATA_BLOB_HPP_
#define KUIPER_INFER_DATA_BLOB_HPP_
#include <glog/logging.h>
#include <armadillo>
#include <memory>
#include <numeric>
#include <vector>
namespace my_kuiper{
template<typename T>
class Tensor{
    public:
    /**
     * @brief Construct a new Tensor object with a ptr of a Already exists array 
     * 
     * @param raw_ptr 
     * @param size 
     */
        explicit Tensor(T * raw_ptr, uint32_t size);
        explicit Tensor(T * raw_ptr, uint32_t rows, uint32_t cols);
        explicit Tensor(T * raw_ptr, uint32_t channels, uint32_t rows, uint32_t cols);
        explicit Tensor(T * raw_ptr, const std::vector<uint32_t>& shapes);

        /**
         * @brief Construct a new Tensor object
         * 
         */
        explicit Tensor() = default;
        explicit Tensor(uint32_t size);
        explicit Tensor(uint32_t rows, uint32_t cols);
        explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);
        explicit Tensor(const std::vector<uint32_t>& shapes);
    

        uint32_t rows() const;
        uint32_t cols() const;
        uint32_t channels() const;
        const std::vector<uint32_t> shapes() const;
        //这两什么区别
        const std::vector<uint32_t>& raw_shapes() const;


        bool empty() const;
        void set_data(arma::Cube<T> & data);

        //不懂为什么都要搞两种类型
        arma::Cube<T> get_data() ;
        ///arma::Cube<T> get_data() const ;会和下面发生重载冲突冲突。因为仅仅返回值不同是不能进行重载的
        const arma::Cube<T> get_data() const;

        arma::Mat<T> get_slice(uint32_t channel);
        const  arma::Mat<T> get_slice(uint32_t channel)const;



        ///这个地方为什么要用&，不用*？
          /**
        * @brief Gets element reference at offset
        *
        * @param offset Element offset
        * @return Element reference
        */
        T& index(uint32_t offset);
        //两种需求，可是调用者怎么知道返回的是哪种类型呢？
        //const T index(uint32_t offset);这样也不行，也不能重载，因为仅仅返回值不同 
        const T index(uint32_t offset)const; 

        const T at(uint32_t channel, uint32_t row, uint32_t col)const;
        T& at(uint32_t channel, uint32_t row, uint32_t col);

        void Padding(const std::vector<uint32_t>& pads, T padding_value);

        void Transpose(std::vector<uint32_t>&shape);
        void Fill(T value);
        void Fill(const std::vector<T> & values,bool row_major=true);

        
        void Flatten(bool row_major);


        void Ones();

        std::vector<T> values(bool row_major = true);

        void RandN(T mean=0,T var =1);

        void RandU(T min=0,T max = 1);

        void Show() const;

        void Reshape(std::vector<uint32_t> & shapes, bool row_major = false);

        void Transform(const std::function<T(T)>& filter);

        T* raw_ptr();

        const T* raw_ptr() const;

        T* raw_ptr(size_t offset);

        const T* raw_ptr(size_t offset) const;
        

        T* matrix_raw_ptr(uint32_t index);

        const T* matrix_raw_ptr(uint32_t index) const;
        
        size_t size() const;
    private:
    std::vector<uint32_t> raw_shape_;
    void Review(const std::vector<uint32_t>& shapes);
    arma::Cube<T> data_;

};


}
#endif