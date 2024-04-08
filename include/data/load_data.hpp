/**
 * @file load_data.hpp
 * @author xiaohou (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-08
 * 
 * @copyright Copyright (c) 2024
 * 
 */
///哦我明白了，给每一个头文件一个唯一标识，这个标识用宏来定义。比如这里load_data.hpp的唯一宏定义标识为MYKUIPER_INFER_INCLUDE_DATA_LOAD_DATA_HPP_
///然后使用ifndef 和 ifdef来判断包含的头文件是否已经宏定义过这个表示，进而来判断load_data.hpp是否已被包含。
#ifndef KUIPER_INFER_INCLUDE_DATA_LOAD_DATA_HPP_
#define KUIPER_INFER_INCLUDE_DATA_LOAD_DATA_HPP_
#include <glog/logging.h>
#include <armadillo>
#include <string>
namespace mykuiper{
class CSVDataLoader {
 public:
  template <typename T>
  static arma::Mat<T> LoadData(const std::string& file_path, char split_char = ',');
 private:
  static std::pair<size_t, size_t> GetMatrixSize(std::ifstream& file, char split_char);
};

template <typename T>
arma::Mat<T> CSVDataLoader::LoadData(const std::string& file_path, const char split_char) {
  arma::Mat<T> data;
  if (file_path.empty()) {
    LOG(ERROR) << "CSV file path is empty: " << file_path;
    return data;
  }

  std::ifstream in(file_path);
  if (!in.is_open() || !in.good()) {
    LOG(ERROR) << "File open failed: " << file_path;
    return data;
  }
  std::stringstream line_stream;

  const auto& [rows, cols] = CSVDataLoader::GetMatrixSize(in, split_char);
  data.zeros(rows, cols);

  size_t row = 0;
  while (in.good()) {
    std::getline(in, line_str);
    if (line_str.empty()) {
      break;
    }

    std::string token;
    line_stream.clear();
    line_stream.str(line_str);

    size_t col = 0;
    while (line_stream.good()) {
      std::getline(line_stream, token, split_char);
      try {
        if (std::is_same_v<T, float>) {
          data.at(row, col) = std::stof(token);
        } else if (std::is_same_v<T, int8_t> || std::is_same_v<T, int32_t>) {
          data.at(row, col) = std::stoi(token);
        } else {
          LOG(FATAL) << "Unsupported data type \n";
        }
      } catch (std::exception& e) {
        DLOG(ERROR) << "Parse CSV File meet error: " << e.what() << " row:" << row
                    << " col:" << col;
      }
      col += 1;
      CHECK(col <= cols) << "There are excessive elements on the column";
    }

    row += 1;
    CHECK(row <= rows) << "There are excessive elements on the row";
  }
  return data;
}
}// namespace mykuiper

#endif  // MYKUIPER_INFER_INCLUDE_DATA_LOAD_DATA_HPP_