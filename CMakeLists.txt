cmake_minimum_required(VERSION 3.16)
project(mykuiper)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(./include)

find_package(benchmark REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Armadillo REQUIRED)
find_package(glog REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
aux_source_directory(./source/data DIR_DATA)
aux_source_directory(./source/runtime DIR_RUNTIME)
aux_source_directory(./source/runtime/pnnx DIR_RUNTIME)
aux_source_directory(./source/layer/abstract DIR_ABSTRACT_LAYER)
aux_source_directory(./source/layer/details DIR_BINOCULAR_LAYER)
aux_source_directory(./source/parser DIR_PARSER)
aux_source_directory(./source/utils/time DIR_UTILS)
aux_source_directory(./source/utils/math DIR_MATH)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

set(link_lib glog::glog)
IF (!WIN32)
    set(link_lib ${link_lib} pthread)
ENDIF ()
set(link_math_lib ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

add_library(kuiper SHARED ${DIR_DATA} ${DIR_PARSER} ${DIR_MATH} ${DIR_UTILS} ${DIR_RUNTIME} ${DIR_ABSTRACT_LAYER} ${DIR_BINOCULAR_LAYER} ${DIR_PARSER} )

target_link_libraries(kuiper ${link_lib} ${link_math_lib} OpenMP::OpenMP_CXX)
target_include_directories(kuiper PUBLIC ${benchmark_INCLUDE_DIRS})
target_include_directories(kuiper PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(kuiper PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(kuiper PUBLIC ${Armadillo_INCLUDE_DIR})


