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

# add_executable(main main.cpp ${DIR_DATA} ${DIR_PARSER} ${DIR_MATH} ${DIR_UTILS} ${DIR_RUNTIME} ${DIR_ABSTRACT_LAYER} ${DIR_BINOCULAR_LAYER} ${DIR_PARSER} )
# target_link_libraries(main ${link_lib} ${link_math_lib})
# target_link_directories(main PUBLIC ${PROJECT_SOURCE_DIR}/lib)
# target_link_libraries(main kuiper)
# target_include_directories(main PUBLIC ${glog_INCLUDE_DIR})
# target_include_directories(main PUBLIC ${GTest_INCLUDE_DIR})
# target_include_directories(main PUBLIC ${Armadillo_INCLUDE_DIR})

# # mathfun library defines，这是干嘛的
# add_compile_definitions(SSE_MATHFUN_WITH_CODE USE_SSE_AUTO)


# # 本项目的开发者请使用set(DEVELOPMENT ON)或者在cmake中添加-DDEVELOPMENT=ON将选项打开
# # 如果是，打开了test和bench功能，增加了bench和test的directory。
# option(DEVELOPMENT ON)
# if (${DEVELOPMENT})
#     message(STATUS "DEVELOPMENT MODE ON")
#     enable_testing()
#     add_subdirectory(bench)
#     add_subdirectory(test)
# endif ()

# 本项目的开发者请使用set(DEVELOPMENT ON)或者在cmake中添加-DDEVELOPMENT=ON将选项打开
# 如果是，打开了test和bench功能，增加了bench和test的directory。
#option(DEVELOPMENT ON)
#if (${DEVELOPMENT}) 我们这里直接打开
message(STATUS "DEVELOPMENT MODE ON")
enable_testing()
# 这里add_subdirectory之后的子文件夹，就需要在里面包含CMakeLists文件。 相对于上面的aux_source_directory是不一样的。
add_subdirectory(bench)
add_subdirectory(test)
#endif ()
