cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(Extract_main)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 添加 nlohmann/json 头文件的包含目录
include_directories(${PROJECT_SOURCE_DIR}/include)

# 设置 Eigen 库的路径（假设 Eigen 库已下载并解压到指定目录）
set(EIGEN3_INCLUDE_DIR "C:/source/eigen-3.4.0")
include_directories(${EIGEN3_INCLUDE_DIR})

# 添加 src 子目录
add_subdirectory(src)
