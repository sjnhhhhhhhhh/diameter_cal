# 项目名称
project(Extract_main)

# 查找源文件
set(SOURCE_FILES extract_c++.cpp)

# 添加可执行文件
add_executable(Extract ${SOURCE_FILES})

# 添加 include 目录
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${EIGEN3_INCLUDE_DIR})
