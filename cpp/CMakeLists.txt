cmake_minimum_required(VERSION 3.16)
project(NCNN_YOLO LANGUAGES CXX)

# 设置 C++ 17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置指定项目生成的可执行文件路径
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 查找所需的包
find_package(OpenCV CONFIG REQUIRED)
find_package(ncnn CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

if(MSVC)
    # 增加 UTF-8 支持
    add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
    # 严格检查
    add_compile_options(/W4 /WX)
else()
    # 严格检查
    add_compile_options(-Wall -Wextra)
endif()

add_compile_definitions(
    # 如果是 Debug 模式，定义 IS_DEBUG
    $<$<CONFIG:Debug>:IS_DEBUG>
    # 如果是 Release 模式，定义 IS_RELEASE
    $<$<CONFIG:Release>:IS_RELEASE>
)

include_directories(./include)

add_library(NCNN_YOLO SHARED
    src/ncnn_yolo.cpp
    src/yolo_cls.cpp
    src/yolo_det.cpp
    src/yolo_obb.cpp
    src/yolo_pose.cpp
    src/yolo_seg.cpp
)
target_link_libraries(NCNN_YOLO PUBLIC
    ncnn
    nlohmann_json::nlohmann_json
    ${OpenCV_LIBS}
)

add_executable(main src/main.cpp)
target_link_libraries(main NCNN_YOLO)