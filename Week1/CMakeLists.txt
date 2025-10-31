cmake_minimum_required(VERSION 3.10)
project(task)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 OpenCV
find_package(OpenCV REQUIRED)

if(OpenCV_FOUND)
    message(STATUS "Found OpenCV ${OpenCV_VERSION}")
endif()

# 设置所有可执行文件输出到 bin 目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Task1
add_executable(Task1 Task1/code/task1.cpp)
target_include_directories(Task1 PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(Task1 ${OpenCV_LIBS})

add_custom_command(
    TARGET Task1
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/Task1/output"
        "$<TARGET_FILE_DIR:Task1>/images"
)

# Task2  
add_executable(Task2 Task2/code/task2.cpp)
target_include_directories(Task2 PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(Task2 ${OpenCV_LIBS})

add_custom_command(
    TARGET Task2
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/Task2/output"
        "$<TARGET_FILE_DIR:Task2>/images"
)

# Task3
add_executable(Task3 Task3/code/task3.cpp) 
target_include_directories(Task3 PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(Task3 ${OpenCV_LIBS})

add_custom_command(
    TARGET Task3
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/Task3/output"
        "$<TARGET_FILE_DIR:Task3>/images"
)