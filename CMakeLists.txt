cmake_minimum_required(VERSION 3.18)

project(ImageClassification)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

set(TENSORFLOW_LIB_DIR "/opt/tensorflow/lib")

add_executable(classification main.cpp)
# target_include_directories(get_prediction PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_include_directories(classification PRIVATE
    ${TENSORFLOW_LIB_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(classification
    ${TENSORFLOW_LIB_DIR}/libtensorflow.so
    ${TENSORFLOW_LIB_DIR}/libtensorflow_cc.so
    ${TENSORFLOW_LIB_DIR}/libtensorflow_framework.so
    ${OpenCV_LIBRARIES}
)

