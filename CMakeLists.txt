cmake_minimum_required(VERSION 3.18)

project(DeepLearning)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

set(TENSORFLOW_LIB_DIR "/opt/tensorflow/lib")

add_executable(image_classification image_classification.cpp)
add_executable(object_detection object_detection.cpp)

target_include_directories(image_classification PRIVATE
    ${TENSORFLOW_LIB_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(image_classification
    ${TENSORFLOW_LIB_DIR}/libtensorflow.so
    ${TENSORFLOW_LIB_DIR}/libtensorflow_cc.so
    ${TENSORFLOW_LIB_DIR}/libtensorflow_framework.so
    ${OpenCV_LIBRARIES}
)

target_include_directories(object_detection PRIVATE
    ${TENSORFLOW_LIB_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(object_detection
    ${TENSORFLOW_LIB_DIR}/libtensorflow.so
    ${TENSORFLOW_LIB_DIR}/libtensorflow_cc.so
    ${TENSORFLOW_LIB_DIR}/libtensorflow_framework.so
    ${OpenCV_LIBRARIES}
)
