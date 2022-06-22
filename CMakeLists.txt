cmake_minimum_required(VERSION 3.18)

project(DeepLearning)

set(CMAKE_CXX_STANDARD 11)
set(USR_LOCAL_LIB_DIR "/usr/local/lib")
set(USR_LOCAL_INCLUDE_DIR "/usr/local/include")
set(TENSORFLOW_LIB_DIR "/opt/tensorflow/lib")

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

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
    ${USR_LOCAL_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(object_detection
    ${TENSORFLOW_LIB_DIR}/libtensorflow.so
    ${TENSORFLOW_LIB_DIR}/libtensorflow_cc.so
    ${TENSORFLOW_LIB_DIR}/libtensorflow_framework.so
    ${USR_LOCAL_LIB_DIR}/libavcodec.so
    ${USR_LOCAL_LIB_DIR}/libavformat.so
    ${USR_LOCAL_LIB_DIR}/libavutil.so
    ${USR_LOCAL_LIB_DIR}/libswscale.so
    ${OpenCV_LIBRARIES}
)
