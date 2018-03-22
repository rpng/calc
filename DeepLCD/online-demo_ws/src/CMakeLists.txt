
cmake_minimum_required(VERSION 2.8)
project(deeplcd)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

# colors, because why not :)
string(ASCII 27 Esc)
set(RED "${Esc}[31m")
set(GREEN "${Esc}[32m")
set(ENDL "${Esc}[m")

MESSAGE("${GREEN}Build type: ${CMAKE_BUILD_TYPE}${ENDL}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native -std=c++11")

find_package(Boost COMPONENTS filesystem REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 3 REQUIRED)
find_package(CUDA QUIET)

if (NOT Caffe_ROOT_DIR)
	set(Caffe_ROOT_DIR $ENV{HOME}/caffe)
endif()

set(Caffe_LIBS ${Caffe_ROOT_DIR}/build/lib/libcaffe.so)
set(Caffe_INCLUDE_DIRS ${Caffe_ROOT_DIR}/build/install/include)

find_package(catkin REQUIRED COMPONENTS roscpp cv_bridge geometry_msgs image_transport)

include_directories(
	${OpenCV_INCLUDE_DIR}
	${Boost_INLCLUDE_DIRS}
	${Caffe_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIR}
	${PROJECT_SOURCE_DIR}/../../include/${PROJECT_NAME}
	${catkin_INCLUDE_DIRS}
	${PROJECT_SOURCE_DIR}
)
if (${CUDA_FOUND})
 	include_directories(${CUDA_INCLUDE_DIRS})
endif()

catkin_package()


add_executable(online-demo online-demo.cpp)
target_link_libraries( online-demo
	${PROJECT_SOURCE_DIR}/../../build/lib${PROJECT_NAME}.so
	${catkin_LIBRARIES}
	${OpenCV_LIBS}
	${Boost_LIBRARIES}
	${EIGEN3_LIBS}
)
if (${CUDA_FOUND})
	target_link_libraries(online-demo
		${CUDA_LIBRARIES}
		${CUDA_CUBLAS_LIBRARIES}
		${CUDA_curand_LIBRARY}
	)
endif()


execute_process(COMMAND cp -r ../../src/images ${CATKIN_DEVEL_PREFIX})
execute_process(COMMAND cp -r ../../calc_model ${CATKIN_DEVEL_PREFIX})
