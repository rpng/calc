
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

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native -std=c++11")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules) # FindEigen3.cmake

find_package(Boost COMPONENTS filesystem REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(CUDA QUIET)

if (NOT Caffe_ROOT_DIR)
	set(Caffe_ROOT_DIR $ENV{HOME}/caffe)
endif()

MESSAGE("${GREEN}Using Caffe_ROOT_DIR: ${Caffe_ROOT_DIR}${ENDL}")
set(Caffe_LIBS ${Caffe_ROOT_DIR}/build/lib/libcaffe.so)
set(Caffe_INCLUDE_DIRS ${Caffe_ROOT_DIR}/build/install/include)
MESSAGE("Caffe_LIBS: ${Caffe_LIBS}")
MESSAGE("Caffe_INCLUDE_DIRS: ${Caffe_INCLUDE_DIRS}")

include_directories(
	${OpenCV_INCLUDE_DIR}
	${Boost_INLCLUDE_DIRS}
	${Caffe_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIR}
	${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}
)


if (${CUDA_FOUND})
 	include_directories(${CUDA_INCLUDE_DIRS})
endif()

file(GLOB SOURCES "src/${PROJECT_NAME}/[A-z]*.cpp")

add_library(${PROJECT_NAME} SHARED
	${SOURCES}
)

target_link_libraries(${PROJECT_NAME}
	${OpenCV_LIBS}
	${Caffe_LIBS}
	${Boost_LIBRARIES}
	${EIGEN3_LIBS}
	glog
	pthread
)
if (${CUDA_FOUND})
	target_link_libraries(${PROJECT_NAME}
		${CUDA_LIBRARIES}
		${CUDA_CUBLAS_LIBRARIES}
		${CUDA_curand_LIBRARY}
	)
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
	${Caffe_INCLUDE_DIRS}
)


### Default model download ###########################

message( "\n\n${GREEN}Downloading default model if not already downloaded\n\n${ENDL}" )
execute_process(
	COMMAND ./get_model.sh
	WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)


#### Demo ###########################################

message( "\n\nCopying images and model to build directory for demo and testing....\n\n" )
execute_process(COMMAND cp -r ../src/images .)
execute_process(COMMAND cp -r ../calc_model .)

add_executable( demo src/demo/demo.cpp )
target_link_libraries( demo
	${PROJECT_NAME}
	${OpenCV_LIBS}
)


#### Speed Test ###########################################

add_executable( speed-test src/speed-test/speed-test.cpp )
target_link_libraries( speed-test
	${PROJECT_NAME}
	${OpenCV_LIBS}
	${Boost_LIBRARIES}
)

add_executable( vary-db-size src/speed-test/vary-db-size.cpp )
target_link_libraries( vary-db-size
	${PROJECT_NAME}
	${OpenCV_LIBS}
	${Boost_LIBRARIES}
)
	
#### Tests ##########################################

option(BUILD_TESTS "Build gtest module and python tests (if python wrappers are built)" ON)

if (BUILD_TESTS)

	message( "\n\n\n${GREEN}Building tests\n\n\n${ENDL}")

	set(PROJECT_TEST_NAME ${PROJECT_NAME}-test)
	
	include(ExternalProject)

	set(GTEST_FORCE_SHARED_CRT ON)
	set(GTEST_DISABLE_PTHREADS OFF)

	ExternalProject_Add(googletest
	    GIT_REPOSITORY https://github.com/nmerrill67/googletest.git
	    CMAKE_ARGS -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG:PATH=DebugLibs
	    -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE:PATH=ReleaseLibs
	    -Dgtest_force_shared_crt=${GTEST_FORCE_SHARED_CRT}
	    -Dgtest_disable_pthreads=${GTEST_DISABLE_PTHREADS}
	    -DBUILD_GTEST=ON
	    PREFIX "${CMAKE_CURRENT_BINARY_DIR}"
	    # Disable install step
	    INSTALL_COMMAND ""
	)

	# Specify include dir
	ExternalProject_Get_Property(googletest source_dir)
	set(GTEST_INCLUDE_DIRS ${source_dir}/googletest/include)

	# Specify MainTest's link libraries
	ExternalProject_Get_Property(googletest binary_dir)
	set(GTEST_LIBS_DIR ${binary_dir}/googlemock/gtest)

	enable_testing()
	
	include_directories(${GTEST_INCLUDE_DIRS})

	file(GLOB TEST_SRC_FILES ${PROJECT_SOURCE_DIR}/src/test/*.cpp)

	add_executable(${PROJECT_TEST_NAME} ${TEST_SRC_FILES})
	add_dependencies(${PROJECT_TEST_NAME} googletest)

	target_link_libraries(${PROJECT_TEST_NAME}
                ${PROJECT_NAME}
		${OpenCV_LIBS}
		${Caffe_LIBS}
		${Boost_LIBRARIES}
		${EIGEN3_LIBS}
		glog
	 	${GTEST_LIBS_DIR}/libgtest.a
        	${GTEST_LIBS_DIR}/libgtest_main.a
        )
	
	target_link_libraries(${PROJECT_TEST_NAME} ${CMAKE_THREAD_LIBS_INIT})

 	add_custom_target(tests COMMAND ./${PROJECT_TEST_NAME})
else()
	message( "\n\n\n${RED}Skipping test compilation \n\n\n${ENDL}")	
endif()


















