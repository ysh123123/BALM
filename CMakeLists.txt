cmake_minimum_required(VERSION 3.0.2)
project(balm2)

# Set the build type
set(CMAKE_BUILD_TYPE "Release")

# Set the C++ standard to C++17 and add compiler options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options(-std=c++17)

# Set C++ compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -fexceptions -pthread")

# Set C flags for exceptions
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")

# Define shared library path
set(SHARED_LIB_PATH ${CMAKE_BINARY_DIR}/lib)
message(STATUS "SHARED_LIB_PATH: ${SHARED_LIB_PATH}")

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(Ceres REQUIRED)
find_package(PATC REQUIRED)

# Add subdirectory for proto_msg
add_subdirectory(proto_msg)

# Include directories
include_directories(
    include
    ${PCL_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${PATC_INCLUDE_DIR}
)

# Add the benchmark executable and link libraries
add_executable(${PROJECT_NAME}_benchmark_realworld src/benchmark/benchmark_realworld.cpp)
target_link_libraries(${PROJECT_NAME}_benchmark_realworld PRIVATE proto_msg ${PCL_LIBRARIES} ${PATC_LIBRARIES})


#add_executable(consistency src/simulation/consistency.cpp)
#target_link_libraries(consistency ${catkin_LIBRARIES} ${PCL_LIBRARIES})
#
#add_executable(benchmark_virtual src/benchmark/benchmark_virtual.cpp)
#target_link_libraries(benchmark_virtual ${catkin_LIBRARIES} ${PCL_LIBRARIES})



#add_executable(PA_test src/compare_test/PA_test.cpp)
#target_link_libraries(PA_test ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES})
#
#add_executable(BAREG_test src/compare_test/BAREG_test.cpp)
#target_link_libraries(BAREG_test ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES})
#
#add_executable(BALM1_test src/compare_test/BALM1_test.cpp)
#target_link_libraries(BALM1_test ${catkin_LIBRARIES} ${PCL_LIBRARIES})
#
#add_executable(EF_test src/compare_test/EF_test.cpp src/compare_test/SE3/SO3.cpp src/compare_test/SE3/SE3.cpp)
#target_link_libraries(EF_test ${catkin_LIBRARIES} ${PCL_LIBRARIES})


