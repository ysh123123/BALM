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
add_executable(${PROJECT_NAME}_slid_ba src/balm1.cpp
include/myso3/myso3.cpp)
target_link_libraries(${PROJECT_NAME}_slid_ba PRIVATE proto_msg ${PCL_LIBRARIES} ${PATC_LIBRARIES})

add_executable(${PROJECT_NAME}_slid_ba2 src/balm2.cpp
    include/myso3/myso3.cpp)
target_link_libraries(${PROJECT_NAME}_slid_ba2 PRIVATE proto_msg ${PCL_LIBRARIES} ${PATC_LIBRARIES})

#add_executable(${PROJECT_NAME}_consistency src/simulation/consistency.cpp)
#target_link_libraries(${PROJECT_NAME}_consistency PRIVATE proto_msg ${PCL_LIBRARIES} ${PATC_LIBRARIES})


