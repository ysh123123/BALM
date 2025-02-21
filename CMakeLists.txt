cmake_minimum_required(VERSION 3.0.2)
project(ba_test)

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
find_package(GTSAM REQUIRED QUIET)
# Add subdirectory for proto_msg
add_subdirectory(proto_msg)

# Include directories
include_directories(
    include
    ${PCL_INCLUDE_DIRS}
    ${GTSAM_INCLUDE_DIR}
    ${PATC_INCLUDE_DIR}
)
file(GLOB_RECURSE SOURCE_FILES "include/**/*.cpp" "include/*.cpp" "src/*.cpp")

# Add the benchmark executable and link libraries
add_executable(${PROJECT_NAME}_slid_ba1 app/balm1_test.cpp ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME}_slid_ba1 PRIVATE proto_msg ${PCL_LIBRARIES} ${PATC_LIBRARIES})

add_executable(${PROJECT_NAME}_slid_ba2 app/balm2_test.cpp ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME}_slid_ba2 PRIVATE proto_msg gtsam ${PCL_LIBRARIES} ${PATC_LIBRARIES})

add_executable(${PROJECT_NAME}_hba app/hba_test.cpp ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME}_hba #[[${catkin_LIBRARIES}]] gtsam ${PCL_LIBRARIES} proto_msg ${PATC_LIBRARIES})

