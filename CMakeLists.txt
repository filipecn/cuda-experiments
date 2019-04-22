cmake_minimum_required(VERSION 3.7)

project(CUDA_EXPERIMENTS)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread")
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE "ON")

set(DEBUG_FLAGS "${DEBUG_FLAGS} -g -Wall -Wextra  -fPIC")
set(RELEASE_FLAGS "-O3 -Wall -Wextra  -fPIC")
set(CMAKE_CXX_FLAGS ${RELEASE_FLAGS})
set(CMAKE_CXX_FLAGS_DEBUG ${DEBUG_FLAGS})


include(ExternalProject)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/ext")
include(ponos)
message("PONOS_INCLUDE_DIR ${PONOS_INCLUDE_DIR}")
message("PONOS_LIBRARIES ${PONOS_LIBRARIES}")

enable_language("CUDA")
add_subdirectory(experiments)