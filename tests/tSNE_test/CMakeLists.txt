cmake_minimum_required(VERSION 3.21)
project(tsne_tester)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

message (STATUS "Fetching argparse...")
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
    GIT_TAG "v3.2"
    EXCLUDE_FROM_ALL

)
FetchContent_MakeAvailable(argparse)
message(STATUS "hnswlib at ${argparse_SOURCE_DIR}")

message (STATUS "Fetching rapidcsv...")
FetchContent_Declare(
  rapidcsv
  GIT_REPOSITORY "https://github.com/d99kris/rapidcsv.git"
  GIT_TAG        "v8.88"
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(rapidcsv)
message(STATUS "hnswlib at ${rapidcsv_SOURCE_DIR}")

add_executable(
    tsne_tester
    tSNEtester.cpp
)

target_include_directories(tsne_tester PRIVATE "${argparse_SOURCE_DIR}/include")
target_include_directories(tsne_tester PRIVATE "${rapidcsv_SOURCE_DIR}/src")
target_include_directories(tsne_tester 
    PRIVATE "${CMAKE_SOURCE_DIR}/hdi/data" 
    PRIVATE "${CMAKE_SOURCE_DIR}/hdi/dimensionality_reduction"
    PRIVATE "${CMAKE_SOURCE_DIR}/hdi/utils")
target_link_libraries(tsne_tester PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/../hdi/dimensionality_reduction/$<CONFIG>/hdiutils${CMAKE_STATIC_LIBRARY_SUFFIX})
target_link_libraries(tsne_tester PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/../hdi/dimensionality_reduction/$<CONFIG>/hdidata${CMAKE_STATIC_LIBRARY_SUFFIX})
target_link_libraries(tsne_tester PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/../hdi/dimensionality_reduction/$<CONFIG>/hdidimensionalityreduction${CMAKE_STATIC_LIBRARY_SUFFIX})
