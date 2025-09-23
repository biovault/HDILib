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

find_package(fmt CONFIG REQUIRED)
find_package(kompute CONFIG REQUIRED)

target_include_directories(tsne_tester PRIVATE "${argparse_SOURCE_DIR}/include")
target_include_directories(tsne_tester PRIVATE "${rapidcsv_SOURCE_DIR}/src")
target_include_directories(tsne_tester 
    PRIVATE "${CMAKE_SOURCE_DIR}")
message(STATUS "binary at ${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(tsne_tester PRIVATE kompute::kompute)
target_link_libraries(tsne_tester PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/../../hdi/utils/$<CONFIG>/hdiutils${CMAKE_STATIC_LIBRARY_SUFFIX})
target_link_libraries(tsne_tester PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/../../hdi/data/$<CONFIG>/hdidata${CMAKE_STATIC_LIBRARY_SUFFIX})
target_link_libraries(tsne_tester PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/../../hdi/dimensionality_reduction/$<CONFIG>/hdidimensionalityreduction${CMAKE_STATIC_LIBRARY_SUFFIX})
if(NOT LZ4_TARGET)
    if(TARGET LZ4::lz4_static)
        set(LZ4_TARGET LZ4::lz4_static)
    elseif(TARGET LZ4::lz4_shared)
        set(LZ4_TARGET LZ4::lz4_shared)
    elseif(TARGET lz4::lz4)
        set(LZ4_TARGET lz4::lz4)
    elseif(TARGET LZ4::lz4)    # intentionally UPPERCASE::LOWERCASE
        set(LZ4_TARGET LZ4::lz4)
    else()
        message(FATAL_ERROR "No LZ4 target found.")
    endif()
endif()

#message (STATUS "Flann link library: " ${FLANN_TARGET})
#target_link_libraries(tsne_tester PRIVATE ${FLANN_TARGET})
message (STATUS "Linking lz4 library " ${LZ4_TARGET})
target_link_libraries(tsne_tester PRIVATE ${LZ4_TARGET})