cmake_minimum_required(VERSION 3.21)
project(tsne_tester)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
    GIT_TAG "v3.2"

)
FetchContent_MakeAvailable(argparse)

FetchContent_Declare(
  rapidcsv
  GIT_REPOSITORY "https://github.com/d99kris/rapidcsv.git"
  GIT_TAG        "v8.88"
)
FetchContent_MakeAvailable(rapidcsv)

add_executable(
    ${PROJECT_NAME}
    tSNEtester.cpp
)