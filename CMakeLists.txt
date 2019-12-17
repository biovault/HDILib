cmake_minimum_required (VERSION 3.9)

# Set C++11 language standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable the INSTALL project for building by default in VS
set(CMAKE_VS_INCLUDE_INSTALL_TO_DEFAULT_BUILD 1)

set(PROJECT "HDILib")
PROJECT(${PROJECT})
# If the CMAKE_INSTALL_PREFIX has not been set by the user, set it to the build folder
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    message(STATUS "Default CMAKE_INSTALL_PREFIX detected. Setting to build directory.")
    set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "Default install path" FORCE )
else()
    message(STATUS "Custom CMAKE_INSTALL_PREFIX detected. Leaving it untouched.")
endif()

if (CMAKE_GENERATOR STREQUAL Xcode)
    add_definitions(-DGL_SILENCE_DEPRECATION)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.12" CACHE STRING "Minimum OS X deployment version")
endif()

message(STATUS "CMAKE_GENERATOR: ${CMAKE_GENERATOR}")
#Setting custom compilers for MacOS
if (CMAKE_GENERATOR STREQUAL Xcode)
    find_program(BREW NAMES brew)
    if("${BREW}" STRGREATER "")
        message(STATUS "Running on macOS Darwin using ${BREW}")
        execute_process(COMMAND ${BREW} --prefix llvm OUTPUT_VARIABLE BREW_LLVM_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND ${BREW} --prefix libomp OUTPUT_VARIABLE BREW_LIBOMP_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif("${BREW}" STRGREATER "")
    
    message(STATUS "Trying to find llvm using ${BREW_LLVM_PREFIX}/lib/cmake/llvm")
    find_package(LLVM REQUIRED CONFIG HINTS "${BREW_LLVM_PREFIX}/lib/cmake/llvm")
    if(EXISTS ${LLVM_INSTALL_PREFIX})
        message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} in ${LLVM_INSTALL_PREFIX}")
        #message(STATUS "Using LLVMConfig.cmake in ${LLVM_DIR}")
        #message(STATUS "Found LLVM headers in ${LLVM_INCLUDE_DIRS}")

        find_package(Clang REQUIRED CONFIG HINTS "${BREW_LLVM_PREFIX}/lib/cmake/clang")
        message(STATUS "Found Clang in ${CLANG_INSTALL_PREFIX}")
        #message(STATUS "Found Clang headers in ${CLANG_INCLUDE_DIRS}")

        set(CMAKE_C_COMPILER "${LLVM_INSTALL_PREFIX}/bin/clang")
        set(CMAKE_CXX_COMPILER "${LLVM_INSTALL_PREFIX}/bin/clang++")
        set(CMAKE_XCODE_ATTRIBUTE_CC "${LLVM_INSTALL_PREFIX}/bin/clang")
        set(CMAKE_XCODE_ATTRIBUTE_CXX "${LLVM_INSTALL_PREFIX}/bin/clang++")
        
        set(CMAKE_XCODE_ATTRIBUTE_COMPILER_INDEX_STORE_ENABLE "NO")

        if(EXISTS ${BREW_LIBOMP_PREFIX})
            message(STATUS "Found OpenMP in ${BREW_LIBOMP_PREFIX}")
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY "${BREW_LIBOMP_PREFIX}/lib/libomp.dylib")
            include_directories("${BREW_LIBOMP_PREFIX}/include")
        else()
            message(WARNING "Brew found, but OpenMP support could not be detected, using Grand Central Dispatch instead.")
            add_definitions( -D__USE_GCD__)
        endif(EXISTS ${BREW_LIBOMP_PREFIX})
    else()
        message(WARNING "Could not find custom llvm using base install.")
        message(WARNING "No OpenMP support detected, using Grand Central Dispatch instead.")
        add_definitions( -D__USE_GCD__)
    endif(EXISTS ${LLVM_INSTALL_PREFIX})
#Win/Linux
else()
  find_package(OpenMP REQUIRED)
  if (OPENMP_FOUND)
    message (STATUS "OpenMP found")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
endif()

option(HDI_USE_ROARING "Use roaring bitmaps" ON)

if (HDI_USE_ROARING)
    add_definitions(-DPREPROC_USE_ROARING)
endif(HDI_USE_ROARING)

add_subdirectory (hdi/dimensionality_reduction)
add_subdirectory (hdi/data)
add_subdirectory (hdi/utils)
