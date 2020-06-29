cmake_minimum_required (VERSION 3.9)

# Set C++11 language standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable the INSTALL project for building by default in VS
set(CMAKE_VS_INCLUDE_INSTALL_TO_DEFAULT_BUILD 1)

set(PROJECT "HDILib")
PROJECT(${PROJECT})

# Disallow in-source builds. 
# Build in sub dir e.g. source/build* is still allowed!
if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}" AND NOT $ENV{CI})
   message(FATAL_ERROR "In-source builds are not allowed!\n"
    "Make sure to remove CMakeCache.txt and CMakeFiles/ "
    "from the source directory!")
endif()

# This flag is used to allow conan to install the dependencies
option(HDILIB_BUILD_WITH_CONAN "Should Conan package manager be used?" OFF)
#The cmake make sub directory contains the ConanSetup.cmake 
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
  "${PROJECT_SOURCE_DIR}/cmake")

# CONAN is only used for setting up dependencies if the HDILIB_BUILD_WITH_CONAN is ON
include(ConanSetup)
message(STATUS "*** (HDI_EXTERNAL_FLANN_INCLUDE_DIR) ${HDI_EXTERNAL_FLANN_INCLUDE_DIR}")

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
        execute_process(COMMAND ${BREW} --prefix libomp OUTPUT_VARIABLE BREW_LIBOMP_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif("${BREW}" STRGREATER "")
    
    if(EXISTS ${BREW_LIBOMP_PREFIX})
        message(STATUS "Found OpenMP in ${BREW_LIBOMP_PREFIX}")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${BREW_LIBOMP_PREFIX}/lib/libomp.dylib")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        include_directories("${BREW_LIBOMP_PREFIX}/include")
        link_directories("${BREW_LIBOMP_PREFIX}/lib/")
    else()
        message(WARNING "Brew found, but OpenMP support could not be detected, using Grand Central Dispatch instead.")
        add_definitions( -D__USE_GCD__)
    endif(EXISTS ${BREW_LIBOMP_PREFIX})    

#Win/Linux
endif()

find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
	message (STATUS "OpenMP found ${OpenMP_CXX_LIB_NAMES}")
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}") 
    target_link_libraries(${PROJECT} OpenMP::OpenMP_CXX)
else()
    message(WARNING, "OpenMP not found!")    
endif()

option(HDI_USE_ROARING "Use roaring bitmaps" ON)

if (HDI_USE_ROARING)
    add_definitions(-DPREPROC_USE_ROARING)
endif(HDI_USE_ROARING)

add_subdirectory (hdi/utils)
add_subdirectory (hdi/data)
add_subdirectory (hdi/dimensionality_reduction)

add_dependencies(hdidata hdiutils)
add_dependencies(hdidimensionalityreduction hdiutils)
add_dependencies(hdidimensionalityreduction hdidata)