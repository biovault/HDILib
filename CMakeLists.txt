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

find_package(OpenMP)

if(OPENMP_FOUND)
	message (STATUS "OpenMP found ${OpenMP_CXX_LIB_NAMES}")
	include_directories(${OpenMP_CXX_INCLUDE_DIRS})
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}") 
elseif(CMAKE_GENERATOR STREQUAL Xcode)
	add_definitions( -D__USE_GCD__)
else()
    message(WARNING, "OpenMP not found!")    
endif()

if(MSVC)
	add_definitions(/arch:AVX)
	add_definitions(-DUSE_SSE) # needed so SSE functions are available in HNSWLib
	add_definitions(-DUSE_AVX) # USE_AVX is used both in HNSWLib and Annoy. Setting it from CMake is the most convenient
	message(STATUS "AVX and/or SSE instructions enabled for HNSWLib and/or Annoy")
endif(MSVC)

add_subdirectory (hdi/utils)
add_subdirectory (hdi/data)
add_subdirectory (hdi/dimensionality_reduction)

add_dependencies(hdidata hdiutils)
add_dependencies(hdidimensionalityreduction hdiutils)
add_dependencies(hdidimensionalityreduction hdidata)