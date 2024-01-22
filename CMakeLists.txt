cmake_minimum_required (VERSION 3.15)

# Set C++11 language standard
# flann util/random.h versions 1.8.4, 1.8.5, & 1.9.1 uses std::random_shuffle that is removed in C++17
# So staying at C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable the INSTALL project for building by default in VS
set(CMAKE_VS_INCLUDE_INSTALL_TO_DEFAULT_BUILD 1)

set(PROJECT "HDILib")
PROJECT(${PROJECT})

if (ENABLE_CODE_ANALYSIS AND WIN32)
  message("*******Code Analysis Run*******")
endif()

if (ENABLE_PID)
  message(STATUS "Position independent code code compilation ON")
endif()

set(HDILib_VERSION "undefined" CACHE STRING "HDILib Library version")
# Disallow in-source builds. 
# Build in sub dir e.g. source/build* is still allowed!
if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}" AND NOT $ENV{CI})
   message(FATAL_ERROR "In-source builds are not allowed!\n"
    "Make sure to remove CMakeCache.txt and CMakeFiles/ "
    "from the source directory!")
endif()

# This flag is used to allow conan to install the dependencies
option(INSTALL_PREBUILT_DEPENDENCIES "Use Conan to install prebuilt dependencies?" OFF)
#The cmake make sub directory contains the ConanSetup.cmake 
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
  "${PROJECT_SOURCE_DIR}/cmake")

# CONAN is only used for setting up dependencies if the INSTALL_PREBUILT_DEPENDENCIES is ON
if (INSTALL_PREBUILT_DEPENDENCIES OR DEFINED ENV{CI})
  find_package(lz4 REQUIRED)
  find_package(flann REQUIRED)
elseif(flann_INCLUDE_DIR)
  message(STATUS "Provided the flann_INCLUDE_DIR manually")
else()
  message(FATAL_ERROR "Provide the flann_INCLUDE_DIR pointing to your flann install include")
endif()

message(STATUS "Flann include: ${flann_INCLUDE_DIR}")

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

# NOTE CMake installs are defined in the sub-projects.
# ***Create a HdpsCore package to support the find_package command**

# Helper macro for packaging
include(CMakePackageConfigHelpers)

# Generate the version file for use with find_package
set(hdilib_package_version "${HDILib_VERSION}")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ConfigVersion.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/HDILibConfigVersion.cmake" @ONLY)
#write_basic_package_version_file(
#  "${CMAKE_CURRENT_BINARY_DIR}/HDILibConfigVersion.cmake"
#  VERSION "${HDILib_VERSION}"
#  # When the hdps core is stable compatibility can 
#  # change to AnyNewerVersion or SameMajorVersion
#  COMPATIBILITY ExactVersion
#)

set(INCLUDE_INSTALL_DIR include)
set(LIB_INSTALL_DIR lib)
set(CURRENT_BUILD_DIR "${CMAKE_BINARY_DIR}")
option(ENABLE_CODE_ANALYSIS "Use Static Code Analysis on build" OFF)

# create config file
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/HDILibConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/HDILibConfig.cmake"
  PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR CURRENT_BUILD_DIR
  INSTALL_DESTINATION lib/cmake/HDILib
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

# install the package files files
install(FILES
          "${CMAKE_CURRENT_BINARY_DIR}/HDILibConfig.cmake"
          "${CMAKE_CURRENT_BINARY_DIR}/HDILibConfigVersion.cmake"
        DESTINATION lib/cmake/HDILib
        COMPONENT HDI_PACKAGE
)
