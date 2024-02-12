cmake_minimum_required (VERSION 3.15)

option(ENABLE_CODE_ANALYSIS "Use Static Code Analysis on build" OFF)

# Set C++14 language standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable the INSTALL project for building by default in VS
set(CMAKE_VS_INCLUDE_INSTALL_TO_DEFAULT_BUILD 1)

set(PROJECT "HDILib")
PROJECT(${PROJECT})

option(ENABLE_AVX "Enable AVX support" OFF)
if (ENABLE_CODE_ANALYSIS AND WIN32)
  message("*******Code Analysis Run*******")
endif()

if (ENABLE_PID)
  message(STATUS "Position independent code code compilation ON")
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DWIN32 /EHsc /MP /permissive- /Zc:__cplusplus")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
endif(MSVC)

set(HDILib_VERSION "undefined" CACHE STRING "HDILib Library version")
# Disallow in-source builds. 
# Build in sub dir e.g. source/build* is still allowed!
if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}" AND NOT $ENV{CI})
   message(FATAL_ERROR "In-source builds are not allowed!\n"
    "Make sure to remove CMakeCache.txt and CMakeFiles/ "
    "from the source directory!")
endif()

#The cmake make sub directory contains the ConanSetup.cmake 
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
  "${PROJECT_SOURCE_DIR}/cmake")

# Test hardware avx capabilities
include(CMakeSetOptimizationAndAVX)

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

find_package(lz4 REQUIRED)
find_package(flann REQUIRED)
find_package(OpenMP)

if(OpenMP_CXX_FOUND)
	message (STATUS "OpenMP found")
elseif(CMAKE_GENERATOR STREQUAL Xcode)
	add_definitions( -D__USE_GCD__)
else()
    message(WARNING, "OpenMP not found!")
endif()

add_subdirectory (hdi/utils)
add_subdirectory (hdi/data)
add_subdirectory (hdi/dimensionality_reduction)

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
