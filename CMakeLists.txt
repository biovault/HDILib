cmake_minimum_required (VERSION 3.22)

option(USE_VULKAN_KOMPUTE "Build the repo using the Vulkan kompute library - by default ON" ON)

if(USE_VULKAN_KOMPUTE)
    list(APPEND VCPKG_MANIFEST_FEATURES "kompute")

    # Make sure vcpkg toolchain is picked up (or pass -DCMAKE_TOOLCHAIN_FILE=...)
    # Example: cmake .. -DCMAKE_TOOLCHAIN_FILE=[vcpkg-root]/scripts/buildsystems/vcpkg.cmake
    set(VCPKG_OVERLAY_PORTS "${CMAKE_CURRENT_LIST_DIR}/vcpkg-overlays/kompute")

    if(WIN32 AND NOT DEFINED VCPKG_LIBRARY_LINKAGE)
        set(VCPKG_HOST_TRIPLET "x64-windows-static-md" CACHE STRING "")
        set(VCPKG_TARGET_TRIPLET "x64-windows-static-md" CACHE STRING "")
        set(VCPKG_LIBRARY_LINKAGE "static" CACHE STRING "")
    endif()
endif()

set(PROJECT "HDILib")
PROJECT(${PROJECT})

# -----------------------------------------------------------------------------
# User Options
# -----------------------------------------------------------------------------
option(HDILib_ENABLE_AVX "Enable AVX support" OFF)
option(HDILib_ENABLE_CODE_ANALYSIS "Use Static Code Analysis on build with MSVC" OFF)
option(HDILib_ENABLE_PID "Set POSITION_INDEPENDENT_CODE property for all targets" OFF)
option(HDILib_INSTALL "Enable installation of the HDILib" ON)
set(HDILib_OPTIMIZATION_LEVEL "2" CACHE STRING "Optimization level for all targets in release builds, e.g. 0, 1, 2")
set(HDILib_VERSION "1.3.0a1" CACHE STRING "HDILib Library version")

if (HDILib_ENABLE_AVX)
    message(STATUS "ENABLE_AVX is ON")
endif()

if (HDILib_ENABLE_CODE_ANALYSIS AND WIN32)
    message(STATUS "Code Analysis ENABLED")
endif()

if (HDILib_ENABLE_PID)
    message(STATUS "Position independent code compilation ON")
endif()

message(STATUS "Building: HDILib ${HDILib_VERSION}")
message(STATUS "CMAKE_GENERATOR: ${CMAKE_GENERATOR}")

# -----------------------------------------------------------------------------
# CMake Options
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable the INSTALL project for building by default in VS
set(CMAKE_VS_INCLUDE_INSTALL_TO_DEFAULT_BUILD 1)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DWIN32 /EHsc /MP /permissive- /Zc:__cplusplus")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /MD")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
endif(MSVC)

# Disallow in-source builds. 
# Build in sub dir e.g. source/build* is still allowed!
if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}" AND NOT ${IN_CONAN_BUILD})
   message(FATAL_ERROR "In-source builds are not allowed!\n"
    "Make sure to remove CMakeCache.txt and CMakeFiles/ "
    "from the source directory!")
endif()

#The cmake make sub directory contains the i.a. ConanSetup.cmake 
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

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

include(FetchContent)

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
find_package(lz4 CONFIG)
if(NOT ${lz4_FOUND})
    find_package(PkgConfig REQUIRED)

    pkg_check_modules(lz4 REQUIRED IMPORTED_TARGET liblz4)
    set(LZ4_TARGET PkgConfig::lz4)
endif()

if(NOT ${lz4_FOUND})
    message (FATAL_ERROR "lz4 NOT found")
endif()
message (STATUS "Found lz4 version ${lz4_VERSION}")

find_package(flann CONFIG)
if(NOT ${flann_FOUND})
    find_package(PkgConfig REQUIRED)

    pkg_check_modules(flann REQUIRED flann)
    set(FLANN_TARGET flann_cpp)
    set(flann_INCLUDE_DIR "${flann_INCLUDE_DIRS}/flann")
endif()

if(NOT ${flann_FOUND})
    message (FATAL_ERROR "flann NOT found")
endif()
message (STATUS "Found flann version ${flann_VERSION}")

find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    message (STATUS "OpenMP found")
elseif(CMAKE_GENERATOR STREQUAL Xcode)
    add_definitions( -D__USE_GCD__)
else()
    message(WARNING, "OpenMP not found!")
endif()

message (STATUS "Fetching hnswlib...")
FetchContent_Declare(
  hnswlib
  GIT_REPOSITORY https://github.com/nmslib/hnswlib
  GIT_TAG v0.8.0
  EXCLUDE_FROM_ALL
)

FetchContent_MakeAvailable(hnswlib)

message(STATUS "hnswlib at ${hnswlib_SOURCE_DIR}")

message (STATUS "Fetching annoy...")
FetchContent_Declare(
  annoy
  GIT_REPOSITORY https://github.com/spotify/annoy
  GIT_TAG v1.17.2
  EXCLUDE_FROM_ALL
)

FetchContent_MakeAvailable(annoy)

message(STATUS "annoy at ${Annoy_SOURCE_DIR}")

if (USE_VULKAN_KOMPUTE)
    # Neede for fmt library used in vulkan 
    add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
endif()
# -----------------------------------------------------------------------------
# Projects
# -----------------------------------------------------------------------------
add_subdirectory (hdi/utils)
add_subdirectory (hdi/data)
add_subdirectory (hdi/dimensionality_reduction)

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
option(ENABLE_TESTS "Enable building the test executables" ON)
if (ENABLE_TESTS)
    add_subdirectory(tests)
    add_dependencies(tsne_tester hdiutils hdidimensionalityreduction hdidata)
endif()



# -----------------------------------------------------------------------------
# Create HDILib find_package
# -----------------------------------------------------------------------------
# NOTE CMake installs are defined in the sub-projects.

# Helper macro for packaging
include(CMakePackageConfigHelpers)

# Generate the version file for use with find_package
set(hdilib_package_version "${HDILib_VERSION}")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ConfigVersion.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/HDILibConfigVersion.cmake" @ONLY)
#write_basic_package_version_file(
#  "${CMAKE_CURRENT_BINARY_DIR}/HDILibConfigVersion.cmake"
#  VERSION "${HDILib_VERSION}"
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

if(${HDILib_INSTALL})
    install(FILES
              "${CMAKE_CURRENT_BINARY_DIR}/HDILibConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/HDILibConfigVersion.cmake"
            DESTINATION lib/cmake/HDILib
            COMPONENT HDI_PACKAGE
    )
endif()


