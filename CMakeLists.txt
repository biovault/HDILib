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
option(HDILib_BUILD_TESTS "Build tests for HDILib" OFF)
set(HDILib_OPTIMIZATION_LEVEL "2" CACHE STRING "Optimization level for all targets in release builds, e.g. 0, 1, 2")
set(HDILib_VERSION "2.0.0" CACHE STRING "HDILib Library version")

if(HDILib_ENABLE_AVX)
    message(STATUS "ENABLE_AVX is ON")
endif()

if(HDILib_ENABLE_CODE_ANALYSIS AND WIN32)
    message(STATUS "Code Analysis ENABLED")
endif()

if(HDILib_ENABLE_PID)
    message(STATUS "Position independent code compilation ON")
endif()

message(STATUS "Building: HDILib ${HDILib_VERSION}")
message(STATUS "CMAKE_GENERATOR: ${CMAKE_GENERATOR}")

# -----------------------------------------------------------------------------
# CMake Options
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

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

if (NOT TARGET hnswlib::hnswlib)
    message (STATUS "Fetching hnswlib...")
    FetchContent_Declare(
      hnswlib
      GIT_REPOSITORY https://github.com/nmslib/hnswlib
      GIT_TAG v0.8.0
      EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(hnswlib)
else()
    get_target_property(hnswlib_SOURCE_DIR hnswlib::hnswlib INTERFACE_INCLUDE_DIRECTORIES)
endif()

message(STATUS "hnswlib at ${hnswlib_SOURCE_DIR}")

if (NOT TARGET Annoy::Annoy)
    message (STATUS "Fetching annoy...")
    FetchContent_Declare(
      annoy
      GIT_REPOSITORY https://github.com/spotify/annoy
      GIT_TAG v1.17.3
      EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(annoy)
    set(Annoy_SOURCE_DIR "${Annoy_SOURCE_DIR}/src")
else()
    get_target_property(Annoy_SOURCE_DIR Annoy::Annoy INTERFACE_INCLUDE_DIRECTORIES)
    set(Annoy_SOURCE_DIR "${Annoy_SOURCE_DIR}/annoy")
endif()

message(STATUS "annoy at ${Annoy_SOURCE_DIR}")

if (USE_VULKAN_KOMPUTE)
    # Neede for fmt library used in vulkan 
    add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
endif()
# -----------------------------------------------------------------------------
# Projects
# -----------------------------------------------------------------------------
add_subdirectory(hdi/utils)
add_subdirectory(hdi/data)
add_subdirectory(hdi/dimensionality_reduction)

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

# install the package files
# install the vcpkg build files
function(patch_file)
  # Step through the file line by line replacing 
  # the match expression with the replace expression
  # Overwrite the existing file
  cmake_parse_arguments(PARSE_ARGV 0 "patch_arg" "" "INPUT_PATH;MATCH_EXPRESSION;REPLACE_EXPRESSION" "")
  set(eol "\n")
  set(patched_result "")
  file(STRINGS "${patch_arg_INPUT_PATH}" input_lines)
  #message("Unpatched file ${input_lines}")
  foreach(line IN LISTS input_lines)
    string(REPLACE "${patch_arg_MATCH_EXPRESSION}" "${patch_arg_REPLACE_EXPRESSION}" patched_line "${line}")
    set(patched_result "${patched_result}${patched_line}${eol}")
  endforeach()
  #message("New patched file ${patched_result}")
  file(WRITE "${patch_arg_INPUT_PATH}" "${patched_result}")
endfunction()

function(add_config_to_cmake_target_files)
  # 1. Find CMake target files using a glob path (PATH_GLOB).
  # 2. Patch each file found so that the lib path contains a 
  # config (i.e. Release, Debug or RelWithDebInfo) element.
  # This matches the convention we use in multi-build packages

  cmake_parse_arguments(PARSE_ARGV 0 "fix_arg" "" "LIB_NAME;PATH_GLOB" "")
  #message("globbing: ${fix_arg_PATH_GLOB}")
  file(GLOB _cmake_config_files "${fix_arg_PATH_GLOB}")
  #message("glob list: ${_cmake_config_files}")
  foreach(_cmake_config_file IN LISTS _cmake_config_files)
      set(_upper_file "")
      string(TOUPPER "${_cmake_config_file}" _upper_file)
      #message("Checking: ${_upper_file}")
      # Identify if each file is a Release, Debug or RelWithDebugInfo
      # and apply the appropriate patch
      set(_found -1)
      string(FIND "${_upper_file}" "RELEASE.CMAKE" _found)
      if (_found GREATER -1)
        patch_file(
          INPUT_PATH "${_cmake_config_file}" 
          MATCH_EXPRESSION "lib/${fix_arg_LIB_NAME}" 
          REPLACE_EXPRESSION "Release/${fix_arg_LIB_NAME}"
        )
      endif()
      string(FIND "${_upper_file}" "DEBUG.CMAKE" _found)
      if (_found GREATER -1)
        # Eliminate the "debug" element - we package this differently
        message(STATUS "Change debug/lib for ${fix_arg_LIB_NAME} in ${_upper_file}")
        patch_file(
          INPUT_PATH "${_cmake_config_file}" 
          MATCH_EXPRESSION "debug/lib/${fix_arg_LIB_NAME}" 
          REPLACE_EXPRESSION "lib/${fix_arg_LIB_NAME}"
        )
        patch_file(
          INPUT_PATH "${_cmake_config_file}" 
          MATCH_EXPRESSION "lib/${fix_arg_LIB_NAME}" 
          REPLACE_EXPRESSION "Debug/${fix_arg_LIB_NAME}"
        )
      endif()
      string(FIND "${_upper_file}" "RELWITHDEBINFO.CMAKE" _found)
      if (_found GREATER -1)
        patch_file(
          INPUT_PATH "${_cmake_config_file}" 
          MATCH_EXPRESSION "lib/${fix_arg_LIB_NAME}" 
          REPLACE_EXPRESSION "RelWithDebInfo/${fix_arg_LIB_NAME}"
        )
      endif()
  endforeach()
endfunction()

function(fix_include_path)
  # Insert a dependency name element into the include path
  # In the package include each extra dependency has its own sub-dir
  # under include. 
  cmake_parse_arguments(PARSE_ARGV 0 "inc_arg" "" "DEPENDENCY_NAME;TARGET_PATH" "")

  patch_file(
    INPUT_PATH "${inc_arg_TARGET_PATH}" 
    MATCH_EXPRESSION "\${_IMPORT_PREFIX}/include" 
    REPLACE_EXPRESSION "\${_IMPORT_PREFIX}/../include"
  )
endfunction()

if(${HDILib_INSTALL})
    install(FILES
              "${CMAKE_CURRENT_BINARY_DIR}/HDILibConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/HDILibConfigVersion.cmake"
            DESTINATION lib/cmake/HDILib
            COMPONENT HDI_PACKAGE
    )
    if (USE_VULKAN_KOMPUTE)
      # TBD possibly move this to an external script
      # Install the config.cmake files for the vcpkg dependencies
      message("Fix config in ${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/share/kompute/komputeTargets-*.cmake")
      add_config_to_cmake_target_files(
        LIB_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}kompute${CMAKE_STATIC_LIBRARY_SUFFIX}" 
        PATH_GLOB "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/share/kompute/komputeTargets-*.cmake"
      )
      add_config_to_cmake_target_files(
        LIB_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}kp_logger${CMAKE_STATIC_LIBRARY_SUFFIX}" 
        PATH_GLOB "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/share/kompute/komputeTargets-*.cmake"
      )
      fix_include_path(
        DEPENDENCY_NAME "kompute"
        TARGET_PATH "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/share/kompute/komputeTargets.cmake"
      )
      install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/share/kompute" 
        DESTINATION lib/cmake COMPONENT HDI_PACKAGE)
      add_config_to_cmake_target_files(
        LIB_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}fmtd${CMAKE_STATIC_LIBRARY_SUFFIX}" 
        PATH_GLOB "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/share/fmt/fmt-targets-*.cmake")
      add_config_to_cmake_target_files(
        LIB_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}fmt${CMAKE_STATIC_LIBRARY_SUFFIX}" 
        PATH_GLOB "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/share/fmt/fmt-targets-*.cmake")
      fix_include_path(
        DEPENDENCY_NAME "fmt"
        TARGET_PATH "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/share/fmt/fmt-targets.cmake"
      )
      install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/share/fmt" 
        DESTINATION lib/cmake COMPONENT HDI_PACKAGE)
      add_config_to_cmake_target_files(
        LIB_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}glfw3${CMAKE_STATIC_LIBRARY_SUFFIX}" 
        PATH_GLOB "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/share/glfw3/glfw3Targets-*.cmake")
      install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/share/glfw3" 
        DESTINATION lib/cmake COMPONENT HDI_PACKAGE)

      # Install the include files
      install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/include/kompute"
        DESTINATION include)
      # Additional generated kompute headers
      install(FILES 
        "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/include/ShaderLogisticRegression.hpp"
        "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/include/ShaderOpMult.hpp"
        DESTINATION include)
      install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/include/fmt"
        DESTINATION include)
      install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/include/GLFW"
        DESTINATION include)
      install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/include/vulkan"
        DESTINATION include)
      install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/include/vk_video"
        DESTINATION include)

      # Install the static libraries
      install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}kompute${CMAKE_STATIC_LIBRARY_SUFFIX}"
        DESTINATION lib/Release COMPONENT HDI_PACKAGE)
      install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}kp_logger${CMAKE_STATIC_LIBRARY_SUFFIX}"
        DESTINATION lib/Release COMPONENT HDI_PACKAGE)
      # Install vulkan so lib?
      install(CODE "
        execute_process(COMMAND ls ${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/lib/)
      ")
      if(WIN32)
        install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}vulkan-1${CMAKE_STATIC_LIBRARY_SUFFIX}"
        DESTINATION lib/Release COMPONENT HDI_PACKAGE)
      else()
        install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}vulkan${CMAKE_SHARED_LIBRARY_SUFFIX}"
        DESTINATION lib/Release COMPONENT HDI_PACKAGE)
      endif()
      install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}fmt${CMAKE_STATIC_LIBRARY_SUFFIX}"
        DESTINATION lib/Release COMPONENT HDI_PACKAGE)
      install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}glfw3${CMAKE_STATIC_LIBRARY_SUFFIX}"
        DESTINATION lib/Release COMPONENT HDI_PACKAGE)
      install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/debug/lib/${CMAKE_STATIC_LIBRARY_PREFIX}kompute${CMAKE_STATIC_LIBRARY_SUFFIX}"
        DESTINATION lib/Debug COMPONENT HDI_PACKAGE)
      install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/debug/lib/${CMAKE_STATIC_LIBRARY_PREFIX}kp_logger${CMAKE_STATIC_LIBRARY_SUFFIX}"
        DESTINATION lib/Debug COMPONENT HDI_PACKAGE)
      install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/debug/lib/${CMAKE_STATIC_LIBRARY_PREFIX}fmtd${CMAKE_STATIC_LIBRARY_SUFFIX}"
        DESTINATION lib/Debug COMPONENT HDI_PACKAGE)
      install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/debug/lib/${CMAKE_STATIC_LIBRARY_PREFIX}glfw3${CMAKE_STATIC_LIBRARY_SUFFIX}"
        DESTINATION lib/Debug COMPONENT HDI_PACKAGE)
    endif()
endif()

# -----------------------------------------------------------------------------
# Unit testing
# -----------------------------------------------------------------------------
if(HDILib_BUILD_TESTS)
	message(STATUS "Build HDLIB unit tests")
	add_subdirectory("test")
endif()
