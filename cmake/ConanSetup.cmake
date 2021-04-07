if(NOT HDILIB_BUILD_WITH_CONAN)
    message(STATUS "***********Not using ConanSetup**************")
    return()
endif()

message(STATUS "Start ConanSetup")


# Build-with-Conan only supports one configuration at a time (per .sln file, for Visual Studio).
if(CMAKE_BUILD_TYPE AND CMAKE_CONFIGURATION_TYPES)
    get_property(hdilib_generator_is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(hdilib_generator_is_multi_config)
        set(CMAKE_CONFIGURATION_TYPES ${CMAKE_BUILD_TYPE})
    endif()
endif()


# Download the conan cmake macros automatically.
#This is the location for conan_cmake_run
if(NOT EXISTS "${CMAKE_BINARY_DIR}/cmake/conan.cmake")
   message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
   file(DOWNLOAD "https://github.com/conan-io/cmake-conan/raw/v0.15/conan.cmake"
                 "${CMAKE_BINARY_DIR}/cmake/conan.cmake")
endif()

include(${CMAKE_BINARY_DIR}/cmake/conan.cmake)

# env variable CI exists and is true for Appevor, Travis, CircleCI, GitLab, GitHub Actions
# on Azure it's TF_BUILD :-(
if($ENV{CI} or $ENV{TF_BUILD})
    set(IS_CI TRUE)
else()
    set(IS_CI FALSE)
endif()

if(MSVC)
    set(CC_CACHE $ENV{CC})
    set(CXX_CACHE $ENV{CXX})
    unset(ENV{CC}) # Disable clcache, e.g. for building qt
    unset(ENV{CXX})
endif()

set(CONAN_SETTINGS "")

if(UNIX)
    if(LIBCXX) 
        set(CONAN_SETTINGS ${CONAN_SETTINGS} "compiler.libcxx=${LIBCXX}")
    endif()    
endif()

message(STATUS "Install dependencies with conan")
conan_cmake_run(
    CONANFILE conanfile.py
    BASIC_SETUP ${CONAN_UPDATE}
    BUILD missing
    BUILD_TYPE ${CMAKE_BUILD_TYPE}
)

conan_load_buildinfo()

message(STATUS "Installed dependencies with conan")
set(HDI_LIB_ROOT "${CONAN_HDILIB_ROOT}")
set(HDI_EXTERNAL_FLANN_INCLUDE_DIR "${CONAN_INCLUDE_DIRS_FLANN}" CACHE PATH "External Flann Include Dir (Required)")
message(STATUS "Conan Flann include (HDI_EXTERNAL_FLANN_INCLUDE_DIR) ${HDI_EXTERNAL_FLANN_INCLUDE_DIR}")
if(WIN32)
    set(FLANN_BUILD_DIR "${CONAN_FLANN_ROOT}")
    set(GLFW_ROOT "${CONAN_GLFW_ROOT}")
endif()

if(MSVC)
    set(ENV{CC} ${CC_CACHE}) # Restore vars
    set(ENV{CXX} ${CXX_CACHE})
endif()

message(STATUS "End ConanSetup")
