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

# If using RenderDoc (https://renderdoc.org) or XCode GPU debugging capturing
# GPU calls is a powerful debugging tool for Vulkan
option(TSNETESTER_GPUDEBUG_CAPTURE " Enable GPU debug capture for the test program (Windows or MacOS)" OFF)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)
configure_file(tSNETester_config.h.in tSNETester_config.h)
if(APPLE AND TSNETESTER_GPUDEBUG_CAPTURE)
    enable_language(OBJCXX)
endif()

set(TESTER_SOURCES
    tSNETester.cpp
    tSNETester_config.h
)

set(RENDERDOC_SOURCES
    RenderDebug.cpp
    RenderDebug.h
    renderdoc_app.h
)


set(METALCAPTURE_SOURCES
    MetalCaptureBridge.mm
    MetalCaptureBridge.h
    MetalDevice.cpp
    MetalDevice.h
)
 
add_executable(
    tsne_tester
)

target_sources(tsne_tester PRIVATE ${TESTER_SOURCES})

if(WIN32 AND TSNETESTER_GPUDEBUG_CAPTURE)
    target_sources(tsne_tester PRIVATE ${RENDERDOC_SOURCES})
endif()

if(APPLE AND TSNETESTER_GPUDEBUG_CAPTURE)
    set_source_files_properties(
        MetalCaptureBridge.mm
        PROPERTIES
            LANGUAGE OBJCXX
    )
    target_sources(tsne_tester PRIVATE ${METALCAPTURE_SOURCES})
endif()

find_package(fmt CONFIG REQUIRED)
find_package(kompute CONFIG REQUIRED)
find_package(glfw3 REQUIRED)

macro(print_all_variables)
    message(STATUS "print_all_variables------------------------------------------{")
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
    message(STATUS "print_all_variables------------------------------------------}")
endmacro()
message("***********************************")
message("***********************************")
print_all_variables()
message("***********************************")
message("***********************************")
target_include_directories(tsne_tester PRIVATE "${argparse_SOURCE_DIR}/include")
target_include_directories(tsne_tester PRIVATE "${rapidcsv_SOURCE_DIR}/src")
target_include_directories(tsne_tester 
    PRIVATE "${CMAKE_SOURCE_DIR}")
message(STATUS "binary at ${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(tsne_tester PRIVATE kompute::kompute)
target_link_libraries(tsne_tester PRIVATE glfw)
target_link_libraries(tsne_tester PRIVATE ${OPENGL_LIBRARIES})
target_link_libraries(tsne_tester PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/../../hdi/utils/$<CONFIG>/${CMAKE_STATIC_LIBRARY_PREFIX}hdiutils${CMAKE_STATIC_LIBRARY_SUFFIX})
target_link_libraries(tsne_tester PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/../../hdi/data/$<CONFIG>/${CMAKE_STATIC_LIBRARY_PREFIX}hdidata${CMAKE_STATIC_LIBRARY_SUFFIX})
target_link_libraries(tsne_tester PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/../../hdi/dimensionality_reduction/$<CONFIG>/${CMAKE_STATIC_LIBRARY_PREFIX}hdidimensionalityreduction${CMAKE_STATIC_LIBRARY_SUFFIX})
if (APPLE)
    target_link_libraries(tsne_tester PRIVATE "-framework OpenGL" "-framework Metal" "-framework Foundation" )
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(tsne_tester PRIVATE OpenMP::OpenMP_CXX)
endif()
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
