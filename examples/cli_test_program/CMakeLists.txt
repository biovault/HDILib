cmake_minimum_required(VERSION 3.22)
project(tsne_tester)

set(CMAKE_CXX_STANDARD 17)

# If using RenderDoc (https://renderdoc.org) or XCode GPU debugging capturing
# GPU calls is a powerful debugging tool for Vulkan
option(TSNETESTER_GPUDEBUG_CAPTURE " Enable GPU debug capture for the test program (Windows or MacOS)" OFF)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
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
find_package(glfw3 CONFIG REQUIRED)
find_package(argparse CONFIG REQUIRED)
find_path(RAPIDCSV_INCLUDE_DIRS "rapidcsv.h")

target_include_directories(tsne_tester PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
target_include_directories(tsne_tester PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../..")
target_include_directories(tsne_tester PRIVATE "${RAPIDCSV_INCLUDE_DIRS}")

# For static library linking the consumer lib comes first (gcc)
target_link_libraries(tsne_tester PRIVATE Vulkan::Vulkan)
target_link_libraries(tsne_tester PRIVATE glfw)
target_link_libraries(tsne_tester PRIVATE ${OPENGL_LIBRARIES})
target_link_libraries(tsne_tester PRIVATE hdidimensionalityreduction hdiutils hdidata)
target_link_libraries(tsne_tester PRIVATE kompute::kompute)
target_link_libraries(tsne_tester PRIVATE lz4::lz4 flann::flann_cpp_s)
target_link_libraries(tsne_tester PRIVATE argparse::argparse)

if (APPLE)
    target_link_libraries(tsne_tester PRIVATE "-framework OpenGL" "-framework Metal" "-framework Foundation" )
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(tsne_tester PRIVATE OpenMP::OpenMP_CXX)
endif()
