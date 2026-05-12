cmake_minimum_required(VERSION 3.22)

# find_package() uses uppercase <PackageName>_ROOT variables.
# Remove once cmake minimum is 3.27
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(PackageTest CXX)

message(STATUS "HDILib at ${HDILIB_ROOT}")
find_package(HDILib COMPONENTS hdiutils hdidata hdidimensionalityreduction PATHS ${HDILIB_ROOT} CONFIG REQUIRED)
find_package(lz4 CONFIG REQUIRED)
find_package(flann CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(kompute CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)

find_package(OpenGL REQUIRED)
find_package(OpenMP)

set(SOURCES
    src/minimal_setup.cpp
    src/offscreenWindow.h
    src/offscreenWindow.cpp
)

add_executable(example ${SOURCES})

target_include_directories(example PRIVATE ${HDILib_INCLUDE_DIR})

# Link the libraries individually or use the predefined HDILib_LINK_LIBS
target_link_libraries(example PRIVATE 
    HDI::hdidimensionalityreduction HDI::hdiutils HDI::hdidata  # Same as ${HDILib_LINK_LIBS}
    lz4::lz4 flann::flann_cpp_s kompute::kompute                # HDILib drags several dependencies in
    ${OPENGL_LIBRARIES}                                         # We need openGL
    glfw                                                        # Use GLFW to create offscreenWindow
    )

if(OpenMP_CXX_FOUND)
    target_link_libraries(example PRIVATE OpenMP::OpenMP_CXX)
endif()

if(UNIX)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(example PRIVATE Threads::Threads)
endif(UNIX)
