cmake_minimum_required(VERSION 3.22)

# find_package() uses uppercase <PackageName>_ROOT variables.
# Remove once cmake minimum is 3.27
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(PackageTest CXX)

message(STATUS "HDILib root path ${HDILIB_ROOT}")
find_package(HDILib COMPONENTS hdiutils hdidata hdidimensionalityreduction PATHS ${HDILIB_ROOT} CONFIG REQUIRED)
find_package(lz4 CONFIG REQUIRED)
find_package(flann CONFIG REQUIRED)

find_package(OpenGL REQUIRED)
find_package(OpenMP)

include(FetchContent)
FetchContent_Declare (
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
)

set(GLFW_BUILD_DOCS OFF CACHE BOOL  "")
set(GLFW_INSTALL OFF CACHE BOOL  "")

FetchContent_MakeAvailable(glfw)

if(NOT FLANN_TARGET)
    if(TARGET flann::flann_cpp_s)
        set(FLANN_TARGET flann::flann_cpp_s)
    elseif(TARGET flann::flann_cpp)
        set(FLANN_TARGET flann::flann_cpp)
    elseif(TARGET flann::flann_s)
        set(FLANN_TARGET flann::flann_s)
    elseif(TARGET flann::flann)
        set(FLANN_TARGET flann::flann)
    else()
        message(FATAL_ERROR "No Flann target found.")
    endif()
endif()

set(SOURCES
    src/example.cpp
    src/offscreenWindow.h
    src/offscreenWindow.cpp
)

add_executable(example ${SOURCES})

target_include_directories(example PRIVATE ${HDILib_INCLUDE_DIR})
target_include_directories(example PRIVATE "${GLFW_SOURCE_DIR}/include")

# Link the libraries individually or use the predefined HDILib_LINK_LIBS
target_link_libraries(example PRIVATE 
    HDI::hdidimensionalityreduction HDI::hdiutils HDI::hdidata  # Same as ${HDILib_LINK_LIBS}
    lz4::lz4 ${FLANN_TARGET}                                    # HDILib drags FLANN in
    ${OPENGL_LIBRARIES}                                         # We need openGL
    glfw                                                        # Use GLFW to create offscreenWindow
    )

if(UNIX)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(example PRIVATE Threads::Threads)
endif(UNIX)

install(TARGETS example
    RUNTIME DESTINATION bin)

add_custom_command(TARGET example POST_BUILD
	COMMAND "${CMAKE_COMMAND}"
	--install ${CMAKE_BINARY_DIR}
	--config $<CONFIGURATION>
	--prefix ${CMAKE_BINARY_DIR}
)
