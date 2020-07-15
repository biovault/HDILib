cmake_minimum_required(VERSION 3.9)

# Set C++11 language standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(PackageTest CXX)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()
message(status, "Conan Libs: ${CONAN_LIBS}")
find_package(OpenMP)
if (APPLE)
    find_package(OpenMP)
    find_package(OpenGL REQUIRED)
else()
    find_package(OpenMP REQUIRED)
endif()

add_executable(example example.cpp)
if (WIN32)
    target_link_libraries(example PRIVATE ${CONAN_LIBS} OpenMP::OpenMP_CXX ${CMAKE_DL_LIBS})
else()
    # Force the lib order for linking
    if (CMAKE_HOST_SYSTEM_NAME MATCHES "Linux")  
        target_link_libraries(example PRIVATE lz4;hdidimensionalityreduction;hdidata;hdiutils OpenMP::OpenMP_CXX ${CMAKE_DL_LIBS})
    elseif (OPENMP_FOUND)
        target_link_libraries(example PRIVATE lz4;hdidimensionalityreduction;hdidata;hdiutils OpenMP::OpenMP_CXX ${CMAKE_DL_LIBS}  ${OPENGL_LIBRARIES})
    else()
        target_link_libraries(example PRIVATE lz4;hdidimensionalityreduction;hdidata;hdiutils ${CMAKE_DL_LIBS}  ${OPENGL_LIBRARIES})
    endif()
endif()

