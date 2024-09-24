cmake_minimum_required(VERSION 3.15)

# Set C++11 language standard
# flann util/random.h versions 1.8.4, 1.8.5, & 1.9.1 uses std::random_shuffle that is removed in C++17
# So staying at C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(PackageTest CXX)

# CONAN_HDILIB_ROOT should be at the root of the HDILib install PATHS ${CONAN_HDILIB_ROOT} CONFIG
message(STATUS "HDILib root path ${HDILib_ROOT}")
find_package(HDILib COMPONENTS hdiutils hdidata hdidimensionalityreduction PATHS ${HDILib_ROOT} CONFIG REQUIRED)
message("test_package list dir ${CMAKE_CURRENT_LIST_DIR}")
find_package(lz4 REQUIRED)
find_package(flann REQUIRED)

find_package(OpenMP)
if (APPLE)
    find_package(OpenMP)
    find_package(OpenGL REQUIRED)
else()
    find_package(OpenMP REQUIRED)
endif()

add_executable(example example.cpp)
# Link the libraries individually or use the predefined HDILib_LINK_LIBS
if (WIN32)
    target_link_libraries(example PRIVATE lz4::lz4 HDI::hdidimensionalityreduction HDI::hdiutils HDI::hdidata flann::flann_cpp OpenMP::OpenMP_CXX ${CMAKE_DL_LIBS})
else()
    # Force the lib order for linking
    if (CMAKE_HOST_SYSTEM_NAME MATCHES "Linux")
        target_link_libraries(example PRIVATE lz4::lz4 ${HDILib_LINK_LIBS} OpenMP::OpenMP_CXX ${CMAKE_DL_LIBS})
    elseif (OPENMP_FOUND)
        target_link_libraries(example PRIVATE lz4::lz4 ${HDILib_LINK_LIBS} OpenMP::OpenMP_CXX ${CMAKE_DL_LIBS}  ${OPENGL_LIBRARIES})
    else()
        target_link_libraries(example PRIVATE lz4::lz4 ${HDILib_LINK_LIBS} HDI::hdidata ${CMAKE_DL_LIBS}  ${OPENGL_LIBRARIES})
    endif()
endif()
if(UNIX)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(example PRIVATE Threads::Threads)
endif(UNIX)

target_include_directories(example PRIVATE ${HDILib_INCLUDE_DIR})

install(TARGETS example
    RUNTIME DESTINATION bin)

add_custom_command(TARGET example POST_BUILD
	COMMAND "${CMAKE_COMMAND}"
	--install ${CMAKE_BINARY_DIR}
	--config $<CONFIGURATION>
	--prefix ${CMAKE_BINARY_DIR}
)
