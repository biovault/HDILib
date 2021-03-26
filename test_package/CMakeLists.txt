cmake_minimum_required(VERSION 3.15)

# Set C++11 language standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(PackageTest CXX)

# Test the cmake package files using the path to the HDILibConfig.cmake
find_package(HDILib COMPONENTS hdiutils hdidata hdidimensionalityreduction PATHS ${CMAKE_MODULE_PATH} REQUIRED)
include(${CMAKE_BINARY_DIR}/Findlz4.cmake)

message(status, "Conan Libs: ${CONAN_LIBS}")
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
    target_link_libraries(example PRIVATE HDI::hdidimensionalityreduction HDI::hdiutils HDI::hdidata ${CONAN_FLANN_ROOT}/lib/flann.lib OpenMP::OpenMP_CXX ${CMAKE_DL_LIBS})
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
target_include_directories(example PRIVATE ${HDILib_INCLUDE_DIR})

install(TARGETS example
    RUNTIME DESTINATION bin)

add_custom_command(TARGET example POST_BUILD
	COMMAND "${CMAKE_COMMAND}"
	--install ${CMAKE_BINARY_DIR}
	--config $<CONFIGURATION>
	--prefix ${CMAKE_BINARY_DIR}
)
