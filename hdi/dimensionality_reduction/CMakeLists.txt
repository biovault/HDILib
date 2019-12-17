include_directories ("${CMAKE_SOURCE_DIR}")

set(PROJECT "hdidimensionalityreduction")

option(HNSWLIB_INCLUDE_DIR "The include folder for HNSWLIB" OFF)

if( "${HNSWLIB_INCLUDE_DIR}" STREQUAL "OFF")
    message(STATUS "HNSWLIB Not Found!")
else("${HNSWLIB_INCLUDE_DIR}" STREQUAL "OFF")
    include_directories(${HNSWLIB_INCLUDE_DIR})
    message (STATUS "HNSWLIB: " ${HNSWLIB_INCLUDE_DIR})
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHNSWLIB_FOUND")
endif("${HNSWLIB_INCLUDE_DIR}" STREQUAL "OFF")

include_directories("$ENV{FLANN_DIR}/include")
message (STATUS "$ENV{FLANN_DIR}/include")

# Gather list of all .h files in "/"
file(GLOB HeaderFiles *.h gpgpu_sne/*.h)
file(GLOB SourceFiles *.cpp gpgpu_sne/*.cpp)

add_library(${PROJECT} ${HeaderFiles} ${SourceFiles} )

########### INSTALL ##############
set_target_properties(${PROJECT} PROPERTIES PUBLIC_HEADER "${HeaderFiles}")

install(TARGETS ${PROJECT}
    CONFIGURATIONS Debug
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/Debug
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/hdi/dimensionality_reduction
)
install(TARGETS ${PROJECT}
    CONFIGURATIONS Release
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/Release
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/hdi/dimensionality_reduction
)
