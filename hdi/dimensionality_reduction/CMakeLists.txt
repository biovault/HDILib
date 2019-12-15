include_directories ("${CMAKE_SOURCE_DIR}")

get_filename_component(HNSWLIB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../external/hnsw" REALPATH)
include_directories(${HNSWLIB_INCLUDE_DIR})
message (STATUS "HNSWLIB: " ${HNSWLIB_INCLUDE_DIR})
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHNSWLIB_FOUND")

# Gather list of all .h files in "/"
file(GLOB HeaderFiles *.h gpgpu_sne/*.h)
file(GLOB SourceFiles *.cpp gpgpu_sne/*.cpp)

add_library   (hdidimensionalityreduction ${HeaderFiles} ${SourceFiles} )

########### INSTALL ##############
install (TARGETS hdidimensionalityreduction DESTINATION bin/hdi)
install (FILES ${HeaderFiles} ${UiHeaders} DESTINATION include/hdi/dimensionality_reduction)
