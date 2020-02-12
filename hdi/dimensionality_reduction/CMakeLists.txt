include_directories ("${CMAKE_SOURCE_DIR}")

set(PROJECT "hdidimensionalityreduction")

#FLANN
SET(HDI_EXTERNAL_FLANN_INCLUDE_DIR  "HDI_EXTERNAL_FLANN_INCLUDE_DIR-NOTFOUND" CACHE PATH "External Flann Include Dir (Required)")
include_directories(${HDI_EXTERNAL_FLANN_INCLUDE_DIR})
message (STATUS "Flann: " ${HDI_EXTERNAL_FLANN_INCLUDE_DIR})

get_filename_component(HNSWLIB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../external/hnsw" REALPATH)
include_directories(${HNSWLIB_INCLUDE_DIR})
message (STATUS "HNSWLIB: " ${HNSWLIB_INCLUDE_DIR})
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHNSWLIB_FOUND")

get_filename_component(ANNOY_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../external/annoy/src" REALPATH)
include_directories(${ANNOY_INCLUDE_DIR})
message (STATUS "Annoy: " ${ANNOY_INCLUDE_DIR})
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__USE_ANNOY__")



# Gather list of all .h files in "/"
file(GLOB HeaderFiles *.h gpgpu_sne/*.h)
file(GLOB SourceFiles *.cpp gpgpu_sne/*.cpp)

add_library(${PROJECT} ${HeaderFiles} ${SourceFiles} )

# Need to wait until add_library call before depenency of Flann can be set in case Flann is to be build from scratch.
if(DEFINED(${HDI_FLANN_BUILD_INCLUDE_DIR}))
	add_dependencies(${PROJECT} flann)
endif(DEFINED(${HDI_FLANN_BUILD_INCLUDE_DIR}))

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
