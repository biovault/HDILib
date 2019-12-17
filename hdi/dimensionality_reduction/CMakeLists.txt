include_directories ("${CMAKE_SOURCE_DIR}")

set(PROJECT "hdidimensionalityreduction")

get_filename_component(HNSWLIB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../external/hnsw" REALPATH)
include_directories(${HNSWLIB_INCLUDE_DIR})
message (STATUS "HNSWLIB: " ${HNSWLIB_INCLUDE_DIR})
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHNSWLIB_FOUND")

get_filename_component(ANNOY_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../external/annoy/src" REALPATH)
include_directories(${ANNOY_INCLUDE_DIR})
message (STATUS "Annoy: " ${ANNOY_INCLUDE_DIR})
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__USE_ANNOY__")

#Setting custom compilers for MacOS
if (CMAKE_GENERATOR STREQUAL Xcode)
    find_program(BREW NAMES brew)
    if("${BREW}" STRGREATER "")
        message(STATUS "Running on macOS Darwin using ${BREW}")
        execute_process(COMMAND ${BREW} --prefix flann OUTPUT_VARIABLE BREW_FLANN_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif("${BREW}" STRGREATER "")
    
    #finding flann
    if(EXISTS ${BREW_FLANN_PREFIX})
        message(STATUS "Flann found from brew at ${BREW_FLANN_PREFIX}.")
        include_directories("${BREW_FLANN_PREFIX}/include")
    else()
        # there should probably be an alternative in here for setting the flann path
        message(WARNING "Flann not found.")
    endif(EXISTS ${BREW_FLANN_PREFIX})
#Win/Linux
else(CMAKE_GENERATOR STREQUAL Xcode)
    include_directories("$ENV{FLANN_DIR}/include")
    message (STATUS "$ENV{FLANN_DIR}/include")
endif(CMAKE_GENERATOR STREQUAL Xcode)

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
