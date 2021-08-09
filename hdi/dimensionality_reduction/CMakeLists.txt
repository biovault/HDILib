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
file(GLOB HeaderFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h gpgpu_sne/*.h)
message(STATUS "################# ${HeaderFiles} ##############")
file(GLOB SourceFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp gpgpu_sne/*.cpp)

add_library(${PROJECT} STATIC ${HeaderFiles} ${SourceFiles} )

# Need to wait until add_library call before depenency of Flann can be set in case Flann is to be build from scratch.
if(DEFINED(${HDI_FLANN_BUILD_INCLUDE_DIR}))
	add_dependencies(${PROJECT} flann)
endif(DEFINED(${HDI_FLANN_BUILD_INCLUDE_DIR}))

if(UNIX)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT} PRIVATE Threads::Threads)
endif(UNIX)

if(ENABLE_CODE_ANALYSIS AND WIN32)
        target_compile_options(${PROJECT} PRIVATE /analyze)
endif()

########### INSTALL ##############
#set_target_properties(${PROJECT} PROPERTIES PUBLIC_HEADER "${HeaderFiles}")

install(TARGETS ${PROJECT}
    EXPORT ${PROJECT}Targets
    LIBRARY DESTINATION lib/$<CONFIGURATION>
    ARCHIVE DESTINATION lib/$<CONFIGURATION>
    COMPONENT hdidimensionalityreduction
)

# Preserve the header hierarchy by explicit install
# the CMake PUBLIC_HEADER target property flattens it if used.
install(CODE "foreach(HEADER ${HeaderFiles})
        message(STATUS \"Installing: \${HEADER} to \${CMAKE_INSTALL_PREFIX}/include/hdi/dimensionality_reduction\")
        execute_process(
            COMMAND \"${CMAKE_COMMAND}\" -E copy_if_different 
                \"${CMAKE_CURRENT_SOURCE_DIR}/\${HEADER}\" 
                \"${CMAKE_INSTALL_PREFIX}/include/hdi/dimensionality_reduction/\${HEADER}\"
        )

    endforeach()"
    COMPONENT PUBLIC_HEADERS
)

install(EXPORT ${PROJECT}Targets
        FILE ${PROJECT}Targets.cmake
        NAMESPACE HDI::
        DESTINATION lib/cmake/HDILib
        COMPONENT HDIDIMENSIONALITYREDUCTION_TARGET
)
