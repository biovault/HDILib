set(PROJECT "hdidimensionalityreduction")

# Gather list of all .h files in "/"
file(GLOB HeaderFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h gpgpu_sne/*.h)
message(STATUS "################# ${HeaderFiles} ##############")
file(GLOB SourceFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp gpgpu_sne/*.cpp)

add_library(${PROJECT} STATIC ${HeaderFiles} ${SourceFiles} )

target_include_directories(${PROJECT} PRIVATE "${CMAKE_SOURCE_DIR}")

get_filename_component(HNSWLIB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../external/hnsw" REALPATH)
target_include_directories(${PROJECT} PRIVATE ${HNSWLIB_INCLUDE_DIR})
message (STATUS "HNSWLIB: " ${HNSWLIB_INCLUDE_DIR})
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHNSWLIB_FOUND")

get_filename_component(ANNOY_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../external/annoy/src" REALPATH)
target_include_directories(${PROJECT} PRIVATE ${ANNOY_INCLUDE_DIR})
message (STATUS "Annoy: " ${ANNOY_INCLUDE_DIR})
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__USE_ANNOY__")

message (STATUS "Flann: " ${flann_INCLUDE_DIR})
target_include_directories(${PROJECT} PRIVATE ${flann_INCLUDE_DIR} ${lz4_INCLUDE_DIR})
target_link_libraries(${PROJECT} PRIVATE flann::flann_cpp_s LZ4::lz4_static)

if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT} PRIVATE OpenMP::OpenMP_CXX)
endif()

if(ENABLE_CODE_ANALYSIS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(${PROJECT} PRIVATE /analyze)
    endif()
endif()

if(UNIX)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT} PRIVATE Threads::Threads)
endif(UNIX)

if(ENABLE_CODE_ANALYSIS AND WIN32)
    target_compile_options(${PROJECT} PRIVATE /analyze)
endif()

if(ENABLE_PID)
    set_target_properties(${PROJECT} PROPERTIES POSITION_INDEPENDENT_CODE ON)
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
