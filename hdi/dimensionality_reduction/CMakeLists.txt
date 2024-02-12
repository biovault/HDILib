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

get_filename_component(ANNOY_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../external/annoy/src" REALPATH)
target_include_directories(${PROJECT} PRIVATE ${ANNOY_INCLUDE_DIR})
message (STATUS "Annoy: " ${ANNOY_INCLUDE_DIR})

if(DEFINED ENV{CI})
    message (STATUS "Flann include: " ${flann_INCLUDE_DIR})
    message (STATUS "lz4 include: " ${lz4_INCLUDE_DIR})
    target_include_directories(${PROJECT} PRIVATE ${flann_INCLUDE_DIR} ${lz4_INCLUDE_DIR})
else()
    message (STATUS "Flann path: " ${flann_DIR})
    if(WIN32)
        target_link_libraries(${PROJECT} PRIVATE flann::flann_cpp_s)
    else()
        target_link_libraries(${PROJECT} PRIVATE flann::flann)
    endif()
endif()

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
install(CODE "foreach(DR_HEADER ${HeaderFiles})
        message(STATUS \"Installing: \${DR_HEADER} to \${CMAKE_INSTALL_PREFIX}/include/hdi/dimensionality_reduction\")
        execute_process(
            COMMAND \"${CMAKE_COMMAND}\" -E copy_if_different 
                \"${CMAKE_CURRENT_SOURCE_DIR}/\${DR_HEADER}\" 
                \"${CMAKE_INSTALL_PREFIX}/include/hdi/dimensionality_reduction/\${DR_HEADER}\"
        )

    endforeach()"
    COMPONENT PUBLIC_HEADERS
)

# Install dependency headers
if(NOT DEFINED ENV{CI})
    install(DIRECTORY "${flann_INCLUDE_DIR}/" DESTINATION "${CMAKE_INSTALL_PREFIX}/include" COMPONENT FLANN_HEADERS)
    install(DIRECTORY "${lz4_INCLUDE_DIR}/" DESTINATION "${CMAKE_INSTALL_PREFIX}/include" COMPONENT LZ4_HEADERS)
endif()

install(DIRECTORY "${HNSWLIB_INCLUDE_DIR}/hnswlib/" DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hnswlib" COMPONENT HNSW_HEADERS)
install(CODE "foreach(ANNOY_HEADER annoylib.h kissrandom.h mman.h)
        execute_process(
            COMMAND \"${CMAKE_COMMAND}\" -E copy_if_different 
                \"${ANNOY_INCLUDE_DIR}/\${ANNOY_HEADER}\" 
                \"${CMAKE_INSTALL_PREFIX}/include/\${ANNOY_HEADER}\"
        )
    endforeach()"
    COMPONENT ANNOY_HEADERS
)

# Install cmake targets file
install(EXPORT ${PROJECT}Targets
        FILE ${PROJECT}Targets.cmake
        NAMESPACE HDI::
        DESTINATION lib/cmake/HDILib
        COMPONENT HDIDIMENSIONALITYREDUCTION_TARGET
)
