set(PROJECT "hdidimensionalityreduction")

# Gather list of all .h files in "/"
file(GLOB HeaderFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h gpgpu_sne/*.h)
message(STATUS "################# ${HeaderFiles} ##############")
file(GLOB SourceFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp gpgpu_sne/*.cpp)

add_library(${PROJECT} STATIC ${HeaderFiles} ${SourceFiles} )

target_include_directories(${PROJECT} PRIVATE "${PROJECT_SOURCE_DIR}")

get_filename_component(HNSWLIB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../external/hnsw" REALPATH)
target_include_directories(${PROJECT} PRIVATE ${HNSWLIB_INCLUDE_DIR})
message (STATUS "HNSWLIB: " ${HNSWLIB_INCLUDE_DIR})

get_filename_component(ANNOY_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../external/annoy/src" REALPATH)
target_include_directories(${PROJECT} PRIVATE ${ANNOY_INCLUDE_DIR})
message (STATUS "Annoy: " ${ANNOY_INCLUDE_DIR})

# prefer static linking
if(NOT FLANN_TARGET)
    if(TARGET flann::flann_cpp_s)
        set(FLANN_TARGET flann::flann_cpp_s)
    elseif(TARGET flann::flann_cpp)
        set(FLANN_TARGET flann::flann_cpp)
    elseif(TARGET flann::flann_s)
        set(FLANN_TARGET flann::flann_s)
    elseif(TARGET flann::flann)
        set(FLANN_TARGET flann::flann)
    else()
        message(FATAL_ERROR "No Flann target found.")
    endif()
endif()

if(NOT LZ4_TARGET)
    if(TARGET LZ4::lz4_static)
        set(LZ4_TARGET LZ4::lz4_static)
    elseif(TARGET LZ4::lz4_shared)
        set(LZ4_TARGET LZ4::lz4_shared)
    elseif(TARGET lz4::lz4)
        set(LZ4_TARGET lz4::lz4)
    elseif(TARGET LZ4::lz4)    # intentionally UPPERCASE::LOWERCASE
        set(LZ4_TARGET LZ4::lz4)
    else()
        message(FATAL_ERROR "No LZ4 target found.")
    endif()
endif()

message (STATUS "Flann link library: " ${FLANN_TARGET})
target_link_libraries(${PROJECT} PRIVATE ${FLANN_TARGET})
message (STATUS "Linking lz4 library " ${LZ4_TARGET})
target_link_libraries(${PROJECT} PRIVATE ${LZ4_TARGET})

if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT} PRIVATE OpenMP::OpenMP_CXX)
endif()

if(HDILib_ENABLE_PID)
    set_target_properties(${PROJECT} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if(HDILib_ENABLE_CODE_ANALYSIS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(${PROJECT} PRIVATE /analyze)
    endif()
endif()

set_optimization_level(${PROJECT} ${HDILib_OPTIMIZATION_LEVEL})
check_and_set_AVX(${PROJECT} ${HDILib_ENABLE_AVX})

if(UNIX)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT} PRIVATE Threads::Threads)
endif(UNIX)

########### INSTALL ##############
if(${HDILib_INSTALL})
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
    if(NOT flann_INCLUDE_DIR)
        get_target_property(flann_INCLUDE_DIR ${FLANN_TARGET} INTERFACE_INCLUDE_DIRECTORIES)
        if(flann_INCLUDE_DIR)
            if(flann_INCLUDE_DIR STREQUAL "/usr/include")
                set(flann_INCLUDE_DIR "${flann_INCLUDE_DIR}/flann")
            endif()
        endif()
    endif()

    if(flann_INCLUDE_DIR)
        message(STATUS "Using flann includes at: ${flann_INCLUDE_DIR}")
        install(DIRECTORY "${flann_INCLUDE_DIR}/" DESTINATION "${CMAKE_INSTALL_PREFIX}/include" COMPONENT FLANN_HEADERS)
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
endif()
