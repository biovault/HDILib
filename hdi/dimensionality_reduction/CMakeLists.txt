set(PROJECT "hdidimensionalityreduction")

# Gather list of all .h files in "/"
configure_file(dr_config.h.in dr_config.h)

if(USE_VULKAN_KOMPUTE)
    file(GLOB HeaderFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h gpgpu_sne/*.h gpgpu_vulkan/*.h gpgpu_vulkan/shaders/*.h)
    message(STATUS "################# ${HeaderFiles} ##############")
    file(GLOB SourceFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp gpgpu_sne/*.cpp gpgpu_vulkan/*.cpp gpgpu_vulkan/shaders/*.cpp)
    message(STATUS "################# ${HeaderFiles} ##############")
    file(GLOB_RECURSE ShaderFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/gpgpu_vulkan/shaders/*.comp")
    add_library(${PROJECT} STATIC ${HeaderFiles} ${SourceFiles} ${ShaderFiles})
    source_group(Shaders FILES ${ShaderFiles})
    add_dependencies(${PROJECT} compile_shaders)
    find_package(fmt CONFIG REQUIRED)
    find_package(kompute CONFIG REQUIRED)
    # choose the algorithm implementation slow PUSH_CONSTANTS or fast UBOs 
    set(SHADER_USE_PUSH_CONSTANTS FALSE)
else()
    file(GLOB HeaderFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h gpgpu_sne/*.h)
    message(STATUS "################# ${HeaderFiles} ##############")
    file(GLOB SourceFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp gpgpu_sne/*.cpp)
    add_library(${PROJECT} STATIC ${HeaderFiles} ${SourceFiles} )
endif()



target_include_directories(${PROJECT} PRIVATE "${PROJECT_SOURCE_DIR}")
target_include_directories(${PROJECT} PRIVATE "${hnswlib_SOURCE_DIR}")
target_include_directories(${PROJECT} PRIVATE "${Annoy_SOURCE_DIR}/src")
#include the generated dr_config.h file and any others
target_include_directories(${PROJECT} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})


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

target_compile_definitions(${PROJECT} PRIVATE _ISOC99_SOURCE _GNU_SOURCE)
if (SHADER_USE_PUSH_CONSTANTS) 
  target_compile_definitions(${PROJECT} PRIVATE SHADER_USE_PUSH_CONSTANTS=1)
endif()

if(HDILib_ENABLE_PID)
    set_target_properties(${PROJECT} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if(HDILib_ENABLE_CODE_ANALYSIS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(${PROJECT} PRIVATE /analyze)
    endif()
endif()

hdi_check_and_set_AVX(${PROJECT} ${HDILib_ENABLE_AVX})
hdi_set_optimization_level(${PROJECT} ${HDILib_OPTIMIZATION_LEVEL})

if (USE_VULKAN_KOMPUTE)
    target_link_libraries(${PROJECT} PRIVATE kompute::kompute)
endif()

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

    # Install cmake targets file
    install(EXPORT ${PROJECT}Targets
            FILE ${PROJECT}Targets.cmake
            NAMESPACE HDI::
            DESTINATION lib/cmake/HDILib
            COMPONENT HDIDIMENSIONALITYREDUCTION_TARGET
    )
endif()

if(USE_VULKAN_KOMPUTE)
    message(STATUS "Add vulkan subdir")
    add_subdirectory(gpgpu_vulkan)
endif()