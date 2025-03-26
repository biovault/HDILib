set(PROJECT "hdidata")

# Gather list of all .h files in "/"
file(GLOB HeaderFiles *.h)
file(GLOB SourceFiles *.cpp)

add_library(${PROJECT} STATIC ${HeaderFiles} ${SourceFiles} )

target_include_directories(${PROJECT} PRIVATE "${PROJECT_SOURCE_DIR}")

target_link_libraries(${PROJECT} PRIVATE ${OPENGL_LIBRARIES})

set_target_properties(${PROJECT} PROPERTIES PUBLIC_HEADER "${HeaderFiles}")

if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT} PRIVATE OpenMP::OpenMP_CXX)
endif()

if(HDILib_ENABLE_PID)
    set_target_properties(${PROJECT} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# For compatibility with cibuildwheel consumer
target_compile_options(${target} PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-std=c99>   # For GCC and Clang
    $<$<CXX_COMPILER_ID:MSVC>:/std:c99>        # For MSVC
)

if(HDILib_ENABLE_CODE_ANALYSIS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(${PROJECT} PRIVATE /analyze)
    endif()
endif()

hdi_check_and_set_AVX(${PROJECT} ${HDILib_ENABLE_AVX})
hdi_set_optimization_level(${PROJECT} ${HDILib_OPTIMIZATION_LEVEL})

########### INSTALL ##############
if(${HDILib_INSTALL})
    install(TARGETS ${PROJECT}
        EXPORT ${PROJECT}Targets
        LIBRARY DESTINATION lib/$<CONFIGURATION>>
        ARCHIVE DESTINATION lib/$<CONFIGURATION>
        PUBLIC_HEADER DESTINATION include/hdi/data
        COMPONENT hdidata
    )

    install(EXPORT ${PROJECT}Targets
            FILE ${PROJECT}Targets.cmake
            NAMESPACE HDI::
            DESTINATION lib/cmake/HDILib
            COMPONENT HDIDATA_TARGET
    )
endif()
