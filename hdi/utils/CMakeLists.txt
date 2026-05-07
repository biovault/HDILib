set(PROJECT "hdiutils")

set(HeaderFilesUtils
    abstract_log.h
    assert_by_exception.h
    cout_log.h
    dataset_utils.h
    graph_algorithms.h
    graph_algorithms_inl.h
    log_helper_functions.h
    log_progress.h
    math_utils.h
    math_utils_inl.h
    memory_utils.h
    scoped_timers.h
    scoped_timers_inl.h
    timers.h
    timing_utils.h
    vector_utils.h
)

set(HeaderFilesGlad
    glad/glad.h
    glad/KHR/khrplatform.h
)

set(SourceFilesUtils
    cout_log.cpp
    dataset_utils.cpp
    glad.cpp
    graph_algorithms.cpp
    log_progress.cpp
    math_utils.cpp
    memory_utils.cpp
    timers.cpp
)

if(APPLE)
    # Do not include or build glad on Apple
    set(HeaderFiles ${HeaderFilesUtils})
    list(REMOVE_ITEM SourceFilesUtils "glad.cpp")
else()
    set(HeaderFiles ${HeaderFilesUtils} ${HeaderFilesGlad})
endif()

add_library(${PROJECT} STATIC ${HeaderFiles} ${SourceFilesUtils} ${Resources} )

target_include_directories(${PROJECT} PRIVATE "${PROJECT_SOURCE_DIR}")

#set_target_properties(${PROJECT} PROPERTIES PUBLIC_HEADER "${HeaderFiles}")

if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT} PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_definitions(${PROJECT} PRIVATE _ISOC99_SOURCE _GNU_SOURCE)

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

if (UNIX)
    target_link_libraries (hdiutils PRIVATE ${CMAKE_DL_LIBS}) # glad.cpp requires -ldl
endif (UNIX)

########### INSTALL ##############
if(${HDILib_INSTALL})
    install(TARGETS ${PROJECT}
        EXPORT ${PROJECT}Targets
        LIBRARY DESTINATION lib/$<CONFIGURATION>
        ARCHIVE DESTINATION lib/$<CONFIGURATION>
        COMPONENT hdiutils
    )

    # Preserve the header hierarchy by explicit install
    # the CMake PUBLIC_HEADER target property flattens it if used.
    install(CODE "foreach(HEADER ${HeaderFiles})
            message(STATUS \"Installing: \${HEADER} to \${CMAKE_INSTALL_PREFIX}/include/hdi/utils\")
            execute_process(
                COMMAND \"${CMAKE_COMMAND}\" -E copy_if_different 
                    \"${CMAKE_CURRENT_SOURCE_DIR}/\${HEADER}\" 
                    \"${CMAKE_INSTALL_PREFIX}/include/hdi/utils/\${HEADER}\"
            )

        endforeach()"
        COMPONENT PUBLIC_HEADERS
    )

    install(EXPORT ${PROJECT}Targets
            FILE ${PROJECT}Targets.cmake
            NAMESPACE HDI::
            DESTINATION lib/cmake/HDILib
            COMPONENT HDIUTILS_TARGET
    )
endif()

