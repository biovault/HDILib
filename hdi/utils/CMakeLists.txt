set(PROJECT "hdiutils")

# Gather list of all .h files in "/"
file(GLOB HeaderFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h glad/*.h glad/KHR/*.h)
file(GLOB SourceFiles *.cpp)

add_library(${PROJECT} STATIC ${HeaderFiles} ${SourceFiles} ${Resources} )

target_include_directories(${PROJECT} PRIVATE "${PROJECT_SOURCE_DIR}")

#set_target_properties(${PROJECT} PROPERTIES PUBLIC_HEADER "${HeaderFiles}")

if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT} PRIVATE OpenMP::OpenMP_CXX)
endif()

if(ENABLE_PID)
    set_target_properties(${PROJECT} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if(ENABLE_CODE_ANALYSIS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(${PROJECT} PRIVATE /analyze)
    endif()
endif()

set_optimization_level(${PROJECT} ${OPTIMIZATION_LEVEL})
check_and_set_AVX(${PROJECT} ${ENABLE_AVX})

if (UNIX)
    target_link_libraries (hdiutils PRIVATE ${CMAKE_DL_LIBS}) # glad.cpp requires -ldl
endif (UNIX)

########### INSTALL ##############
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
