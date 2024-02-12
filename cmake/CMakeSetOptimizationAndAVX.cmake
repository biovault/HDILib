# -----------------------------------------------------------------------------
# Check for and link to AVX instruction sets
# -----------------------------------------------------------------------------
macro(check_and_set_AVX target useavx)
    message(STATUS "Set instruction sets for ${target}, MV_USE_AVX is ${useavx}")

    if(${useavx})
        # Use cmake hardware checks to see whether AVX should be activated
        include(CheckCXXCompilerFlag)

        if(MSVC)
            set(AXV_CompileOption "/arch:AVX")
            set(AXV2_CompileOption "/arch:AVX2")
        else()
            set(AXV_CompileOption "-DUSE_AVX")
            set(AXV2_CompileOption "-DUSE_AVX2")
        endif()
        
        if(NOT DEFINED COMPILER_OPT_AVX_SUPPORTED OR NOT DEFINED COMPILER_OPT_AVX2_SUPPORTED)
            check_cxx_compiler_flag(${AXV_CompileOption} COMPILER_OPT_AVX_SUPPORTED)
            check_cxx_compiler_flag(${AXV2_CompileOption} COMPILER_OPT_AVX2_SUPPORTED)
        endif()

        if(COMPILER_OPT_AVX2_SUPPORTED)
            message( STATUS "Use AXV2 for ${target}")
            target_compile_options(${target} PRIVATE ${AXV2_CompileOption})
        elseif(COMPILER_OPT_AVX_SUPPORTED)
            message( STATUS "Use AXV for ${target}")
            target_compile_options(${target} PRIVATE ${AXV_CompileOption})
        endif()
    endif()
endmacro()

# -----------------------------------------------------------------------------
# Sets the optimization level
# -----------------------------------------------------------------------------
macro(set_optimization_level target level)
    message(STATUS "Set optimization level in release for ${target} to ${level}")

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        set(OPTIMIZATION_LEVEL_FLAG "-O${level}")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        if(${level} EQUAL 0)
            set(OPTIMIZATION_LEVEL_FLAG "/Od")
        else()
            set(OPTIMIZATION_LEVEL_FLAG "/O${level}")
        endif()
    endif()

    target_compile_options(${target} PRIVATE "$<$<CONFIG:Release>:${OPTIMIZATION_LEVEL_FLAG}>")

    message( STATUS "Optimization level for ${target} (release) is ${OPTIMIZATION_LEVEL_FLAG}")
endmacro()
