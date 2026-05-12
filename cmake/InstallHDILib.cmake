# Helper macro for packaging
include(CMakePackageConfigHelpers)

# Generate the version file for use with find_package
set(hdilib_package_version "${HDILib_VERSION}")
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ConfigVersion.cmake.in 
    "${CMAKE_CURRENT_BINARY_DIR}/HDILibConfigVersion.cmake" 
    @ONLY
)

#write_basic_package_version_file(
#  "${CMAKE_CURRENT_BINARY_DIR}/HDILibConfigVersion.cmake"
#  VERSION "${HDILib_VERSION}"
#  COMPATIBILITY ExactVersion
#)

set(INCLUDE_INSTALL_DIR include)
set(LIB_INSTALL_DIR lib)
set(CURRENT_BUILD_DIR "${CMAKE_BINARY_DIR}")

# create config file
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/HDILibConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/HDILibConfig.cmake"
  PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR CURRENT_BUILD_DIR
  INSTALL_DESTINATION lib/cmake/HDILib
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/HDILibConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/HDILibConfigVersion.cmake"
    DESTINATION lib/cmake/HDILib
    COMPONENT HDI_PACKAGE
)

if(HDILib_INSTALL_DEPENDENCIES)

    function(patch_file)
    # Step through the file line by line replacing 
    # the match expression with the replace expression
    # Overwrite the existing file
    cmake_parse_arguments(PARSE_ARGV 0 "patch_arg" "" "INPUT_PATH;MATCH_EXPRESSION;REPLACE_EXPRESSION" "")
    set(eol "\n")
    set(patched_result "")
    file(STRINGS "${patch_arg_INPUT_PATH}" input_lines)
    #message("Unpatched file ${input_lines}")
    foreach(line IN LISTS input_lines)
        string(REPLACE "${patch_arg_MATCH_EXPRESSION}" "${patch_arg_REPLACE_EXPRESSION}" patched_line "${line}")
        set(patched_result "${patched_result}${patched_line}${eol}")
    endforeach()
    #message("New patched file ${patched_result}")
    file(WRITE "${patch_arg_INPUT_PATH}" "${patched_result}")
    endfunction()

    function(add_config_to_cmake_target_files)
    # 1. Find CMake target files using a glob path (PATH_GLOB).
    # 2. Patch each file found so that the lib path contains a 
    # config (i.e. Release, Debug or RelWithDebInfo) element.
    # This matches the convention we use in multi-build packages

    cmake_parse_arguments(PARSE_ARGV 0 "fix_arg" "" "LIB_NAME;PATH_GLOB" "")
    #message("globbing: ${fix_arg_PATH_GLOB}")
    file(GLOB _cmake_config_files "${fix_arg_PATH_GLOB}")
    #message("glob list: ${_cmake_config_files}")
    foreach(_cmake_config_file IN LISTS _cmake_config_files)
        set(_upper_file "")
        string(TOUPPER "${_cmake_config_file}" _upper_file)
        #message("Checking: ${_upper_file}")
        # Identify if each file is a Release, Debug or RelWithDebugInfo
        # and apply the appropriate patch
        set(_found -1)
        string(FIND "${_upper_file}" "RELEASE.CMAKE" _found)
        if (_found GREATER -1)
            patch_file(
            INPUT_PATH "${_cmake_config_file}" 
            MATCH_EXPRESSION "lib/${fix_arg_LIB_NAME}" 
            REPLACE_EXPRESSION "Release/${fix_arg_LIB_NAME}"
            )
        endif()
        string(FIND "${_upper_file}" "DEBUG.CMAKE" _found)
        if (_found GREATER -1)
            # Eliminate the "debug" element - we package this differently
            message(STATUS "Change debug/lib for ${fix_arg_LIB_NAME} in ${_upper_file}")
            patch_file(
            INPUT_PATH "${_cmake_config_file}" 
            MATCH_EXPRESSION "debug/lib/${fix_arg_LIB_NAME}" 
            REPLACE_EXPRESSION "lib/${fix_arg_LIB_NAME}"
            )
            patch_file(
            INPUT_PATH "${_cmake_config_file}" 
            MATCH_EXPRESSION "lib/${fix_arg_LIB_NAME}" 
            REPLACE_EXPRESSION "Debug/${fix_arg_LIB_NAME}"
            )
        endif()
        string(FIND "${_upper_file}" "RELWITHDEBINFO.CMAKE" _found)
        if (_found GREATER -1)
            patch_file(
            INPUT_PATH "${_cmake_config_file}" 
            MATCH_EXPRESSION "lib/${fix_arg_LIB_NAME}" 
            REPLACE_EXPRESSION "RelWithDebInfo/${fix_arg_LIB_NAME}"
            )
        endif()
    endforeach()
    endfunction()

    function(fix_include_path)
    # Insert a dependency name element into the include path
    # In the package include each extra dependency has its own sub-dir
    # under include. 
    cmake_parse_arguments(PARSE_ARGV 0 "inc_arg" "" "DEPENDENCY_NAME;TARGET_PATH" "")

    patch_file(
        INPUT_PATH "${inc_arg_TARGET_PATH}" 
        MATCH_EXPRESSION "\${_IMPORT_PREFIX}/include" 
        REPLACE_EXPRESSION "\${_IMPORT_PREFIX}/../include"
    )
    endfunction()

    # Install the config.cmake files for the vcpkg dependencies
    message(STATUS "Fix config in ${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/share/kompute/komputeTargets-*.cmake")
    add_config_to_cmake_target_files(
        LIB_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}kompute${CMAKE_STATIC_LIBRARY_SUFFIX}" 
        PATH_GLOB "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/share/kompute/komputeTargets-*.cmake"
    )
    add_config_to_cmake_target_files(
        LIB_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}kp_logger${CMAKE_STATIC_LIBRARY_SUFFIX}" 
        PATH_GLOB "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/share/kompute/komputeTargets-*.cmake"
    )
    fix_include_path(
        DEPENDENCY_NAME "kompute"
        TARGET_PATH "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/share/kompute/komputeTargets.cmake"
    )
    install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/share/kompute" 
        DESTINATION lib/cmake COMPONENT HDI_PACKAGE)
    add_config_to_cmake_target_files(
        LIB_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}fmtd${CMAKE_STATIC_LIBRARY_SUFFIX}" 
        PATH_GLOB "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/share/fmt/fmt-targets-*.cmake")
    add_config_to_cmake_target_files(
        LIB_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}fmt${CMAKE_STATIC_LIBRARY_SUFFIX}" 
        PATH_GLOB "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/share/fmt/fmt-targets-*.cmake")
    fix_include_path(
        DEPENDENCY_NAME "fmt"
        TARGET_PATH "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/share/fmt/fmt-targets.cmake"
      )
    install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/share/fmt" 
        DESTINATION lib/cmake COMPONENT HDI_PACKAGE)
    add_config_to_cmake_target_files(
        LIB_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}glfw3${CMAKE_STATIC_LIBRARY_SUFFIX}" 
        PATH_GLOB "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/share/glfw3/glfw3Targets-*.cmake")
    install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/share/glfw3" 
        DESTINATION lib/cmake COMPONENT HDI_PACKAGE)

    # Install the include files
    install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/include/kompute"
        DESTINATION include
    )
    # Additional generated kompute headers
    install(FILES 
        "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/include/ShaderLogisticRegression.hpp"
        "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/include/ShaderOpMult.hpp"
        DESTINATION include
    )
    install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/include/fmt"
        DESTINATION include
    )
    install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/include/GLFW"
        DESTINATION include
    )
    install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/include/vulkan"
        DESTINATION include
    )
    install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/include/vk_video"
        DESTINATION include
    )

    # Install the static libraries
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}kompute${CMAKE_STATIC_LIBRARY_SUFFIX}"
        DESTINATION lib/Release 
        COMPONENT HDI_PACKAGE
    )
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}kp_logger${CMAKE_STATIC_LIBRARY_SUFFIX}"
        DESTINATION lib/Release 
        COMPONENT HDI_PACKAGE
    )
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}fmt${CMAKE_STATIC_LIBRARY_SUFFIX}"
        DESTINATION lib/Release 
        COMPONENT HDI_PACKAGE
    )

    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/debug/lib/${CMAKE_STATIC_LIBRARY_PREFIX}kompute${CMAKE_STATIC_LIBRARY_SUFFIX}"
        DESTINATION lib/Debug 
        COMPONENT HDI_PACKAGE
    )
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/debug/lib/${CMAKE_STATIC_LIBRARY_PREFIX}kp_logger${CMAKE_STATIC_LIBRARY_SUFFIX}"
        DESTINATION lib/Debug 
        COMPONENT HDI_PACKAGE
    )
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/debug/lib/${CMAKE_STATIC_LIBRARY_PREFIX}fmtd${CMAKE_STATIC_LIBRARY_SUFFIX}"
        DESTINATION lib/Debug 
        COMPONENT HDI_PACKAGE
    )

    # Install vulkan so lib?
    if(WIN32)
        install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}vulkan-1${CMAKE_STATIC_LIBRARY_SUFFIX}"
            DESTINATION lib/Release 
            COMPONENT HDI_PACKAGE
        )
        install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}vulkan-1${CMAKE_STATIC_LIBRARY_SUFFIX}"
            DESTINATION lib/Debug 
            COMPONENT HDI_PACKAGE
        )
    else()
        install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}vulkan${CMAKE_SHARED_LIBRARY_SUFFIX}"
            DESTINATION lib/Release 
            COMPONENT HDI_PACKAGE
        )
        install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}vulkan${CMAKE_SHARED_LIBRARY_SUFFIX}"
            DESTINATION lib/Debug 
            COMPONENT HDI_PACKAGE
        )
    endif()

endif()
