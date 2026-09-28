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
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/HDILibConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/HDILibConfig.cmake"
    PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR CURRENT_BUILD_DIR
    INSTALL_DESTINATION share/HDILib
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/HDILibConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/HDILibConfigVersion.cmake"
    DESTINATION share/HDILib
    COMPONENT HDI_PACKAGE
)

if(HDILib_INSTALL_DEPENDENCIES AND HDILib_USE_VULKAN_KOMPUTE)

    # Install the config.cmake files for the vcpkg dependencies
    message("Merge dependencies to the HDILib package")
    install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/share/kompute" 
      DESTINATION share COMPONENT HDI_PACKAGE)
    install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/share/fmt" 
      DESTINATION share COMPONENT HDI_PACKAGE)
    install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/share/glfw3" 
      DESTINATION share COMPONENT HDI_PACKAGE)

    # Install the include files
    install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/include/kompute"
      DESTINATION include)
    # Additional generated kompute headers
    install(FILES 
      "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/include/ShaderLogisticRegression.hpp"
      "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/include/ShaderOpMult.hpp"
      DESTINATION include)
    install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/include/fmt"
      DESTINATION include)
    install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/include/GLFW"
      DESTINATION include)
    install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/include/vulkan"
      DESTINATION include)
    install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/include/vk_video"
      DESTINATION include)

    # Install the release libraries
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}kompute${CMAKE_STATIC_LIBRARY_SUFFIX}"
      DESTINATION lib COMPONENT HDI_PACKAGE)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}kp_logger${CMAKE_STATIC_LIBRARY_SUFFIX}"
      DESTINATION lib COMPONENT HDI_PACKAGE)
    # Install vulkan so lib?
    install(CODE "
      execute_process(COMMAND ls ${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/lib/)
    ")
    if(WIN32)
      install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}vulkan-1${CMAKE_STATIC_LIBRARY_SUFFIX}"
      DESTINATION lib COMPONENT HDI_PACKAGE)
    else()
      install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}vulkan${CMAKE_SHARED_LIBRARY_SUFFIX}"
      DESTINATION lib COMPONENT HDI_PACKAGE)
    endif()
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}fmt${CMAKE_STATIC_LIBRARY_SUFFIX}"
      DESTINATION lib COMPONENT HDI_PACKAGE)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}glfw3${CMAKE_STATIC_LIBRARY_SUFFIX}"
      DESTINATION lib COMPONENT HDI_PACKAGE)

    # Install the debug libraries
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/debug/lib/${CMAKE_STATIC_LIBRARY_PREFIX}kompute${CMAKE_STATIC_LIBRARY_SUFFIX}"
      DESTINATION debug/lib COMPONENT HDI_PACKAGE)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/debug/lib/${CMAKE_STATIC_LIBRARY_PREFIX}kp_logger${CMAKE_STATIC_LIBRARY_SUFFIX}"
      DESTINATION debug/lib COMPONENT HDI_PACKAGE)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/debug/lib/${CMAKE_STATIC_LIBRARY_PREFIX}fmtd${CMAKE_STATIC_LIBRARY_SUFFIX}"
      DESTINATION debug/lib COMPONENT HDI_PACKAGE)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/debug/lib/${CMAKE_STATIC_LIBRARY_PREFIX}glfw3${CMAKE_STATIC_LIBRARY_SUFFIX}"
      DESTINATION debug/lib COMPONENT HDI_PACKAGE)
endif()

