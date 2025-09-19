# All comp files in this directory will be converted to SPIRV binary
# in a cpp header

file(GLOB COMP_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.comp)
find_package(kompute CONFIG REQUIRED) # imports the vulkan compiler
set(_shader_hpp_list)
foreach(_shader ${COMP_FILES})
    string(REPLACE ".comp" ".hpp" _shader_hpp ${_shader})
    get_filename_component(_namespace ${_shader} NAME_WE)
    message(STATUS "Vulkan shader ${_shader} to ${_shader_hpp} with namespace ${_namespace}")
    list(APPEND _shader_hpp_list "${_shader_hpp}")
    vulkan_compile_shader(
        INFILE ${_shader}
        OUTFILE ${_shader_hpp}
        NAMESPACE ${_namespace}
    )   
endforeach()

add_custom_target(compile_shaders DEPENDS ${_shader_hpp_list})

add_library(vulkan_shader INTERFACE)
set_target_properties(vulkan_shader PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(vulkan_shader INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
add_dependencies(vulkan_shader compile_shaders)

