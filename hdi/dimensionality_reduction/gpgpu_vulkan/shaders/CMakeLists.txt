# All comp files in this directory will be converted to SPIRV binary
# in a cpp header

file(GLOB COMP_FILES *.comp RELATIVE ${CMAKE_CURRENT_SOURCE_DIR})

foreach(IN_SHADER COMP_FILES)
    string(REPLACE ".comp" ".hpp" OUT_SHADER ${IN_SHADER})
    vulkan_compile_shader(
        INFILE ${IN_SHADER}
        OUTFILE ${OUT_SHADER}
        NAMESPACE "shader"
    )   
endforeach()
