set(PROJECT_DR "hdidimensionalityreduction")

configure_file(dr_config.h.in "${CMAKE_CURRENT_LIST_DIR}/dr_config.h")

set(HeaderFilesDR
    dr_config.h
    embedding_equalizer.h
    embedding_equalizer_inl.h
    evaluation.h
    evaluation_inl.h
    gpgpu_sne/field_computation.h
    gpgpu_sne/gpgpu_sne_compute.h
    gpgpu_sne/gpgpu_sne_raster.h
    gradient_descent_tsne_texture.h
    gradient_descent_tsne_texture_inl.h
    hd_joint_probability_generator.h
    hd_joint_probability_generator_inl.h
    hierarchical_sne.h
    hierarchical_sne_inl.h
    knn_utils.h
    sparse_tsne_user_def_probabilities.h
    sparse_tsne_user_def_probabilities_inl.h
    sptree.h
    sptree_inl.h
    tsne.h
    tsne_inl.h
    tsne_parameters.h
    tsne_random_walks.h
    tsne_random_walks_inl.h
    weighted_sptree.h
    weighted_sptree_inl.h
    wtsne.h
    wtsne_inl.h 
)

set(SourceFilesDR
    embedding_equalizer.cpp
    evaluation.cpp
    gpgpu_sne/field_computation.cpp
    gpgpu_sne/gpgpu_sne_compute.cpp
    gpgpu_sne/gpgpu_sne_raster.cpp
    gradient_descent_tsne_texture.cpp
    hd_joint_probability_generator.cpp
    hierarchical_sne.cpp
    knn_utils.cpp
    sparse_tsne_user_def_probabilities.cpp
    sptree.cpp
    tsne.cpp
    tsne_random_walks.cpp
    weighted_sptree.cpp
    wtsne.cpp
)

set(ShaderFilesVulkan "")
set(HeaderFilesVulkan "")

if(HDILib_USE_VULKAN_KOMPUTE)
    list(APPEND HeaderFilesDR
        gpgpu_vulkan/gpgpu_sne_comp_vulkan.h
        gpgpu_vulkan/shaders/shader_progs.h
        gpgpu_vulkan/shaders/shaders.h
        gpgpu_vulkan/shaders/vkUniformBufferHelper.h
        gpgpu_vulkan/shaders/vulkan_extension.h
        gpgpu_vulkan/tensor_config.h
    )

    list(APPEND SourceFilesDR
        gpgpu_vulkan/gpgpu_sne_comp_vulkan.cpp
        gpgpu_vulkan/shaders/shader_progs.cpp
        gpgpu_vulkan/shaders/shaders.cpp
        gpgpu_vulkan/shaders/vulkan_extension.cpp
    )

    list(APPEND ShaderFilesVulkan
        gpgpu_vulkan/shaders/bounds.comp
        gpgpu_vulkan/shaders/bounds_reduce.comp
        gpgpu_vulkan/shaders/center_scale.comp
        gpgpu_vulkan/shaders/compute_field_workgroup.comp
        gpgpu_vulkan/shaders/compute_fields.comp
        gpgpu_vulkan/shaders/compute_fields_enh.comp
        gpgpu_vulkan/shaders/compute_forces.comp
        gpgpu_vulkan/shaders/interp1.comp
        gpgpu_vulkan/shaders/interp2.comp
        gpgpu_vulkan/shaders/interp_fields.comp
        gpgpu_vulkan/shaders/reset_counter.comp
        gpgpu_vulkan/shaders/stencil.comp
        gpgpu_vulkan/shaders/stencil2active.comp
        gpgpu_vulkan/shaders/update.comp
    )

    # each *.comp file will be glslang converted to a SPIRV .hpp file
    set(HeaderFilesVulkan "")
    foreach(_shader_comp IN LISTS ShaderFilesVulkan)
      get_filename_component(_COMP_ROOT ${_shader_comp} NAME_WE)
      list(APPEND HeaderFilesVulkan "gpgpu_vulkan/shaders/${_COMP_ROOT}.hpp")
    endforeach()
    
    source_group(ShadersVulkan FILES ${ShaderFilesVulkan})
    message(STATUS "Add vulkan subdir")
    add_subdirectory(gpgpu_vulkan)
endif()

add_library(${PROJECT_DR} STATIC ${HeaderFilesDR} ${SourceFilesDR} ${ShaderFilesVulkan})

target_include_directories(${PROJECT_DR} PRIVATE "${PROJECT_SOURCE_DIR}")

target_link_libraries(${PROJECT_DR} PRIVATE lz4::lz4)
target_link_libraries(${PROJECT_DR} PRIVATE flann::flann_cpp_s)
target_link_libraries(${PROJECT_DR} PRIVATE Annoy::Annoy)
target_link_libraries(${PROJECT_DR} PRIVATE hnswlib::hnswlib)

if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_DR} PRIVATE OpenMP::OpenMP_CXX)
endif()

if (HDILib_USE_VULKAN_KOMPUTE)
    target_link_libraries(${PROJECT_DR} PRIVATE kompute::kompute)
    add_dependencies(${PROJECT_DR} compile_shaders)
endif()

if(UNIX)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_DR} PRIVATE Threads::Threads)
endif(UNIX)

target_compile_definitions(${PROJECT_DR} PRIVATE 
    _ISOC99_SOURCE 
    _GNU_SOURCE
    $<$<BOOL:${HDILib_USE_VULKAN_KOMPUTE}>:VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1>
)

if(HDILib_ENABLE_PID)
    set_target_properties(${PROJECT_DR} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if(HDILib_ENABLE_CODE_ANALYSIS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(${PROJECT_DR} PRIVATE /analyze)
    endif()
endif()

hdi_check_and_set_AVX(${PROJECT_DR} ${HDILib_ENABLE_AVX})
hdi_set_optimization_level(${PROJECT_DR} ${HDILib_OPTIMIZATION_LEVEL})

########### INSTALL ##############
if(${HDILib_INSTALL})
    install(TARGETS ${PROJECT_DR}
        EXPORT ${PROJECT_DR}Targets
        LIBRARY DESTINATION lib/$<CONFIGURATION>
        ARCHIVE DESTINATION lib/$<CONFIGURATION>
        COMPONENT hdidimensionalityreduction
    )

    # Preserve the header hierarchy by explicit install
    # the CMake PUBLIC_HEADER target property flattens it if used.
    set(_ALL_HEADERS ${HeaderFilesDR} ${HeaderFilesVulkan})

    install(CODE "foreach(DR_HEADER ${_ALL_HEADERS})
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
    install(EXPORT ${PROJECT_DR}Targets
            FILE ${PROJECT_DR}Targets.cmake
            NAMESPACE HDI::
            DESTINATION lib/cmake/HDILib
            COMPONENT HDIDIMENSIONALITYREDUCTION_TARGET
    )
endif()
