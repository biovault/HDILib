set(PROJECT "hdidata")

set(HeaderFilesData
    abstract_data.h
    abstract_panel_data.h
    cluster.h
    embedding.h
    embedding_inl.h
    empty_data.h
    flow_model.h
    flow_model_inl.h
    histogram.h
    histogram_inl.h
    io.h
    map_helpers.h
    map_mem_eff.h
    nano_flann.h
    panel_data.h
    panel_data_inl.h
    pixel_data.h
    shader.h
    text_data.h
    voxel_data.h
    vptree.h
)

set(SourceFilesData 
    embedding.cpp
    flow_model.cpp
    histogram.cpp
    panel_data.cpp
    shader.cpp
)

add_library(${PROJECT} STATIC ${HeaderFilesData} ${SourceFilesData} )

target_include_directories(${PROJECT} PRIVATE "${PROJECT_SOURCE_DIR}")

target_link_libraries(${PROJECT} PRIVATE ${OPENGL_LIBRARIES})

set_target_properties(${PROJECT} PROPERTIES PUBLIC_HEADER "${HeaderFilesData}")

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
