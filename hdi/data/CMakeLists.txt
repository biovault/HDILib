set(PROJECT_DATA "hdidata")

set(HeaderFilesData
    abstract_data.h
    abstract_panel_data.h
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

add_library(${PROJECT_DATA} STATIC ${HeaderFilesData} ${SourceFilesData} )

target_include_directories(${PROJECT_DATA} PRIVATE "${PROJECT_SOURCE_DIR}")

target_link_libraries(${PROJECT_DATA} PRIVATE ${OPENGL_LIBRARIES})

set_target_properties(${PROJECT_DATA} PROPERTIES PUBLIC_HEADER "${HeaderFilesData}")

if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_DATA} PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_definitions(${PROJECT_DATA} PRIVATE _ISOC99_SOURCE _GNU_SOURCE)

if(HDILib_ENABLE_PID)
    set_target_properties(${PROJECT_DATA} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

hdi_check_and_set_AVX(${PROJECT_DATA} ${HDILib_ENABLE_AVX})
hdi_set_optimization_level(${PROJECT_DATA} ${HDILib_OPTIMIZATION_LEVEL})

########### INSTALL ##############
if(${HDILib_INSTALL})
    install(TARGETS ${PROJECT_DATA}
        EXPORT ${PROJECT_DATA}Targets
        LIBRARY DESTINATION lib/$<CONFIGURATION>>
        ARCHIVE DESTINATION lib/$<CONFIGURATION>
        PUBLIC_HEADER DESTINATION include/hdi/data
        COMPONENT hdidata
    )

    install(EXPORT ${PROJECT_DATA}Targets
            FILE ${PROJECT_DATA}Targets.cmake
            NAMESPACE HDI::
            DESTINATION lib/cmake/HDILib
            COMPONENT HDIDATA_TARGET
    )
endif()
