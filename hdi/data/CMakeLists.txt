include_directories ("${CMAKE_SOURCE_DIR}")

set(PROJECT "hdidata")

# Gather list of all .h files in "/"
file(GLOB HeaderFiles *.h)
file(GLOB SourceFiles *.cpp)

add_library(${PROJECT} STATIC ${HeaderFiles} ${SourceFiles} )

########### INSTALL ##############
set_target_properties(${PROJECT} PROPERTIES PUBLIC_HEADER "${HeaderFiles}")
target_link_libraries(${PROJECT} ${OPENGL_LIBRARIES})

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
