include_directories ("${CMAKE_SOURCE_DIR}")

set(PROJECT "hdidata")

# Gather list of all .h files in "/"
file(GLOB HeaderFiles *.h)
file(GLOB SourceFiles *.cpp)

add_library(${PROJECT} ${HeaderFiles} ${SourceFiles} )

########### INSTALL ##############
set_target_properties(${PROJECT} PROPERTIES PUBLIC_HEADER "${HeaderFiles}")

install(TARGETS ${PROJECT}
    CONFIGURATIONS Debug
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/Debug
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/hdi/data
)
install(TARGETS ${PROJECT}
    CONFIGURATIONS Release
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/Release
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/hdi/data
)
