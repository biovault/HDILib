include_directories ("${CMAKE_SOURCE_DIR}")

set(PROJECT "hdiutils")

# Gather list of all .h files in "/"
file(GLOB HeaderFiles *.h)
file(GLOB SourceFiles *.cpp)

add_library(${PROJECT} ${HeaderFiles} ${SourceFiles} ${Resources} )

if (UNIX)
    target_link_libraries (hdiutils ${CMAKE_DL_LIBS}) # glad.cpp requires -ldl
endif (UNIX)

########### INSTALL ##############
set_target_properties(${PROJECT} PROPERTIES PUBLIC_HEADER "${HeaderFiles}")

install(TARGETS ${PROJECT}
    CONFIGURATIONS Debug
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/Debug
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/hdi/utils
)
install(TARGETS ${PROJECT}
    CONFIGURATIONS Release
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/Release
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/hdi/utils
)
