include_directories ("${CMAKE_SOURCE_DIR}")

# Gather list of all .h files in "/"
file(GLOB HeaderFiles *.h)
file(GLOB SourceFiles *.cpp)

add_library(hdiutils ${HeaderFiles} ${SourceFiles} ${Resources} )

if (UNIX)
    target_link_libraries (hdiutils ${CMAKE_DL_LIBS}) # glad.cpp requires -ldl
endif (UNIX)

########### INSTALL ##############
install (TARGETS hdiutils DESTINATION bin/hdi)
install (FILES ${HeaderFiles} DESTINATION include/hdi/utils)
