include_directories ("${CMAKE_SOURCE_DIR}")

# Gather list of all .h files in "/"
file(GLOB HeaderFiles *.h)
file(GLOB SourceFiles *.cpp)

add_library   (hdidata ${HeaderFiles} ${SourceFiles} )

########### INSTALL ##############
install (TARGETS hdidata DESTINATION bin/hdi)
install (FILES ${HeaderFiles} ${UiHeaders} DESTINATION include/hdi/data)
