set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories ("${CMAKE_SOURCE_DIR}")
include_directories ("${CMAKE_CURRENT_BINARY_DIR}")

if (CMAKE_GENERATOR STREQUAL Xcode)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "10.12" CACHE STRING "Minimum OS X deployment version")
  set_target_properties(${PROJECT} PROPERTIES XCODE_ATTRIBUTE_COMPILER_INDEX_STORE_ENABLE "NO")
  if(DEFINED ENV{CONAN_IS_TRAVIS_OSX})
    find_program(BREW NAMES brew)
    message(STATUS, "In Darwin with ${BREW}")
    execute_process(COMMAND ${BREW} --prefix libomp OUTPUT_VARIABLE BREW_LIBOMP_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS, "BREW LIBOMP ${BREW_LIBOMP_PREFIX}")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "${BREW_LIBOMP_PREFIX}/lib/libomp.dylib")
    include_directories("${BREW_LIBOMP_PREFIX}/include") 
  else() 
      find_package(LLVM)
      if( LLVM_FOUND )
        include_directories (${LLVM_LIBRARY_DIRS}/clang/${LLVM_VERSION_BASE_STRING}/include)
        
        set(CMAKE_CXX_COMPILER "/usr/local/opt/llvm/bin/clang++")
        set(CMAKE_C_COMPILER "/usr/local/opt/llvm/bin/clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
        set(CMAKE_XCODE_ATTRIBUTE_CC "/usr/local/opt/llvm/bin/clang")
        set(CMAKE_XCODE_ATTRIBUTE_CXX "/usr/local/opt/llvm/bin/clang++")

        find_library(IOMP5LIB
          NAMES "iomp5" "iomp5md" "libiomp5" "libiomp5md"
          HINTS ${LLVM_LIBRARY_DIRS})
        set (OMP_LIBRARIES ${OMP_LIBRARIES} ${IOMP5LIB})
      else(LLVM_FOUND)
        message("OS is macOS, no OpenMP support detected, using Grand Central Dispatch instead.")
        add_definitions( -D__USE_GCD__)
      endif( LLVM_FOUND )
  endif()
else()
  find_package(OpenMP REQUIRED)
  if (OPENMP_FOUND)
    message (STATUS "OpenMP found")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
endif()

# Gather list of all .h files in "/"
file(GLOB HeaderFiles *.h)
file(GLOB QOBJHeaderFiles *_qobj.h)
file(GLOB SourceFiles *.cpp)
file(GLOB FormFiles *.ui)
file(GLOB ResourceFiles *.qrc)

add_library(hdiutils ${HeaderFiles} ${SourceFiles} ${Resources} )

if (UNIX)
    target_link_libraries (hdiutils ${CMAKE_DL_LIBS}) # glad.cpp requires -ldl
endif (UNIX)

########### INSTALL ##############
install (TARGETS hdiutils DESTINATION bin/hdi)
install (FILES ${HeaderFiles} DESTINATION include/hdi/utils)
