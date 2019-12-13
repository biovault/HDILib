set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories ("${CMAKE_SOURCE_DIR}")

message(STATUS "CMAKE_GENERATOR: ${CMAKE_GENERATOR}")
#Setting custom compilers for MacOS
if (CMAKE_GENERATOR STREQUAL Xcode)
    find_program(BREW NAMES brew)
    if("${BREW}" STRGREATER "")
        message(STATUS "Running on macOS Darwin using ${BREW}")
        execute_process(COMMAND ${BREW} --prefix llvm OUTPUT_VARIABLE BREW_LLVM_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND ${BREW} --prefix flann OUTPUT_VARIABLE BREW_FLANN_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif("${BREW}" STRGREATER "")
    
    message(STATUS "Trying to find llvm using ${BREW_LLVM_PREFIX}/lib/cmake/llvm")
    find_package(LLVM REQUIRED CONFIG HINTS "${BREW_LLVM_PREFIX}/lib/cmake/llvm")
    if(EXISTS ${LLVM_INSTALL_PREFIX})
        message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} in ${LLVM_INSTALL_PREFIX}")
        #message(STATUS "Using LLVMConfig.cmake in ${LLVM_DIR}")
        #message(STATUS "Found LLVM headers in ${LLVM_INCLUDE_DIRS}")

        find_package(Clang REQUIRED CONFIG HINTS "${BREW_LLVM_PREFIX}/lib/cmake/clang")
        message(STATUS "Found Clang in ${CLANG_INSTALL_PREFIX}")
        #message(STATUS "Found Clang headers in ${CLANG_INCLUDE_DIRS}")

        set(CMAKE_C_COMPILER "${LLVM_INSTALL_PREFIX}/bin/clang")
        set(CMAKE_CXX_COMPILER "${LLVM_INSTALL_PREFIX}/bin/clang++")
        set(CMAKE_XCODE_ATTRIBUTE_CC "${LLVM_INSTALL_PREFIX}/bin/clang")
        set(CMAKE_XCODE_ATTRIBUTE_CXX "${LLVM_INSTALL_PREFIX}/bin/clang++")

        execute_process(COMMAND ${BREW} --prefix libomp OUTPUT_VARIABLE BREW_LIBOMP_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(EXISTS ${BREW_LIBOMP_PREFIX})
            message(STATUS "Found OpenMP in ${BREW_LIBOMP_PREFIX}")
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY "${BREW_LIBOMP_PREFIX}/lib/libomp.dylib")
            include_directories("${BREW_LIBOMP_PREFIX}/include")
        else()
            message(WARNING "Brew found, but OpenMP support could not be detected, using Grand Central Dispatch instead.")
            add_definitions( -D__USE_GCD__)
        endif(EXISTS ${BREW_LIBOMP_PREFIX})
    else()
        message(WARNING "Could not find custom llvm using base install.")
        message(WARNING "No OpenMP support detected, using Grand Central Dispatch instead.")
        add_definitions( -D__USE_GCD__)
    endif(EXISTS ${LLVM_INSTALL_PREFIX})

    #finding flann
    if(EXISTS ${BREW_FLANN_PREFIX})
        message(STATUS "Flann found from brew at ${BREW_FLANN_PREFIX}.")
        include_directories("${BREW_FLANN_PREFIX}/include")
    else()
        # there should probably be an alternative in here for setting the flann path
        message(WARNING "Flann not found.")
    endif(EXISTS ${BREW_FLANN_PREFIX})
#Win/Linux
else()
    find_package(OpenMP REQUIRED)
    if (OPENMP_FOUND)
        message (STATUS "OpenMP found")
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()

    include_directories("$ENV{FLANN_DIR}/include")
    message (STATUS "$ENV{FLANN_DIR}/include")
endif()

option(HNSWLIB_INCLUDE_DIR "The include folder for HNSWLIB" OFF)

if( "${HNSWLIB_INCLUDE_DIR}" STREQUAL "OFF")
    message(STATUS "HNSWLIB Not Found!")
else("${HNSWLIB_INCLUDE_DIR}" STREQUAL "OFF")
    include_directories(${HNSWLIB_INCLUDE_DIR})
    message (STATUS "HNSWLIB: " ${HNSWLIB_INCLUDE_DIR})
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHNSWLIB_FOUND")
endif("${HNSWLIB_INCLUDE_DIR}" STREQUAL "OFF")


# Gather list of all .h files in "/"
file(GLOB HeaderFiles *.h gpgpu_sne/*.h)
file(GLOB SourceFiles *.cpp gpgpu_sne/*.cpp)

add_library   (hdidimensionalityreduction ${HeaderFiles} ${SourceFiles} )

########### INSTALL ##############
install (TARGETS hdidimensionalityreduction DESTINATION bin/hdi)
install (FILES ${HeaderFiles} ${UiHeaders} DESTINATION include/hdi/dimensionality_reduction)
