# -----------------------------------------------------------------------------
# HDILIB testing
# -----------------------------------------------------------------------------
set(HDILIB_TESTS "HDILibTests")

# Setup of test build depends on setup of parent project, the library itself
PROJECT(${HDILIB_TESTS} LANGUAGES CXX)

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
find_package(Catch2 CONFIG REQUIRED)

# -----------------------------------------------------------------------------
# Source files
# -----------------------------------------------------------------------------
set(SOURCES
    test_main.cpp
	test_utils.h
)

source_group(HDILibTests FILES ${SOURCES})

# -----------------------------------------------------------------------------
# CMake Target
# -----------------------------------------------------------------------------
add_executable(${HDILIB_TESTS} ${SOURCES})

# -----------------------------------------------------------------------------
# Target include directories
# -----------------------------------------------------------------------------
target_include_directories(${HDILIB_TESTS} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")

# -----------------------------------------------------------------------------
# Target link directories
# -----------------------------------------------------------------------------
target_link_libraries(${HDILIB_TESTS} PRIVATE Catch2::Catch2WithMain)
target_link_libraries(${HDILIB_TESTS} PRIVATE hdidimensionalityreduction hdiutils hdidata)

if(OpenMP_CXX_FOUND)
	target_link_libraries(${HDILIB_TESTS} PRIVATE OpenMP::OpenMP_CXX)
endif()

# -----------------------------------------------------------------------------
# Target properties
# -----------------------------------------------------------------------------
# Request C++20
target_compile_features(${HDILIB_TESTS} PRIVATE cxx_std_20)
target_compile_features(${HDILIB_TESTS} PRIVATE c_std_17)

# Instruction sets
hdi_check_and_set_AVX(${HDILIB_TESTS} ${HDILib_ENABLE_AVX})
