cmake_minimum_required(VERSION 3.5)
project(DPC-ANN LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP)

# -------------------------- Options ------------------------------------------

set(COMPILE_OPTIONS

    $<$<CONFIG:Debug>:-std=c++17 -DPARLAY_SEQUENTIAL -mcx16 -pthread -march=native -g -O0 -DDEBUG >
    
    $<$<CONFIG:Release>:-std=c++17 -O3 -DHOMEGROWN -mcx16 -pthread -march=native -DNDEBUG>
)

# -------------------------- Includes ------------------------------------------

include_directories("ParlayANN/parlaylib/include")

# --------------------- Create CC Library --------------------------------------

set(CC_SOURCES
    doubling_dpc.cpp
    )


add_library(dpc_ann_cc STATIC ${CC_SOURCES})
target_compile_options(dpc_ann_cc PRIVATE ${COMPILE_OPTIONS})

# ----------------- Nanobind to create python bindings -------------------------

find_package(Python 3 COMPONENTS Interpreter Development.Module REQUIRED)

# Configure CMake to perform an optimized release build by default unless 
# another build type is specified. Without this addition, binding code may 
# run slowly and produce large binaries.
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
  dpc_ann
  python_bindings.cc
)

# Ignore nanobind warnings (temporary)
get_property(
    nanobind_includes TARGET nanobind-static
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES
)
set_target_properties(
    nanobind-static PROPERTIES
    FOLDER nanobind-static
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${nanobind_includes}"
)

# ----------------- Build package ---------------------------------------


if (NOT SKBUILD)

# # Create python so file
# target_link_libraries(dpc_ann PUBLIC dpc_ann_cc)
# target_compile_options(dpc_ann PRIVATE ${COMPILE_OPTIONS})

# Create executable
add_executable(dpc_ann_exe ${CC_SOURCES})
target_compile_options(dpc_ann_exe PRIVATE ${COMPILE_OPTIONS})

else()

# Install python package using scikit-build
install(TARGETS dpc_ann LIBRARY DESTINATION dpc_ann)

endif()