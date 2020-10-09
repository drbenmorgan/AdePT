# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.18)
project(CopCore VERSION 0.1.0)

include(GNUInstallDirs)

# ---------------------
# Device Target Options:
# These should really be set from the external project
# ---------------------

# Device target architecture
set(TARGET_DEVICE CPU CACHE STRING "Target architecture of the device")
set_property(CACHE TARGET_DEVICE PROPERTY STRINGS CPU CUDA HIP CUDACLANG)
set(TARGET_DEFINITION "TARGET_DEVICE_${TARGET_DEVICE}")
add_compile_definitions(${TARGET_DEFINITION})

# CPU manual vectorization target
set(CPU_STATIC_VECTOR_WIDTH OFF CACHE STRING "Define a static vector width for CPU target")
set_property(CACHE CPU_STATIC_VECTOR_WIDTH PROPERTY STRINGS OFF scalar 128bits 256bits 512bits)

# CUDA compute capability
set(CUDA_ARCH COMPATIBILITY CACHE STRING "CUDA target architecture")
set_property(CACHE CUDA_ARCH PROPERTY STRINGS COMPATIBILITY MAX MIN 53 60 61 62 70 72 75)

# -----

# Core library
add_library(CopCore INTERFACE)
target_compile_features(CopCore INTERFACE cxx_std_17)
target_include_directories(CopCore INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Alias to assist in use as a subproject
add_library(CopCore::CopCore ALIAS CopCore)

# Install headers and target(s)
install(DIRECTORY include/CopCore DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(TARGETS CopCore EXPORT CopCoreTargets)

# Support files
include(CMakePackageConfigHelpers)

# -------------------
# Add backend library
# -------------------
file(GLOB backend_sources "src/backend/*.cpp")
add_library(copcore_backend ${backend_sources})
target_include_directories(copcore_backend PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_compile_features(copcore_backend PUBLIC cxx_std_17)
install(TARGETS copcore_backend)


write_basic_package_version_file("${PROJECT_BINARY_DIR}/CopCoreConfigVersion.cmake"
  COMPATIBILITY AnyNewerVersion
  ARCH_INDEPENDENT)

configure_package_config_file(CopCoreConfig.cmake.in "${PROJECT_BINARY_DIR}/CopCoreConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CopCore"
  PATH_VARS CMAKE_INSTALL_INCLUDEDIR)

install(FILES "${PROJECT_BINARY_DIR}/CopCoreConfigVersion.cmake" "${PROJECT_BINARY_DIR}/CopCoreConfig.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CopCore")

install(EXPORT CopCoreTargets
  NAMESPACE CopCore::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CopCore")
