# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.18)
project(CopCore VERSION 0.1.0)

include(GNUInstallDirs)
include(CTest)

# ---------------------
# Device Target Options:
# These can be set from the external project (Namespaced?)
# ---------------------
# Device target architecture
set(TARGET_DEVICE CUDA CACHE STRING "Target architecture of the device")
set_property(CACHE TARGET_DEVICE PROPERTY STRINGS CPU CUDA HIP CUDACLANG)
set(TARGET_DEFINITION "TARGET_DEVICE_${TARGET_DEVICE}")

# CPU manual vectorization target
set(CPU_STATIC_VECTOR_WIDTH OFF CACHE STRING "Define a static vector width for CPU target")
set_property(CACHE CPU_STATIC_VECTOR_WIDTH PROPERTY STRINGS OFF scalar 128bits 256bits 512bits)

# CUDA compute capability
set(CUDA_ARCH COMPATIBILITY CACHE STRING "CUDA target architecture")
set_property(CACHE CUDA_ARCH PROPERTY STRINGS COMPATIBILITY MAX MIN 53 60 61 62 70 72 75)

# -----
# Dependencies
if(TARGET_DEVICE STREQUAL "CUDA")
  enable_language(CUDA)
  find_package(CUDAToolkit 11.0 REQUIRED)
endif()

# Core library
add_library(CopCore STATIC
  include/CopCore/backend/BackendCommon.h
  include/CopCore/backend/CPUBackend.h
  include/CopCore/backend/CPUID.h
  include/CopCore/backend/CUDABackend.h
  include/CopCore/backend/HIPBackend.h
  include/CopCore/backend/Vector.h
  include/CopCore/CopCore.h
  include/CopCore/Invoke.cuh
  include/CopCore/LoggerCommon.h
  include/CopCore/Logger.h
  include/CopCore/TupleTools.cuh
  src/backend/CPUBackend.cpp
  src/backend/CPUID.cpp
  src/backend/HalfType.cpp
  src/backend/Utils.cpp)
target_compile_features(CopCore PUBLIC cxx_std_17 $<$<STREQUAL:"${TARGET_DEVICE}","CUDA">:cuda_std_17>)
target_compile_definitions(CopCore PUBLIC ${TARGET_DEFINITION})
target_include_directories(CopCore PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(CopCore PUBLIC $<$<STREQUAL:"${TARGET_DEVICE}","CUDA">:CUDA::cudart_static>)

# Alias to assist in use as a subproject
add_library(CopCore::CopCore ALIAS CopCore)

# Install headers and target(s)
install(DIRECTORY include/CopCore DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(TARGETS CopCore EXPORT CopCoreTargets)

# Testing
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

# Support files
include(CMakePackageConfigHelpers)

# Support files
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
