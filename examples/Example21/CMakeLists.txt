# SPDX-FileCopyrightText: 2022 CERN
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET G4HepEm::g4HepEm)
  message(STATUS "Disabling example21 (needs G4HepEm)")
  return()
endif()

if(Geant4_FOUND)
  if(NOT Geant4_gdml_FOUND)
    message(STATUS "Disabling example21 (needs Geant4 with GDML support)")
    return()
  endif()
else()
  message(STATUS "Disabling example21 (needs Geant4)")
  return()
endif()

set(sources_g4
  src/ActionInitialisation.cc
  src/DetectorConstruction.cc
  src/DetectorMessenger.cc
  src/EventAction.cc
  src/EventActionMessenger.cc
  src/PrimaryGeneratorAction.cc
  src/PrimaryGeneratorMessenger.cc
  src/RunAction.cc
  src/PhysListHepEm.cc
  src/TrackingAction.cc
  src/FTFP_BERT_HepEm.cc
)

if(HepMC3_FOUND)
  set(sources_hepmc3
      src/HepMC3G4AsciiReader.cc
      src/HepMC3G4AsciiReaderMessenger.cc
      src/HepMC3G4Interface.cc
     )
endif()

# example21
add_executable(example21 example21.cpp 
#AdeptIntegration.cpp AdeptIntegration.cu BasicScoring.cu 
               ${sources_g4} ${sources_hepmc3})

target_include_directories(example21 PRIVATE ${PROJECT_SOURCE_DIR}/examples/Example21/include 
                          ${PROJECT_SOURCE_DIR}/examples/Example21 ${HEPMC3_INCLUDE_DIR})

target_link_libraries(example21
  PRIVATE
    AdePT
    CopCore::CopCore
    VecGeom::vecgeom
    VecGeom::vecgeomcuda_static
    VecGeom::vgdml
    ${Geant4_LIBRARIES}
    G4HepEm::g4HepEm
    G4HepEm::g4HepEmData
    G4HepEm::g4HepEmInit
    G4HepEm::g4HepEmRun
    CUDA::cudart
    ${HEPMC3_LIBRARIES} ${HEPMC3_FIO_LIBRARIES})

set_target_properties(example21 PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)


# Tests
#add_test(NAME example21-adept-em3
#  COMMAND $<TARGET_FILE:example21> -m ${CMAKE_BINARY_DIR}/testEm3.mac)
#add_test(NAME example21-g4-em3
#  COMMAND $<TARGET_FILE:example21> -m ${CMAKE_BINARY_DIR}/testEm3G4.mac)
