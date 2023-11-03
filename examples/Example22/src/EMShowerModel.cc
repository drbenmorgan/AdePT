// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include "EMShowerMessenger.hh"

#include <G4RunManager.hh>
#include "G4Electron.hh"
#include "G4Positron.hh"
#include "G4Gamma.hh"
#include "G4SystemOfUnits.hh"
#include "G4UnitsTable.hh"
#include "G4FastHit.hh"
#include "Randomize.hh"
#include "G4FastSimHitMaker.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"

#include "EMShowerModel.hh"
#include "G4GlobalFastSimulationManager.hh"

EMShowerModel::EMShowerModel(G4String aModelName, G4Region *aEnvelope) : G4VFastSimulationModel(aModelName, aEnvelope)
{
  fRegion = aEnvelope;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

EMShowerModel::EMShowerModel(G4String aModelName) : G4VFastSimulationModel(aModelName) {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

EMShowerModel::~EMShowerModel()
{
  fAdept->Cleanup();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4bool EMShowerModel::IsApplicable(const G4ParticleDefinition &aParticleType)
{
  return &aParticleType == G4Electron::ElectronDefinition() || &aParticleType == G4Positron::PositronDefinition() ||
         &aParticleType == G4Gamma::GammaDefinition();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4bool EMShowerModel::ModelTrigger(const G4FastTrack &aFastTrack)
{

  // The model is invoked for e/e-/gamma, so this has to return true
  return true;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void EMShowerModel::DoIt(const G4FastTrack &aFastTrack, G4FastStep &aFastStep)
{
  auto pdg               = aFastTrack.GetPrimaryTrack()->GetParticleDefinition()->GetPDGEncoding();
  G4double energy        = aFastTrack.GetPrimaryTrack()->GetKineticEnergy();
  auto g4track           = aFastTrack.GetPrimaryTrack();
  auto particlePosition  = g4track->GetPosition();
  auto particleDirection = g4track->GetMomentumDirection();

  // Remove particle from further processing by G4
  // No need to create any deposit, it will be handled by this model (and
  // G4FastSimHitMaker that will call the sensitive detector)
  aFastStep.KillPrimaryTrack();
  aFastStep.SetPrimaryTrackPathLength(0.0);
  aFastStep.SetTotalEnergyDeposited(0);

  fAdept->AddTrack(pdg, energy, particlePosition[0], particlePosition[1], particlePosition[2], particleDirection[0],
                   particleDirection[1], particleDirection[2]);
}

void EMShowerModel::Flush()
{
  fAdept->Shower(G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID());
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void EMShowerModel::Initialize(bool adept)
{
  fAdept = new AdeptIntegration;
  fAdept->SetDebugLevel(fVerbosity);
  fAdept->SetBufferThreshold(fBufferThreshold);
  fAdept->SetMaxBatch(2 * fBufferThreshold);

  G4RunManager::RMType rmType = G4RunManager::GetRunManager()->GetRunManagerType();
  bool sequential             = (rmType == G4RunManager::sequentialRM);

  fAdept->SetSensitiveVolumes(sensitive_volume_index);
  fAdept->SetScoringMap(fScoringMap);
  fAdept->SetRegion(fRegion);

  auto tid = G4Threading::G4GetThreadId();
  if (tid < 0) {
    // This is supposed to set the max batching for Adept to allocate properly the memory
    int num_threads = G4RunManager::GetRunManager()->GetNumberOfThreads();
    int capacity    = 1024 * 1024 * fTrackSlotsGPU / num_threads;
    AdeptIntegration::SetTrackCapacity(capacity);
    fAdept->Initialize(true /*common_data*/);
    if (sequential && adept) fAdept->Initialize();
  } else {
    if (adept) fAdept->Initialize();
  }
}
