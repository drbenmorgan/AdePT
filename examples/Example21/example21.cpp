// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0
//

#include "DetectorConstruction.hh"
#include "ActionInitialisation.hh"

#include "G4RunManagerFactory.hh"
#include "G4Types.hh"
#include "G4UImanager.hh"
#include "FTFP_BERT_HepEm.hh"
#include "G4HadronicProcessStore.hh"
#include "G4EmParameters.hh"
#include "G4FastSimulationPhysics.hh"
#include "G4VisExecutive.hh"
#include "G4UIExecutive.hh"
#include <sstream>

int main(int argc, char **argv)
{
  // Macro name from arguments
  G4String batchMacroName;
  G4bool useInteractiveMode = true;
  G4String helpMsg("Usage: " + G4String(argv[0]) +
                   " [option(s)] \n No additional arguments triggers an interactive mode "
                   "executing vis.mac macro. \n Options:\n\t-h\t\tdisplay this help "
                   "message\n\t-m MACRO\ttriggers a batch mode executing MACRO\n");
  for (G4int i = 1; i < argc; ++i) {
    G4String argument(argv[i]);
    if (argument == "-h" || argument == "--help") {
      G4cout << helpMsg << G4endl;
      return 0;
    } else if (argument == "-m") {
      batchMacroName     = G4String(argv[i + 1]);
      useInteractiveMode = false;
      ++i;
    } else {
      G4Exception("main", "Unknown argument", FatalErrorInArgument,
                  ("Unknown argument passed to " + G4String(argv[0]) + " : " + argument + "\n" + helpMsg).c_str());
    }
  }

  // Initialization of default Run manager
  auto *runManager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Default);

  // Detector geometry:
  auto detector = new DetectorConstruction();
  runManager->SetUserInitialization(detector);

  // Physics list
  auto physicsList = new FTFP_BERT_HepEm();

  // reduce verbosity of physics lists
  G4EmParameters::Instance()->SetVerbose(0);
  G4HadronicProcessStore::Instance()->SetVerbose(0);
  runManager->SetUserInitialization(physicsList);

  //-------------------------------
  // UserAction classes
  //-------------------------------
  runManager->SetUserInitialization(new ActionInitialisation);

  G4UImanager *UImanager = G4UImanager::GetUIpointer();
  G4String command       = "/control/execute ";
  G4int err              = UImanager->ApplyCommand(command + batchMacroName);

  delete runManager;

  return err;
}
