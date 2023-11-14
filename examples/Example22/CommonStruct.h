// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_INTEGRATION_COMMONSTRUCT_H
#define ADEPT_INTEGRATION_COMMONSTRUCT_H

#include <vector>
#include <AdePT/MParray.h>
#include <TrackData.h>

// Common data structures used by the integration with Geant4
namespace adeptint {

/// @brief Auxiliary logical volume data. This stores in the same structure the material-cuts couple index,
/// the sensitive volume handler index and the flag if the region is active for AdePT.
struct VolAuxData {
  int fSensIndex{-1}; ///< index of handler for sensitive volumes (-1 means non-sensitive)
  int fMCIndex{0};    ///< material-cut cuple index in G4HepEm
  int fGPUregion{0};  ///< GPU region index (currently 1 or 0, meaning tracked on GPU or not)
};

/// @brief Buffer holding input tracks to be transported on GPU and output tracks to be
/// re-injected in the Geant4 stack
struct TrackBuffer {
  std::vector<TrackData> toDevice;    ///< Tracks to be transported on the device
  std::vector<TrackData> fromDevice;  ///< Tracks coming from device to be transported on the CPU
  int eventId{-1};                    ///< Index of current transported event
  int startTrack{0};                  ///< Track counter for the current event
  int nelectrons{0};                  ///< Number of electrons in the input buffer
  int npositrons{0};                  ///< Number of positrons in the input buffer
  int ngammas{0};                     ///< Number of gammas in the input buffer

  void Clear()
  {
    toDevice.clear();
    fromDevice.clear();
    nelectrons = npositrons = ngammas = 0;
  }

  void AddTrack(int pdg, double energy, double x, double y, double z, double dirx, double diry, double dirz)
  {
    // Should really validate pdg here...
    toDevice.emplace_back(pdg, energy, x, y, z, dirx, diry, dirz);
    if (pdg == 11)
      nelectrons++;
    else if (pdg == -11)
      npositrons++;
    else if (pdg == 22)
      ngammas++;
  }

  void UpdateEventID(int event)
  {
    if (event != eventId) {
      eventId    = event;
      startTrack = 0;
    } else {
      startTrack += toDevice.size();
    }
  }
};

} // end namespace adeptint
#endif
