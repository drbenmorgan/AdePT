// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef SCORING_H
#define SCORING_H

#include <CopCore/Global.h>
#include <CopCore/PhysicalConstants.h>
#include <VecGeom/navigation/NavStateIndex.h>
#include "CommonStruct.h"

struct BasicScoring;
using AdeptScoring = BasicScoring;

struct BasicScoring {
  using VolAuxData = adeptint::VolAuxData;
  VolAuxData *fAuxData_dev{nullptr};

  __device__ __forceinline__ VolAuxData const &GetAuxData_dev(int volId) const { return fAuxData_dev[volId]; }
  BasicScoring *InitializeOnGPU();

};

#endif
