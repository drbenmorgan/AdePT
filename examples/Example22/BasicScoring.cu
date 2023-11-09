// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include "BasicScoring.h"
#include "AdeptIntegration.h"


BasicScoring *BasicScoring::InitializeOnGPU()
{
  fAuxData_dev = AdeptIntegration::VolAuxArray::GetInstance().fAuxData_dev;
  // Now allocate space for the BasicScoring placeholder on device and only copy the device pointers of components
  BasicScoring *BasicScoring_dev = nullptr;
  COPCORE_CUDA_CHECK(cudaMalloc(&BasicScoring_dev, sizeof(BasicScoring)));
  COPCORE_CUDA_CHECK(cudaMemcpy(BasicScoring_dev, this, sizeof(BasicScoring), cudaMemcpyHostToDevice));
  return BasicScoring_dev;
}
