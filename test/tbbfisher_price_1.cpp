// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0
#include <vector>
#include <random>
#include <iostream>

#include "tbb/parallel_do.h"

struct particle
{
  float energy;
};

// For tbb::parallel_do, only one instance shared between threads
struct FisherPriceKernel
{
  void operator()(particle &p, tbb::parallel_do_feeder<particle> &feeder) const
  {
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    static thread_local std::uniform_real_distribution<float> rng;

    // With parallel_do, not easy to mark particle as "StopAndKill",
    // but enqueued secondaries are destroyed on the fly after processing
    while (p.energy > 0.0f)
    {
      // Essentially "process selection (GPIL)"
      float r = rng(gen);

      if (r < 0.5f)
      {
        // do energy loss
        float eloss = 0.2f * p.energy;
        p.energy = (eloss < 0.001f ? 0.0f : (p.energy - eloss));
      }
      else
      {
        // do "pair production"
        float eloss = 0.5f * p.energy;

        // "queue" secondary
        feeder.add(particle{eloss});

        p.energy -= eloss;
      }
    }
  }
};

int main()
{
  // Initial vector of particles
  std::vector<particle> particleVector;
  particleVector.emplace_back(particle{100.0f});

  tbb::parallel_do(particleVector, FisherPriceKernel());

  return 0;
}