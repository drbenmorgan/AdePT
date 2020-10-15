#include "CopCore/CopCore.h"
#include "CopCore/Invoke.cuh"
#include <tuple>

// Dumb saxpy kernel
__global__ void saxpy(float a, float *x, float *y, unsigned number_of_events)
{
  // A standard grid-stride loop works fine
  // It reduces to the Allen "block-stride" for CPU only (blockIdx.x=0, blockDim.x=1)
  // unsigned start = threadIdx.x;
  // unsigned stride = blockDim.x;

  unsigned start  = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned stride = blockDim.x * gridDim.x;

  for (unsigned i = start; i < number_of_events; i += stride) {
    y[i] = a * x[i] + y[i];
  }
}

int main()
{
  // Can we allocate memory on CPU/GPU with common functions?
  int N      = 1 << 20;
  float *x   = nullptr;
  float *y   = nullptr;
  float *d_x = nullptr;
  float *d_y = nullptr;

  // Initialize
  x = (float *)malloc(N * sizeof(float));
  y = (float *)malloc(N * sizeof(float));
  cudaMalloc((void **)&d_x, N * sizeof(float));
  cudaMalloc((void **)&d_y, N * sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

  // Invoke
  // Allen's example saxpy hardcodes nBlocks(1), but as long as we use
  // a grid-stride loop and 1d data, we can set to anything reasonable
  // Allen's support for CPU/GPU commonality was implemented here:
  // - https://gitlab.cern.ch/lhcb/Allen/-/merge_requests/196
  // where it mentions use of a block-strided loop:
  //
  // for(unsigned i = threadIdx.x; i < N; i += blockDim.x)
  //
  // but this is use-case dependent and seems typically used for
  // a pattern of one block per event (so eventID == blockIdx.x),
  // (e.g.) one thread per track (the block-stride loop)
  dim3 nBlocks(32);
  dim3 nThreads(32);

  auto args           = std::make_tuple(2.0f, d_x, d_y, N);
  constexpr auto size = std::tuple_size_v<decltype(args)>;
  using indices       = std::make_index_sequence<size>;

  invoke_impl(saxpy, nBlocks, nThreads, cudaStream_t{}, args, indices{});

  // Check
  cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++) {
    maxError = max(maxError, abs(y[i] - 4.0f));
  }
  printf("Max error: %f\n", maxError);

  // Tidy
  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
  return 0;
}