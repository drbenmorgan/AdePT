#include "CopCore/CopCore.h"
#include "CopCore/Invoke.cuh"
#include <tuple>

// Dumb saxpy kernel
// But what invokes this, with blockIdx and so on set appropriately?
__global__ void saxpy(int n, float a, float *x, float *y)
{
  // In Allen, to make kernels reproducible on CPU/GPU must use blockDim strided loop:
  for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
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
  auto args           = std::make_tuple(N, 2.0f, d_x, d_y);
  constexpr auto size = std::tuple_size_v<decltype(args)>;
  using indices       = std::make_index_sequence<size>;

  // This is the awkward part - how to set the grid/block appriopriately
  // for CPU vs GPU.
  invoke_impl(saxpy, dim3((N + 255) / 256), dim3(256), cudaStream_t{}, args, indices{});

  // Check

  // Tidy
  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
  return 0;
}