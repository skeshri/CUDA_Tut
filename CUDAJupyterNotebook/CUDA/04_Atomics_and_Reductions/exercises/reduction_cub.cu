#include <stdio.h>

// CUB Headers
#include <cub/cub.cuh>

#define N (32*1048576)

using namespace cub;

CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

int main()
{
  // host data
  double *h_A;
  double* h_sum;
  h_A = new double[N];
  h_sum = new double;

  // device data
  double *d_A;
  double* d_sum;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_sum, sizeof(double)));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_A, N * sizeof(double)));

  // initialize data in host memory
  for (int i = 0; i < N; i++) {
    h_A[i] = 1.0f;
  }
  *h_sum = 0.0f;
  
  // copy data to device memory
  CubDebugExit(cudaMemcpy(d_A, h_A, N * sizeof(double), cudaMemcpyHostToDevice));
  CubDebugExit(cudaMemcpy(d_sum, h_sum, sizeof(double), cudaMemcpyHostToDevice));

  // Request and allocate temporary storage
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_A, d_sum, N));
  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  // Run sum reduce on the device
  CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_A, d_sum, N));

  // copy data back to host from device
  CubDebugExit(cudaMemcpy(h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost));

  // check the host results
  if (*h_sum != (double) N) {
    printf("device reduction with CUB incorrect!\n");
    return -1;
  }
  printf("device reduction with CUB correct!\n");

  free(h_A);
  free(h_sum);
  CubDebugExit(g_allocator.DeviceFree(d_A));
  CubDebugExit(g_allocator.DeviceFree(d_sum));
  CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
  return 0;
}

