#include <stdio.h>

#define N (32*1048576)

__global__ void reduce (double* A, double* sum) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        atomicAdd(sum, A[idx]);
    }
}

int main()
{
    double *h_A;
    double* h_sum;
    h_A = new double[N];
    h_sum = new double;

    double* d_A;
    double* d_sum;
    cudaMalloc(&d_A, N * sizeof(double));
    cudaMalloc(&d_sum, sizeof(double));

    // initialize data in host memory
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
    }
    *h_sum = 0.0f;

    // copy data to device memory
    cudaMemcpy(d_A, h_A, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, h_sum, sizeof(double), cudaMemcpyHostToDevice);

    // sum the array on the device
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    reduce<<<blocks, threads>>>(d_A, d_sum);
    cudaDeviceSynchronize();

    // check the device results
    cudaMemcpy(h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
    if (*h_sum != (double) N) {
        printf("device reduction incorrect!\n");
        return -1;
    }
    printf("device reduction correct!\n");

    free(h_A);
    free(h_sum);
    cudaFree(d_A);
    cudaFree(d_sum);

    return 0;
}
