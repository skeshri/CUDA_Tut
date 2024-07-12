#include <cstdio>

#define N (1024 * 1024 * 32)

__global__ void setval (int* a) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        a[idx] = 1;
    }
}

int main() {

    // The first CUDA call pays for CUDA context creation;
    // insert a non-functional call here to pay for that cost
    // so we can better understand the true cost of the memory
    // allocation below.
    cudaFree(0);

    int* h_a;
    int* d_a;

    h_a = (int*) malloc(N * sizeof(int));
    cudaMalloc(&d_a, N * sizeof(int));

    memset(h_a, 0, N * sizeof(int));

    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    setval<<<blocks, threads>>>(d_a);
    cudaDeviceSynchronize();

    cudaMemcpy(h_a, d_a, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify the data got updated as we expect.
    if (h_a[0] == 1) {
        printf("Success!\n");
    }
    else {
        printf("Failure!\n");
    }

    free(h_a);
    cudaFree(d_a);

    return 0;

}
