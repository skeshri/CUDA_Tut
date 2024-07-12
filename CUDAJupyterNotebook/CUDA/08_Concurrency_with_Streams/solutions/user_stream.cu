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

    cudaMallocHost(&h_a, N * sizeof(int));
    cudaMalloc(&d_a, N * sizeof(int));

    memset(h_a, 0, N * sizeof(int));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice, stream);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    setval<<<blocks, threads, 0, stream>>>(d_a);

    cudaMemcpyAsync(h_a, d_a, N * sizeof(int), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);

    // Verify the data got updated as we expect.
    if (h_a[0] == 1) {
        printf("Success!\n");
    }
    else {
        printf("Failure!\n");
    }

    cudaFreeHost(h_a);
    cudaFree(d_a);

    return 0;

}
