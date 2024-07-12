#include <stdio.h>

// error checking macro
#define cudaCheckErrors(msg)                                    \
    do {                                                        \
        cudaError_t __err = cudaGetLastError();                 \
        if (__err != cudaSuccess) {                             \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n",  \
                    msg, cudaGetErrorString(__err),             \
                    __FILE__, __LINE__);                        \
            fprintf(stderr, "*** FAILED - ABORTING\n");         \
            exit(1);                                            \
        }                                                       \
    } while (0)

const size_t DSIZE = (size_t) 32 * 1048576;

// vector add kernel: C = A + B
__global__ void vadd(const float *A, const float *B, float *C, size_t ds) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    while (idx < ds) {
        C[idx] = A[idx] + B[idx];
        idx += blockDim.x * gridDim.x; // grid-stride loop
    }
}

int main() {

    float *A, *B, *C;
    cudaMallocManaged(&A, DSIZE * sizeof(float));
    cudaMallocManaged(&B, DSIZE * sizeof(float));
    cudaMallocManaged(&C, DSIZE * sizeof(float));
    cudaCheckErrors("cudaMallocManaged failure");

    for (int i = 0; i < DSIZE; i++) {
        A[i] = rand() / (float) RAND_MAX;
        B[i] = rand() / (float) RAND_MAX;
        C[i] = 0;
    }

    vadd<<<80 * 32, 256>>>(A, B, C, DSIZE);
    cudaCheckErrors("kernel launch failure");

    cudaDeviceSynchronize();
    cudaCheckErrors("kernel execution failure");

    printf("A[0] = %f\n", A[0]);
    printf("B[0] = %f\n", B[0]);
    printf("C[0] = %f\n", C[0]);

    return 0;

}
