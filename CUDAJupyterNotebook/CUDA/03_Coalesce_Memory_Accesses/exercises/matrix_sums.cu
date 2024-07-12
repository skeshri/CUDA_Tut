#include <stdio.h>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

const size_t DSIZE = 16384;  // matrix side dimension
const int block_size = 256;  // CUDA maximum is 1024

typedef float real;

// matrix row-sum kernel
__global__ void row_sums(const real *A, real *sums, size_t ds)
{
    int idx = FIXME; // create typical 1D thread index from built-in variables
    if (idx < ds) {
        real sum = 0.0;
        for (size_t i = 0; i < ds; i++) {
            sum += A[FIXME];         // write a for loop that will cause the thread to iterate across a row, keeeping a running sum, and write the result to sums
        }
        sums[idx] = sum;
    }
}

// matrix column-sum kernel
__global__ void column_sums(const real *A, real *sums, size_t ds)
{
    int idx = FIXME; // create typical 1D thread index from built-in variables
    if (idx < ds) {
        real sum = 0.0;
        for (size_t i = 0; i < ds; i++) {
            sum += A[FIXME];         // write a for loop that will cause the thread to iterate down a column, keeeping a running sum, and write the result to sums
        }
        sums[idx] = sum;
    }
}

bool validate(real *data, size_t sz)
{
    for (size_t i = 0; i < sz; i++) {
        if (data[i] != (real) sz) {
            printf("results mismatch at %lu, was: %f, should be: %f\n", i, data[i], (real) sz);
            return false;
        }
    }
    return true;
}

int main()
{
    real *h_A, *h_sums, *d_A, *d_sums;
    h_A = new real[DSIZE*DSIZE];  // allocate space for data in host memory
    h_sums = new real[DSIZE];

    for (int i = 0; i < DSIZE*DSIZE; i++)  // initialize matrix in host memory
        h_A[i] = 1.0;

    cudaMalloc(&d_A, DSIZE*DSIZE*sizeof(real));  // allocate device space for A
    FIXME; // allocate device space for vector d_sums
    cudaCheckErrors("cudaMalloc failure"); // error checking

    // copy matrix A to device:
    cudaMemcpy(d_A, h_A, DSIZE*DSIZE*sizeof(real), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    row_sums<<<(DSIZE+block_size-1)/block_size, block_size>>>(d_A, d_sums, DSIZE);
    cudaCheckErrors("kernel launch failure");

    // copy vector sums from device to host:
    cudaMemcpy(h_sums, d_sums, DSIZE*sizeof(real), cudaMemcpyDeviceToHost);
    cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");

    if (!validate(h_sums, DSIZE)) return -1;
    printf("row sums correct!\n");

    cudaMemset(d_sums, 0, DSIZE*sizeof(real));

    column_sums<<<(DSIZE+block_size-1)/block_size, block_size>>>(d_A, d_sums, DSIZE);
    cudaCheckErrors("kernel launch failure");

    // copy vector sums from device to host:
    cudaMemcpy(h_sums, d_sums, DSIZE*sizeof(real), cudaMemcpyDeviceToHost);
    cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");

    if (!validate(h_sums, DSIZE))
        return -1;

    printf("column sums correct!\n");

    return 0;
}

