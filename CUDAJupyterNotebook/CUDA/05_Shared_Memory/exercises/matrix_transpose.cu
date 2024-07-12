#include <stdio.h>
#include <assert.h>

#define N 4096

#define TILE_WIDTH 32
#define BLOCK_SIZE TILE_SIZE * TILE_SIZE

// Our 2D matrix will be row-major, so the actual
// offset into memory should be continuous in the
// column index and strided in the row index.
#define IDX(ROW, COL) ROW * N + COL

#define NUM_REPS 100

void checkCudaErrors(cudaError_t result)
{
    if (result != cudaSuccess) {
        printf("CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
}

// Verify that the transposed matrix has the results we expect
void check_results(const float* B)
{
    bool passed = true;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (B[IDX(i, j)] != IDX(j, i)) {
                passed = false;
            }
        }
    }

    if (passed) {
        printf("Success!\n");
    } else {
        printf("Incorrect result\n");
        exit(1);
    }
}

// Naive matrix transpose from global memory
__global__ void gmemTranspose(float* B, const float* A)
{
    // In CUDA, threads in the x dimension are contiguous 
    // and the y-dimension is strided (by the number of 
    // threads in the x-dimension). So we want the x dimension
    // to correspond to contiguous entries in a row, and therefore
    // x corresponds to the column index, while y will correspond
    // to the row index.
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Write this code so the read is a coalesced access 
    // from global memory and the write is uncoalesced.
    // (This choice is arbitrary and we could have made
    //  the other choice. Feel free to experiment.)
    B[FIXME] = A[FIXME];
}

// Matrix transpose using shared memory
__global__ void smemTranspose(float* B, const float* A)
{
    // Declare a 32x32 tile. The total size of the 2D tile
    // is equal to the number of threads in a block. This is
    // an intentional choice made to make the exercise simple.
    __shared__ float tile[TILE_WIDTH][TILE_WIDTH];

    // As before, x is the column dimension and y is the
    // row dimension.
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Read the data into our shared memory tile.
    // As before, this read should be coalesced from 
    // global memory, and we want to maintain the same
    // indexing in the shared memory tile (the tile should
    // have the same orientation as the part of global 
    // memory it is copying from).
    tile[FIXME][FIXME] = A[IDX(row, col)];

    // Synchronize to ensure all threads in the block
    // have had an opportunity to read in their data.
    __syncthreads();

    // Locate the part of the output matrix that our
    // transposed tile should reside in. Note that we
    // need to reverse the block indexing (so that the
    // location of the tile in the matrix is transposed) 
    // but we do NOT reverse the thread indexing with
    // respect to global memory (so that we are still 
    // accessing contiguous locations in global memory).
    // The transpose within the tile must be done using
    // our indexing into the shared memory tile.
    col = FIXME;
    row = FIXME;

    // Write data from the tile back to global memory.
    // This write is coalesced. Note that the indexing
    // is the same as the indexing into A.
    B[IDX(row, col)] = tile[FIXME][FIXME];
}

int main()
{
    // Define a matrix size of width N rows by N columns
    std::size_t sz = N * N * sizeof(float);

    // Define a two dimensional grid and block layout
    // Assumes N is evenly divisible by TILE_WIDTH
    dim3 dimGrid(N / TILE_WIDTH, N / TILE_WIDTH, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    // Initialize data on the host
    // A is input, B is output (transpose of A)
    float* h_A = (float*)malloc(sz);
    float* h_B = (float*)malloc(sz);

    // A and B should be thought of as row-major
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            h_A[IDX(i, j)] = IDX(i, j);
        }
    }

    // Copy host data to the device
    float* d_A;
    float* d_B;

    checkCudaErrors(cudaMalloc(&d_A, sz));
    checkCudaErrors(cudaMalloc(&d_B, sz));

    checkCudaErrors(cudaMemcpy(d_A, h_A, sz, cudaMemcpyHostToDevice));



    // First do the matrix transpose naively, entirely in global memory

    printf("Beginning global memory transpose\n");

    checkCudaErrors(cudaMemset(d_B, 0, sz));

    // Run the kernel NUM_REPS times to get good timing
    for (int i = 0; i < NUM_REPS; ++i) {
        gmemTranspose<<<dimGrid, dimBlock>>>(d_B, d_A);
    }

    checkCudaErrors(cudaDeviceSynchronize());

    // Copy data back to the host
    checkCudaErrors(cudaMemcpy(h_B, d_B, sz, cudaMemcpyDeviceToHost));

    check_results(h_B);
    


    // Now do the matrix transpose using shared memory

    printf("Beginning shared memory transpose\n");

    checkCudaErrors(cudaMemset(d_B, 0, sz));

    // Run the kernel NUM_REPS times to get good timing
    for (int i = 0; i < NUM_REPS; ++i) {
        smemTranspose<<<dimGrid, dimBlock>>>(d_B, d_A);
    }

    checkCudaErrors(cudaDeviceSynchronize());

    // Copy data back to the host
    checkCudaErrors(cudaMemcpy(h_B, d_B, sz, cudaMemcpyDeviceToHost));

    check_results(h_B);



    // Clean up
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    free(h_A);
    free(h_B);
}