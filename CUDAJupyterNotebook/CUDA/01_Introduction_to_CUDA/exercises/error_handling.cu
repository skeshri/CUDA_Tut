#include <cstdio>

__global__ void kernel (int* a) {
    a[-1] = 1;
}

int main() {
    int* a;
    cudaMalloc(&a, -sizeof(int));

    kernel<<<1, -1>>>(a);

    cudaDeviceSynchronize();

    free(a);
}
