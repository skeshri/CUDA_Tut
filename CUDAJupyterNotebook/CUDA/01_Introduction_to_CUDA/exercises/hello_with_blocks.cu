#include <stdio.h>

__global__ void hello() {

    printf("Hello from block: %u\n", FIXME);

}

int main() {

    hello<<<FIXME>>>();
    cudaDeviceSynchronize();

}
