#include <stdio.h>

__global__ void hello() {

    int index = FIXME;
    if (index == FIXME) {
        printf("Hello from unique thread index: %u\n", index);
    }

}

int main(){

    hello<<<FIXME>>>();
    cudaDeviceSynchronize();

}
