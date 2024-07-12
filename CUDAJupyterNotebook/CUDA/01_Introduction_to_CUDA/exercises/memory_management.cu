#include <cstdio>

__global__ void set_value (int* a) {
    FIXME;
}

int main() {
    int* a;
    int* d_a;

    // Allocate the host copy of a
    a = (int*) malloc(sizeof(int));
    // Allocate the device copy of a
    // By convention, device copies of 
    // variables are often prefixed with d_
    FIXME;

    // Set the host value of a
    *a = 1;

    // Copy the value of a to the device
    cudaMemcpy(FIXME, a, FIXME, cudaMemcpyHostToDevice);

    // Launch the kernel to set the value
    set_value<<<FIXME>>>(d_a);
    cudaDeviceSynchronize();

    // Copy the value of a back to the host
    cudaMemcpy(a, FIXME, sizeof(int), FIXME);
    
    // Check that the value of a is correct
    if (*a == 2) {
        printf("Success!\n");
    }
    else {
        printf("Failure\n");
    }
    
    // Clean up a
    free(a);
    // Clean up d_a
    FIXME;
}
