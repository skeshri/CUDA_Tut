#include <stdio.h>

#define N (32*1048576)

int main()
{
    double *h_A;
    double* h_sum;
    h_A = new double[N];
    h_sum = new double;

    // initialize data in host memory
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
    }
    *h_sum = 0.0f;

    // sum the array on the host
    for (int i = 0; i < N; i++) {
        *h_sum += h_A[i];
    }

    // check the host results
    if (*h_sum != (double) N) {
        printf("host reduction incorrect!\n");
        return -1;
    }
    printf("host reduction correct!\n");

    free(h_A);
    free(h_sum);

    return 0;
}
