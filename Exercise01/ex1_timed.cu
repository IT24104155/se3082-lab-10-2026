#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

#define N 512

__global__ void add(int *a, int *b, int *c) {
    // Each block runs only one addition
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

void random_ints(int *x, int size) {
    for (int i = 0; i < size; i++)
        x[i] = rand() % 100;
}

// Track CPU wall-clock time in seconds
double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

int main(void) {
    int *a, *b, *c;             // Host copies of a, b, c
    int *d_a, *d_b, *d_c;       // Device copies of a, b, c
    int size = N * sizeof(int);
    double dStart, dElaps;

    // Allocate space for device copies
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Allocate host copies and initialize input values
    a = (int *)malloc(size);
    random_ints(a, N);

    b = (int *)malloc(size);
    random_ints(b, N);

    c = (int *)malloc(size);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Start timer immediately before the kernel launch
    dStart = cpuSecond();

    // Launch add() with N blocks and one thread per block
    add<<<N, 1>>>(d_a, d_b, d_c);

    // Wait for the GPU to finish before stopping the timer
    cudaDeviceSynchronize();

    dElaps = cpuSecond() - dStart;

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Print the first five results
    for (int r = 0; r < 5; r++)
        printf("%d + %d = %d\n", a[r], b[r], c[r]);

    // Check the GPU results against the CPU
    int errors = 0;

    for (int r = 0; r < N; r++) {
        if (c[r] != a[r] + b[r])
            errors++;
    }

    printf("Verification: %d mismatches out of %d\n", errors, N);
    printf("Time taken - %.7f\n", dElaps);

    // Cleanup
    free(a);
    free(b);
    free(c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
