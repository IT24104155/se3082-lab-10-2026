#include <stdio.h>
#include <cuda_runtime.h>

// Macro to check CUDA runtime errors
#define CHECK(call)                                                   \
do                                                                    \
{                                                                     \
    cudaError_t error = call;                                         \
    if (error != cudaSuccess)                                         \
    {                                                                 \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n",                 \
                __FILE__, __LINE__, cudaGetErrorString(error));       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

__global__ void add(int *a, int *b, int *c)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main(void)
{
    const int N = 5;
    int a[N] = {1, 2, 3, 4, 5};
    int b[N] = {10, 20, 30, 40, 50};
    int c[N];

    int *d_a, *d_b, *d_c;
    size_t size = N * sizeof(int);

    // Allocate GPU memory and check for errors
    CHECK(cudaMalloc((void **)&d_a, size));
    CHECK(cudaMalloc((void **)&d_b, size));
    CHECK(cudaMalloc((void **)&d_c, size));

    // Copy input arrays to the GPU
    CHECK(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));

    // Launch the kernel
    add<<<1, N>>>(d_a, d_b, d_c);

    // Check for kernel launch errors and execution errors
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Copy results back to the CPU
    CHECK(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost));

    // Display results
    for (int i = 0; i < N; i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    // Release GPU memory
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));

    return 0;
}
