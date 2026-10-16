#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define N 500
#define THREADS_PER_BLOCK 8

// Macro to check CUDA errors
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

// Return the current time in seconds
double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);

    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

// CUDA kernel for element-wise 3D matrix multiplication
__global__ void matrixMul3D(const int *a, const int *b, int *c, int n)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < n && y < n && z < n)
    {
        int index = z * n * n + y * n + x;
        c[index] = a[index] * b[index];
    }
}

int main(void)
{
    size_t totalElements = (size_t)N * N * N;
    size_t size = totalElements * sizeof(int);

    int *a, *b, *c;
    int *d_a, *d_b, *d_c;

    // Allocate host memory
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    if (a == NULL || b == NULL || c == NULL)
    {
        printf("Host memory allocation failed.\n");
        free(a);
        free(b);
        free(c);
        return 1;
    }

    // Initialise input matrices with small values
    for (size_t i = 0; i < totalElements; i++)
    {
        a[i] = (int)(i % 10);
        b[i] = (int)((i % 10) + 1);
    }

    // Allocate GPU memory
    CHECK(cudaMalloc((void **)&d_a, size));
    CHECK(cudaMalloc((void **)&d_b, size));
    CHECK(cudaMalloc((void **)&d_c, size));

    // Copy input matrices to GPU
    CHECK(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));

    // Configure a 3D grid and 3D blocks
    dim3 threadsPerBlock(THREADS_PER_BLOCK,
                         THREADS_PER_BLOCK,
                         THREADS_PER_BLOCK);

    dim3 numBlocks(
        (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
        (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
        (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK
    );

    // Measure kernel execution time
    double start = cpuSecond();

    matrixMul3D<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, N);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    double kernelTime = cpuSecond() - start;

    // Copy result back to host
    CHECK(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost));

    // Verify every element
    size_t errors = 0;

    for (size_t i = 0; i < totalElements; i++)
    {
        if (c[i] != a[i] * b[i])
        {
            errors++;
        }
    }

    printf("Matrix dimensions: %d x %d x %d\n", N, N, N);
    printf("Blocks: %d x %d x %d\n",
           numBlocks.x, numBlocks.y, numBlocks.z);
    printf("Threads per block: %d x %d x %d\n",
           threadsPerBlock.x,
           threadsPerBlock.y,
           threadsPerBlock.z);

    printf("Verification: %zu mismatches out of %zu elements\n",
           errors, totalElements);
    printf("Kernel execution time: %.7f seconds\n", kernelTime);

    // Display the first 10 results
    printf("First 10 results:\n");

    for (int i = 0; i < 10; i++)
    {
        printf("%d ", c[i]);
    }

    printf("\n");

    // Free memory
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));

    free(a);
    free(b);
    free(c);

    return 0;
}
