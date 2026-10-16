#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <cuda_runtime.h>

#define N 100000000
#define THREADS_PER_BLOCK 512

__global__ void multiply(int *a, int *b, int *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        c[i] = a[i] * b[i];
    }
}

void random_ints(int *x, int size)
{
    for (int i = 0; i < size; i++)
    {
        x[i] = rand() % 100;
    }
}

double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);

    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

int main(void)
{
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;

    size_t size = (size_t)N * sizeof(int);
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    double start, kernelTime, totalTime;

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

    // Initialise input arrays
    random_ints(a, N);
    random_ints(b, N);

    // Allocate GPU memory
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Start total timer: transfers + kernel + result transfer
    start = cpuSecond();

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Measure only GPU kernel execution
    double kernelStart = cpuSecond();

    multiply<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    kernelTime = cpuSecond() - kernelStart;

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    totalTime = cpuSecond() - start;

    // Display a few results
    for (int i = 0; i < 5; i++)
    {
        printf("%d * %d = %d\n", a[i], b[i], c[i]);
    }

    // Verify the result
    int errors = 0;

    for (int i = 0; i < N; i++)
    {
        if (c[i] != a[i] * b[i])
        {
            errors++;
        }
    }

    printf("Verification: %d mismatches out of %d\n", errors, N);
    printf("Kernel execution time: %.7f seconds\n", kernelTime);
    printf("Total execution time:  %.7f seconds\n", totalTime);

    // Free host memory
    free(a);
    free(b);
    free(c);

    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
