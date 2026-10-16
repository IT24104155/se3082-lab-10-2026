#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

#define ROWS 10000
#define COLS 10000

__global__ void mulElem(int *a, int *b, int *c,
                        int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        size_t idx = (size_t)row * cols + col;
        c[idx] = a[idx] * b[idx];
    }
}

void random_ints(int *x, size_t size) {
    for (size_t i = 0; i < size; i++)
        x[i] = rand() % 100;
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

int main(void) {
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    size_t count = (size_t)ROWS * COLS;
    size_t size = count * sizeof(int);
    double dStart, dElaps;

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    a = (int *)malloc(size);
    random_ints(a, count);

    b = (int *)malloc(size);
    random_ints(b, count);

    c = (int *)malloc(size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 16);

    dim3 numBlocks(
        (COLS + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (ROWS + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    printf("Grid: %d x %d blocks of %d x %d threads\n",
           numBlocks.x, numBlocks.y,
           threadsPerBlock.x, threadsPerBlock.y);

    // Time the kernel only
    dStart = cpuSecond();
    mulElem<<<numBlocks, threadsPerBlock>>>(
        d_a, d_b, d_c, ROWS, COLS
    );
    cudaDeviceSynchronize();
    dElaps = cpuSecond() - dStart;

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Print the last 1000 elements of the final row
    printf("Last 1000 matrix elements:\n");

    for (int col = COLS - 1000; col < COLS; col++) {
        size_t idx = (size_t)(ROWS - 1) * COLS + col;

        printf("C[%d][%d] = %d x %d = %d\n",
               ROWS - 1, col, a[idx], b[idx], c[idx]);
    }

    int errors = 0;
    for (size_t i = 0; i < count; i++) {
        if (c[i] != a[i] * b[i])
            errors++;
    }

    printf("Verification: %d mismatches out of %zu\n",
           errors, count);
    printf("Time taken - %.7f\n", dElaps);

    free(a);
    free(b);
    free(c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
