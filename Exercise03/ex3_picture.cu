#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define WIDTH 800
#define HEIGHT 600
#define THREADS_PER_BLOCK 16

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

// CUDA kernel to process picture pixels
__global__ void processPicture(unsigned char *picture,
                               int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int index = y * width + x;
        int value = picture[index] * 2;

        if (value > 255)
        {
            value = 255;
        }

        picture[index] = (unsigned char)value;
    }
}

int main(void)
{
    size_t size = (size_t)WIDTH * HEIGHT * sizeof(unsigned char);

    unsigned char *picture;
    unsigned char *d_picture;

    // Allocate host memory
    picture = (unsigned char *)malloc(size);

    if (picture == NULL)
    {
        printf("Host memory allocation failed.\n");
        return 1;
    }

    // Initialise the picture with sample pixel values
    for (int i = 0; i < WIDTH * HEIGHT; i++)
    {
        picture[i] = i % 256;
    }

    // Allocate GPU memory
    CHECK(cudaMalloc((void **)&d_picture, size));

    // Copy picture from CPU to GPU
    CHECK(cudaMemcpy(d_picture, picture, size,
                     cudaMemcpyHostToDevice));

    // Configure the 2D grid and blocks
    dim3 threadsPerBlock(16, 16);

    dim3 numBlocks(
        (WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    // Execute the CUDA kernel
    processPicture<<<numBlocks, threadsPerBlock>>>(
        d_picture, WIDTH, HEIGHT);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Copy processed picture back to CPU
    CHECK(cudaMemcpy(picture, d_picture, size,
                     cudaMemcpyDeviceToHost));

    // Display the first 10 processed pixel values
    printf("First 10 processed pixel values:\n");

    for (int i = 0; i < 10; i++)
    {
        printf("%u ", (unsigned int)picture[i]);
    }

    printf("\n");

    // Verify the processed pixels
    int errors = 0;

    for (int i = 0; i < WIDTH * HEIGHT; i++)
    {
        unsigned int original = i % 256;
        unsigned int expected = original * 2;

        if (expected > 255)
        {
            expected = 255;
        }

        if (picture[i] != expected)
        {
            errors++;
        }
    }

    printf("Verification: %d mismatches out of %d pixels\n",
           errors, WIDTH * HEIGHT);

    printf("Image dimensions: %d x %d\n", WIDTH, HEIGHT);
    printf("Blocks: %d x %d\n", numBlocks.x, numBlocks.y);
    printf("Threads per block: %d x %d\n",
           threadsPerBlock.x, threadsPerBlock.y);

    // Free memory
    CHECK(cudaFree(d_picture));
    free(picture);

    return 0;
}
