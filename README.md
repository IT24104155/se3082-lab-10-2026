# SE3082 — Lab 10: CUDA Performance and Error Handling

**Student ID:** IT24104155

This lab extends the CUDA activities from Lab 09 by measuring execution time, checking runtime errors, and using two- and three-dimensional thread grids.

## Activities

| Folder | Work included |
| --- | --- |
| [Exercise01](Exercise01) | Timed vector addition; timed Lab 09 block, thread, vector and matrix examples; comparison of kernel time and transfer-plus-kernel time for 100,000,000 vector elements |
| [Exercise02](Exercise02) | CUDA runtime and kernel error checking with the `CHECK()` macro |
| [Exercise03](Exercise03) | Double grayscale pixel intensity with saturation at 255, using a 2D grid on an 800 × 600 sample image |
| [Exercise04](Exercise04) | Element-wise multiplication of two 500 × 500 × 500 arrays using a 3D grid |

Exercise 3 changes pixel intensity; it does not resize image dimensions. Exercise 4 is an element-wise product, not a matrix dot product.

## Run in Colab

[Open SE3082_Lab10_IT24104155.ipynb](https://colab.research.google.com/github/IT24104155/se3082-lab-10-2026/blob/main/notebooks/SE3082_Lab10_IT24104155.ipynb)

1. Select a GPU runtime.
2. Run the GPU checks at the start of the notebook.
3. Execute the cells in order. The supplied compile commands target `sm_75`; adjust this flag for the selected GPU if needed.
4. Check program output and CPU comparison mismatch counts, and record fresh timing measurements.

The exercise headings and supplied explanatory text are preserved in the notebook. The source README contained only a title; a copy with this submission's student ID is in [docs/original_readme.md](docs/original_readme.md).

## Build locally

Requires a Linux CUDA environment (the timing examples use `gettimeofday`), a compatible NVIDIA GPU, and GNU make.

```sh
make
make run
make clean
# Optional GPU architecture override:
make CUDAFLAGS='-O2 -arch=sm_75'
```

Programs are built under `bin/`. Large-vector and 2D-matrix examples need approximately 1.2 GB each of host and device array storage. Exercise 4 needs approximately 1.5 GB each, plus runtime overhead.

## Progress and verification

- Nine supplied CUDA programs are organized into four exercise folders.
- The notebook has been renamed for IT24104155 and links to this repository.
- Saved code outputs and execution counts were cleared for fresh runs.
- Existing timing tables are explicitly labeled as reference measurements from the source author.
- CUDA compilation and GPU execution have not been verified in this environment. A successful run is not claimed here.

Kernel timing uses a host wall-clock timer around the launch and synchronization. It includes launch and synchronization overhead. The transfer-plus-kernel measurement in Part 1(c) excludes allocation, input initialization, and CPU verification.

See [SOURCES.md](SOURCES.md) for attribution.
