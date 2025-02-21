# Communication-Avoiding QR (CAQR) Factorization Assignment



## Project Overview

### Q1. TSQR Factorization in Matlab
- Open `Matlab/Q1_TSQR Factorization.mlx` in Matlab.
- Run the script to see the TSQR factorization process and the final Q and R matrices.
- This is a basic implementation of the TSQR (Tall Skinny QR) factorization of a tall, narrow matrix without using parallel programming.
- The matrix is divided into four blocks, and the QR factorization is applied sequentially using Matlab’s built-in functions.

### Q2. TSQR Implementation in C
- The main implementation is in `C_Language/TSQR_Q2.c`.
- This code performs a **communication-avoiding tall-skinny QR (TSQR) factorization** using **MPI** for parallel processing and **LAPACK** routines  with Householder reflections for local QR factorizations.
- The matrix rows are evenly distributed across four compute nodes.
- The code can be run with MPI (`mpirun -np 4 ./TSQR_Q2`), and the provided Makefile simplifies the build process.
- **Key Steps**:
  1. Initializes MPI and ensures exactly **4 processes** are used.
  2. The **root process** generates a random **global matrix A** and distributes it evenly to all processes.
  3. Each process computes a **local QR factorization** using `LAPACK’s dgeqrf`, extracting the **R factor** and generating the explicit **Q factor** with `dorgqr`.
  4. The local **R factors** are gathered by the root process, which performs a **second-level QR factorization** to compute the **final global Q and R factors**.
  5. The code verifies the correctness by **reconstructing the original matrix** and checking the **orthogonality** of the **Q matrix**.
  6. Finally, the **Q and R matrices** are displayed along with the **factorization error** and **orthogonality error**.

The implementation is designed to handle **tall and narrow matrices** efficiently, distributing the computation evenly and minimizing communication overhead.



### Q3. Parallel TSQR and Scaling Analysis

- Implemented in `C_Language/parallel_TSQR.c` and `C_Language/tsqr_scaling_test.c`.

- Demonstrates the parallel implementation of the TSQR algorithm and evaluates performance with random test matrices of varying dimensions.

- Results are stored in `scaling_results.csv`, and performance plots can be generated using `Matlab/scaling_plots.mlx`.

- Test results are pre-saved, allowing quick analysis without re-running the C tests.

  ### Performance Analysis

  #### Communication Time

  - When the matrix column count `n` is small (e.g., `n = 10` ), communication time remains relatively low and stable as the matrix row count `m` increases.

  - When `n = 50`, the communication time plot shows an unusual dip and rise (a "bend" in the curve).

  - Possible reasons for this anomaly include:

    1. **Data Communication vs. Computation Balance**: Small matrix sizes may lead to high communication overhead, while larger matrices improve efficiency.
    2. **MPI Initialization Overhead**: At smaller data sizes, MPI setup costs might dominate, leading to non-linear behavior.
    3. **Load Balancing Issues**: Certain matrix dimensions may lead to uneven data distribution, impacting communication performance.
    4. **Caching and Memory Behavior**: Data size relative to cache capacity might affect communication times.
    5. **Random Variance or Measurement Noise**: External factors or system noise could also contribute to this behavior.

  - For larger `n` (e.g., `n = 500`), communication time increases significantly, indicating a potential communication bottleneck for wide matrices.

![image](https://github.com/user-attachments/assets/7c55ee17-a97c-4a12-bf81-e05e1146f693)

  #### Compute Time

  - Compute time is minimal for small `n` values, demonstrating efficient parallel computation.

  - However, when `n = 500`, compute time increases sharply, suggesting that the computational load is heavily influenced by the matrix column count.

    ![image-20250221221410983](/Users/sunlishuang/Library/Application Support/typora-user-images/image-20250221221410983.png)

  #### Total Execution Time

  - The total execution time reflects the combined impact of communication and computation times.
  - For small `n`, communication overhead is the primary factor, while for large `n`, computation time dominates.
  - This suggests potential optimizations by reducing communication costs for narrow matrices or improving computational efficiency for wide matrices.
  - ![image-20250221221619734](/Users/sunlishuang/Library/Application Support/typora-user-images/image-20250221221619734.png)



## Makefile Usage （for Q2 & Q3)

- **Compile the Code:**
```bash
make
```
- **Run TSQR Implementation (C Part):**
```bash
make run_Q2
```
- **Run Parallel TSQR and Scaling Tests:**
```bash
make run_Q3
```
- **Clean Up the Directory:**
```bash
make clean
```

## Dependencies
- **Matlab** (for the first and third parts)
- **C Compiler** with support for MPI (e.g., mpicc)
- **LAPACK** library for QR factorization
- **MPI** (e.g., mpirun for running on multiple processors)


