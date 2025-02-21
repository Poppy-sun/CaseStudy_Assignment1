/**
 * @file TSQR_Q2.c
 * @brief Communication-Avoiding TSQR Factorization Implementation in C.
 *
 * This file implements a communication-avoiding tall-skinny QR (TSQR) factorization
 * using MPI and LAPACK routines. The matrix is distributed evenly across 4 processes.
 * Each process performs a local QR factorization, and the local R factors are gathered
 * and further factorized to produce the final global Q and R factors. The final solution
 * is verified by reconstructing the original matrix and checking the orthogonality of Q.
 *
 * Assignment: Communication-Avoiding QR Factorization (CAQR)
 * Course: MAP55672 - Case Studies in HPC
 * Author: Sun Lishuang
 * Date: 21/02/2025
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <lapacke.h>
#include <math.h>
#include <string.h>
#include <parallel_TSQR.h>



/**
 * @brief Main function that executes the TSQR factorization.
 *
 * This function performs the following steps:
 *  - Initializes MPI and checks for exactly 4 processes.
 *  - The root process generates a global matrix A with random entries.
 *  - Distributes the matrix A across the processes.
 *  - Each process computes a local QR factorization using LAPACK's dgeqrf.
 *  - Extracts the local R factor and computes the explicit Q using dorgqr.
 *  - Gathers local R factors and performs a second-level QR factorization to obtain the global R and Q.
 *  - Broadcasts the global factors and computes the final local Q factors.
 *  - Reconstructs the original matrix and computes the factorization error and orthogonality error.
 *  - Displays the final Q and R matrices.
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return 0 on successful execution.
 */
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Ensure exactly 4 processes are used.
    if (size != 4) {
        if (rank == 0)
            printf("Need exactly 4 processes!\n");
        MPI_Finalize();
        return 1;
    }
    
    // Define matrix dimensions: m rows (tall) and n columns (narrow).
    int m = 16, n = 4;
    int m_local = m / 4;  // Each process will handle m_local rows.
    
    // Allocate memory for local matrices and workspace (row-major order).
    double *A_local   = malloc(m_local * n * sizeof(double));
    double *Q_local   = malloc(m_local * n * sizeof(double));
    double *R_local   = malloc(n * n * sizeof(double));
    double *tau_local = malloc(n * sizeof(double));
    if (!A_local || !Q_local || !R_local || !tau_local) {
        printf("Memory allocation failed at rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    
    // Root process generates the global matrix A (in row-major order)
    // and saves a copy for error verification.
    double *A_global = NULL;
    if (rank == 0) {
        A_global = malloc(m * n * sizeof(double));
        if (!A_global) {
            printf("Memory allocation failed for A_global\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        // Fill the global matrix A with random numbers in the range [-0.001, 0.001].
        for (int i = 0; i < m; i++) {
           for (int j = 0; j < n; j++) {
               A_global[i * n + j] = ((double)rand() / RAND_MAX) * 0.002 - 0.001;
           }
        }
        // Print the original matrix A.
        printf("Original Matrix A (%d x %d):\n", m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                printf("%8.4f ", A_global[i * n + j]);
            }
            printf("\n");
        }
    }
    
    // Scatter the global matrix A (in row-major order) to all processes.
    MPI_Scatter(A_global, m_local * n, MPI_DOUBLE,
                A_local,   m_local * n, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    // Each process performs a local QR factorization on its block A_local (m_local x n).
    int info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m_local, n, A_local, n, tau_local);
    if (info != 0) {
       printf("LAPACKE_dgeqrf failed at rank %d with info = %d\n", rank, info);
       MPI_Abort(MPI_COMM_WORLD, info);
    }
    
    // Extract the local R factor from A_local (stored in the upper triangular part).
    // For row i and column j, if i <= j then it's an element of R, otherwise set to 0.
    for (int i = 0; i < m_local; i++) {
       for (int j = 0; j < n; j++) {
           if (i <= j)
              R_local[i * n + j] = A_local[i * n + j];
           else
              R_local[i * n + j] = 0.0;
       }
    }
    
    // Use LAPACKE_dorgqr to generate the explicit local Q factor from A_local.
    memcpy(Q_local, A_local, m_local * n * sizeof(double));
    info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, m_local, n, n, Q_local, n, tau_local);
    if (info != 0) {
       printf("LAPACKE_dorgqr failed at rank %d with info = %d\n", rank, info);
       MPI_Abort(MPI_COMM_WORLD, info);
    }
    
    // Gather all processes' R_local matrices to the root process to form the stacked matrix R_stack.
    // R_stack has dimensions (4*n) x n (row-major order).
    double *R_stack = NULL;
    if (rank == 0) {
       R_stack = malloc(4 * n * n * sizeof(double));
       if (!R_stack) {
          printf("Memory allocation failed for R_stack\n");
          MPI_Abort(MPI_COMM_WORLD, -1);
       }
    }
    MPI_Gather(R_local, n * n, MPI_DOUBLE,
               R_stack,  n * n, MPI_DOUBLE,
               0, MPI_COMM_WORLD);
    
    // Root process performs a global QR factorization on the stacked matrix R_stack.
    double *tau_global = NULL;
    double *R_final    = NULL; // Final upper triangular factor R.
    double *Q_global   = NULL; // Global Q factor (dimensions: (4*n) x n, row-major).
    if (rank == 0) {
       int rows_global = 4 * n;
       tau_global = malloc(n * sizeof(double));
       if (!tau_global) {
          printf("Memory allocation failed for tau_global\n");
          MPI_Abort(MPI_COMM_WORLD, -1);
       }
       info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, rows_global, n, R_stack, n, tau_global);
       if (info != 0) {
          printf("Global LAPACKE_dgeqrf failed with info = %d\n", info);
          MPI_Abort(MPI_COMM_WORLD, info);
       }
       // Extract the final R: take the upper triangular part of the first n rows of R_stack.
       R_final = malloc(n * n * sizeof(double));
       if (!R_final) {
          printf("Memory allocation failed for R_final\n");
          MPI_Abort(MPI_COMM_WORLD, -1);
       }
       for (int i = 0; i < n; i++) {
          for (int j = 0; j < n; j++) {
             if (i <= j)
                R_final[i * n + j] = R_stack[i * n + j];
             else
                R_final[i * n + j] = 0.0;
          }
       }
       // Use LAPACKE_dorgqr to generate the global Q factor from R_stack.
       Q_global = malloc(rows_global * n * sizeof(double));
       if (!Q_global) {
          printf("Memory allocation failed for Q_global\n");
          MPI_Abort(MPI_COMM_WORLD, -1);
       }
       memcpy(Q_global, R_stack, rows_global * n * sizeof(double));
       info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, rows_global, n, n, Q_global, n, tau_global);
       if (info != 0) {
          printf("Global LAPACKE_dorgqr failed with info = %d\n", info);
          MPI_Abort(MPI_COMM_WORLD, info);
       }
    }
    
    // Broadcast the final R factor to all processes.
    double *R_final_bcast = malloc(n * n * sizeof(double));
    if (rank == 0)
       memcpy(R_final_bcast, R_final, n * n * sizeof(double));
    MPI_Bcast(R_final_bcast, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Broadcast the global Q factor to all processes.
    // Q_global has dimensions (4*n) x n (row-major order), where each block is of size n*n.
    double *Q_global_bcast = malloc(4 * n * n * sizeof(double));
    if (rank == 0)
       memcpy(Q_global_bcast, Q_global, 4 * n * n * sizeof(double));
    MPI_Bcast(Q_global_bcast, 4 * n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Each process multiplies its local Q with the corresponding block from the global Q
    // to obtain the final local Q factor:
    // Q_final_local = Q_local (m_local x n) * Q_block (n x n).
    double *Q_final_local = malloc(m_local * n * sizeof(double));
    if (!Q_final_local) {
       printf("Memory allocation failed for Q_final_local at rank %d\n", rank);
       MPI_Abort(MPI_COMM_WORLD, -1);
    }
    // The corresponding global Q block: for process rank r, take the block at offset r in Q_global_bcast.
    int block_offset = rank * n * n;
    matrix_multiply(Q_local, Q_global_bcast + block_offset, Q_final_local, m_local, n, n);
    
    // Gather the final local Q factors from each process to form the global Q_final (m x n, row-major).
    double *Q_final = NULL;
    if (rank == 0) {
       Q_final = malloc(m * n * sizeof(double));
       if (!Q_final) {
          printf("Memory allocation failed for Q_final\n");
          MPI_Abort(MPI_COMM_WORLD, -1);
       }
    }
    MPI_Gather(Q_final_local, m_local * n, MPI_DOUBLE,
               Q_final, m_local * n, MPI_DOUBLE,
               0, MPI_COMM_WORLD);
    
    // The root process verifies the TSQR factorization result:
    // Reconstruct A and compute the error, then check the orthogonality of Q,
    // and finally display Q and R.
    if (rank == 0) {
       double *A_reconstructed = malloc(m * n * sizeof(double));
       if (!A_reconstructed) {
          printf("Memory allocation failed for A_reconstructed\n");
          MPI_Abort(MPI_COMM_WORLD, -1);
       }
       // Reconstruct A: A_reconstructed = Q_final * R_final_bcast.
       matrix_multiply(Q_final, R_final_bcast, A_reconstructed, m, n, n);
       double error = 0.0;
       for (int i = 0; i < m; i++)
          for (int j = 0; j < n; j++) {
              double diff = A_global[i * n + j] - A_reconstructed[i * n + j];
              error += diff * diff;
          }
       error = sqrt(error);
       printf("\nTSQR Factorization Error ||A - Q*R||_F = %e\n", error);
       
       // Compute the orthogonality error: ||Q^T * Q - I||_F.
       double *QtQ = malloc(n * n * sizeof(double));
       compute_orthogonality(Q_final, m, n, QtQ);
       double ortho_error = 0.0;
       for (int i = 0; i < n; i++)
          for (int j = 0; j < n; j++) {
             double expected = (i == j) ? 1.0 : 0.0;
             double diff = QtQ[i * n + j] - expected;
             ortho_error += diff * diff;
          }
       ortho_error = sqrt(ortho_error);
       printf("Orthogonality Error ||Q^T*Q - I||_F = %e\n\n", ortho_error);
       
       // Display the final Q matrix (dimensions: m x n).
       printf("Final Q Matrix (%d x %d):\n", m, n);
       for (int i = 0; i < m; i++) {
           for (int j = 0; j < n; j++) {
               printf("%8.4f ", Q_final[i * n + j]);
           }
           printf("\n");
       }
       
       // Display the final R matrix (dimensions: n x n).
       printf("\nFinal R Matrix (%d x %d):\n", n, n);
       for (int i = 0; i < n; i++) {
           for (int j = 0; j < n; j++) {
               printf("%8.4f ", R_final_bcast[i * n + j]);
           }
           printf("\n");
       }
       
       free(A_reconstructed);
       free(QtQ);
    }
    
    // Free all allocated memory.
    free(A_local);
    free(Q_local);
    free(R_local);
    free(tau_local);
    free(Q_final_local);
    free(R_final_bcast);
    free(Q_global_bcast);
    if (rank == 0) {
       free(A_global);
       free(R_stack);
       free(tau_global);
       free(R_final);
       free(Q_final);
       free(Q_global);
    }
    
    MPI_Finalize();
    return 0;
}
