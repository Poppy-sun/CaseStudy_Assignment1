/**
 * @file parallel_TSQR.c
 * @brief Implementation of a parallel TSQR (Tall Skinny QR) algorithm.
 *
 * This file provides an implementation of the parallel TSQR algorithm using MPI for distributed
 * computing and LAPACK for performing local QR decompositions. The algorithm involves:
 *  - Local QR decomposition on submatrices.
 *  - Extraction of the upper triangular R matrices.
 *  - Gathering local R matrices from all processes and performing a global QR decomposition.
 *  - Broadcasting the final R and Q matrices to all processes.
 *
 * Detailed functionalities are implemented in the functions below.
 */

#include <parallel_TSQR.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <lapacke.h>
#include <math.h>
#include <string.h>

/**
 * @brief Perform matrix multiplication C = A * B for row-major matrices.
 *
 * This function multiplies two matrices A and B and stores the result in matrix C.
 *
 * @param A Pointer to the first input matrix A of size m x n in row-major order.
 * @param B Pointer to the second input matrix B of size n x p in row-major order.
 * @param C Pointer to the output matrix C of size m x p in row-major order.
 * @param m Number of rows in matrix A and C.
 * @param n Number of columns in matrix A and rows in matrix B.
 * @param p Number of columns in matrix B and C.
 */
void matrix_multiply(double *A, double *B, double *C, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
       for (int j = 0; j < p; j++) {
           double sum = 0.0;
           for (int k = 0; k < n; k++) {
               sum += A[i * n + k] * B[k * p + j];
           }
           C[i * p + j] = sum;
       }
    }
}

/**
 * @brief Compute the Frobenius norm of a row-major matrix A.
 *
 * This function calculates the Frobenius norm of a given matrix A.
 *
 * @param A Pointer to the input matrix A of size m x n in row-major order.
 * @param m Number of rows in matrix A.
 * @param n Number of columns in matrix A.
 * @return The Frobenius norm of the matrix A.
 */
double frobenius_norm(double *A, int m, int n) {
    double sum = 0.0;
    for (int i = 0; i < m; i++)
       for (int j = 0; j < n; j++) {
           double val = A[i * n + j];
           sum += val * val;
       }
    return sqrt(sum);
}

/**
 * @brief Compute Q^T * Q to check the orthogonality of matrix Q.
 *
 * This function multiplies the transpose of Q by Q and stores the result in QtQ.
 * It is used to verify that Q is orthogonal.
 *
 * @param Q Pointer to the input matrix Q of size m x n in row-major order.
 * @param m Number of rows in matrix Q.
 * @param n Number of columns in matrix Q.
 * @param QtQ Pointer to the output matrix of size n x n (row-major order) that stores Q^T * Q.
 */
void compute_orthogonality(double *Q, int m, int n, double *QtQ) {
    for (int i = 0; i < n; i++) {
       for (int j = 0; j < n; j++) {
           double sum = 0.0;
           for (int k = 0; k < m; k++) {
               sum += Q[k * n + i] * Q[k * n + j];
           }
           QtQ[i * n + j] = sum;
       }
    }
}

/**
 * @brief Parallel TSQR wrapper function.
 *
 * This function performs a parallel TSQR algorithm. It first computes a local QR
 * decomposition on each process's submatrix, then gathers the local R matrices, and
 * performs a global QR decomposition on the gathered matrix. The final R and Q matrices
 * are then broadcast to all processes.
 *
 * @param A_local Local submatrix of A on each process, stored in row-major order.
 * @param m_local Number of rows of the local submatrix.
 * @param n Number of columns of the submatrix.
 */
void parallel_TSQR(double *A_local, int m_local, int n) 
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Memory allocation for local Q, local R, and local tau vector
    double *Q_local   = malloc(m_local * n * sizeof(double));
    double *R_local   = malloc(n * n * sizeof(double));
    double *tau_local = malloc(n * sizeof(double));
    if (!Q_local) {
        printf("Memory allocation failed for Q_local at rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    // Local QR decomposition using LAPACK
    int info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m_local, n, A_local, n, tau_local);
    if (info != 0) {
       printf("LAPACKE_dgeqrf failed at rank %d with info = %d\n", rank, info);
       MPI_Abort(MPI_COMM_WORLD, info);
    }

    // Extract R matrix (upper triangular part)
    for (int i = 0; i < n; i++) {
       for (int j = 0; j < n; j++) {
           if (i <= j)
              R_local[i * n + j] = A_local[i * n + j];
           else
              R_local[i * n + j] = 0.0;
       }
    }

    // Generate explicit local Q
    memcpy(Q_local, A_local, m_local * n * sizeof(double));
    if (m_local < n || n <= 0) {
        printf("Error: Invalid matrix dimensions. m_local = %d, n = %d\n", m_local, n);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, m_local, n, n, Q_local, n, tau_local);
    if (info != 0) {
       printf("LAPACKE_dorgqr failed at rank %d with info = %d\n", rank, info);
       MPI_Abort(MPI_COMM_WORLD, info);
    }

    // Global QR decomposition variables
    double *R_stack = NULL;
    double *R_final = NULL;
    double *Q_global = NULL;
    double *tau_global = NULL;

    // Allocate memory for global R and Q matrices on all processes to avoid MPI_Bcast segmentation fault
    R_final = malloc(n * n * sizeof(double));
    Q_global = malloc(size * n * n * sizeof(double));
    if (!R_final || !Q_global) {
        printf("Memory allocation failed at rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    if (rank == 0) {
        R_stack = malloc(size * n * n * sizeof(double));
        tau_global = malloc(n * sizeof(double));
        if (!R_stack || !tau_global) {
            printf("Memory allocation failed at rank %d\n", rank);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    // Gather R matrices from all processes
    MPI_Gather(R_local, n * n, MPI_DOUBLE,
               R_stack, n * n, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Perform global QR decomposition on the root process
        int rows_global = size * n;
        info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, rows_global, n, R_stack, n, tau_global);
        if (info != 0) {
            printf("Global LAPACKE_dgeqrf failed with info = %d\n", info);
            MPI_Abort(MPI_COMM_WORLD, info);
        }

        // Extract final R matrix (upper triangular part)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i <= j)
                    R_final[i * n + j] = R_stack[i * n + j];
                else
                    R_final[i * n + j] = 0.0;
            }
        }

        // Generate explicit global Q
        memcpy(Q_global, R_stack, rows_global * n * sizeof(double));
        info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, rows_global, n, n, Q_global, n, tau_global);
        if (info != 0) {
             printf("LAPACKE_dorgqr failed at rank %d with info = %d\n", rank, info);
            MPI_Abort(MPI_COMM_WORLD, info);
        }
    }

    // Broadcast global R and Q matrices to all processes
    MPI_Bcast(R_final, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(Q_global, size * n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    printf("Calling DORGQR with m_local = %d, n = %d\n", m_local, n);

    // Free allocated memory
    free(Q_local);
    free(R_local);
    free(tau_local);
    if (rank == 0) {
        free(R_stack);
        free(R_final);
        free(Q_global);
        free(tau_global);
    }
}
