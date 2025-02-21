/**
 * @file tsqr_scaling_test.c
 * @brief Test program for parallel TSQR scalability.
 *
 * This program calls the parallel TSQR function to test the scalability of different matrix dimensions.
 * It measures computation time and communication time separately, and exports the results to a CSV file.
 */

#include <parallel_TSQR.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <lapacke.h>
#include <math.h>
#include <time.h>

int main(int argc, char **argv) {
    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Require exactly 4 processes for this test
    if (size != 4) {
        if (rank == 0)
            printf("Exactly 4 processes are required!\n");
        MPI_Finalize();
        return 1;
    }
    
    // Define different matrix sizes for testing scalability
    int m_values[] = {100, 500, 1000, 5000, 10000}; // Different row counts
    int n_values[] = {10, 50, 500}; // Different column counts (narrow matrices)
    int m_local;
    double *A_global = NULL;
    double *A_local = NULL;
    double start_time, end_time;
    double compute_start, compute_end, comm_start, comm_end;
    double compute_time, comm_time;

    // Rank 0 creates the CSV file to export results
    FILE *csv_file = NULL;
    if (rank == 0) {
        csv_file = fopen("scaling_results.csv", "w");
        if (csv_file == NULL) {
            printf("Unable to create CSV file!\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        // Write CSV header
        fprintf(csv_file, "m,n,total_time,compute_time,comm_time\n");
    }
    
    // Loop over different matrix dimensions
    for (int i = 0; i < sizeof(m_values) / sizeof(m_values[0]); i++) {
        for (int j = 0; j < sizeof(n_values) / sizeof(n_values[0]); j++) {
            int m = m_values[i];
            int n = n_values[j];
            m_local = m / size;
            // Check if local matrix dimensions are valid for QR decomposition
            if (m_local < n) {
                if (rank == 0) {
                    printf("Skipping invalid matrix dimensions: m_local = %d, n = %d\n", m_local, n);
                }
                continue;
            }

            // Allocate memory for the local submatrix
            A_local = (double *)malloc(m_local * n * sizeof(double));
            if (rank == 0) {
                // Allocate and initialize the global matrix with random values in the range [0, 10)
                A_global = (double *)malloc(m * n * sizeof(double));
                for (int k = 0; k < m * n; k++) {
                    A_global[k] = ((double)rand() / RAND_MAX) * 10.0;
                }
            }
            
            // Measure communication time (MPI_Scatter)
            comm_start = MPI_Wtime();
            MPI_Scatter(A_global, m_local * n, MPI_DOUBLE,
                        A_local,   m_local * n, MPI_DOUBLE,
                        0, MPI_COMM_WORLD);
            comm_end = MPI_Wtime();

            // Measure computation time (TSQR computation)
            compute_start = MPI_Wtime();
            parallel_TSQR(A_local, m_local, n);
            compute_end = MPI_Wtime();

            // Calculate total time, computation time, and communication time
            start_time = comm_start;
            end_time = compute_end;
            compute_time = compute_end - compute_start;
            comm_time = (comm_end - comm_start);

            // Print and write results on the root process
            if (rank == 0) {
                printf("Matrix size: %dx%d, Total time: %f seconds, Compute time: %f seconds, Communication time: %f seconds\n", 
                       m, n, end_time - start_time, compute_time, comm_time);
                fprintf(csv_file, "%d,%d,%f,%f,%f\n", 
                        m, n, end_time - start_time, compute_time, comm_time);
            }

            // Free allocated memory for the local submatrix and the global matrix on the root process
            free(A_local);
            if (rank == 0) {
                free(A_global);
            }
        }
    }
    
    // Close the CSV file and inform the user that data has been exported
    if (rank == 0 && csv_file != NULL) {
        fclose(csv_file);
        printf("Timing data has been exported to scaling_results.csv\n");
    }

    MPI_Finalize();
    return 0;
}
