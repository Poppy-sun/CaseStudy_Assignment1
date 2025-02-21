#ifndef PARALLEL_TSQR_H
#define PARALLEL_TSQR_H

void parallel_TSQR(double *A_local, int m_local, int n);
void matrix_multiply(double *A, double *B, double *C, int m, int n, int p);
double frobenius_norm(double *A, int m, int n);
void compute_orthogonality(double *Q, int m, int n, double *QtQ);

#endif /* PARALLEL_TSQR_H */
