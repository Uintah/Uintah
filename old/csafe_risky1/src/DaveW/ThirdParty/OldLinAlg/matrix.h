#ifndef MATRIX_DAVE_H
#define MATRIX_DAVE_H
#include <stdio.h>

double norm(double **a, int nx, int ny);
double normI(double **a, int nx, int ny);
void matMatMult(int x, int y, int z, double **a, double **b, double **c);
void spMatMatMult(int x, int y, int z, double **a, double **b, double **c);
void matVecMult(int x, int y, double **a, double *b, double *c);
void svdTruncateW(double *w, int n, double truncMin);
void svdCompose(double **u, int ny, int nx, double *w, double **v, 
		double **uTr, double **ui);
int isIdentity(int x, double **a);
void printMatrix(double **a, int u, int v);
void printMatrix2(double **a, int u, int v);
void printMatrixToFile(char *name, double **a, int u, int v);
void printSparseMatrix(double **a, int u);
void printSparseMatrix2(char *name, double **a, int u);
FILE *readSizeOfMatrix(int *ns, int *nv, int *nc, FILE *f, char *prob);
void readMatrixVV(double **Ass, double **Asv, double **Asc, double **Avs, 
		double **Avv, double **Avc, double **Azz, int ns, int nv, int nc, FILE *f);
void matSub(int nx, int ny, double **a, double **b, double **c);
double percentFull(double **a, int nx, int ny);
double **makeMatrix(int nx, int ny);
void freeMatrix(double **m);
void getColumn(double **a, int u, int idx, double *c);
void putColumn(double **a, int u, int idx, double *c);
void copyMatrix(double **a, double **b, int nx, int ny);
void sortem2(int *idx, double *data, int n);

#endif
