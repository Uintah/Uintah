#ifndef VECTOR_DAVE_H
#define VECTOR_DAVE_H
#include <stdio.h>

double *makeVector(int nx);
void freeVector(double *v);
void vectorSub(int n, double *a, double *b, double *c);
void printVector(double *b, int u);
void printVectorF(FILE **f, double *v, int n);
void printVectorToFile(char *name, double *v, int n);
void printIntVector(int *b, int u);
void remapVector(double *b, int *map, int u);
void copyVector(double *a, double *b, int n);
double normV(double *a, double *b, int n);
void fPrintVector(FILE *f, double *b, int n);
void fPrintIndexedVector(FILE *f, double *b, int n);

#endif
