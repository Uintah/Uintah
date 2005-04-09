#include <stdio.h>
#include <stdlib.h>

double *makeVector(int nx) {
    return (double*) malloc ((nx+1)*sizeof(double));
}

void freeVector(double *v) {
    free(v);
}

void vectorSub(int n, double *a, double *b, double *c) {
    for (int i=1; i<=n; i++) c[n]=a[n]-b[n];
}

void printVectorToFile(char *name, double *v, int n) {
    FILE *f = fopen(name, "wt");
    fprintf(f, "%d\n", n);
    for (int i=1; i<=n; i++) {
	fprintf(f, "%lf\n", v[i]);
    }
    fclose(f);
}

void printVector(double *b, int u) {
    for (int i=1; i<=u; i++) {
	printf("%lf\n", b[i]);
    }
    printf("\n");
}

void printVectorF(FILE **f, double *v, int n) {
    for (int i=1; i<=n; i++) fprintf(*f, "%lf ", v[i]);
    fprintf(*f, "\n");
}

void printIntVector(int *b, int u) {
    for (int i=1; i<=u; i++) {
	printf("%d\n", b[i]);
    }
    printf("\n");
}

void remapVector(double *b, int *map, int u) {
    double *v=makeVector(u);
    for (int i=1; i<=u; i++) {
	v[i]=b[map[i]];
    }
    for (i=1; i<=u; i++) {
	b[i]=v[i];
    }
    free(v);
}

void copyVector(double *a, double *b, int n) {
    for (int i=1; i<=n; i++) {
	b[i]=a[i];
    }
}

double normV(double *a, double *b, int n) {
    double total=0;
    for (int i=1; i<=n; i++) {
	total+=(b[n]-a[n])*(b[n]-a[n]);
    }
    return total;
}

void fPrintVector(FILE *f, double *b, int n) {
    for (int i=1; i<=n; i++) {
	fprintf(f, "%lf\n", b[i]);
    }
}

void fPrintIndexedVector(FILE *f, double *b, int n) {
    for (int i=1; i<=n; i++) {
	fprintf(f, "%d %lf\n", i, b[i]);
    }
}

