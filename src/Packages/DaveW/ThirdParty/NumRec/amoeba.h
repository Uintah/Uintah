#ifndef AMOEBA_H
#define AMOEBA_H

void amoeba(double **p, double y[], int ndim, double ftol,
	    double *(*funk)(int), int *nfunk, int extra, int *stop);

#endif
