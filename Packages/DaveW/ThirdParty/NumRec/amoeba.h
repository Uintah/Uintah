#ifndef AMOEBA_H
#define AMOEBA_H

void amoeba(double **p, double y[], int ndim, double ftol,
	    double *(*funk)(double []), int *nfunk, int extra);

#endif
