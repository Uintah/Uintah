#ifndef PROTOZOA_H
#define PROTOZOA_H

void protozoa(double **p, double y[], int ndim, double ftol,
	    double *(*funk)(double []), int *nfunk, int extra);

#endif
