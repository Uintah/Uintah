#ifndef AMOTRY_H
#define AMOTRY_H

double* amotry(double **p, double  y[], double  psum[], int ndim,
	       double *(*funk)(double []), int ihi, double fac, int extra);

#endif
