#define NRANSI
#include "nrutil.h"

double* amotry(double **p, double  y[], double  psum[], int ndim,
	       double *(*funk)(int), int ihi, double fac, int extra)
{
    int j;
    double  fac1,fac2,*ytry,*ptry;
    
#if 0
    cerr << "Here are the dipoles (from in amotry0):\n";
    for (int i=1; i<=ndim+2; i++) {
	cerr << "   "<<i<<" = ";
	for (int jj=1; jj<=ndim+3; jj++) {
	    cerr << p[i][jj] << " ";
	}
	cerr << "\n";
    }

/*  ptry=dvector(1,ndim);  */

#endif

    ptry=p[ndim+2];
//    cerr << "ndim+2="<<ndim+2<<" ptry[1]="<<ptry[1]<<"\n";
    fac1=(1.0-fac)/ndim;
    fac2=fac1-fac;
    for (j=1;j<=ndim;j++) ptry[j]=psum[j]*fac1-p[ihi][j]*fac2;

#if 0
    cerr << "Here are the dipoles (from in amotry1):\n";
    for (i=1; i<=ndim+2; i++) {
	cerr << "   "<<i<<" = ";
	for (int jj=1; jj<=ndim+3; jj++) {
	    cerr << p[i][jj] << " ";
	}
	cerr << "\n";
    }
#endif

    ytry=(*funk)(ndim+2);

#if 0
    cerr << "Here are the dipoles (from in amotry2):\n";
    for (i=1; i<=ndim+2; i++) {
	cerr << "   "<<i<<" = ";
	for (int jj=1; jj<=ndim+3; jj++) {
	    cerr << p[i][jj] << " ";
	}
	cerr << "\n";
    }
#endif

    if (ytry[0] < y[ihi]) {
	y[ihi]=ytry[0];
	for (j=1;j<=ndim;j++) {
	    psum[j] += ptry[j]-p[ihi][j];
	    p[ihi][j]=ptry[j];
	}
	for (;j<=ndim+extra;j++) {
	    p[ihi][j]=ytry[j-ndim];
	}
    }
/*  free_dvector(ptry,1,ndim);   */
    return ytry;

#if 0
    cerr << "Here are the dipoles (from in amotry3):\n";
    for (i=1; i<=ndim+2; i++) {
	cerr << "   "<<i<<" = ";
	for (int jj=1; jj<=ndim+3; jj++) {
	    cerr << p[i][jj] << " ";
	}
	cerr << "\n";
    }
#endif
}
#undef NRANSI
/* (C) Copr. 1986-92 Numerical Recipes Software Z1.)$. */
