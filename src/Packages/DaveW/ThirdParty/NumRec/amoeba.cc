#include <math.h>
#define NRANSI
#include "nrutil.h"
//#define NMAX 100000
#include <iostream>

using namespace std;

#define GET_PSUM \
for (j=1;j<=ndim;j++) {\
   for (sum=0.0,i=1;i<=mpts;i++) sum += p[i][j];\
      psum[j]=sum;}
#define SWAP(a,b) {swap=(a);(a)=(b);(b)=swap;}

void amoeba(double **p, double y[], int ndim, double ftol,
	    double *(*funk)(double []), int *nfunk, int extra)
{
    double *amotry(double **p, double y[], double psum[], int ndim,
		   double *(*funk)(double []), int ihi, double fac, int extra);
    int i,ihi,ilo,inhi,j,mpts=ndim+1;
    double rtol,sum,swap,ysave,*ytry,*psum;
    
    int NMAX=*nfunk;
    
    
    psum=dvector(1,ndim);
    *nfunk=0;
    GET_PSUM
	for (;;) {
	    ilo=1;
	    ihi = y[1]>y[2] ? (inhi=2,1) : (inhi=1,2);
	    for (i=1;i<=mpts;i++) {
		if (y[i] <= y[ilo]) ilo=i;
		if (y[i] > y[ihi]) {
		    inhi=ihi;
		    ihi=i;
		} else if (y[i] > y[inhi] && i != ihi) inhi=i;
	    }
	    rtol=2.0*fabs(y[ihi]-y[ilo])/(fabs(y[ihi])+fabs(y[ilo]));
	    cerr << "\n\nAMOEBA   rtol="<<rtol<<"  ftol="<<ftol<<"\n";
	    cerr << "   y="<<y[1]<<" "<<y[2]<<" "<<y[3]<<" "<<y[4]<<"\n\n\n";
#if 0
	    cerr << "   ilo="<<ilo<<" ihi="<<ihi<<" inhi="<<inhi<<"\n";
	    cerr << "   ndim="<<ndim<<" extra="<<extra<<" mpts="<<mpts<<"\n";
	    cerr << "Here are the dipoles (from in amoeba):\n";
	    for (i=1; i<=ndim+2; i++) {
		cerr << "   "<<i<<" = ";
		for (int jj=1; jj<=ndim+3; jj++) {
		    cerr << p[i][jj] << " ";
		}
		cerr << "\n";
	    }
#endif
	    if (rtol < ftol) {
		SWAP(y[1],y[ilo])
		    for (i=1;i<=ndim+extra;i++) SWAP(p[1][i],p[ilo][i])
						    break;
	    }
#if 0
	    cerr << "Here are the dipoles (from in amoeba2):\n";
	    for (i=1; i<=ndim+2; i++) {
		cerr << "   "<<i<<" = ";
		for (int jj=1; jj<=ndim+3; jj++) {
		    cerr << p[i][jj] << " ";
		}
		cerr << "\n";
	    }
#endif
	    if (*nfunk >= NMAX) {
		cerr << "NMAX exceeded";
		break;
	    }
	    *nfunk += 2;
	    
	    ytry=amotry(p,y,psum,ndim,funk,ihi,-1.0,extra);
	    if (ytry[0] <= y[ilo])
		ytry=amotry(p,y,psum,ndim,funk,ihi,2.0,extra);
	    else if (ytry[0] >= y[inhi]) {
		ysave=y[ihi];
		ytry=amotry(p,y,psum,ndim,funk,ihi,0.5,extra);
		if (ytry[0] >= ysave) {
		    for (i=1;i<=mpts;i++) {
			if (i != ilo) {
			    for (j=1;j<=ndim;j++)
				p[i][j]=p[mpts+1][j]=0.5*(p[i][j]+p[ilo][j]);
			    ytry=(*funk)(p[i]);
			    y[i]=ytry[0];
			    for (;j<=ndim+extra;j++) {
				p[i][j]=ytry[j-ndim];
			    }
			}
		    }
		    *nfunk += ndim;
		    GET_PSUM
			}
	    } else --(*nfunk);
	}
    free_dvector(psum,1,ndim);
}
#undef SWAP
#undef GET_PSUM
#undef NMAX
#undef NRANSI
/* (C) Copr. 1986-92 Numerical Recipes Software Z1.)$. */
