/****************************************************************
 * Protozoa module (improved simplex search)"for the SCIRun              *
 *                                                              *
 *  Written by:                                                 *
 *   Kris Zyp                                              *
 *   Department of Computer Science                             *
 *   University of Utah                                         *
 *   Sept 2000                                                   *
 *                                                              *
 *  Copyright (C) 1999 SCI Group                                *
 *                                                              *
 *                                                              *
 ****************************************************************/
#include <math.h>
#include <iostream>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/Array2.h>
#define NRANSI
using namespace std;
using namespace SCICore::Containers;
#define GET_PSUM \
for (j=1;j<=ndim;j++) {\
   for (sum=0.0,i=1;i<=ndim;i++) sum += p[i][j];\
      psum[j]=sum;}
#define SWAP(a,b) {swap=(a);(a)=(b);(b)=swap;}

void protozoa(double **p, double y[], int ndim, double ftol,
	    double *(*funk)(double []), int *nfunk, int extra)
{
    double *amotry(double **p, double y[], double psum[], int ndim,
		   double *(*funk)(double []), int ihi, double fac, int extra);
    int i,k,maxrow,ihi,ilo,inhi,j,mpts=2*ndim+1;
    double rtol,sum,swap,ysave,*ytry,tmp;
    Array1<double> x(mpts+1);
    Array1<double> fac(mpts+1);
    Array1<double> row(mpts+2);
    Array2<double> A(mpts+2,mpts+1);
    Array1<double> psum(mpts);
    Array1<int> indx(mpts+1);
    
    int NMAX=*nfunk;
    *nfunk=0;
    cerr << "calculate sum\n";
    //  GET_PSUM
	for (;;) {
	    //    cerr << "find hi\n";
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
	    for (i=1; i<=ndim+4; i++) {
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
	    //    cerr << "assign the matrix\n";
	    for (i=1;i<=mpts;i++) {
		A(mpts,i)=1;
		A(mpts+1,i)= y[i];
		for (j=1;j<=ndim;j++) {
		    A(j,i)= p[i][j];
		    A(j+ndim,i) = p[i][j]*p[i][j];
		}
	    }
	    //    cerr << "invert the matrix\n";
	
	    // a little gaussian elimination:
   for (i=1;i<=mpts;i++) {

      /* Find the row with the largest first value */
      maxrow = i;
      for (j=i+1;j<=mpts;j++) {
         if (abs(A(i,j)) > abs(A(i,maxrow)))
            maxrow = j;
      }

      /* Swap the maxrow and ith row */
      for (k=i;k<=mpts+1;k++) {
         tmp = A(k,i);
         A(k,i) = A(k,maxrow);
         A(k,maxrow) = tmp;
      }

      /* Singular matrix? */
      if (abs(A(i,i)) < 0.0000001)
	  cerr << "urghh... This matrix isn't looking very healthy" << endl;

      /* Eliminate the ith element of the jth row */
      for (j=i+1;j<=mpts;j++) {
         for (k=mpts+1;k>=i;k--) {
            A(k,j) -= A(k,i) * A(i,j) / A(i,i);
         }
      }
   }

   /* Do the back substitution */
   for (j=mpts;j>=1;j--) {
      tmp = 0;
      for (k=j+1;k<=mpts;k++)
         tmp += A(k,j) * x[k];
      x[j] = (A(mpts+1,j) - tmp) / A(j,j);
   }
   for (i=1;i<=ndim;i++){
       p[ihi][i] = -x[i]/(2*x[i+ndim]);	
		//		cerr << " " << p[ihi][i];
   }
	    //	    cerr << "ihi " << ihi << endl;
   y[ihi] = *((*funk)(p[ihi]));
   //  cerr << "Error (from in Amoeba)" << y[ihi] << endl;
   *nfunk += ndim;
		    //		    GET_PSUM
}
}


#undef NRANSI

#undef SWAP
#undef GET_PSUM
