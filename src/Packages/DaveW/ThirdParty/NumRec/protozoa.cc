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
   for (sum=0.0,k=1;k<=mpts;k++) sum += p[k][j];\
      psum[j]=sum;}
#define SWAP(a,b) {swap=(a);(a)=(b);(b)=swap;}

void protozoa(double **p, double y[], int ndim, double ftol,
	    double *(*funk)(int), int *nfunk, int extra)
{

    int i,k,maxrow,ihi,ilo,inhi,j,mpts=2*ndim+1;
    double rtol,sum,swap,ysave,*ytry,tmp;
    Array1<double> x(mpts+1);
    Array1<double> fac(mpts+1);
    Array1<double> row(mpts+2);
    Array2<double> A(mpts+2,mpts+1);
    Array1<double> psum(mpts+1);
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

	    if (rtol < ftol) {
		SWAP(y[1],y[ilo])
		    for (i=1;i<=ndim+extra;i++) SWAP(p[1][i],p[ilo][i])
						    break;
	    }
	    if (*nfunk >= NMAX) {
		cerr << "NMAX exceeded";
		break;
	    }
	    //*nfunk += 2;
	    //    cerr << "assign the matrix\n";

	    // This parts creates the matrix of of ax^2 + bx + cy^2 + dy +... + 1 = error

	    for (i=1;i<=mpts;i++) {
		A(mpts,i)=1;
		A(mpts+1,i)= y[i]*y[i];
		for (j=1;j<=ndim;j++) {
		    A(j,i)= p[i][j];
		    A(j+ndim,i) = p[i][j]*p[i][j];
		}
	    }
	    //    cerr << "invert the matrix\n";
	
	    //do some gaussian elimination:
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
      if (abs(A(i,i)) < 0.000000001)
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

   // so you have the matrix solution.  Now to find the minimum value for the quadratic equation you take the derivative of ax^2 + bx (for each dimension x,y,z) and set it to zero so x = -b/2a
   for (i=1;i<=ndim;i++){
       if (x[i+ndim] <=  0) {  // it is convex instead of concave
	   cerr << "convexity detected on axis " << i << endl;
	   GET_PSUM
	   p[ihi][i] = psum[i]/mpts;
       }
       else
	   p[ihi][i] = -x[i]/(2*x[i+ndim]);	
       cerr << " " << p[ihi][i];
   }
	    //	    cerr << "ihi " << ihi << endl;
   y[ihi] = *((*funk)(ihi));
   cerr << " " << y[ihi] << endl;
   //  cerr << "Error (from in Amoeba)" << y[ihi] << endl;
   *nfunk += 1;
		    //		    GET_PSUM
}
}


#undef NRANSI

#undef SWAP
#undef GET_PSUM
