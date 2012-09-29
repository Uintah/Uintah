/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/*
 *  sci_lapack.cc
 * 
 *  Written by:
 *   Author: Andrew Shafer
 *   Department of Computer Science
 *   University of Utah
 *   Date: Oct 21, 2003
 *
*/

#include <sci_defs/lapack_defs.h>

#include <cmath>
#include <Core/Math/sci_lapack.h>
#include <Core/Util/Assert.h>

//Functions to switch between Fortran and C style matrices

namespace SCIRun {

double *ctof(double **c, int rows, int cols)
{
  double *f;
  int i, j;

  f = new double[rows*cols];

  for (i=0; i<rows; i++){ 
    for(j=0; j<cols; j++){
      f[i+(j*rows)] = c[i][j];
    }
  }
  return(f);
}


void ftoc(double *f, double **c, int rows, int cols)
{
  int i, j;
  for (i=0; i<rows; i++) for (j=0; j<cols; j++) c[i][j] = f[i+j*cols];
}

void sort_eigens(double *Er, double *Ei, int N, double **Evecs=0)
{
  double temp, *E2;
  int i, j, k;
  
  E2 = new double[N];
  for (i=0; i<N; i++) E2[i] = Er[i]*Er[i]+Ei[i]*Ei[i];
  
  for (j=0; j<N; j++) for (i=0; i<N-1; i++)
    if (fabs(E2[i])<fabs(E2[i+1])) {
      temp = E2[i]; E2[i] = E2[i+1]; E2[i+1] = temp;
      temp = Er[i]; Er[i] = Er[i+1]; Er[i+1] = temp;
      temp = Ei[i]; Ei[i] = Ei[i+1]; Ei[i+1] = temp;

      if (Evecs) {
	for (k=0; k<N; k++) {
	  temp = Evecs[k][i];
	  Evecs[k][i] = Evecs[k][i+1];
	  Evecs[k][i+1] = temp;
	}
      }
    }

  delete [] E2;
}

#if defined(HAVE_LAPACK)

extern "C" {
#if defined(REDSTORM)
#  define DGETRF dgetrf
#else
#  define DGETRF dgetrf_
#endif
  int DGETRF(int *m, int *n, double *a, int *lda, int *ipiv, int *info);
  int dgetri_(int *m, double *a, int *lda, int *ipiv, 
	      double *work, int *lwork, int *info);
  int dgesvd_(char *jobu, char *jobvt, int *m, int *n, double *a, int *lda, 
	      double *S, double *u, int *ldu, double *vt, int *ldvt, 
	      double *work, int *lwork, int *info);
  int dgeev_(char *jobvl, char *jobvr, int *n, double *a, int *lda,
	     double *Er, double *Ei, double *vl, int *ldvl, double *vr, 
	     int *ldvr, double *work, int *lwork, int *info);
}

bool lapackinvert(double *A, int n)
{
  // A is the matrix
  // n is the order of A (A is n*n)
  // P an int array to store the permutations

  int lda, lwork, info;  //The leading dimension of the matrix a.

  int *P = new int[n];  //int array that stores permutations.
 
  lwork = n*64;
  double * work = new double[lwork];
 
  lda = n;
  lwork = n;
  
  DGETRF(&n, &n, A, &lda, P, &info);  
  dgetri_(&n, A, &lda, P, work, &lwork, &info); 

  delete [] work;
  delete [] P;

  if(info == 0)
    return true;
  else
    return false;
}


void lapacksvd(double **A, int m, int n, double *S, double **U, double **VT)
{
  char jobu, jobvt;
  int lda, ldu, ldvt, lwork, info;
  double *a, *u, *vt, *work;

  int minmn, maxmn;

  jobu = 'A'; 
  /* Specifies options for computing U.
		 A: all M columns of U are returned in array U;
		 S: the first min(m,n) columns of U (the left
		    singular vectors) are returned in the array U;
		 O: the first min(m,n) columns of U (the left
		    singular vectors) are overwritten on the array A;
		 N: no columns of U (no left singular vectors) are
		    computed. */

  jobvt = 'A'; 
  /* Specifies options for computing VT.
		  A: all N rows of V**T are returned in the array
		     VT;
		  S: the first min(m,n) rows of V**T (the right
		     singular vectors) are returned in the array VT;
		  O: the first min(m,n) rows of V**T (the right
		     singular vectors) are overwritten on the array A;
		  N: no rows of V**T (no right singular vectors) are
		     computed. */

  lda = m; // The leading dimension of the matrix a.
  a = ctof(A, m, n); // Convert the matrix A from double pointer
			                 // C form to single pointer Fortran form.


  /* Since A is not a square matrix, we have to make some decisions
     based on which dimension is shorter. */

  if (m>=n) { minmn = n; maxmn = m; } else { minmn = m; maxmn = n; }

  ldu = m; // Left singular vector matrix
  u = new double[ldu*m];

  ldvt = n; // Right singular vector matrix
  vt = new double[ldvt*n];

  lwork = 5*maxmn; // Set up the work array, larger than needed.
  work = new double[lwork];

  dgesvd_(&jobu, &jobvt, &m, &n, a, &lda, S, u,
	  &ldu, vt, &ldvt, work, &lwork, &info);

  ftoc(u, U, ldu, m);
  ftoc(vt, VT, ldvt, n);

  delete [] a;
  delete [] u;
  delete [] vt;
  delete [] work;
}


void lapackeigen(double **H, int n, double *Er, double *Ei, double **Evecs)
{
  char jobvl, jobvr;
  int lda,  ldvl, ldvr, lwork, info;
  double *a, *vl=0, *vr=0, *work;
  
  jobvl = 'N'; /* V/N to calculate/not calculate the left eigenvectors
		  of the matrix H.*/

  if (Evecs) 
    jobvr = 'V'; // As above, but for the right eigenvectors.
  else
    jobvr = 'N';

  lda = n; // The leading dimension of the matrix a.
  a = ctof(H, n, lda); /* Convert the matrix H from double pointer
				C form to single pointer Fortran form. */

  /* Whether we want them or not, we need to define the matrices
     for the eigenvectors, and give their leading dimensions.
     We also create a vector for work space. */

  ldvl = n;
  ldvr = n;
  if (Evecs)
    vr = new double[n*n];
  lwork = 4*n;

  work = new double[lwork];
  
  dgeev_(&jobvl, &jobvr, &n, a, &lda, Er, Ei, vl,
	 &ldvl, vr, &ldvr, work, &lwork, &info);

  if (Evecs) {
    ftoc(vr, Evecs, n, ldvr);
    sort_eigens(Er, Ei, n, Evecs); /* Sort the results by eigenvalue in
					 decreasing magnitude. */
  } else {
    sort_eigens(Er, Ei, n);
  }

  delete [] a;
  if (Evecs) delete [] vr;
  delete [] work;
}

#else

// stubs for LAPACK wrappers
bool lapackinvert(double *A, int n)
{
  ASSERTFAIL("Build was not configured with LAPACK");
  return false;
}

void lapacksvd(double **A, int m, int n, double *S, double **U, double **VT)
{
  ASSERTFAIL("Build was not configured with LAPACK");
}

void lapackeigen(double **H, int n, double *Er, double *Ei, double **Evecs)
{
  ASSERTFAIL("Build was not configured with LAPACK");
}

#endif

} // End namespace SCIRun
