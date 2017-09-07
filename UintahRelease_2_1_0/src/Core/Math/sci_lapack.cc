/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
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

#include <sci_defs/cuda_defs.h>
#include <sci_defs/lapack_defs.h>
#include <sci_defs/magma_defs.h>
#include <sci_defs/uintah_defs.h> // For FIX_NAME

#include <Core/Math/sci_lapack.h>
#include <Core/Util/Assert.h>

#include <cmath>

#include <stdio.h>

// Functions to switch between Fortran and C style matrices

namespace Uintah {

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

#  define DGETRF FIX_NAME(dgetrf)
#  define DGETRI FIX_NAME(dgetri)
#  define DGESVD FIX_NAME(dgesvd)
#  define DGEEV  FIX_NAME(dgeev)

extern "C" {
  int DGETRF( int *m, int *n, double *a, int *lda, int *ipiv, int *info );
  int DGETRI( int *m, double *a, int *lda, int *ipiv, 
              double *work, int *lwork, int *info );
  int DGESVD( char *jobu, char *jobvt, int *m, int *n, double *a, int *lda, 
              double *S, double *u, int *ldu, double *vt, int *ldvt, 
              double *work, int *lwork, int *info );
  int DGEEV( char *jobvl, char *jobvr, int *n, double *a, int *lda,
             double *Er, double *Ei, double *vl, int *ldvl, double *vr, 
             int *ldvr, double *work, int *lwork, int *info );
}

bool
lapackinvert(double *A, int n)
{
  
#if defined(HAVE_MAGMA)

  // d_A    is the device matrix
  // h_R    is the host result (pinned buffer)
  // n      is the order of A (A is n*n)
  // ipiv   an int array to store the permutations
  // lda,   lwork, info The leading dimension of the host matrix A.
  // ldda,  ldwork, info The leading dimension of the device matrix A.

  // CUDA and CUBLAS initialization
  MAGMA_CUDA_INIT();

  // things we'll be working with
  double* h_R = 0;
  double* d_A = 0;
  double* dwork = 0;
  magma_int_t n2, lda, ldda;
  magma_int_t info;
  double* work;
  magma_int_t *ipiv;
  magma_int_t lwork, ldwork;

  // query for Magma workspace size and pad for device memory
  lwork = int(n * 64);
  ldwork = n * magma_get_dgetri_nb(n);
  n2 = n * n;
  ldda = ((n + 31) / 32) * 32;
  lda = n;
  lwork = n;

  // allocate host memory, pinned host memory and device memory
  ipiv = new int[n];
  work = new double[n*64];
  MAGMA_HOSTALLOC(h_R, double, n2);
  MAGMA_DEVALLOC(d_A, double, n * ldda);
  MAGMA_DEVALLOC(dwork, int, ldwork);

  // make the MAGMA calls and get results back host-side
  magma_dsetmatrix(n, n, A, lda, d_A, ldda);
  magma_dgetrf_gpu(n, n, d_A, ldda, ipiv, &info);
  magma_dgetmatrix(n, n, d_A, ldda, A, lda);
  magma_dgetri_gpu(n, d_A, ldda, ipiv, dwork, ldwork, &info);
  magma_dgetmatrix(n, n, d_A, ldda, h_R, lda);

  // swap pointers with pinned memory host version of A and A itself
  A = h_R;

  // clean up CPU memory allocations
  delete[] work;
  delete[] ipiv;

  // clean up device memory allocations
  MAGMA_HOSTFREE(h_R);
  MAGMA_DEVFREE(d_A);
  MAGMA_DEVFREE(dwork);

  // shutdown CUDA and CUBLAS
  MAGMA_CUDA_FINALIZE();

  if (info == 0) {
    return true;
  } else {
    return false;
  }

#else

  // A is the matrix
  // n is the order of A (A is n*n)
  // P an int array to store the permutations

  int lda, lwork, info;  //The leading dimension of the matrix A.

  int* P = new int[n];  //int array that stores permutations.

  lwork = n*64;
  double* work = new double[lwork];

  lda = n;
  lwork = n;

  DGETRF(&n, &n, A, &lda, P, &info);
  DGETRI(&n, A, &lda, P, work, &lwork, &info);

  delete [] work;
  delete [] P;

  if(info == 0)
    return true;
  else
    return false;

#endif

}

// This is for Vulcan@LLNL:
#undef HAVE_DGESVD

void
lapacksvd( double **A, int m, int n, double *S, double **U, double **VT )
{
#if defined(HAVE_MAGMA)

//  magma_dgesvd(char jobu,
//               char jobvt,
//               magma_int_t m,
//               magma_int_t n,
//               double *A,
//               magma_int_t lda,
//               double *s,
//               double *U,
//               magma_int_t ldu,
//               double *VT,
//               magma_int_t ldvt,
//               double *work,
//               magma_int_t lwork,
//               magma_int_t *info )

//  magma_dgesvd(jobu, jobvt, m, n, a, lda, S, u,
//               ldu, vt, ldvt, work, lwork, &info);

  printf( "!!!WARNING!!! (magma) lapacksvd (in sci_lapack.cc) called, but not implemented!!!\n" );

#else

# if defined( HAVE_DGESVD )
  char jobu, jobvt;
  int lda, ldu, ldvt, lwork, info;
  double *a, *u, *vt, *work;

  int maxmn;

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

  if (m >= n) { maxmn = m; } else { maxmn = n; }

  ldu = m; // Left singular vector matrix
  u = new double[ldu*m];

  ldvt = n; // Right singular vector matrix
  vt = new double[ldvt*n];

  lwork = 5*maxmn; // Set up the work array, larger than needed.
  work = new double[lwork];

  DGESVD( &jobu, &jobvt, &m, &n, a, &lda, S, u,
          &ldu, vt, &ldvt, work, &lwork, &info );

  ftoc(u, U, ldu, m);
  ftoc(vt, VT, ldvt, n);

  delete [] a;
  delete [] u;
  delete [] vt;
  delete [] work;
# else
  printf( "!!!WARNING!!! lapacksvd (in sci_lapack.cc) called, but not implemented!!!\n" );
# endif
#endif

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
  
#if defined(HAVE_MAGMA)

//  magma_dgeev(char jobvl,
//              char jobvr,
//              magma_int_t n,
//              double *a,
//              magma_int_t lda,
//              double *WR,
//              double *WI,
//              double *vl,
//              magma_int_t ldvl,
//              double *vr,
//              magma_int_t ldvr,
//              double *work,
//              magma_int_t lwork,
//              magma_int_t *info)

  magma_dgeev(jobvl, jobvr, n, a, lda, Er, Ei, vl,
              ldvl, vr, ldvr, work, lwork, &info);

#else

  DGEEV( &jobvl, &jobvr, &n, a, &lda, Er, Ei, vl,
         &ldvl, vr, &ldvr, work, &lwork, &info );

#endif

  if (Evecs) {
    ftoc(vr, Evecs, n, ldvr);
    sort_eigens(Er, Ei, n, Evecs); // Sort the results by eigenvalue in decreasing magnitude.
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

} // End namespace Uintah
