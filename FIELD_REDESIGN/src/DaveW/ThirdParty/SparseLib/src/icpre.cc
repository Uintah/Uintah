/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/*             ********   ***                                 SparseLib++    */
/*          *******  **  ***       ***      ***               v. 1.5c        */
/*           *****      ***     ******** ********                            */
/*            *****    ***     ******** ********              R. Pozo        */
/*       **  *******  ***   **   ***      ***                 K. Remington   */
/*        ********   ********                                 A. Lumsdaine   */
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/*                                                                           */
/*                                                                           */
/*                     SparseLib++ : Sparse Matrix Library                   */
/*                                                                           */
/*               National Institute of Standards and Technology              */
/*                        University of Notre Dame                           */
/*              Authors: R. Pozo, K. Remington, A. Lumsdaine                 */
/*                                                                           */
/*                                 NOTICE                                    */
/*                                                                           */
/* Permission to use, copy, modify, and distribute this software and         */
/* its documentation for any purpose and without fee is hereby granted       */
/* provided that the above notice appear in all copies and supporting        */
/* documentation.                                                            */
/*                                                                           */
/* Neither the Institutions (National Institute of Standards and Technology, */
/* University of Notre Dame) nor the Authors make any representations about  */
/* the suitability of this software for any purpose.  This software is       */
/* provided ``as is'' without expressed or implied warranty.                 */
/*                                                                           */
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

#include <stdlib.h>
#include <math.h>
#include "icpre.h"
#include "qsort.h"

#include "spblas.h"

static void
ICFactor(const MV_Vector<int> &ia, const MV_Vector<int> &ja, MV_Vector<double> &sa);
static void
ICSolve(const MV_Vector<int> &ia, const MV_Vector<int> &ja, 
    const MV_Vector<double> &sa, MV_Vector<double> &dest);


ICPreconditioner::ICPreconditioner(const CompCol_MatDouble &A)
  : nz_(0), pntr_(A.dim(1)+1), indx_(0), val_(0) 
{
  dim_[0] = A.dim(0);
  dim_[1] = A.dim(1);

  int i, j, k;

  for (k = 0; k < dim_[1]; k++)
    for (j = A.col_ptr(k); j < A.col_ptr(k+1); j++)
      if (A.row_ind(j) >= k)
    nz_++;

  val_.newsize(nz_);
  indx_.newsize(nz_);

  // Copy just triangular part
  pntr_(0) = 0;
  for (k = 0; k < dim_[1]; k++) {
    pntr_(k+1) = pntr_(k);
    for (j = A.col_ptr(k); j < A.col_ptr(k+1); j++) {
      if (A.row_ind(j) >= k) {
    i = pntr_(k+1)++;
    val_(i) = A.val(j);
    indx_(i) = A.row_ind(j);
      }
    }
  }

  for (i = 0; i < dim_[1]; i++)
    QSort(indx_, val_, pntr_[i], pntr_[i+1] - pntr_[i]);

  for (i = 0; i < dim_[1]; i++)
    if (indx_[pntr_(i)] != i) {
      cerr << "IC Preconditioner: diagonal not found!" << i << endl;
      exit(1);
    }
  
  ICFactor(pntr_, indx_, val_);
}


ICPreconditioner::ICPreconditioner(const CompRow_MatDouble &A)
  : nz_(0), pntr_(A.dim(0)+1), indx_(0), val_(0) 
{
  dim_[0] = A.dim(0);
  dim_[1] = A.dim(1);

  int i, j, k;

  for (k = 0; k < dim_[0]; k++)
    for (j = A.row_ptr(k); j < A.row_ptr(k+1); j++)
      if (A.col_ind(j) >= k)
    nz_++;

  val_.newsize(nz_);
  indx_.newsize(nz_);

  // Copy just triangular part (including diagonal)
  pntr_(0) = 0;
  for (k = 0; k < dim_[0]; k++) {
    pntr_(k+1) = pntr_(k);
    for (j = A.row_ptr(k); j < A.row_ptr(k+1); j++) {
      if (A.col_ind(j) >= k) {
    i = pntr_(k+1)++;
    val_(i) = A.val(j);
    indx_(i) = A.col_ind(j);
      }
    }
  }

  for (i = 0; i < dim_[0]; i++)
    QSort(indx_, val_, pntr_[i], pntr_[i+1] - pntr_[i]);

  for (i = 0; i < dim_[0]; i++)
    if (indx_[pntr_(i)] != i) {
      cerr << "IC Preconditioner: diagonal not found!" << i << endl;
      exit(1);
    }
  
  ICFactor(pntr_, indx_, val_);
}


MV_Vector<double>
ICPreconditioner::solve(const MV_Vector<double> &x) const
{
  MV_Vector<double> y(x);

  ICSolve(pntr_, indx_, val_, y);

  return y;
}


MV_Vector<double>
ICPreconditioner::trans_solve(const MV_Vector<double> &x) const
{
  MV_Vector<double> y(x);

  ICSolve(pntr_, indx_, val_, y);

  return y;
}


static void
ICFactor(const MV_Vector<int> &pntr, const MV_Vector<int> &indx, 
     MV_Vector<double> &val)
{
  int d, g, h, i, j, k, n = pntr.size() - 1;
  double z;

  for (k = 0; k < n - 1; k++) {
    d = pntr[k];
    z = val[d] = sqrt(val[d]);

    for (i = d + 1; i < pntr[k+1]; i++)
      val[i] /= z;

    for (i = d + 1; i < pntr[k+1]; i++) {
      z = val[i];
      h = indx[i];
      g = i;

      for (j = pntr[h] ; j < pntr[h+1]; j++)
    for ( ; g < pntr[k+1] && indx[g+1] <= indx[j]; g++)
      if (indx[g] == indx[j])
        val[j] -= z * val[g];
    }
  }
  d = pntr[n-1];
  val[d] = sqrt(val[d]);
}


static void
ICSolve(const MV_Vector<int> &pntr, const MV_Vector<int> &indx, 
    const MV_Vector<double> &val, MV_Vector<double> &dest)
{
  int M = dest.size();
  MV_Vector<double> work(M);

  int descra[9];

  descra[0] = 0;

  // lower diag
  descra[1] = 1;
  descra[2] = 0;

  F77NAME(dcscsm) (0, M, 1, 1, NULL, 1.0,
           descra, &val(0), &indx(0), &pntr(0),
           &dest(0), M, 0.0, &dest(1), M,
           &work(0), M);

  // lower diag transpose
  F77NAME(dcscsm) (1, M, 1, 1, NULL, 1.0,
           descra, &val(0), &indx(0), &pntr(0),
           &dest(0), M, 0.0, &dest(1), M,
           &work(0), M);
}
