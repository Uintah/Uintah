/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 *  MatrixOperations.cc: Matrix Operations
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   August 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/MatrixOperations.h>


namespace SCIRun {


MatrixHandle
operator+(MatrixHandle A, MatrixHandle B)
{
  ASSERT(A.get_rep());
  ASSERT(B.get_rep());

  ASSERTEQ(A->ncols(), B->ncols());
  ASSERTEQ(A->nrows(), B->nrows());

  if (A->as_column() && B->as_column())
  {
    ColumnMatrix *ac = A->as_column();
    ColumnMatrix *bc = B->as_column();
    ColumnMatrix *cc = scinew ColumnMatrix(ac->nrows());
    Add(*cc, *ac, *bc);
    return cc;
  }
  else if (A->is_sparse() && B->is_sparse())
  {
    SparseRowMatrix *as = A->as_sparse();
    SparseRowMatrix *bs = B->as_sparse();
    SparseRowMatrix *cs = AddSparse(*as, *bs);
    return cs;
  }
  else
  {
    DenseMatrix *ad = A->dense();
    DenseMatrix *bd = B->dense();
    DenseMatrix *cd = scinew DenseMatrix(ad->nrows(), bd->ncols());
    Add(*cd, *ad, *bd);
    if (!(A->is_dense())) { delete ad; }
    if (!(B->is_dense())) { delete bd; }
    return cd;
  }
}



MatrixHandle
operator-(MatrixHandle A, MatrixHandle B)
{
  ASSERT(A.get_rep());
  ASSERT(B.get_rep());

  ASSERTEQ(A->ncols(), B->ncols());
  ASSERTEQ(A->nrows(), B->nrows());

  if (A->as_column() && B->as_column())
  {
    ColumnMatrix *ac = A->as_column();
    ColumnMatrix *bc = B->as_column();
    ColumnMatrix *cc = scinew ColumnMatrix(ac->nrows());
    Sub(*cc, *ac, *bc);
    return cc;
  }
  else if (A->is_sparse() && B->is_sparse())
  {
    SparseRowMatrix *as = A->as_sparse();
    SparseRowMatrix *bs = B->as_sparse();
    SparseRowMatrix *cs = SubSparse(*as, *bs);
    return cs;
  }
  else
  {
    DenseMatrix *ad = A->dense();
    DenseMatrix *bd = B->dense();
    DenseMatrix *cd = scinew DenseMatrix(ad->nrows(), bd->ncols());
    Sub(*cd, *ad, *bd);
    if (!(A->is_dense())) { delete ad; }
    if (!(B->is_dense())) { delete bd; }
    return cd;
  }
}


MatrixHandle
operator*(MatrixHandle A, MatrixHandle B)
{
  ASSERT(A.get_rep());
  ASSERT(B.get_rep());

  ASSERTEQ(A->ncols(), B->nrows());

  DenseMatrix *ad = A->dense();
  DenseMatrix *bd = B->dense();
  DenseMatrix *cd = scinew DenseMatrix(ad->nrows(), bd->ncols());
  Mult(*cd, *ad, *bd);
  if (!(A->is_dense())) { delete ad; }
  if (!(B->is_dense())) { delete bd; }
  return cd;
}


MatrixHandle
operator*(double a, MatrixHandle B)
{
  ASSERT(B.get_rep());

  MatrixHandle C = B->clone();
  C->scalar_multiply(a);
  return C;
}


MatrixHandle
operator*(MatrixHandle A, double b)
{
  return b*A;
}


} // End namespace SCIRun
