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
 *  MatrixOperations.cc: Matrix Operations
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   August 2003
 *
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

  if (B->is_column())
  {
    ColumnMatrix *cd = scinew ColumnMatrix(A->nrows());
    Mult(*cd, *(A.get_rep()), *(B->column()));
    return cd;
  }
  else if (A->is_sparse() && B->is_sparse())
  {
    SparseRowMatrix *as = A->sparse();
    SparseRowMatrix *bs = B->sparse();
    return as->sparse_sparse_mult(*bs);
  }
  else if (A->is_sparse())
  {
    SparseRowMatrix *ad = A->sparse();
    DenseMatrix *bd = B->dense();
    DenseMatrix *cd = scinew DenseMatrix(ad->nrows(), bd->ncols());
    ad->sparse_mult(*bd, *cd);
    if (!(B->is_dense())) { delete bd; }
    return cd;
  }
  else if (B->is_sparse())
  {
    DenseMatrix *ad = A->dense();
    SparseRowMatrix *bst = B->sparse()->transpose();
    DenseMatrix *cd = scinew DenseMatrix(A->nrows(), B->ncols());
    bst->sparse_mult_transXB(*ad, *cd);
    if (!A->is_dense()) { delete ad; }
    delete bst;
    return cd;
  }
  else
  {
    DenseMatrix *ad = A->dense();
    DenseMatrix *bd = B->dense();
    DenseMatrix *cd = scinew DenseMatrix(ad->nrows(), bd->ncols());
    Mult(*cd, *ad, *bd);

    if (!(A->is_dense())) { delete ad; }
    if (!(B->is_dense())) { delete bd; }
    return cd;
  }
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
