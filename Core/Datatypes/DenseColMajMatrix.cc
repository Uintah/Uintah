/*
  For more information, please see: http://software.sci.utah.edu

  The MIT License

  Copyright (c) 2004 Scientific Computing and Imaging Institute,
  University of Utah.

  License for the specific language governing rights and limitations under
  Permission is hereby granted, free of charge, to any person obtaining a
  copy of this software and associated documentation files (the "Software"),
  to deal in the Software without restriction, including without limitation
  the rights to use, copy, modify, merge, publish, distribute, sublicense,
  and/or sell copies of the Software, and to permit persons to whom the
  Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included
  in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
  DEALINGS IN THE SOFTWARE.
*/


/*
 *  DenseColMajMatrix.cc: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <cstdio>

#include <sci_defs/lapack_defs.h>
#include <sci_defs/blas_defs.h>

#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseColMajMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Util/Assert.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/Assert.h>
#include <Core/Exceptions/FileNotFound.h>
#include <iostream>
#include <vector>
#include <cstring>

#if defined(HAVE_LAPACK)
#  include <Core/Math/sci_lapack.h>
#endif

using std::cout;
using std::endl;
using std::vector;

namespace SCIRun {

DenseColMajMatrix*
DenseColMajMatrix::clone()
{
  return scinew DenseColMajMatrix(*this);
}


//! constructors
DenseColMajMatrix::DenseColMajMatrix() :
  dataptr_(0)
{
}


DenseColMajMatrix::DenseColMajMatrix(int r, int c) :
  Matrix(r, c)
{
  dataptr_ = scinew double[nrows_ * ncols_];
}


DenseColMajMatrix::DenseColMajMatrix(const DenseColMajMatrix& m) :
  Matrix(m.nrows_, m.ncols_)
{
  dataptr_ = scinew double[nrows_ * ncols_];
  memcpy(dataptr_, m.dataptr_, sizeof(double) * nrows_ * ncols_);
}


DenseMatrix *
DenseColMajMatrix::dense()
{
  DenseMatrix *m = scinew DenseMatrix(nrows_, ncols_);
  for (int i = 0; i < nrows_; i++)
    for (int j = 0; j < ncols_; j++)
      (*m)[i][j] = iget(i, j);
  return m;
}


ColumnMatrix *
DenseColMajMatrix::column()
{
  ColumnMatrix *cm = scinew ColumnMatrix(nrows_);
  for (int i=0; i<nrows_; i++)
    (*cm)[i] = iget(i, 0);
  return cm;
}


SparseRowMatrix *
DenseColMajMatrix::sparse()
{
  int nnz = 0;
  int r, c;
  int *rows = scinew int[nrows_ + 1];
  for (r=0; r<nrows_; r++)
    for (c=0; c<ncols_; c++)
      if (iget(r, c) != 0.0) nnz++;

  int *columns = scinew int[nnz];
  double *a = scinew double[nnz];

  int count = 0;
  for (r=0; r<nrows_; r++)
  {
    rows[r] = count;
    for (c=0; c<ncols_; c++)
      if (iget(r, c) != 0)
      {
        columns[count] = c;
        a[count] = iget(r, c);
        count++;
      }
  }
  rows[nrows_] = count;

  return scinew SparseRowMatrix(nrows_, ncols_, rows, columns, nnz, a);
}


DenseColMajMatrix *
DenseColMajMatrix::dense_col_maj()
{
  return this;
}


double *
DenseColMajMatrix::get_data_pointer()
{
  return dataptr_;
}


size_t
DenseColMajMatrix::get_data_size()
{
  return nrows() * ncols();
}


//! destructor
DenseColMajMatrix::~DenseColMajMatrix()
{
  if (dataptr_) { delete[] dataptr_; }
}


//! assignment operator
DenseColMajMatrix&
DenseColMajMatrix::operator=(const DenseColMajMatrix& m)
{
  if (dataptr_) { delete[] dataptr_; }

  nrows_ = m.nrows_;
  ncols_ = m.ncols_;
  dataptr_ = scinew double[nrows_ * ncols_];
  memcpy(dataptr_, m.dataptr_, sizeof(double) * nrows_ * ncols_);

  return *this;
}


double
DenseColMajMatrix::get(int r, int c) const
{
  ASSERTRANGE(r, 0, nrows_);
  ASSERTRANGE(c, 0, ncols_);
  return iget(r, c);
}


void
DenseColMajMatrix::put(int r, int c, double d)
{
  ASSERTRANGE(r, 0, nrows_);
  ASSERTRANGE(c, 0, ncols_);
  iget(r, c) = d;
}


void
DenseColMajMatrix::add(int r, int c, double d)
{
  ASSERTRANGE(r, 0, nrows_);
  ASSERTRANGE(c, 0, ncols_);
  iget(r, c) += d;
}


DenseColMajMatrix *
DenseColMajMatrix::transpose()
{
  DenseColMajMatrix *m = scinew DenseColMajMatrix(ncols_, nrows_);
  for (int c=0; c<ncols_; c++)
    for (int r=0; r<nrows_; r++)
      m->iget(c, r) = iget(r, c);
  return m;
}


void
DenseColMajMatrix::getRowNonzerosNoCopy(int r, int &size, int &stride,
                                        int *&cols, double *&vals)
{
  size = ncols_;
  stride = nrows_;
  cols = NULL;
  vals = dataptr_ + r;
}


void
DenseColMajMatrix::zero()
{
  memset(dataptr_, 0, sizeof(double) * nrows_ * ncols_);
}


DenseColMajMatrix *
DenseColMajMatrix::identity(int size)
{
  DenseColMajMatrix *result = scinew DenseColMajMatrix(size, size);
  result->zero();
  for (int i = 0; i < size; i++)
  {
    result->iget(i, i) = 1.0;
  }

  return result;
}


void
DenseColMajMatrix::print() const
{
  std::cout << "DenseColMaj Matrix: " << nrows_ << " by " << ncols_ << std::endl;
  print(std::cout);
}


void
DenseColMajMatrix::print(ostream& ostr) const
{
  for (int i=0; i<nrows_; i++)
  {
    for (int j=0; j<ncols_; j++)
    {
      ostr << iget(i, j) << "\t";
    }
    ostr << endl;
  }
}


MatrixHandle
DenseColMajMatrix::submatrix(int r1, int c1, int r2, int c2)
{
  ASSERTRANGE(r1, 0, r2+1);
  ASSERTRANGE(r2, r1, nrows_);
  ASSERTRANGE(c1, 0, c2+1);
  ASSERTRANGE(c2, c1, ncols_);
  DenseColMajMatrix *mat = scinew DenseColMajMatrix(r2 - r1 + 1, c2 - c1 + 1);
  for (int i = c1; i <= c2; i++)
  {
    // TODO: Test this.
    memcpy(mat->dataptr_ + (i - c1) * (r2 - r1 + 1),
           dataptr_ + c1 * nrows_ + r1,
           (r2 - r1 + 1) * sizeof(double));
  }
  return mat;
}


void
DenseColMajMatrix::mult(const ColumnMatrix& x, ColumnMatrix& b,
                        int& flops, int& memrefs, int beg, int end,
                        int spVec) const
{
  // Compute A*x=b
  ASSERTEQ(x.nrows(), ncols_);
  ASSERTEQ(b.nrows(), nrows_);
  if (beg == -1) beg = 0;
  if (end == -1) end = nrows_;
  int i, j;
  if (!spVec)
  {
    for (i=beg; i<end; i++)
    {
      double sum = 0.0;
      for (j=0; j<ncols_; j++)
      {
        sum += iget(i, j) * x[j];
      }
      b[i] = sum;
    }
  }
  else
  {
    for (i=beg; i<end; i++) b[i] = 0.0;
    for (j=0; j<ncols_; j++)
    {
      if (x[j])
      {
        for (i=beg; i<end; i++)
        {
          b[i] += iget(i, j) * x[j];
        }
      }
    }
  }
  flops += (end-beg) * ncols_ * 2;
  memrefs += (end-beg) * ncols_ * 2 *sizeof(double)+(end-beg)*sizeof(double);
}


void
DenseColMajMatrix::mult_transpose(const ColumnMatrix& x, ColumnMatrix& b,
                                  int& flops, int& memrefs, int beg, int end,
                                  int spVec) const
{
  // Compute At*x=b
  ASSERT(x.nrows() == nrows_);
  ASSERT(b.nrows() == ncols_);
  if (beg == -1) beg = 0;
  if (end == -1) end = ncols_;
  int i, j;
  if (!spVec)
  {
    for (i=beg; i<end; i++)
    {
      double sum = 0.0;
      for (j=0; j<nrows_; j++)
      {
        sum += iget(j, i) * x[j];
      }
      b[i] = sum;
    }
  }
  else
  {
    for (i=beg; i<end; i++) b[i] = 0.0;
    for (j=0; j<nrows_; j++)
    {
      if (x[j])
      {
        for (i=beg; i<end; i++)
        {
          b[i] += iget(j, i) * x[j];
        }
      }
    }
  }
  flops+=(end-beg)*nrows_*2;
  memrefs+=(end-beg)*nrows_*2*sizeof(double)+(end-beg)*sizeof(double);
}

} // End namespace SCIRun
