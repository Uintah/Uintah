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
 *  SparseRowMatrix.cc: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Math/ssmult.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>
#include <Core/Util/Assert.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using std::cerr;
using std::endl;
#include <stdio.h>
#include <memory.h>

namespace SCIRun {

Persistent*
SparseRowMatrix::maker()
{
  return scinew SparseRowMatrix;
}


PersistentTypeID SparseRowMatrix::type_id("SparseRowMatrix", "Matrix",
					  SparseRowMatrix::maker);


SparseRowMatrix*
SparseRowMatrix::clone()
{
  return scinew SparseRowMatrix(*this);
}


SparseRowMatrix::SparseRowMatrix()
  : nnrows(0),
    nncols(0),
    rows(0),
    columns(0),
    nnz(0),
    a(0)
{
}


SparseRowMatrix::SparseRowMatrix(int nnrows, int nncols,
				 int* rows, int* columns,
				 int nnz, double* a_)
  : nnrows(nnrows),
    nncols(nncols),
    rows(rows),
    columns(columns),
    nnz(nnz),
    a(a_)
{
  if (a == 0) { a = scinew double[nnz]; }
}


SparseRowMatrix::SparseRowMatrix(const SparseRowMatrix& copy)
  : nnrows(copy.nnrows),
    nncols(copy.nncols),
    nnz(copy.nnz)
{
  rows = scinew int[nnrows+1];
  columns = scinew int[nnz];
  a = scinew double[nnz];
  memcpy(a, copy.a, sizeof(double)*nnz);
  memcpy(rows, copy.rows, sizeof(int)*(nnrows+1));
  memcpy(columns, copy.columns, sizeof(int)*nnz);
}


SparseRowMatrix::~SparseRowMatrix()
{
  if (a)
  {
    delete[] a;
  }
  if (columns)
  {
    delete[] columns;
  }
  if (rows)
  {
    delete[] rows;
  }
}


SparseRowMatrix *
SparseRowMatrix::sparse()
{
  return this;
}


DenseMatrix *
SparseRowMatrix::dense()
{
  DenseMatrix *dm = scinew DenseMatrix(nnrows,nncols);
  if (nnrows==0) return dm;
  dm->zero();
  int count=0;
  int nextRow;
  for (int r=0; r<nnrows; r++)
  {
    nextRow = rows[r+1];
    while (count<nextRow)
    {
      (*dm)[r][columns[count]]=a[count];
      count++;
    }
  }
  return dm;
}


ColumnMatrix *
SparseRowMatrix::column()
{
  ColumnMatrix *cm = scinew ColumnMatrix(nnrows);
  if (nnrows)
  {
    cm->zero();
    for (int i=0; i<nnrows; i++)
    {
      // If the first column entry for the row is a zero.
      if (columns[rows[i]] == 0)
      {
	(*cm)[i] = a[rows[i]];
      }
      else
      {
	(*cm)[i] = 0;
      }
    }
  }
  return cm;
}


SparseRowMatrix *
SparseRowMatrix::transpose()
{
  double *t_a = scinew double[nnz];
  int *t_columns = scinew int[nnz];
  int *t_rows = scinew int[nncols+1];
  int t_nnz = nnz;
  int t_nncols = nnrows;
  int t_nnrows = nncols;
  SparseRowMatrix *t = scinew SparseRowMatrix(t_nnrows, t_nncols, t_rows,
					      t_columns, t_nnz, t_a);

  int *at = scinew int[t_nnrows+1];
  int i;
  for (i=0; i<t_nnrows+1;i++)
  {
    at[i] = 0;
  }
  for (i=0; i<t_nnz;i++)
  {
    at[columns[i]+1]++;
  }
  t_rows[0] = 0;
  for (i=1; i<t_nnrows+1; i++)
  {
    at[i] += at[i-1];
    t_rows[i] = at[i];
  }

  int c = 0;
  for (int r=0; r<nnrows; r++)
  {
    for (; c<rows[r+1]; c++)
    {
      int mcol = columns[c];
      t_columns[at[mcol]] = r;
      t_a[at[mcol]] = a[c];
      at[mcol]++;
    }
  }

  delete at;
  return t;
}


int
SparseRowMatrix::getIdx(int i, int j)
{
  int row_idx=rows[i];
  int next_idx=rows[i+1];
  int l=row_idx;
  int h=next_idx-1;
  for (;;)
  {
    if (h<l)
    {
      return -1;
    }
    int m=(l+h)/2;
    if (j<columns[m])
    {
      h=m-1;
    }
    else if (j>columns[m])
    {
      l=m+1;
    }
    else
    {
      return m;
    }
  }
}


double&
SparseRowMatrix::get(int i, int j) const
{
  int row_idx=rows[i];
  int next_idx=rows[i+1];
  int l=row_idx;
  int h=next_idx-1;
  for (;;)
  {
    if (h<l)
    {
#if 0
      cerr << "column " << j << " not found in row "<<i << ": ";
      for (int idx=row_idx;idx<next_idx;idx++)
	cerr << columns[idx] << " ";
      cerr << endl;
      ASSERTFAIL("Column not found");
#endif
      static double zero;
      zero=0;
      return zero;
    }
    int m=(l+h)/2;
    if (j<columns[m])
    {
      h=m-1;
    }
    else if (j>columns[m])
    {
      l=m+1;
    }
    else
    {
      return a[m];
    }
  }
}


void
SparseRowMatrix::put(int i, int j, double d)
{
  get(i,j)=d;
}


void
SparseRowMatrix::add(int i, int j, double d)
{
  get(i,j)+=d;
}


int
SparseRowMatrix::nrows() const
{
  return nnrows;
}


int
SparseRowMatrix::ncols() const
{
  return nncols;
}


void
SparseRowMatrix::getRowNonzeros(int r, Array1<int>& idx, Array1<double>& val)
{
  int row_idx=rows[r];
  int next_idx=rows[r+1];
  idx.resize(next_idx-row_idx);
  val.resize(next_idx-row_idx);
  int i=0;
  for (int c=row_idx; c<next_idx; c++, i++)
  {
    idx[i]=columns[c];
    val[i]=a[c];
  }
}


void
SparseRowMatrix::zero()
{
  double* ptr=a;
  for (int i=0;i<nnz;i++)
    *ptr++=0.0;
}


void
SparseRowMatrix::solve(ColumnMatrix&)
{
  ASSERTFAIL("SparseRowMatrix can't do a direct solve!");
}


void
SparseRowMatrix::mult(const ColumnMatrix& x, ColumnMatrix& b,
		      int& flops, int& memrefs, int beg, int end,
		      int) const
{
  // Compute A*x=b
  ASSERT(x.nrows() == nncols);
  ASSERT(b.nrows() == nnrows);
  if (beg==-1)beg=0;
  if (end==-1)end=nnrows;
  double* xp=&x[0];
  double* bp=&b[0];
  ssmult(beg, end, rows, columns, a, xp, bp);

  int nnz=2*(rows[end]-rows[beg]);
  flops+=2*(rows[end]-rows[beg]);
  int nr=end-beg;
  memrefs+=2*sizeof(int)*nr+nnz*sizeof(int)+2*nnz*sizeof(double)+nr*sizeof(double);
}


void
SparseRowMatrix::mult_transpose(const ColumnMatrix& x, ColumnMatrix& b,
				int& flops, int& memrefs,
				int beg, int end, int) const
{
  // Compute At*x=b
  ASSERT(x.nrows() == nnrows);
  ASSERT(b.nrows() == nncols);
  if (beg==-1)beg=0;
  if (end==-1)end=nnrows;
  double* bp=&b[0];
  for (int i=beg; i<end; i++)
    bp[i] = 0;
  for (int j=0; j<nnrows; j++)
  {
    if (!x[j]) continue;
    double xj = x[j];
    int row_idx = rows[j];
    int next_idx = rows[j+1];
    int i=row_idx;
    for (; i<next_idx && columns[i] < beg; i++);
    for (; i<next_idx && columns[i] < end; i++)
      bp[columns[i]] += a[i]*xj;
  }
  int nnz=2*(rows[end]-rows[beg]);
  flops+=2*(rows[end]-rows[beg]);
  int nr=end-beg;
  memrefs+=2*sizeof(int)*nr+nnz*sizeof(int)+2*nnz*sizeof(double)+nr*sizeof(double);
}


void
SparseRowMatrix::sparse_mult(const DenseMatrix& x, DenseMatrix& b) const
{
  // Compute A*x=b
  ASSERT(x.nrows() == nncols);
  ASSERT(b.nrows() == nnrows);
  ASSERT(x.ncols() == b.ncols());
  int i, j, k;
  //  cout << "x size = " << x.nrows() << " " << x.ncols() << "\n";
  //  cout << "b size = " << b.nrows() << " " << b.ncols() << "\n";

  for (j = 0; j < b.ncols(); j++)
  {
    for (i = 0; i < b.nrows(); i++)
    {
      double sum = 0.0;
      for (k = rows[i]; k < rows[i+1]; k++)
      {
	sum += a[k] * x.get(columns[k], j);
      }
      b.put(i, j, sum);
    }
  }
}


void
SparseRowMatrix::print() const
{
  cerr << "Sparse RowMatrix: " << endl;
}


void SparseRowMatrix::print(std::ostream&) const
{

}


#define SPARSEROWMATRIX_VERSION 1

void
SparseRowMatrix::io(Piostream& stream)
{
  stream.begin_class("SparseRowMatrix", SPARSEROWMATRIX_VERSION);
  // Do the base class first...
  Matrix::io(stream);

  stream.io(nnrows);
  stream.io(nncols);
  stream.io(nnz);
  if (stream.reading())
  {
    a=new double[nnz];
    columns=new int[nnz];
    rows=new int[nnrows+1];
  }
  int i;
  stream.begin_cheap_delim();
  for (i=0;i<=nnrows;i++)
    stream.io(rows[i]);
  stream.end_cheap_delim();

  stream.begin_cheap_delim();
  for (i=0;i<nnz;i++)
    stream.io(columns[i]);
  stream.end_cheap_delim();

  stream.begin_cheap_delim();
  for (i=0;i<nnz;i++)
    stream.io(a[i]);
  stream.end_cheap_delim();

  stream.end_class();
}


SparseRowMatrix *
AddSparse(const SparseRowMatrix &a, const SparseRowMatrix &b)
{
  ASSERT(a.nrows() == b.nrows() && a.ncols() == b.ncols());

  int *rows = scinew int[a.nrows() + 1];
  vector<int> cols;
  vector<double> vals;

  int r, ca, cb;

  rows[0] = 0;
  for (r = 0; r < a.nrows(); r++)
  {
    rows[r+1] = rows[r];
    ca = a.rows[r];
    cb = b.rows[r];
    while (1)
    {
      if (ca >= a.rows[r+1] && cb >= b.rows[r+1])
      {
	break;
      }
      else if (ca >= a.rows[r+1])
      {
	cols.push_back(b.columns[cb]);
	vals.push_back(b.a[cb]);
	rows[r+1]++;
	cb++;
      }
      else if (cb >= b.rows[r+1])
      {
	cols.push_back(a.columns[ca]);
	vals.push_back(a.a[ca]);
	rows[r+1]++;
	ca++;
      }
      else if (a.columns[ca] < b.columns[cb])
      {
	cols.push_back(a.columns[ca]);
	vals.push_back(a.a[ca]);
	rows[r+1]++;
	ca++;
      }
      else if (a.columns[ca] > b.columns[cb])
      {
	cols.push_back(b.columns[cb]);
	vals.push_back(b.a[cb]);
	rows[r+1]++;
	cb++;
      }
      else
      {
	cols.push_back(a.columns[ca]);
	vals.push_back(a.a[ca] + b.a[cb]);
	rows[r+1]++;
	ca++;
	cb++;
      }
    }
  }

  int *vcols = scinew int[cols.size()];
  for (unsigned int i = 0; i < cols.size(); i++)
  {
    vcols[i] = cols[i];
  }

  double *vvals = scinew double[vals.size()];
  for (unsigned int i = 0; i < vals.size(); i++)
  {
    vvals[i] = vals[i];
  }

  return scinew SparseRowMatrix(a.nrows(), a.ncols(), rows,
				vcols, vals.size(), vvals);
}


SparseRowMatrix *
SubSparse(const SparseRowMatrix &a, const SparseRowMatrix &b)
{
  ASSERT(a.nrows() == b.nrows() && a.ncols() == b.ncols());

  int *rows = scinew int[a.nrows() + 1];
  vector<int> cols;
  vector<double> vals;

  int r, ca, cb;

  rows[0] = 0;
  for (r = 0; r < a.nrows(); r++)
  {
    rows[r+1] = rows[r];
    ca = a.rows[r];
    cb = b.rows[r];
    while (1)
    {
      if (ca >= a.rows[r+1] && cb >= b.rows[r+1])
      {
	break;
      }
      else if (ca >= a.rows[r+1])
      {
	cols.push_back(b.columns[cb]);
	vals.push_back(-b.a[cb]);
	rows[r+1]++;
	cb++;
      }
      else if (cb >= b.rows[r+1])
      {
	cols.push_back(a.columns[ca]);
	vals.push_back(a.a[ca]);
	rows[r+1]++;
	ca++;
      }
      else if (a.columns[ca] < b.columns[cb])
      {
	cols.push_back(a.columns[ca]);
	vals.push_back(a.a[ca]);
	rows[r+1]++;
	ca++;
      }
      else if (a.columns[ca] > b.columns[cb])
      {
	cols.push_back(b.columns[cb]);
	vals.push_back(-b.a[cb]);
	rows[r+1]++;
	cb++;
      }
      else
      {
	cols.push_back(a.columns[ca]);
	vals.push_back(a.a[ca] - b.a[cb]);
	rows[r+1]++;
	ca++;
	cb++;
      }
    }
  }

  unsigned int i;
  int *vcols = scinew int[cols.size()];
  for (i = 0; i < cols.size(); i++)
  {
    vcols[i] = cols[i];
  }

  double *vvals = scinew double[vals.size()];
  for (i = 0; i < vals.size(); i++)
  {
    vvals[i] = vals[i];
  }

  return scinew SparseRowMatrix(a.nrows(), a.ncols(), rows,
				vcols, vals.size(), vvals);
}


void SparseRowMatrix::scalar_multiply(double s)
{
  for (int i=0;i<nnz;i++)
  {
    a[i] *= s;
  }
}


MatrixHandle
SparseRowMatrix::submatrix(int r1, int c1, int r2, int c2)
{
  ASSERTRANGE(r1, 0, r2+1);
  ASSERTRANGE(r2, r1, nnrows);
  ASSERTRANGE(c1, 0, c2+1);
  ASSERTRANGE(c2, c1, nncols);

  int i, j;
  int *rs = scinew int[r2-r1+2];
  vector<int> csv;
  vector<double> valsv;

  rs[0] = 0;
  for (i = r1; i < nnrows; i++)
  {
    rs[i-r1+1] = rs[i-r1];
    for (j = rows[i]; j < rows[i+1]; j++)
    {
      if (columns[j] >= c1 && columns[j] <= c2)
      {
	csv.push_back(columns[j] - c1);
	valsv.push_back(a[j]);
	rs[i-r1+1]++;
      }
    }
  }

  int *cs = scinew int[csv.size()];
  for (i = 0; (unsigned int)i < csv.size(); i++)
  {
    cs[i] = csv[i];
  }

  double *vals = scinew double[valsv.size()];
  for (i = 0; (unsigned int)i < valsv.size(); i++)
  {
    vals[i] = valsv[i];
  }

  return scinew SparseRowMatrix(r2-r1+1, c2-c1+1, rs, cs, valsv.size(), vals);
}

} // End namespace SCIRun
