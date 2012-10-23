/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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
 *  SparseRowMatrix.cc: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 */

#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/DenseColMajMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Math/ssmult.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>
#include <Core/Util/Assert.h>
#include <Core/Malloc/Allocator.h>


#include <iostream>
#include <algorithm>

using namespace std;

#include <cstdio>
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


SparseRowMatrix::SparseRowMatrix() :
  rows(0),
  columns(0),
  nnz(0),
  a(0)
{
}


SparseRowMatrix::SparseRowMatrix(int nnrows, int nncols,
				 int* rows, int* columns,
				 int nnz, double* a_) :
  Matrix(nnrows, nncols),
  rows(rows),
  columns(columns),
  nnz(nnz),
  a(a_)
{
  if (a == 0) { a = scinew double[nnz]; }
  //validate();
}


SparseRowMatrix::SparseRowMatrix(const SparseRowMatrix& copy) :
  Matrix(copy.nrows_, copy.ncols_),
  nnz(copy.nnz)
{
  rows = scinew int[nrows_+1];
  columns = scinew int[nnz];
  a = scinew double[nnz];
  memcpy(a, copy.a, sizeof(double)*nnz);
  memcpy(rows, copy.rows, sizeof(int)*(nrows_+1));
  memcpy(columns, copy.columns, sizeof(int)*nnz);
  //validate();
}


SparseRowMatrix::~SparseRowMatrix()
{
  if (a)
  {
    delete[] a;
  }
  //if (columns)
  //{
  //  delete[] columns;
  //}
  //if (rows)
  //{
  //  delete[] rows;
  //}
}


void
SparseRowMatrix::validate()
{
  int i, j;

  ASSERTMSG(rows[0] == 0, "Row start is nonzero.");
  for (i = 0; i< nrows_; i++)
  {
    ASSERTMSG(rows[i] <= rows[i+1], "Malformed rows, not increasing.");
    for (j = rows[i]; j < rows[i+1]; j++)
    {
      ASSERTMSG(columns[j] >= 0 && columns[j] < ncols_, "Column out of range.");
      if (j != rows[i])
      {
        if (columns[j-1] >= columns[j])
        {
          cout << i << " : " << columns[j-1] << " " << columns[j] << "\n";
          cout << i << " : " << a[j-1] << " " << a[j] << "\n";
        }
        ASSERTMSG(columns[j-1] < columns[j], "Column out of order.");
      }
    }
  }
  ASSERTMSG(rows[nrows_] == nnz, "Row end is incorrect.");
}


SparseRowMatrix *
SparseRowMatrix::sparse()
{
  return this;
}


DenseMatrix *
SparseRowMatrix::dense()
{
  DenseMatrix *dm = scinew DenseMatrix(nrows_, ncols_);
  if (nrows_ == 0) return dm;
  dm->zero();
  int count=0;
  int nextRow;
  for (int r=0; r<nrows_; r++)
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


DenseColMajMatrix *
SparseRowMatrix::dense_col_maj()
{
  DenseColMajMatrix *dm = scinew DenseColMajMatrix(nrows_, ncols_);
  if (nrows_ == 0) return dm;
  dm->zero();
  int count = 0;
  int nextRow;
  for (int r = 0; r<nrows_; r++)
  {
    nextRow = rows[r+1];
    while (count<nextRow)
    {
      dm->iget(r, columns[count]) = a[count];
      count++;
    }
  }
  return dm;
}


ColumnMatrix *
SparseRowMatrix::column()
{
  ColumnMatrix *cm = scinew ColumnMatrix(nrows_);
  if (nrows_)
  {
    cm->zero();
    for (int i=0; i<nrows_; i++)
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


double *
SparseRowMatrix::get_data_pointer()
{
  return a;
}


size_t
SparseRowMatrix::get_data_size()
{
  return nnz;
}


SparseRowMatrix *
SparseRowMatrix::transpose() const
{
  double *t_a = scinew double[nnz];
  int *t_columns = scinew int[nnz];
  int *t_rows = scinew int[ncols_+1];
  int t_nnz = nnz;
  int t_nncols = nrows_;
  int t_nnrows = ncols_;
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
  for (int r=0; r<nrows_; r++)
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

SparseRowMatrix *
SparseRowMatrix::transpose()
{
  double *t_a = scinew double[nnz];
  int *t_columns = scinew int[nnz];
  int *t_rows = scinew int[ncols_+1];
  int t_nnz = nnz;
  int t_nncols = nrows_;
  int t_nnrows = ncols_;
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
  for (int r=0; r<nrows_; r++)
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


double
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
      return 0.0;
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
  int row_idx=rows[i];
  int next_idx=rows[i+1];
  int l=row_idx;
  int h=next_idx-1;
  for (;;)
  {
    if (h<l)
    {
      ASSERTFAIL("SparseRowMatrix::put into invalid(dataless) location.");
      return;
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
      a[m] = d;
      return;
    }
  }
}


void
SparseRowMatrix::add(int i, int j, double d)
{
  int row_idx=rows[i];
  int next_idx=rows[i+1];
  int l=row_idx;
  int h=next_idx-1;
  for (;;)
  {
    if (h<l)
    {
      ASSERTFAIL("SparseRowMatrix::add into invalid(dataless) location.");
      return;
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
      a[m] += d;
      return;
    }
  }
}


void
SparseRowMatrix::getRowNonzerosNoCopy(int r, int &size, int &stride,
                                      int *&cols, double *&vals)
{
  size = rows[r+1] - rows[r];
  stride = 1;
  cols = columns + rows[r];
  vals = a + rows[r];
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
  ASSERT(x.nrows() == ncols_);
  ASSERT(b.nrows() == nrows_);
  if (beg==-1) beg = 0;
  if (end==-1) end = nrows_;
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
  ASSERT(x.nrows() == nrows_);
  ASSERT(b.nrows() == ncols_);
  if (beg==-1) beg = 0;
  if (end==-1) end = nrows_;
  double* bp=&b[0];
  for (int i=beg; i<end; i++)
    bp[i] = 0;
  for (int j=0; j<nrows_; j++)
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
  ASSERT(x.nrows() == ncols_);
  ASSERT(b.nrows() == nrows_);
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
SparseRowMatrix::sparse_mult_transXB(const DenseMatrix& x,
                                     DenseMatrix& b) const
{
  // Compute A*xT=bT
  ASSERT(x.ncols() == ncols_);
  ASSERT(b.ncols() == nrows_);
  ASSERT(x.nrows() == b.nrows());
  int i, j, k;

  for (j = 0; j < b.nrows(); j++)
  {
    for (i = 0; i < b.ncols(); i++)
    {
      double sum = 0.0;
      for (k = rows[i]; k < rows[i+1]; k++)
      {
       sum += a[k] * x.get(j, columns[k]);
      }
      b.put(j, i, sum);
    }
  }
}


class SparseSparseElement {
public:
  int     row; 
  int     col;
  double  val;
};

inline bool operator<(const SparseSparseElement& s1, const SparseSparseElement& s2)
{
  if (s1.row < s2.row) return(true);
  if ((s1.row == s2.row)&&(s1.col < s2.col)) return(true);
  return (false);
}

inline bool operator==(const SparseSparseElement& s1, const SparseSparseElement& s2)
{
  if ((s1.row == s2.row)&&(s1.col == s2.col)) return(true);
  return (false);
}

MatrixHandle
SparseRowMatrix::sparse_sparse_mult(const SparseRowMatrix &b) const
{
  // Compute A*B=C
  ASSERT(b.nrows() == ncols_);
  
  MatrixHandle output = 0;
  
  int bncols = b.ncols_;
  int *brows = b.rows;
  int *bcolumns = b.columns;
  double* ba = b.a;  
  
  // Rough estimate
  std::vector<SparseSparseElement> elems;
  
  for (int r =0; r<nrows_; r++)
  {
    int ps = rows[r];
    int pe = rows[r+1];   
    for (int p = ps; p < pe; p++)
    {
      int s = columns[p];
      double v = a[p];
      int qs = brows[s];
      int qe = brows[s+1];
      for (int q=qs; q<qe; q++)
      {
        SparseSparseElement el;
        el.row = r;
        el.col = bcolumns[q];
        el.val = v*ba[q];
        elems.push_back(el);
      }
    }
  }
  
  std::sort(elems.begin(),elems.end());
  
  int s = 0;
  int nnz = 0;
  if (elems.size()) nnz = 1;
  for (unsigned int r=1; r<elems.size(); r++)
  {
    if (elems[s] == elems[r])
    {
      elems[s].val += elems[r].val;
      elems[r].val = 0.0;
    }
    else
    {
      nnz++;
      s = r;
    }
  }
  
  int *rr = scinew int[nrows_+1];
  int *cc = scinew int[nnz];
  double *vv = scinew double[nnz];
  
  if ((rr == 0)||(cc == 0)||(vv == 0))
  {
    if (rr) delete[] rr;
    if (cc) delete[] cc;
    if (vv) delete[] vv;
    return (output);  
  }
  
  rr[0] = 0;
  int q = 0;
  unsigned int k = 0;
  for( int p=0; p < nrows_; p++ )
  {
    while ((k < elems.size())&&(elems[k].row == p)) { 
      if (elems[k].val) { 
        cc[q] = elems[k].col; vv[q] = elems[k].val; q++;
      } 
      k++; 
    }
    rr[p+1] = q;
  }   
  
  output = dynamic_cast<Matrix *>(scinew SparseRowMatrix(nrows_,bncols,rr,cc,nnz,vv));
  
  return (output);
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

  stream.io(nrows_);
  stream.io(ncols_);
  stream.io(nnz);
  if (stream.reading())
  {
    a = scinew double[nnz];
    columns = scinew int[nnz];
    rows = scinew int[nrows_+1];
  }
  int i;
  stream.begin_cheap_delim();
  if (!stream.block_io(rows, sizeof(int), nrows_+1))
  {
    for (i=0;i<=nrows_;i++)
      stream.io(rows[i]);
  }
  stream.end_cheap_delim();

  stream.begin_cheap_delim();
  if (!stream.block_io(columns, sizeof(int), nnz))
  {
    for (i=0;i<nnz;i++)
      stream.io(columns[i]);
  }
  stream.end_cheap_delim();

  stream.begin_cheap_delim();
  if (!stream.block_io(a, sizeof(double), nnz))
  {
    for (i=0;i<nnz;i++)
      stream.io(a[i]);
  }
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
    for (;;)
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
				vcols, (int)vals.size(), vvals);
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
    for( ;; )
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
				vcols, (int)vals.size(), vvals);
}


MatrixHandle
SparseRowMatrix::submatrix(int r1, int c1, int r2, int c2)
{
  ASSERTRANGE(r1, 0, r2+1);
  ASSERTRANGE(r2, r1, nrows_);
  ASSERTRANGE(c1, 0, c2+1);
  ASSERTRANGE(c2, c1, ncols_);

  int i, j;
  int *rs = scinew int[r2-r1+2];
  vector<int> csv;
  vector<double> valsv;

  rs[0] = 0;
  for (i = r1; i <= r2; i++)
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
  double *vals = scinew double[valsv.size()];
  for (i = 0; (unsigned int)i < csv.size(); i++)
  {
    cs[i] = csv[i];
    vals[i] = valsv[i];
  }

  return scinew SparseRowMatrix(r2-r1+1, c2-c1+1, rs, cs,
                                (int)valsv.size(), vals);
}


SparseRowMatrix *
SparseRowMatrix::identity(int size)
{ 
  int *r = scinew int[size+1];
  int *c = scinew int[size];
  double *d = scinew double[size];

  int i;
  for (i=0; i<size; i++)
  {
    c[i] = r[i] = i;
    d[i] = 1.0;
  }
  r[i] = i;

  return scinew SparseRowMatrix(size, size, r, c, size, d);
}

} // End namespace SCIRun

