/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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
 *  DenseColMajMatrix.cc: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
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

//using namespace std;
using std::cout;
using std::endl;
using std::vector;

namespace SCIRun {

static Persistent* maker()
{
  return scinew DenseColMajMatrix;
}

PersistentTypeID DenseColMajMatrix::type_id("DenseColMajMatrix", "Matrix", maker);

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
DenseColMajMatrix::print(std::ostream& ostr) const
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


#define DENSEMATRIX_VERSION 3

void
DenseColMajMatrix::io(Piostream& stream)
{
  int version=stream.begin_class("DenseColMajMatrix", DENSEMATRIX_VERSION);

  // Do the base class first.
  Matrix::io(stream);

  stream.io(nrows_);
  stream.io(ncols_);
  if (stream.reading())
  {
    dataptr_ = scinew double[nrows_ * ncols_];
  }
  stream.begin_cheap_delim();

  int split;
  if (stream.reading())
  {
    if (version > 2)
    {
      Pio(stream, separate_raw_);
      if (separate_raw_)
      {
        Pio(stream, raw_filename_);
        FILE *f = fopen(raw_filename_.c_str(), "r");
        if (f)
        {
          fread(dataptr_, sizeof(double), nrows_ * ncols_, f);
          fclose(f);
        }
        else
        {
          const string errmsg = "Error reading separated file '" +
            raw_filename_ + "'";
          std::cerr << errmsg << "\n";
          throw FileNotFound(errmsg, __FILE__, __LINE__);
        }
      }
    }
    else
    {
      separate_raw_ = false;
    }
    split = separate_raw_;
  }
  else
  {     // writing
    string filename = raw_filename_;
    split = separate_raw_;
    if (split)
    {
      if (filename == "")
      {
        if (stream.file_name.c_str())
        {
          char *tmp=strdup(stream.file_name.c_str());
          char *dot = strrchr( tmp, '.' );
          if (!dot ) dot = strrchr( tmp, 0);
          filename = stream.file_name.substr(0,dot-tmp) + ".raw";
          delete tmp;
        } else split=0;
      }
    }
    Pio(stream, split);
    if (split)
    {
      Pio(stream, filename);
      FILE *f = fopen(filename.c_str(), "w");
      fwrite(dataptr_, sizeof(double), nrows_ * ncols_, f);
      fclose(f);
    }
  }

  if (!split)
  {
    if (!stream.block_io(dataptr_, sizeof(double), (size_t)(nrows_ * ncols_)))
    {
      for (size_t i = 0; i < (size_t)(nrows_ * ncols_); i++)
      {
        stream.io(dataptr_[i]);
      }
    }
  }
  stream.end_cheap_delim();
  stream.end_class();
}


#if 0
void
Mult(DenseColMajMatrix& out, const DenseColMajMatrix& m1, const DenseColMajMatrix& m2)
{
  ASSERTEQ(m1.ncols(), m2.nrows());
  ASSERTEQ(out.nrows(), m1.nrows());
  ASSERTEQ(out.ncols(), m2.ncols());
#if defined(HAVE_CBLAS)
  double ALPHA = 1.0;
  double BETA = 0.0;
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m1.nrows(),
              m2.ncols(), m1.ncols(), ALPHA, m1.dataptr_, m1.ncols(),
              m2.dataptr_, m2.ncols(), BETA,
              out.dataptr_, out.ncols());
#else
  int nr = out.nrows();
  int nc = out.ncols();
  int ndot=m1.ncols();
  for (int i=0;i<nr;i++)
  {
    const double* row = m1[i];
    for (int j=0;j<nc;j++)
    {
      double d = 0.0;
      for (int k=0;k<ndot;k++)
      {
        d += row[k] * m2[k][j];
      }
      out[i][j] = d;
    }
  }
#endif
}


void
Add(DenseColMajMatrix& out, const DenseColMajMatrix& m1, const DenseColMajMatrix& m2)
{
  ASSERTEQ(m1.ncols(), m2.ncols());
  ASSERTEQ(out.ncols(), m2.ncols());
  ASSERTEQ(m1.nrows(), m2.nrows());
  ASSERTEQ(out.nrows(), m2.nrows());

  int nr=out.nrows();
  int nc=out.ncols();

  for (int i=0;i<nr;i++)
    for (int j=0; j<nc; j++)
      out[i][j] = m1[i][j] + m2[i][j];
}


void
Sub(DenseColMajMatrix& out, const DenseColMajMatrix& m1, const DenseColMajMatrix& m2)
{
  ASSERTEQ(m1.ncols(), m2.ncols());
  ASSERTEQ(out.ncols(), m2.ncols());
  ASSERTEQ(m1.nrows(), m2.nrows());
  ASSERTEQ(out.nrows(), m2.nrows());

  int nr=out.nrows();
  int nc=out.ncols();

  for (int i=0;i<nr;i++)
    for (int j=0; j<nc; j++)
      out[i][j] = m1[i][j] - m2[i][j];
}


void
Add(DenseColMajMatrix& out, double f1, const DenseColMajMatrix& m1,
    double f2, const DenseColMajMatrix& m2)
{
  ASSERTEQ(m1.ncols(), m2.ncols());
  ASSERTEQ(out.ncols(), m2.ncols());
  ASSERTEQ(m1.nrows(), m2.nrows());
  ASSERTEQ(out.nrows(), m2.nrows());

  int nr=out.nrows();
  int nc=out.ncols();

  for (int i=0;i<nr;i++)
    for (int j=0; j<nc; j++)
      out[i][j] = f1 * m1[i][j] + f2 * m2[i][j];
}


void
Add(double f1, DenseColMajMatrix& out, double f2, const DenseColMajMatrix& m1)
{
  ASSERTEQ(out.ncols(), m1.ncols());
  ASSERTEQ(out.nrows(), m1.nrows());
  int nr=out.nrows();
  int nc=out.ncols();

  for (int i=0;i<nr;i++)
    for (int j=0; j<nc; j++)
      out[i][j] = f1 * out[i][j] + f2 * m1[i][j];
}


void
Mult_trans_X(DenseColMajMatrix& out, const DenseColMajMatrix& m1, const DenseColMajMatrix& m2)
{
  ASSERTEQ(m1.nrows(), m2.nrows());
  ASSERTEQ(out.nrows(), m1.ncols());
  ASSERTEQ(out.ncols(), m2.ncols());
  int nr=out.nrows();
  int nc=out.ncols();
  int ndot=m1.nrows();
  for (int i=0;i<nr;i++)
  {
    for (int j=0;j<nc;j++)
    {
      double d = 0.0;
      for (int k=0;k<ndot;k++)
      {
        d += m1[k][i] * m2[k][j];
      }
      out[i][j] = d;
    }
  }
}


void
Mult_X_trans(DenseColMajMatrix& out, const DenseColMajMatrix& m1, const DenseColMajMatrix& m2)
{
  ASSERTEQ(m1.ncols(), m2.ncols());
  ASSERTEQ(out.nrows(), m1.nrows());
  ASSERTEQ(out.ncols(), m2.nrows());

  int nr=out.nrows();
  int nc=out.ncols();
  int ndot=m1.ncols();

  for (int i=0;i<nr;i++)
  {
    const double* row = m1[i];
    for (int j=0;j<nc;j++)
    {
      double d = 0.0;
      for (int k=0; k<ndot; k++)
      {
        d += row[k] * m2[j][k];
      }
      out[i][j]=d;
    }
  }
}


double
DenseColMajMatrix::sumOfCol(int n)
{
  ASSERT(n<ncols_);
  ASSERT(n>=0);

  double sum = 0;
  for (int i=0; i<nrows_; i++)
    sum += (*this)[i][n];
  return sum;
}


double
DenseColMajMatrix::sumOfRow(int n)
{
  ASSERT(n < nrows_);
  ASSERT(n >= 0);

  const double* rp = (*this)[n];
  double sum = 0.0;
  for (int i=0; i < ncols_; i++)
  {
    sum += rp[i];
  }
  return sum;
}


double
DenseColMajMatrix::determinant()
{
  double a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p;
  ASSERTMSG(((nrows_ == 4) && (ncols_ == 4)),
            "Number of Rows and Colums for Determinant must equal 4! (Code not completed)");
  a=(*this)[0][0]; b=(*this)[0][1]; c=(*this)[0][2]; d=(*this)[0][3];
  e=(*this)[1][0]; f=(*this)[1][1]; g=(*this)[1][2]; h=(*this)[1][3];
  i=(*this)[2][0]; j=(*this)[2][1]; k=(*this)[2][2]; l=(*this)[2][3];
  m=(*this)[3][0]; n=(*this)[3][1]; o=(*this)[3][2]; p=(*this)[3][3];

  double q=a*f*k*p - a*f*l*o - a*j*g*p + a*j*h*o + a*n*g*l - a*n*h*k
    - e*b*k*p + e*b*l*o + e*j*c*p - e*j*d*o - e*n*c*l + e*n*d*k
    + i*b*g*p - i*b*h*o - i*f*c*p + i*f*d*o + i*n*c*h - i*n*d*g
    - m*b*g*l + m*b*h*k + m*f*c*l - m*f*d*k - m*j*c*h + m*j*d*g;

  return q;
}


// TODO:  Fix the lapack functions.
#if 0 //defined(HAVE_LAPACK)

void
DenseColMajMatrix::svd(DenseColMajMatrix& U, SparseRowMatrix& S, DenseColMajMatrix& VT)
{

  ASSERTEQ(U.ncols(), U.nrows());
  ASSERTEQ(VT.ncols(), VT.nrows());
  ASSERTEQ(U.nrows(), nrows_);
  ASSERTEQ(VT.ncols(), ncols_);
  ASSERTEQ(S.nrows(), nrows_);
  ASSERTEQ(S.ncols(), ncols_);

  // TODO:  Fix this.  Make data?
  //lapacksvd(data, nrows_, ncols_, S.a, U.data, VT.data);
}


void
DenseColMajMatrix::eigenvalues(ColumnMatrix& R, ColumnMatrix& I)
{

  ASSERTEQ(ncols_, nrows_);
  ASSERTEQ(R.nrows(), I.nrows());
  ASSERTEQ(ncols_, R.nrows());

  double *Er = R.get_data_pointer();
  double *Ei = I.get_data_pointer();

  double **data = scinew double*[nrows_];
  for (int i = 0; i < nrows_; i++)
  {
    data[i] = dataptr_ + i*ncols_;
  }

  lapackeigen(data, nrows_, Er, Ei);
}


void
DenseColMajMatrix::eigenvectors(ColumnMatrix& R, ColumnMatrix& I, DenseColMajMatrix& Vecs)
{

  ASSERTEQ(ncols_, nrows_);
  ASSERTEQ(R.nrows(), I.nrows());
  ASSERTEQ(ncols_, R.nrows());

  double *Er = R.get_data_pointer();
  double *Ei = I.get_data_pointer();

  double **data = scinew double*[nrows_];
  for (int i = 0; i < nrows_; i++)
  {
    data[i] = dataptr_ + i*ncols_;
  }

  double **data = scinew double*[nrows_];
  for (int i = 0; i < nrows_; i++)
  {
    data[i] = dataptr_ + i*ncols_;
  }

  lapackeigen(data, nrows_, Er, Ei, Vecs.data);

  for (int i = 0; i<nrows_; i++)
  {
    R[i] = Er[i];
    I[i] = Ei[i];
  }
}


#else

void
DenseColMajMatrix::svd(DenseColMajMatrix& U, SparseRowMatrix& S, DenseColMajMatrix& VT)
{
  ASSERTFAIL("Build was not configured with LAPACK");
}

void
DenseColMajMatrix::eigenvalues(ColumnMatrix& R, ColumnMatrix& I)
{
  ASSERTFAIL("Build was not configured with LAPACK");
}

void
DenseColMajMatrix::eigenvectors(ColumnMatrix& R, ColumnMatrix& I, DenseColMajMatrix& Vecs)
{
  ASSERTFAIL("Build was not configured with LAPACK");
}

#endif

#endif
} // End namespace SCIRun
