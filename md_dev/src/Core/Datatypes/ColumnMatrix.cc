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
 *  ColumnMatrix.cc: for RHS and LHS
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/DenseColMajMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Util/Assert.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/Expon.h>
#include <Core/Math/LinAlg.h>
#include <iostream>
#include <cstring>
using std::endl;
using std::cerr;

namespace SCIRun {

static Persistent* maker()
{
    return scinew ColumnMatrix(0);
}

PersistentTypeID ColumnMatrix::type_id("ColumnMatrix", "Matrix", maker);

ColumnMatrix::ColumnMatrix(int rows) :
  Matrix(rows, 1)
{
    if(nrows_)
	data=scinew double[nrows_];
    else
	data=0;
}

ColumnMatrix::ColumnMatrix(const ColumnMatrix& c) :
  Matrix(c.nrows_, 1)
{
  if(nrows_){
    data=scinew double[nrows_];
    for(int i=0;i<nrows_;i++)
      data[i]=c.data[i];
  } else {
    data=0;
  }
}


ColumnMatrix *
ColumnMatrix::column()
{
  return this;
}


DenseMatrix *
ColumnMatrix::dense()
{
  DenseMatrix *dm = scinew DenseMatrix(nrows_, 1);
  for (int i = 0; i < nrows_; i++)
  {
    (*dm)[i][0] = data[i];
  }
  return dm;
}


DenseColMajMatrix *
ColumnMatrix::dense_col_maj()
{
  DenseColMajMatrix *dm = scinew DenseColMajMatrix(nrows_, 1);
  for (int i = 0; i < nrows_; i++)
  {
    dm->iget(i, 0) = data[i];
  }
  return dm;
}


SparseRowMatrix *
ColumnMatrix::sparse()
{
  int nnz = 0;
  int r;
  int *row = scinew int[nrows_+1];
  for (r=0; r<nrows_; r++)
    if (data[r] != 0) nnz++;
  
  int *columns = scinew int[nnz];
  double *a = scinew double[nnz];
  
  int count=0;
  for (r=0; r<nrows_; r++) {
    row[r] = count;
    if (data[r] != 0) {
      columns[count]=0;
      a[count]=data[r];
      count++;
    }
  }
  row[nrows_]=count;
  return scinew SparseRowMatrix(nrows_, 1, row, columns, nnz, a);
}


double *
ColumnMatrix::get_data_pointer()
{
  return data;
}


size_t
ColumnMatrix::get_data_size()
{
  return nrows();
}


Matrix *ColumnMatrix::transpose() {
  DenseMatrix *dm = scinew DenseMatrix(1, nrows_);
  for (int i=0; i<nrows_; i++)
    (*dm)[0][i] = data[i];
  return dm;
}

ColumnMatrix* ColumnMatrix::clone() {
    return scinew ColumnMatrix(*this);
}

ColumnMatrix& ColumnMatrix::operator=(const ColumnMatrix& c)
{
    if(nrows_ != c.nrows_){
	if(data)delete[] data;
	nrows_=c.nrows_;
	data=scinew double[nrows_];
    }
    for(int i=0;i<nrows_;i++)
	data[i]=c.data[i];
    return *this;
}

ColumnMatrix::~ColumnMatrix()
{
    if(data)
	delete[] data;
}

void ColumnMatrix::resize(int new_rows)
{
    if(data)
	delete[] data;
    if(new_rows)
	data=new double[new_rows];
    else
	data=0;
    nrows_ = new_rows;
}

void ColumnMatrix::zero()
{
  memset(data, 0, sizeof(double) * nrows_);
}

extern "C" double cm_vnorm(int beg, int end, double* data);
extern "C" double dnrm2(int n, double* x, int incx);

double ColumnMatrix::vector_norm() const
{
  return linalg_norm2(nrows_, data);
}

double ColumnMatrix::vector_norm(int& flops, int& memrefs) const
{
    flops += nrows_ * 2;
    memrefs += nrows_ * sizeof(double);
    return vector_norm();
}

double ColumnMatrix::vector_norm(int& flops, int& memrefs, int beg, int end) const
{
    ASSERTRANGE(end, 0, nrows_+1);
    ASSERTRANGE(beg, 0, end);
    flops+=(end-beg)*2;
    memrefs+=(end-beg)*sizeof(double);
    return linalg_norm2((end-beg), data+beg);
}

void ColumnMatrix::print() const
{
    std::cout << "Column Matrix: " << nrows_ << endl;
    print(std::cout);
}

void ColumnMatrix::print(std::ostream& str) const
{
    for(int i=0; i<nrows_; i++)
    {
	str << data[i] << endl;
    }
}

void ColumnMatrix::print() {
  print(cerr);
}

double ColumnMatrix::get(int r, int c) const
{
    ASSERTRANGE(r, 0, nrows_);
    ASSERTEQ(c, 0);
    return data[r];
}

void ColumnMatrix::put(int r, int c, double d)
{
    ASSERTRANGE(r, 0, nrows_);
    ASSERTEQ(c, 0);
    data[r] = d;
}

void ColumnMatrix::add(int r, int c, double d)
{
    ASSERTRANGE(r, 0, nrows_);
    ASSERTEQ(c, 0);
    data[r] += d;
}

double ColumnMatrix::sumOfCol(int c) {
  ASSERTEQ(c, 0);
  double sum = 0;
  for (int i=0; i< nrows_; i++)
    sum+=data[i];
  return sum;
}

void ColumnMatrix::getRowNonzeros(int r, Array1<int>& idx, 
				  Array1<double>& val) { 
  idx.resize(1);
  val.resize(1);
  idx[0]=0;
  val[0]=data[r];
}

void
ColumnMatrix::getRowNonzerosNoCopy(int r, int &size, int &stride,
                                   int *&cols, double *&vals)
{
  size = 1;
  stride = 1;
  cols = NULL;
  vals = data + r;
}

int ColumnMatrix::solve(ColumnMatrix&) { 
  ASSERTFAIL("Error - called solve on a columnmatrix.\n");
}

int ColumnMatrix::solve(vector<double>&) { 
  ASSERTFAIL("Error - called solve on a columnmatrix.\n");
}

void ColumnMatrix::mult(const ColumnMatrix&, ColumnMatrix&,
			int& , int& , int , int , int) const {
  ASSERTFAIL("Error - called mult on a columnmatrix.\n");
}

void ColumnMatrix::mult_transpose(const ColumnMatrix&, ColumnMatrix&,
				  int&, int&, int, int, int) const {
  ASSERTFAIL("Error - called mult_transpose on a columnmatrix.\n");
}

#define COLUMNMATRIX_VERSION 2

void ColumnMatrix::io(Piostream& stream)
{
    int version=stream.begin_class("ColumnMatrix", COLUMNMATRIX_VERSION);
    
    if (version > 1)
    {
      // New version inherits from Matrix
      Matrix::io(stream);
    }

    stream.io(nrows_);

    if (stream.reading())
    {
      data = scinew double[nrows_];
    }

    if (!stream.block_io(data, sizeof(double), nrows_))
    {
      for (int i=0; i<nrows_; i++)
        stream.io(data[i]);
    }
    stream.end_class();
}

void Mult(ColumnMatrix& result, const ColumnMatrix& a, const ColumnMatrix& b)
{
    ASSERTEQ(result.nrows_, a.nrows_);
    ASSERTEQ(result.nrows_, b.nrows_);
    linalg_mult(result.nrows_, result.data, a.data, b.data);
}

void Mult(ColumnMatrix& result, const ColumnMatrix& a, const ColumnMatrix& b,
	  int& flops, int& memrefs)
{
    ASSERTEQ(result.nrows_, a.nrows_);
    ASSERTEQ(result.nrows_, b.nrows_);
    linalg_mult(result.nrows_, result.data, a.data, b.data);
    flops+=result.nrows_;
    memrefs+=result.nrows_ * 3 * sizeof(double);
}

void Mult(ColumnMatrix& result, const ColumnMatrix& a, const ColumnMatrix& b,
	  int& flops, int& memrefs, int beg, int end)
{
    ASSERTEQ(result.nrows_, a.nrows_);
    ASSERTEQ(result.nrows_, b.nrows_);
    ASSERTRANGE(end, 0, result.nrows_+1);
    ASSERTRANGE(beg, 0, end);
    linalg_mult(end-beg, result.data+beg, a.data+beg, b.data+beg);
    flops+=(end-beg);
    memrefs+=(end-beg)*3*sizeof(double);
}

void Sub(ColumnMatrix& result, const ColumnMatrix& a, const ColumnMatrix& b)
{
    ASSERTEQ(result.nrows_, a.nrows_);
    ASSERTEQ(result.nrows_, b.nrows_);
    linalg_sub(result.nrows_, result.data, a.data, b.data);
}

void Sub(ColumnMatrix& result, const ColumnMatrix& a, const ColumnMatrix& b,
	  int& flops, int& memrefs)
{
    ASSERTEQ(result.nrows_, a.nrows_);
    ASSERTEQ(result.nrows_, b.nrows_);
    linalg_sub(result.nrows_, result.data, a.data, b.data);
    flops+=result.nrows_;
    memrefs += result.nrows_ * 3 * sizeof(double);
}

void ScMult_Add(ColumnMatrix& result, double s, const ColumnMatrix& a,
		const ColumnMatrix& b)
{
    ASSERTEQ(result.nrows_, a.nrows_);
    ASSERTEQ(result.nrows_, b.nrows_);
    linalg_smadd(result.nrows_, result.data, s, a.data, b.data);
}

void ScMult_Add(ColumnMatrix& result, double s, const ColumnMatrix& a,
		const ColumnMatrix& b, int& flops, int& memrefs)
{
    ASSERTEQ(result.nrows_, a.nrows_);
    ASSERTEQ(result.nrows_, b.nrows_);
    linalg_smadd(result.nrows_, result.data, s, a.data, b.data);
    flops+=result.nrows_ * 2;
    memrefs+=result.nrows_ * 3 * sizeof(double);
}

void ScMult_Add(ColumnMatrix& result, double s, const ColumnMatrix& a,
		const ColumnMatrix& b, int& flops, int& memrefs,
		int beg, int end)
{
    ASSERTEQ(result.nrows_, a.nrows_);
    ASSERTEQ(result.nrows_, b.nrows_);
    ASSERTRANGE(end, 0, result.nrows_+1);
    ASSERTRANGE(beg, 0, end);
    linalg_smadd(end-beg, result.data+beg, s, a.data+beg, b.data+beg);
    flops+=(end-beg)*2;
    memrefs+=(end-beg)*3*sizeof(double);
}

double Dot(const ColumnMatrix& a, const ColumnMatrix& b)
{
    ASSERTEQ(a.nrows_, b.nrows_);
    return linalg_dot(a.nrows_, a.data, b.data);
}

double Dot(const ColumnMatrix& a, const ColumnMatrix& b,
	   int& flops, int& memrefs)
{
    ASSERTEQ(a.nrows_, b.nrows_);
    flops += a.nrows_ * 2;
    memrefs += 2 * sizeof(double) * a.nrows_;
    return linalg_dot(a.nrows_, a.data, b.data);
}

double Dot(const ColumnMatrix& a, const ColumnMatrix& b,
	   int& flops, int& memrefs, int beg, int end)
{
    ASSERTEQ(a.nrows_, b.nrows_);
    ASSERTRANGE(end, 0, a.nrows_+1);
    ASSERTRANGE(beg, 0, end);
    flops+=(end-beg)*2;
    memrefs+=2*sizeof(double)*(end-beg);
    return linalg_dot((end-beg), a.data+beg, b.data+beg);
}

void Copy(ColumnMatrix& out, const ColumnMatrix& in)
{
    ASSERTEQ(out.nrows_, in.nrows_);
    for(int i=0; i<out.nrows_; i++)
	out.data[i] = in.data[i];
}

void Copy(ColumnMatrix& out, const ColumnMatrix& in, int&, int& refs)
{
    ASSERTEQ(out.nrows_, in.nrows_);
    for(int i=0; i<out.nrows_; i++)
	out.data[i]=in.data[i];
    refs+=sizeof(double)*out.nrows_;
}

void Copy(ColumnMatrix& out, const ColumnMatrix& in, int&, int& refs,
	  int beg, int end)
{
    ASSERTEQ(out.nrows_, in.nrows_);
    ASSERTRANGE(end, 0, out.nrows_+1);
    ASSERTRANGE(beg, 0, end);
    for(int i=beg;i<end;i++)
	out.data[i]=in.data[i];
    refs+=sizeof(double)*(end-beg);
}

void AddScMult(ColumnMatrix& result, const ColumnMatrix& a,
	       double s, const ColumnMatrix& b)
{
    ASSERTEQ(result.nrows_, a.nrows_);
    ASSERTEQ(result.nrows_, b.nrows_);
    linalg_smadd(result.nrows_, result.data, s, b.data, a.data);
}

void Add(ColumnMatrix& result, const ColumnMatrix& a, const ColumnMatrix& b)
{
    ASSERTEQ(result.nrows_, a.nrows_);
    ASSERTEQ(result.nrows_, b.nrows_);
    linalg_add(result.nrows_, result.data, a.data, b.data);
}

void Add(ColumnMatrix& result, const ColumnMatrix& a, const ColumnMatrix& b,
	 const ColumnMatrix& c)
{
    ASSERTEQ(result.nrows_, a.nrows_);
    ASSERTEQ(result.nrows_, b.nrows_);
    ASSERTEQ(result.nrows_, c.nrows_);
    for(int i=0;i<result.nrows_;i++)
	result.data[i]=a.data[i]+b.data[i]+c.data[i];
}

void Mult(ColumnMatrix& result, const ColumnMatrix& a, double s)
{
    ASSERTEQ(result.nrows_, a.nrows_);
    for(int i=0; i<result.nrows_; i++)
	result.data[i] = a.data[i]*s;
}

void ColumnMatrix::scalar_multiply(double s)
{
  for (int i=0; i<nrows_; i++)
  {
    data[i] *= s;
  }
}


MatrixHandle
ColumnMatrix::submatrix(int r1, int c1, int r2, int c2)
{
  ASSERTRANGE(r1, 0, r2+1);
  ASSERTRANGE(r2, r1, nrows_);
  ASSERTEQ(c1, 0);
  ASSERTEQ(c2, 0);

  ColumnMatrix *mat = scinew ColumnMatrix(r2 - r1 + 1);
  memcpy(mat->data, data + r1, (r2 - r1 + 1) * sizeof(double));

  return mat;
}


DenseMatrix
ColumnMatrix::exterior(const ColumnMatrix &m) const
{
  DenseMatrix ret(nrows_, nrows_);

  if (nrows_ != m.nrows())
  {
    ASSERTFAIL("Cannot compute exterior of two vectors of unequal dimensions.");
  }
  for (int i=0; i < nrows_; i++)
    for (int j=0; j< nrows_; j++)
      ret.put(i, j, get(j) * m.get(i));

  return ret;
}

} // End namespace SCIRun


