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

#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Math/ssmult.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>
#include <Core/Util/Assert.h>
#include <Core/Containers/String.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using std::cerr;
using std::endl;
#include <stdio.h>
#include <memory.h>

namespace SCIRun {

static Persistent* maker()
{
    return scinew SparseRowMatrix;
}

PersistentTypeID SparseRowMatrix::type_id("SparseRowMatrix", "Matrix", maker);

SparseRowMatrix* SparseRowMatrix::clone(){
  return scinew SparseRowMatrix(*this);
}

SparseRowMatrix::SparseRowMatrix()
: Matrix(Matrix::SYMMETRIC, Matrix::SPARSE),
  nnrows(0),
  nncols(0),
  rows(0),
  columns(0),
  nnz(0),
  a(0)
{
}

SparseRowMatrix::SparseRowMatrix(int nnrows, int nncols,
				       Array1<int>& in_rows,
				       Array1<int>& in_cols)
: Matrix(Matrix::SYMMETRIC, Matrix::SPARSE),
  nnrows(nnrows), 
  nncols(nncols)
{
    nnz=in_cols.size();
    a=scinew double[nnz];
    columns=scinew int[nnz];
    rows=scinew int[nnrows+1];
    int i;
    for(i=0;i<in_rows.size();i++){
	rows[i]=in_rows[i];
    }
    for(i=0;i<in_cols.size();i++){
	columns[i]=in_cols[i];
    }
}

SparseRowMatrix::SparseRowMatrix(int nnrows, int nncols,
				 int* rows, int* columns,
				 int nnz, double* a)
  : Matrix(Matrix::SYMMETRIC, Matrix::SPARSE),
    nnrows(nnrows),
    nncols(nncols),
    rows(rows),
    columns(columns),
    nnz(nnz), a(a)
{
}

SparseRowMatrix::SparseRowMatrix(int nnrows, int nncols,
				 int* rows, int* columns,
				 int nnz)
  : Matrix(Matrix::SYMMETRIC, Matrix::SPARSE),
    nnrows(nnrows),
    nncols(nncols),
    rows(rows),
    columns(columns),
    nnz(nnz)
{
    a=scinew double[nnz];
}

SparseRowMatrix::SparseRowMatrix(const SparseRowMatrix& copy)
  : Matrix(copy),
    nnrows(copy.nnrows),
    nncols(copy.nncols),
    dummy(copy.dummy),
    minVal(copy.minVal),
    maxVal(copy.maxVal),
    nnz(copy.nnz)
{
  rows = scinew int[nnrows];
  columns = scinew int[nncols];
  a = scinew double[nnz];
  memcpy(a, copy.a, sizeof(double)*nnz);
  memcpy(rows, copy.rows, sizeof(int)*nnrows);
  memcpy(columns, copy.columns, sizeof(int)*nncols);
}

void
SparseRowMatrix::transpose( SparseRowMatrix &m )
{
  int i;

  nnz = m.nnz;
  nncols = m.nnrows;
  nnrows = m.nncols;

  a = scinew double[nnz];
  
  columns = scinew int[nnz];
  rows = scinew int[nnrows+1];
  
  int *at = scinew int[nnrows+1];
  for (i=0; i<nnrows+1;i++) 
    at[i] = 0;
  for (i=0; i<nnz;i++)
    at[m.columns[i]+1]++;
  rows[0] = 0;
  for (i=1; i<nnrows+1; i++) {
    at[i] += at[i-1];
    rows[i] = at[i];
  }

  int c = 0;
  for (int r=0; r<m.nnrows; r++) {
    for (; c<m.rows[r+1]; c++) {
      int mcol = m.columns[c];
      columns[at[mcol]] = r;
      a[at[mcol]] = m.a[c];
      at[mcol]++;
    }
  }

  delete at;
}

SparseRowMatrix::~SparseRowMatrix()
{
    if(a)
	delete[] a;
    if(columns)
	delete[] columns;
    if(rows)
	delete[] rows;
}

  
int SparseRowMatrix::getIdx(int i, int j) {
    int row_idx=rows[i];
    int next_idx=rows[i+1];
    int l=row_idx;
    int h=next_idx-1;
    for(;;){
	if(h<l){
	  return -1;
	}
	int m=(l+h)/2;
	if(j<columns[m]){
	    h=m-1;
	} else if(j>columns[m]){
	    l=m+1;
	} else {
	    return m;
	}
    }
}
  
double& SparseRowMatrix::get(int i, int j)
{
    int row_idx=rows[i];
    int next_idx=rows[i+1];
    int l=row_idx;
    int h=next_idx-1;
    for(;;){
	if(h<l){
    #if 0
	  cerr << "column " << j << " not found in row "<<i << ": ";
	    for(int idx=row_idx;idx<next_idx;idx++)
		cerr << columns[idx] << " ";
	    cerr << endl;
	    ASSERTFAIL("Column not found");
    #endif
	    static double zero;
	    zero=0;
	    return zero;
	}
	int m=(l+h)/2;
	if(j<columns[m]){
	    h=m-1;
	} else if(j>columns[m]){
	    l=m+1;
	} else {
	    return a[m];
	}
    }
}

void SparseRowMatrix::put(int i, int j, const double& d)
{
    get(i,j)=d;
}

void SparseRowMatrix::add(int i, int j, const double& d)
{
    get(i,j)+=d;
}

int SparseRowMatrix::nrows() const
{
    return nnrows;
}

int SparseRowMatrix::ncols() const
{
    return nncols;
}

double SparseRowMatrix::density()
{	
    return (1.*nnz)/(1.*nnrows*nncols);
}

double SparseRowMatrix::minValue() {
    if (extremaCurrent_)
	return minVal;
    if (nnz == 0) return 0;
    minVal=maxVal=a[0];
    for (int idx=0; idx<nnz; idx++) {
	if (a[idx] < minVal)
	    minVal = a[idx];
	if (a[idx] > maxVal)
	    maxVal = a[idx];
    }
    extremaCurrent_ = true;
    return minVal;
}

double SparseRowMatrix::maxValue() {
    if (extremaCurrent_)
	return maxVal;
    if (nnz == 0) return 0;
    minVal=maxVal=a[0];
    for (int idx=0; idx<nnz; idx++) {
	if (a[idx] < minVal)
	    minVal = a[idx];
	if (a[idx] > maxVal)
	    maxVal = a[idx];
    }
    extremaCurrent_ = true;
    return maxVal;
}

void SparseRowMatrix::getRowNonzeros(int r, Array1<int>& idx, 
					Array1<double>& val)
{
    int row_idx=rows[r];
    int next_idx=rows[r+1];
    idx.resize(next_idx-row_idx);
    val.resize(next_idx-row_idx);
    int i=0;
    for (int c=row_idx; c<next_idx; c++, i++) {
	idx[i]=columns[c];
	val[i]=a[c];
    }
}
    
void SparseRowMatrix::zero()
{
    double* ptr=a;
    for(int i=0;i<nnz;i++)
	*ptr++=0.0;
}

void SparseRowMatrix::solve(ColumnMatrix&)
{
    ASSERTFAIL("SparseRowMatrix can't do a direct solve!");
}

void SparseRowMatrix::mult(const ColumnMatrix& x, ColumnMatrix& b,
			   int& flops, int& memrefs, int beg, int end, 
			   int) const
{
    // Compute A*x=b
    ASSERT(x.nrows() == nncols);
    ASSERT(b.nrows() == nnrows);
    if(beg==-1)beg=0;
    if(end==-1)end=nnrows;
    double* xp=&x[0];
    double* bp=&b[0];
    ssmult(beg, end, rows, columns, a, xp, bp);

    int nnz=2*(rows[end]-rows[beg]);
    flops+=2*(rows[end]-rows[beg]);
    int nr=end-beg;
    memrefs+=2*sizeof(int)*nr+nnz*sizeof(int)+2*nnz*sizeof(double)+nr*sizeof(double);
}

void SparseRowMatrix::mult_transpose(const ColumnMatrix& x, ColumnMatrix& b,
					int& flops, int& memrefs,
					int beg, int end, int)
{
    // Compute At*x=b
    ASSERT(x.nrows() == nnrows);
    ASSERT(b.nrows() == nncols);
    if(beg==-1)beg=0;
    if(end==-1)end=nnrows;
    double* bp=&b[0];
    for (int i=beg; i<end; i++)
      bp[i] = 0;
    for (int j=0; j<nnrows; j++) {
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

void SparseRowMatrix::print() const
{
    cerr << "Sparse RowMatrix: " << endl;
}

void SparseRowMatrix::print(std::ostream&) const
{
  
}


#define SPARSEROWMATRIX_VERSION 1

void SparseRowMatrix::io(Piostream& stream)
{
    stream.begin_class("SparseRowMatrix", SPARSEROWMATRIX_VERSION);
    // Do the base class first...
    Matrix::io(stream);

    stream.io(nnrows);
    stream.io(nncols);
    stream.io(nnz);
    if(stream.reading()){
	a=new double[nnz];
	columns=new int[nnz];
	rows=new int[nnrows+1];
    }
    int i;
    stream.begin_cheap_delim();
    for(i=0;i<=nnrows;i++)
	stream.io(rows[i]);
    stream.end_cheap_delim();

    stream.begin_cheap_delim();
    for(i=0;i<nnz;i++)
	stream.io(columns[i]);
    stream.end_cheap_delim();

    stream.begin_cheap_delim();
    for(i=0;i<nnz;i++)
	stream.io(a[i]);
    stream.end_cheap_delim();

    stream.end_class();
}

} // End namespace SCIRun
