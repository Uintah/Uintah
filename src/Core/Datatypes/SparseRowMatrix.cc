//static char *id="@(#) $Id$";

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

#include <CoreDatatypes/SparseRowMatrix.h>
#include <Math/ssmult.h>
#include <Math/MiscMath.h>
#include <Math/MinMax.h>
#include <Util/Assert.h>
#include <Exceptions/Exceptions.h>
#include <Containers/String.h>
#include <CoreDatatypes/ColumnMatrix.h>
#include <Malloc/Allocator.h>
#include <iostream.h>
#include <stdio.h>

namespace SCICore {
namespace CoreDatatypes {

static Persistent* maker()
{
    return scinew SparseRowMatrix;
}

PersistentTypeID SparseRowMatrix::type_id("SparseRowMatrix", "Matrix", maker);

SparseRowMatrix::SparseRowMatrix()
: Matrix(Matrix::symmetric, Matrix::sparse), nnrows(0), nncols(0), a(0),
  columns(0), rows(0), nnz(0)
{
}

SparseRowMatrix::SparseRowMatrix(int nnrows, int nncols,
				       Array1<int>& in_rows,
				       Array1<int>& in_cols)
: Matrix(Matrix::symmetric, Matrix::sparse), nnrows(nnrows), 
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
: Matrix(Matrix::symmetric, Matrix::sparse), nnrows(nnrows),
  nncols(nncols), rows(rows), columns(columns), nnz(nnz), a(a)
{
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

  
double& SparseRowMatrix::get(int i, int j)
{
    int row_idx=rows[i];
    int next_idx=rows[i+1];
    int l=row_idx;
    int h=next_idx-1;
    for(;;){
	if(h<l){
	  //#if 0
	  cerr << "column " << j << " not found in row "<<i << ": ";
	    for(int idx=row_idx;idx<next_idx;idx++)
		cerr << columns[idx] << " ";
	    cerr << endl;
	    ASSERT(0);
	    //#endif
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
    if (extremaCurrent)
	return minVal;
    if (nnz == 0) return 0;
    minVal=maxVal=a[0];
    for (int idx=0; idx<nnz; idx++) {
	if (a[idx] < minVal)
	    minVal = a[idx];
	if (a[idx] > maxVal)
	    maxVal = a[idx];
    }
    extremaCurrent=1;
    return minVal;
}

double SparseRowMatrix::maxValue() {
    if (extremaCurrent)
	return maxVal;
    if (nnz == 0) return 0;
    minVal=maxVal=a[0];
    for (int idx=0; idx<nnz; idx++) {
	if (a[idx] < minVal)
	    minVal = a[idx];
	if (a[idx] > maxVal)
	    maxVal = a[idx];
    }
    extremaCurrent=1;
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
    EXCEPTION(General("SparseRowMatrix can't do a direct solve!"));
}

void SparseRowMatrix::mult(const ColumnMatrix& x, ColumnMatrix& b,
			      int& flops, int& memrefs, int beg, int end) const
{
    // Compute A*x=b
    ASSERT(x.nrows() == nnrows);
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
					int beg, int end)
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
      int row_idx = rows[j];
      int next_idx = rows[j+1];
      double xj = x[j];
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

void SparseRowMatrix::print()
{
    cerr << "Sparse RowMatrix: " << endl;
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

} // End namespace CoreDatatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:28  mcq
// Initial commit
//
// Revision 1.1  1999/04/25 04:07:17  dav
// Moved files into CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:51  dav
// Import sources
//
//
