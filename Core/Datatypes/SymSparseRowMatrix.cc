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
 *  SymSparseRowMatrix.cc:  Symmetric Sparse Row Matrices
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   November 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Datatypes/SymSparseRowMatrix.h>
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

namespace SCIRun {

static Persistent* maker()
{
    return scinew SymSparseRowMatrix;
}

PersistentTypeID SymSparseRowMatrix::type_id("SymSparseRowMatrix", "Matrix", maker);

SymSparseRowMatrix::SymSparseRowMatrix()
  : Matrix(Matrix::SYMMETRIC, Matrix::SYMSPARSE),
    nnrows(0),
    nncols(0),
    rows(0),
    columns(0),
    nnz(0),
    a(0)
{
    upper_rows=0;
    upper_columns=0;
    upper_a=0;
    upper_nnz=0;
}

SymSparseRowMatrix::SymSparseRowMatrix(int nnrows, int nncols,
				       Array1<int>& in_rows,
				       Array1<int>& in_cols)
: Matrix(Matrix::SYMMETRIC, Matrix::SYMSPARSE),
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
    //compute_upper();
}

SymSparseRowMatrix::SymSparseRowMatrix(int nnrows, int nncols,
				       int* rows, int* columns,
				       int nnz)
: Matrix(Matrix::SYMMETRIC, Matrix::SYMSPARSE),
  nnrows(nnrows),
  nncols(nncols),
  rows(rows),
  columns(columns),
  nnz(nnz)
{
    a=scinew double[nnz];
    //compute_upper();
}

SymSparseRowMatrix::~SymSparseRowMatrix()
{
    if(a)
	delete[] a;
    if(columns)
	delete[] columns;
    if(rows)
	delete[] rows;
}

int SymSparseRowMatrix::getIdx(int i, int j) {
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
  
double& SymSparseRowMatrix::get(int i, int j)
{
    int row_idx=rows[i];
    int next_idx=rows[i+1];
    int l=row_idx;
    int h=next_idx-1;
    for(;;){
	if(h<l){
#if 0
	    cerr << "column " << j << " not found in row: ";
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

void SymSparseRowMatrix::put(int i, int j, const double& d)
{
    get(i,j)=d;
}

int SymSparseRowMatrix::nrows() const
{
    return nnrows;
}

int SymSparseRowMatrix::ncols() const
{
    return nncols;
}

double SymSparseRowMatrix::density()
{	
    return (1.*nnz)/(1.*nnrows*nncols);
}

double SymSparseRowMatrix::minValue() {
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

double SymSparseRowMatrix::maxValue() {
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

void SymSparseRowMatrix::getRowNonzeros(int r, Array1<int>& idx, 
					Array1<double>& val)
{
    int skip=0;
    int row_idx=rows[r];
    int next_idx=rows[r+1];
    idx.resize(next_idx-row_idx);
    val.resize(next_idx-row_idx);
    int i=0;
    for (int c=row_idx; c<next_idx; c++, i++) {
	if (a[c]) {
	    idx[i]=columns[c];
	    val[i]=a[c];
	} else {
	    i--;
	    skip++;
	}
    }
    idx.resize(idx.size()-skip);
    val.resize(val.size()-skip);
}
    
void SymSparseRowMatrix::zero()
{
    double* ptr=a;
    for(int i=0;i<nnz;i++)
	*ptr++=0.0;
}

void SymSparseRowMatrix::solve(ColumnMatrix&)
{
    ASSERTFAIL("SymSparseRowMatrix can't do a direct solve!");
}

void SymSparseRowMatrix::mult(const ColumnMatrix& x, ColumnMatrix& b,
			      int& flops, int& memrefs, int beg, int end,
			      int) const
{
    // Compute A*x=b
    ASSERT(x.nrows() == nnrows);
    ASSERT(b.nrows() == nnrows);
    if(beg==-1)beg=0;
    if(end==-1)end=nnrows;
    double* xp=&x[0];
    double* bp=&b[0];
    //if(beg==0 && end==nnrows)
	//ssmult_upper(beg, end, upper_rows, upper_columns, upper_a, xp, bp);
    //else
	//ssmult_uppersub(nnrows, beg, end, upper_rows, upper_columns, upper_a, xp, bp);
        ssmult(beg, end, rows, columns, a, xp, bp);

    int nnz=2*(rows[end]-rows[beg]);
    flops+=2*(rows[end]-rows[beg]);
    int nr=end-beg;
    memrefs+=2*sizeof(int)*nr+nnz*sizeof(int)+2*nnz*sizeof(double)+nr*sizeof(double);
}

void SymSparseRowMatrix::mult_transpose(const ColumnMatrix& x, ColumnMatrix& b,
					int& flops, int& memrefs,
					int beg, int end, int)
{
    // Compute At*x=b
    // This is the same as Ax=b since the matrix is symmetric
    ASSERT(x.nrows() == nnrows);
    ASSERT(b.nrows() == nnrows);
    if(beg==-1)beg=0;
    if(end==-1)end=nnrows;
    double* xp=&x[0];
    double* bp=&b[0];
    for(int i=beg;i<end;i++){
	double sum=0;
	int row_idx=rows[i];
	int next_idx=rows[i+1];
	for(int j=row_idx;j<next_idx;j++){
	    sum+=a[j]*xp[columns[j]];
	}
	bp[i]=sum;
    }
    int nnz=2*(rows[end]-rows[beg]);
    flops+=2*(rows[end]-rows[beg]);
    int nr=end-beg;
    memrefs+=2*sizeof(int)*nr+nnz*sizeof(int)+2*nnz*sizeof(double)+nr*sizeof(double);
}

void SymSparseRowMatrix::print() const
{
    cerr << "Sparse RowMatrix: " << endl;
}

#define SYMSPARSEROWMATRIX_VERSION 1

void SymSparseRowMatrix::io(Piostream& stream)
{
    stream.begin_class("SymSparseRowMatrix", SYMSPARSEROWMATRIX_VERSION);
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
    //compute_upper();
}

void SymSparseRowMatrix::compute_upper()
{
    if(upper_rows)
	delete[] upper_rows;
    if(upper_columns)
	delete[] upper_columns;
    if(upper_a)
	delete[] upper_a;
    upper_nnz=(nnz-nnrows)/2+nnrows;
    upper_columns=new int[upper_nnz];
    upper_rows=new int[nnrows+1];
    upper_a=new double[upper_nnz];
    int idx=0;
    for(int i=0;i<nnrows;i++){
	upper_rows[i]=idx;
	int start=rows[i];
	int last=rows[i+1];
	while(columns[start]<i)start++;
	for(;start<last;start++){
	    upper_a[idx]=a[start];
	    upper_columns[idx]=columns[start];
	    idx++;
	}
    }
    ASSERTEQ(idx, upper_nnz);
    upper_rows[nnrows]=idx;
}

} // End namespace SCIRun

