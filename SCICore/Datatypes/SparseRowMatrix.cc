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

#include <SCICore/Datatypes/SparseRowMatrix.h>
#include <SCICore/Math/ssmult.h>
#include <SCICore/Math/MiscMath.h>
#include <SCICore/Math/MinMax.h>
#include <SCICore/Util/Assert.h>
#include <SCICore/Exceptions/Exceptions.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Datatypes/ColumnMatrix.h>
#include <SCICore/Malloc/Allocator.h>
#include <iostream>
using std::cerr;
using std::endl;
#include <stdio.h>

namespace SCICore {
namespace Datatypes {

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

SparseRowMatrix::SparseRowMatrix(int nnrows, int nncols,
				 int* rows, int* columns,
				 int nnz)
: Matrix(Matrix::symmetric, Matrix::sparse), nnrows(nnrows),
  nncols(nncols), rows(rows), columns(columns), nnz(nnz)
{
    a=scinew double[nnz];
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
    EXCEPTION(SCICore::ExceptionsSpace::General("SparseRowMatrix can't do a direct solve!"));
}

void SparseRowMatrix::mult(const ColumnMatrix& x, ColumnMatrix& b,
			      int& flops, int& memrefs, int beg, int end) const
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

} // End namespace Datatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.7  2000/03/04 00:18:30  dmw
// added new Mesh BC and fixed sparserowmatrix bug
//
// Revision 1.6  1999/12/11 05:47:41  dmw
// sparserowmatrix -- someone had commented out the code that lets you get() a zero entry... I put it back in.    densematrix -- just cleaned up some comments
//
// Revision 1.5  1999/10/07 02:07:34  sparker
// use standard iostreams and complex type
//
// Revision 1.4  1999/08/25 03:48:41  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.3  1999/08/18 20:20:19  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:38:54  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:28  mcq
// Initial commit
//
// Revision 1.1  1999/04/25 04:07:17  dav
// Moved files into Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:51  dav
// Import sources
//
//
