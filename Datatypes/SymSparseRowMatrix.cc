
#include <Datatypes/SymSparseRowMatrix.h>
#include <Math/MiscMath.h>
#include <Math/MinMax.h>
#include <Classlib/Assert.h>
#include <Classlib/Exceptions.h>
#include <Classlib/String.h>
#include <Datatypes/ColumnMatrix.h>
#include <Malloc/Allocator.h>
#include <iostream.h>

static Persistent* maker()
{
    return scinew SymSparseRowMatrix;
}

PersistentTypeID SymSparseRowMatrix::type_id("SymSparseRowMatrix", "Matrix", maker);

SymSparseRowMatrix::SymSparseRowMatrix()
: Matrix(Matrix::symmetric, Matrix::symsparse), nnrows(0), nncols(0), a(0),
  columns(0), rows(0), nnz(0)
{
}

SymSparseRowMatrix::SymSparseRowMatrix(int nnrows, int nncols,
				       Array1<int>& in_rows,
				       Array1<int>& in_cols)
: Matrix(Matrix::symmetric, Matrix::symsparse), nnrows(nnrows), 
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

SymSparseRowMatrix::SymSparseRowMatrix(int nnrows, int nncols,
				       int* rows, int* columns,
				       int nnz)
: Matrix(Matrix::symmetric, Matrix::symsparse), nnrows(nnrows),
  nncols(nncols), rows(rows), columns(columns), nnz(nnz)
{
    a=scinew double[nnz];
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

double& SymSparseRowMatrix::get(int i, int j)
{
    int row_idx=rows[i];
    int next_idx=rows[i+1];
    int l=row_idx;
    int h=next_idx-1;
    while(1){
	if(h<l){
#if 0
	    cerr << "column " << j << " not found in row: ";
	    for(int idx=row_idx;idx<next_idx;idx++)
		cerr << columns[idx] << " ";
	    cerr << endl;
	    ASSERT(0);
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

int SymSparseRowMatrix::nrows()
{
    return nnrows;
}

int SymSparseRowMatrix::ncols()
{
    return nncols;
}

double SymSparseRowMatrix::density()
{	
    return (1.*nnz)/(1.*nnrows*nncols);
}

double SymSparseRowMatrix::minValue() {
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

double SymSparseRowMatrix::maxValue() {
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

void SymSparseRowMatrix::getRowNonzeros(int r, Array1<int>& idx, 
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
    
void SymSparseRowMatrix::zero()
{
    double* ptr=a;
    for(int i=0;i<nnz;i++)
	*ptr++=0.0;
}

void SymSparseRowMatrix::solve(ColumnMatrix&)
{
    EXCEPTION(General("SymSparseRowMatrix can't do a direct solve!"));
}

extern "C" void ssmult(int beg, int end, int* rows, int* columns,
		       double* a, double* xp, double* bp);

void SymSparseRowMatrix::mult(const ColumnMatrix& x, ColumnMatrix& b,
			      int& flops, int& memrefs, int beg, int end)
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

void SymSparseRowMatrix::mult_transpose(const ColumnMatrix& x, ColumnMatrix& b,
					int& flops, int& memrefs,
					int beg, int end)
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

void SymSparseRowMatrix::print()
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
}

