
#include <Datatypes/SymSparseRowMatrix.h>
#include <Math/MiscMath.h>
#include <Math/MinMax.h>
#include <Classlib/Assert.h>
#include <Classlib/Exceptions.h>
#include <Datatypes/ColumnMatrix.h>
#include <iostream.h>

SymSparseRowMatrix::SymSparseRowMatrix(int nnrows, int nncols,
				       Array1<int>& in_rows,
				       Array1<int>& in_cols)
: Matrix(Matrix::symmetric), nnrows(nnrows), nncols(nncols)
{
    nnz=in_cols.size();
    a=new double[nnz];
    columns=new int[nnz];
    rows=new int[nnz];
    for(int i=0;i<in_rows.size();i++){
	rows[i]=in_rows[i];
    }
    for(i=0;i<in_cols.size();i++){
	columns[i]=in_cols[i];
    }
}

SymSparseRowMatrix::~SymSparseRowMatrix()
{
    delete[] a;
    delete[] columns;
    delete[] rows;
}

double& SymSparseRowMatrix::get(int i, int j)
{
    int row_idx=rows[i];
    int next_idx=rows[i+1];
    int l=row_idx;
    int h=next_idx-1;
    while(1){
	int m=(l+h)/2;
	if(j<columns[m]){
	    h=m-1;
	} else if(j>columns[m]){
	    l=m+1;
	} else {
	    return a[m];
	}
	if(h<l){
	    cerr << "column " << j << " not found in row: ";
	    for(int idx=row_idx;idx<next_idx;idx++)
		cerr << columns[idx] << " ";
	    cerr << endl;
	    ASSERT(0);
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

void SymSparseRowMatrix::mult(ColumnMatrix& x, ColumnMatrix& b,
			      int beg, int end)
{
    // Compute A*x=b
    ASSERT(x.nrows() == nnrows);
    ASSERT(b.nrows() == nnrows);
    if(beg==-1)beg=0;
    if(end==-1)end=nnrows;
    for(int i=beg;i<end;i++){
	double sum=0;
	int row_idx=rows[i];
	int next_idx=rows[i+1];
	for(int j=row_idx;j<next_idx;j++){
	    sum+=a[j]*x[columns[j]];
	}
	b[i]=sum;
    }
}

void SymSparseRowMatrix::mult_transpose(ColumnMatrix& x, ColumnMatrix& b,
					int beg, int end)
{
    // Compute At*x=b
    // This is the same as Ax=b since the matrix is symmetric
    ASSERT(x.nrows() == nnrows);
    ASSERT(b.nrows() == nnrows);
    if(beg==-1)beg=0;
    if(end==-1)end=nnrows;
    for(int i=beg;i<end;i++){
	double sum=0;
	int row_idx=rows[i];
	int next_idx=rows[i+1];
	for(int j=row_idx;j<next_idx;j++){
	    sum+=a[j]*x[columns[j]];
	}
	b[i]=sum;
    }
}

void SymSparseRowMatrix::print()
{
    cerr << "Sparse RowMatrix: " << endl;
}

