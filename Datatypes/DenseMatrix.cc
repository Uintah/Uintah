
#include <Datatypes/DenseMatrix.h>
#include <Classlib/Assert.h>
#include <Datatypes/ColumnMatrix.h>
#include <Malloc/Allocator.h>
#include <Math/MiscMath.h>
#include <iostream.h>

DenseMatrix::DenseMatrix(int r, int c)
: Matrix(Matrix::non_symmetric)
{
    ASSERT(r>0);
    ASSERT(c>0);
    nr=r;
    nc=c;
    data=scinew double*[nr];
    double* tmp=scinew double[nr*nc];
    dataptr=tmp;
    for(int i=0;i<nr;i++){
	data[i]=tmp;
	tmp+=nc;
    }
}

DenseMatrix::~DenseMatrix()
{
    delete[] dataptr;
    delete[] data;
}

DenseMatrix::DenseMatrix(const DenseMatrix& m)
: Matrix(Matrix::non_symmetric)
{
    nc=m.nc;
    nr=m.nr;
    data=scinew double*[nr];
    double* tmp=scinew double[nr*nc];
    dataptr=tmp;
    for(int i=0;i<nr;i++){
	data[i]=tmp;
	double* p=m.data[i];
	for(int j=0;j<nc;j++){
	    *tmp++=*p++;
	}
    }
}

DenseMatrix& DenseMatrix::operator=(const DenseMatrix& m)
{
    delete[] dataptr;
    delete[] data;
    nc=m.nc;
    nr=m.nr;
    data=scinew double*[nr];
    double* tmp=scinew double[nr*nc];
    dataptr=tmp;
    for(int i=0;i<nr;i++){
	data[i]=tmp;
	double* p=m.data[i];
	for(int j=0;j<nc;j++){
	    *tmp++=*p++;
	}
    }
    return *this;
}

double& DenseMatrix::get(int r, int c)
{
    ASSERT(r>=0 && r<nr);
    ASSERT(c>=0 && c<nc);
    return data[r][c];
}

void DenseMatrix::put(int r, int c, const double& d)
{
    ASSERT(r>=0 && r<nr);
    ASSERT(c>=0 && c<nc);
    data[r][c]=d;
}

int DenseMatrix::nrows()
{
    return nr;
}

int DenseMatrix::ncols()
{
    return nc;
}

void DenseMatrix::zero()
{
    for(int r=0;r<nr;r++){
	double* row=data[r];
	for(int c=0;c<nc;c++){
	    row[c]=0.0;
	}
    }
}

void DenseMatrix::solve(ColumnMatrix& sol)
{
    ASSERT(nr==nc);
    ASSERT(sol.nrows()==nc);

    // Gauss-Jordan with partial pivoting
    int i;
    for(i=0;i<nr;i++){
	double max=Abs(data[i][i]);
	int row=i;
	int j;
	for(j=i+1;j<nr;j++){
	    if(Abs(data[j][i]) > max){
		max=Abs(data[j][i]);
		row=j;
	    }
	}
	ASSERT(Abs(max) > 1.e-12);
	if(row != i){
	    // Switch rows (actually their pointers)
	    double* tmp=data[i];
	    data[i]=data[row];
	    data[row]=tmp;
	    double dtmp=sol[i];
	    sol[i]=sol[row];
	    sol[row]=dtmp;
	}
	double denom=1./data[i][i];
	double* r1=data[i];
	double s1=sol[i];
	for(j=i+1;j<nr;j++){
	    double factor=data[j][i]*denom;
	    double* r2=data[j];
	    for(int k=i;k<nr;k++)
		r2[k]-=factor*r1[k];
	    sol[j]-=factor*s1;
	}
    }

    // Back-substitution
    for(i=1;i<nr;i++){
	ASSERT(Abs(data[i][i]) > 1.e-12);
	double denom=1./data[i][i];
	double* r1=data[i];
	double s1=sol[i];
	for(int j=0;j<i;j++){
	    double factor=data[j][i]*denom;
	    double* r2=data[j];
	    for(int k=i;k<nr;k++)
		r2[k]-=factor*r1[k];
	    sol[j]-=factor*s1;
	}
    }

    // Normalize
    for(i=0;i<nr;i++){
	ASSERT(Abs(data[i][i]) > 1.e-12);
	double factor=1./data[i][i];
	for(int j=0;j<nr;j++)
	    data[i][j]*=factor;
	sol[i]*=factor;
    }
}

void DenseMatrix::mult(ColumnMatrix& x, ColumnMatrix& b,
		       int beg, int end)
{
    // Compute A*x=b
    ASSERT(nr == nc);
    ASSERT(x.nrows() == nr);
    ASSERT(b.nrows() == nr);
    if(beg==-1)beg=0;
    if(end==-1)end=nr;
    for(int i=beg;i<end;i++){
	double sum=0;
	double* row=data[i];
	for(int j=0;j<nc;j++){
	    sum+=row[j]*x[j];
	}
	b[i]=sum;
    }
}
    
void DenseMatrix::mult_transpose(ColumnMatrix& x, ColumnMatrix& b,
				 int beg, int end)
{
    // Compute At*x=b
    ASSERT(nr == nc);
    ASSERT(x.nrows() == nr);
    ASSERT(b.nrows() == nr);
    if(beg==-1)beg=0;
    if(end==-1)end=nc;
    for(int i=beg;i<end;i++){
	double sum=0;
	for(int j=0;j<nr;j++){
	    sum+=data[j][i]*x[j];
	}
	b[i]=sum;
    }
}

void DenseMatrix::print()
{
    cerr << "Dense Matrix: " << nr << " by " << nc << endl;
    for(int i=0;i<nr;i++){
	for(int j=0;j<nc;j++){
	    cerr << data[i][j] << "\t";
	}
	cerr << endl;
    }
}

MatrixRow DenseMatrix::operator[](int row)
{
    return MatrixRow(this, row);
}
