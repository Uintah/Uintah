//static char *id="@(#) $Id$";

/*
 *  DenseMatrix.cc: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <SCICore/Datatypes/DenseMatrix.h>
#include <SCICore/Util/Assert.h>
#include <SCICore/Datatypes/ColumnMatrix.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Math/MiscMath.h>
#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

namespace SCICore {
namespace Datatypes {

static Persistent* maker()
{
    return scinew DenseMatrix;
}

PersistentTypeID DenseMatrix::type_id("DenseMatrix", "Matrix", maker);

DenseMatrix::DenseMatrix()
: Matrix(Matrix::non_symmetric, Matrix::dense), nr(0), nc(0)
{
}

DenseMatrix::DenseMatrix(int r, int c)
: Matrix(Matrix::non_symmetric, Matrix::dense)
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
: Matrix(Matrix::non_symmetric, Matrix::dense)
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
    ASSERTRANGE(r, 0, nr);
    ASSERTRANGE(c, 0, nc);
    return data[r][c];
}

void DenseMatrix::put(int r, int c, const double& d)
{
    ASSERTRANGE(r, 0, nr);
    ASSERTRANGE(c, 0, nc);
    extremaCurrent=0;
    data[r][c]=d;
}

double DenseMatrix::minValue() {
    if (extremaCurrent)
	return minVal;
    minVal=maxVal=data[0][0];
    for (int r=0; r<nr; r++) {
	for (int c=0; c<nr; c++) {
	   if (data[r][c] < minVal)
	       minVal = data[r][c];
	   if (data[r][c] > maxVal)
	       maxVal = data[r][c];
       }
    }
    extremaCurrent=1;
    return minVal;
}

double DenseMatrix::maxValue() {
    if (extremaCurrent)
	return maxVal;
    minVal=maxVal=data[0][0];
    for (int r=0; r<nr; r++) {
	for (int c=0; c<nr; c++) {
	   if (data[r][c] < minVal)
	       minVal = data[r][c];
	   if (data[r][c] > maxVal)
	       maxVal = data[r][c];
       }
    }
    extremaCurrent=1;
    return maxVal;
}

int DenseMatrix::nrows() const
{
    return nr;
}

int DenseMatrix::ncols() const
{
    return nc;
}

void DenseMatrix::getRowNonzeros(int r, Array1<int>& idx, Array1<double>& val)
{
    idx.resize(nc);
    val.resize(nc);
    int i=0;
    for (int c=0; c<nc; c++) {
	if (data[r][c]!=0.0) {
	    idx[i]=c;
	    val[i]=data[r][c];
	    i++;
	}
    }
}
    
void DenseMatrix::zero()
{
    for(int r=0;r<nr;r++){
	double* row=data[r];
	for(int c=0;c<nc;c++){
	    row[c]=0.0;
	}
    }
    extremaCurrent=0;
}

void DenseMatrix::solve(ColumnMatrix& sol)
{
    ASSERT(nr==nc);
    ASSERT(sol.nrows()==nc);

    // Gauss-Jordan with partial pivoting
    int i;
    for(i=0;i<nr;i++){
	cout << "Solve: " << i << " of " << nr << endl;
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
	cout << "Solve: " << i << " of " << nr << endl;
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
	cout << "Solve: " << i << " of " << nr << endl;
	ASSERT(Abs(data[i][i]) > 1.e-12);
	double factor=1./data[i][i];
	for(int j=0;j<nr;j++)
	    data[i][j]*=factor;
	sol[i]*=factor;
    }
}

void DenseMatrix::mult(const ColumnMatrix& x, ColumnMatrix& b,
		       int& flops, int& memrefs, int beg, int end) const
{
    // Compute A*x=b
    ASSERTEQ(x.nrows(), nc);
    ASSERTEQ(b.nrows(), nr);
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
    flops+=(end-beg)*nc*2;
    memrefs+=(end-beg)*nc*2*sizeof(double)+(end-beg)*sizeof(double);
}
    
void DenseMatrix::mult_transpose(const ColumnMatrix& x, ColumnMatrix& b,
				 int& flops, int& memrefs, int beg, int end)
{
    // Compute At*x=b
    ASSERT(x.nrows() == nr);
    ASSERT(b.nrows() == nc);
    if(beg==-1)beg=0;
    if(end==-1)end=nc;
    for(int i=beg;i<end;i++){
	double sum=0;
	for(int j=0;j<nr;j++){
	    sum+=data[j][i]*x[j];
	}
	b[i]=sum;
    }
    flops+=(end-beg)*nr*2;
    memrefs+=(end-beg)*nr*2*sizeof(double)+(end-beg)*sizeof(double);
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

#define DENSEMATRIX_VERSION 1

void DenseMatrix::io(Piostream& stream)
{
    stream.begin_class("DenseMatrix", DENSEMATRIX_VERSION);
    // Do the base class first...
    Matrix::io(stream);

    stream.io(nr);
    stream.io(nc);
    if(stream.reading()){
	data=scinew double*[nr];
	double* tmp=scinew double[nr*nc];
	dataptr=tmp;
	for(int i=0;i<nr;i++){
	    data[i]=tmp;
	    tmp+=nc;
	}
    }
    stream.begin_cheap_delim();
    int idx=0;
    for(int i=0;i<nr;i++)
	for (int j=0; j<nc; j++, idx++)
	    stream.io(dataptr[idx]);
    stream.end_cheap_delim();
    stream.end_class();
}

void DenseMatrix::invert()
{
    ASSERTEQ(nr, nc);
    double** newdata=scinew double*[nr];
    double* tmp=scinew double[nr*nc];
    double* newdataptr=tmp;

    int i;
    for(i=0;i<nr;i++){
	newdata[i]=tmp;
	for(int j=0;j<nr;j++){
	    tmp[j]=0;
	}
	tmp[i]=1;
	tmp+=nc;
    }

    // Gauss-Jordan with partial pivoting
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
            double* ntmp=newdata[i];
            newdata[i]=newdata[row];
	    newdata[row]=ntmp;
        }
        double denom=1./data[i][i];
        double* r1=data[i];
        double* n1=newdata[i];
        for(j=i+1;j<nr;j++){
            double factor=data[j][i]*denom;
            double* r2=data[j];
	    double* n2=newdata[j];
            for(int k=0;k<nr;k++){
                r2[k]-=factor*r1[k];
		n2[k]-=factor*n1[k];
	    }
        }
    }

    // Back-substitution
    for(i=1;i<nr;i++){
        ASSERT(Abs(data[i][i]) > 1.e-12);
        double denom=1./data[i][i];
        double* r1=data[i];
        double* n1=newdata[i];
        for(int j=0;j<i;j++){
            double factor=data[j][i]*denom;
            double* r2=data[j];
	    double* n2=newdata[j];
            for(int k=0;k<nr;k++){
                r2[k]-=factor*r1[k];
		n2[k]-=factor*n1[k];
	    }
        }
    }

    // Normalize
    for(i=0;i<nr;i++){
        ASSERT(Abs(data[i][i]) > 1.e-12);
        double factor=1./data[i][i];
        for(int j=0;j<nr;j++){
            data[i][j]*=factor;
	    newdata[i][j]*=factor;
	}
    }


    delete[] dataptr;
    delete[] data;    
    dataptr=newdataptr;
    data=newdata;
}

void Mult(DenseMatrix& out, const DenseMatrix& m1, const DenseMatrix& m2)
{
    ASSERTEQ(m1.ncols(), m2.nrows());
    ASSERTEQ(out.nrows(), m1.nrows());
    ASSERTEQ(out.ncols(), m2.ncols());
    int nr=out.nrows();
    int nc=out.ncols();
    int ndot=m1.ncols();
    for(int i=0;i<nr;i++){
	double* row=m1.data[i];
	for(int j=0;j<nc;j++){
	    double d=0;
	    for(int k=0;k<ndot;k++){
		d+=row[k]*m2.data[k][j];
	    }
	    out[i][j]=d;
	}
    }
}

void Add(DenseMatrix& out, const DenseMatrix& m1, const DenseMatrix& m2)
{
    ASSERTEQ(m1.ncols(), m2.ncols());
    ASSERTEQ(out.ncols(), m2.ncols());
    ASSERTEQ(m1.nrows(), m2.nrows());
    ASSERTEQ(out.nrows(), m2.nrows());

    int nr=out.nrows();
    int nc=out.ncols();

    for(int i=0;i<nr;i++)
	for (int j=0; j<nc; j++)
	    out[i][j]=m1.data[i][j]+m2.data[i][j];
}

void Mult_trans_X(DenseMatrix& out, const DenseMatrix& m1, const DenseMatrix& m2)
{
    ASSERTEQ(m1.nrows(), m2.nrows());
    ASSERTEQ(out.nrows(), m1.ncols());
    ASSERTEQ(out.ncols(), m2.ncols());
    int nr=out.nrows();
    int nc=out.ncols();
    int ndot=m1.nrows();
    for(int i=0;i<nr;i++){
	for(int j=0;j<nc;j++){
	    double d=0;
	    for(int k=0;k<ndot;k++){
		d+=m1.data[k][i]*m2.data[k][j];
	    }
	    out[i][j]=d;
	}
    }
}

void Mult_X_trans(DenseMatrix& out, const DenseMatrix& m1, const DenseMatrix& m2)
{
    ASSERTEQ(m1.ncols(), m2.ncols());
    ASSERTEQ(out.nrows(), m1.nrows());
    ASSERTEQ(out.ncols(), m2.nrows());
    int nr=out.nrows();
    int nc=out.ncols();
    int ndot=m1.ncols();
    for(int i=0;i<nr;i++){
	double* row=m1.data[i];
	for(int j=0;j<nc;j++){
	    double d=0;
	    for(int k=0;k<ndot;k++){
		d+=row[k]*m2.data[j][k];
	    }
	    out[i][j]=d;
	}
    }
}

void DenseMatrix::mult(double s)
{
    for(int i=0;i<nr;i++){
	double* p=data[i];
	for(int j=0;j<nc;j++){
	    p[j]*=s;
	}
    }
}

} // End namespace Datatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.4  1999/10/07 02:07:31  sparker
// use standard iostreams and complex type
//
// Revision 1.3  1999/08/25 03:48:32  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:38:45  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:21  mcq
// Initial commit
//
// Revision 1.1  1999/04/25 04:07:06  dav
// Moved files into Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:49  dav
// Import sources
//
//
