//static char *id="@(#) $Id$";

/*
 *  TriDiagonalMatrix.cc: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <SCICore/Datatypes/TriDiagonalMatrix.h>
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Datatypes/ColumnMatrix.h>
#include <SCICore/Math/LinAlg.h>
#include <SCICore/Math/MinMax.h>
#include <SCICore/Math/MiscMath.h>

namespace SCICore {
namespace Datatypes {

TriDiagonalMatrix::TriDiagonalMatrix(int rows)
: Matrix(non_symmetric, tridiagonal), rows(rows)
{
    data=new TriRow[rows];
}

TriDiagonalMatrix::~TriDiagonalMatrix()
{
    if(data)
	delete[] data;
}

void TriDiagonalMatrix::setrow(int i, double l, double m, double r)
{
    ASSERTRANGE(i, 0, rows);
    data[i][0]=l;
    data[i][1]=m;
    data[i][2]=r;
}

double& TriDiagonalMatrix::get(int r, int c)
{
    ASSERTRANGE(r, 0, rows);
    int off=c-r+1;
    ASSERTRANGE(off, 0, 2);
    return data[r][off];
}

void TriDiagonalMatrix::zero()
{
    for(int i=0;i<rows;i++)
	data[i][0]=data[i][1]=data[i][2]=0.0;
}

int TriDiagonalMatrix::nrows() const
{
    return rows;
}

int TriDiagonalMatrix::ncols() const
{
    return rows;
}

void TriDiagonalMatrix::getRowNonzeros(int r, Array1<int>& idx, Array1<double>& v)
{
    if(r>0){
	idx.add(r-1);
	v.add(data[r][0]);
    }
    idx.add(r);
    v.add(data[r][1]);
    if(r<rows-1){
	idx.add(r+1);
	v.add(data[r][2]);
    }
}

double TriDiagonalMatrix::minValue()
{
    using SCICore::Math::Min;

    double min=data[0][0];
    for(int i=0;i<rows;i++){
	min=Min(data[i][0], min);
	min=Min(data[i][1], min);
	min=Min(data[i][2], min);
    }
    return min;
}

double TriDiagonalMatrix::maxValue()
{
    using SCICore::Math::Max;

    double max=data[0][0];
    for(int i=0;i<rows;i++){
	max=Max(data[i][0], max);
	max=Max(data[i][1], max);
	max=Max(data[i][2], max);
    }
    return max;
}

void TriDiagonalMatrix::mult(const ColumnMatrix& x, ColumnMatrix& b,
			     int& flops, int& memrefs, int, int, int) const
{
    ASSERTEQ(rows, x.nrows());
    ASSERTEQ(rows, b.nrows());
    b[0]=x[0]*data[0][1]+x[1]*data[0][2];
    for(int i=1;i<rows-1;i++){
	b[i]=x[i-1]*data[i][0]+x[i]*data[i][1]+x[i+1]*data[i][2];
    }
    b[rows-1]=x[rows-2]*data[rows-1][0]+x[rows-1]*data[rows-1][1];
    flops+=5*(rows-2)+6;
    memrefs+=6*(rows-2)+8;
}

void TriDiagonalMatrix::mult_transpose(const ColumnMatrix&, ColumnMatrix&,
				       int&, int&, int, int, int)
{
    NOT_FINISHED("TriDiagonal::mult_transpose");
}


void TriDiagonalMatrix::solve(ColumnMatrix& cc)
{
#if 0
    {
	cerr << "Tridiagonal matrix:\n";
	for(int i=0;i<rows;i++)
	    cerr << data[i][0] << "\t" << data[i][1] << "\t" << data[i][2] << endl;
	cerr << "c:\n";
	for(i=0;i<rows;i++)
	    cerr << c[i] << endl;
    }
#endif
    double* c=&cc[0];
    linalg_tridiag(rows, data, c);
#if 0
    for(int i=1;i<rows;i++){
	//ASSERT(Abs(data[i-1][1]) > 1.e-10);
	double factor=data[i][0]/data[i-1][1];

	data[i][1] -= factor*data[i-1][2];
	c[i] -= factor*c[i-1];
    }
    //ASSERT(Abs(data[rows-1][1]) > 1.e-10);
    c[rows-1] = c[rows-1]/data[rows-1][1];
    for(i=rows-2;i>=0;i--){
	c[i] = (c[i]-data[i][2]*c[i+1])/data[i][1];
    }
#endif
#if 0
    {
	cerr << "c:\n";
	for(int i=0;i<rows;i++)
	    cerr << c[i] << endl;
    }
#endif
}

} // End namespace Datatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.4  2000/07/12 15:45:10  dmw
// Added Yarden's raw output thing to matrices, added neighborhood accessors to meshes, added ScalarFieldRGushort
//
// Revision 1.3  1999/08/25 03:48:43  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:38:56  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:30  mcq
// Initial commit
//
// Revision 1.1  1999/04/25 04:07:19  dav
// Moved files into Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:48  dav
// Import sources
//
//
