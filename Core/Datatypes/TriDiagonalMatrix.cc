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

#include <Core/Datatypes/TriDiagonalMatrix.h>
#include <Core/Util/NotFinished.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Math/LinAlg.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/MiscMath.h>

namespace SCIRun {

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

} // End namespace SCIRun
