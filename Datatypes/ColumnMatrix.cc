
/*
 *  ColumnMatrix.h: for RHS and LHS
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Datatypes/ColumnMatrix.h>
#include <Classlib/Assert.h>
#include <Math/Expon.h>
#include <iostream.h>

ColumnMatrix::ColumnMatrix(int rows)
: rows(rows)
{
    data=new double[rows];
}

ColumnMatrix::ColumnMatrix(const ColumnMatrix& c)
: rows(c.rows)
{
    data=new double[rows];
    for(int i=0;i<rows;i++)
	data[i]=c.data[i];
}

ColumnMatrix& ColumnMatrix::operator=(const ColumnMatrix& c)
{
    if(rows != c.rows){
	if(data)delete[] data;
	rows=c.rows;
	data=new double[rows];
    }
    for(int i=0;i<rows;i++)
	data[i]=c.data[i];
    return *this;
}

ColumnMatrix::~ColumnMatrix()
{
    if(data)delete[] data;
}

int ColumnMatrix::nrows()
{
    return rows;
}

void ColumnMatrix::zero()
{
    for(int i=0;i<rows;i++)
	data[i]=0.0;
}

double ColumnMatrix::vector_norm()
{
    double norm=0;
    for(int i=0;i<rows;i++){
	double d=data[i];
	norm+=d*d;
    }
    return Sqrt(norm);
}

void ColumnMatrix::print(ostream& str)
{
    str << "Column Matrix: " << rows << endl;
    for(int i=0;i<rows;i++){
	str << data[i] << endl;
    }
}
