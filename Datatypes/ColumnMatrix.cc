
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
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Malloc/Allocator.h>
#include <Math/Expon.h>
#include <iostream.h>

static Persistent* maker()
{
    return scinew ColumnMatrix(0);
}

PersistentTypeID ColumnMatrix::type_id("ColumnMatrix", "Datatype", maker);

ColumnMatrix::ColumnMatrix(int rows)
: rows(rows)
{
    if(rows)
	data=scinew double[rows];
    else
	data=0;
}

ColumnMatrix::ColumnMatrix(const ColumnMatrix& c)
: rows(c.rows)
{
    if(rows){
	data=scinew double[rows];
	for(int i=0;i<rows;i++)
	    data[i]=c.data[i];
    } else {
	data=0;
    }
}

ColumnMatrix* ColumnMatrix::clone() const
{
    return scinew ColumnMatrix(*this);
}

ColumnMatrix& ColumnMatrix::operator=(const ColumnMatrix& c)
{
    if(rows != c.rows){
	if(data)delete[] data;
	rows=c.rows;
	data=scinew double[rows];
    }
    for(int i=0;i<rows;i++)
	data[i]=c.data[i];
    return *this;
}

ColumnMatrix::~ColumnMatrix()
{
    if(data)
	delete[] data;
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

#define COLUMNMATRIX_VERSION 1

void ColumnMatrix::io(Piostream& stream)
{
    stream.begin_class("ColumnMatrix", COLUMNMATRIX_VERSION);

    stream.io(rows);
    if(stream.reading()){
	data=new double[rows];
    }
    int i;
    for(i=0;i<rows;i++)
	stream.io(data[i]);
    stream.end_class();
}

#ifdef __GNUG__

#include <Classlib/LockingHandle.cc>

template class LockingHandle<ColumnMatrix>;

#endif
