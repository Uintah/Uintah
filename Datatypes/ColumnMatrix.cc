
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
#include <Math/LinAlg.h>
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

int ColumnMatrix::nrows() const
{
    return rows;
}

void ColumnMatrix::zero()
{
    for(int i=0;i<rows;i++)
	data[i]=0.0;
}

extern "C" double cm_vnorm(int beg, int end, double* data);
extern "C" double dnrm2(int n, double* x, int incx);

double ColumnMatrix::vector_norm()
{
//    double norm=Sqrt(cm_vnorm(0, rows, data));
//    double norm=dnrm2(rows, data, 1);
    double norm=linalg_norm2(rows, data);
    return norm; 
}

double ColumnMatrix::vector_norm(int& flops, int& memrefs)
{
    flops+=rows*2;
    memrefs+=rows*sizeof(double);
    return vector_norm();
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
	data=scinew double[rows];
    }
    int i;
    for(i=0;i<rows;i++)
	stream.io(data[i]);
    stream.end_class();
}

void Mult(ColumnMatrix& result, const ColumnMatrix& a, const ColumnMatrix& b)
{
    ASSERT(result.rows == a.rows && result.rows == b.rows);
    linalg_mult(result.rows, result.data, a.data, b.data);
}

void Mult(ColumnMatrix& result, const ColumnMatrix& a, const ColumnMatrix& b,
	  int& flops, int& memrefs)
{
    ASSERT(result.rows == a.rows && result.rows == b.rows);
    linalg_mult(result.rows, result.data, a.data, b.data);
    flops+=result.rows;
    memrefs+=result.rows*3*sizeof(double);
}

void Sub(ColumnMatrix& result, const ColumnMatrix& a, const ColumnMatrix& b)
{
    ASSERT(result.rows == a.rows && result.rows == b.rows);
    linalg_sub(result.rows, result.data, a.data, b.data);
}

void Sub(ColumnMatrix& result, const ColumnMatrix& a, const ColumnMatrix& b,
	  int& flops, int& memrefs)
{
    ASSERT(result.rows == a.rows && result.rows == b.rows);
    linalg_sub(result.rows, result.data, a.data, b.data);
    flops+=result.rows;
    memrefs+=result.rows*3*sizeof(double);
}

void ScMult_Add(ColumnMatrix& result, double s, const ColumnMatrix& a,
		const ColumnMatrix& b)
{
    ASSERT(result.rows == a.rows && result.rows == b.rows);
    linalg_smadd(result.rows, result.data, s, a.data, b.data);
}

void ScMult_Add(ColumnMatrix& result, double s, const ColumnMatrix& a,
		const ColumnMatrix& b, int& flops, int& memrefs)
{
    ASSERT(result.rows == a.rows && result.rows == b.rows);
    linalg_smadd(result.rows, result.data, s, a.data, b.data);
    flops+=result.rows*2;
    memrefs+=result.rows*3*sizeof(double);
}

double Dot(const ColumnMatrix& a, const ColumnMatrix& b)
{
    ASSERT(a.rows == b.rows);
    return linalg_dot(a.rows, a.data, b.data);
}

double Dot(const ColumnMatrix& a, const ColumnMatrix& b,
	   int& flops, int& memrefs)
{
    ASSERT(a.rows == b.rows);
    flops+=a.rows*2;
    memrefs+=2*sizeof(double)*a.rows;
    return linalg_dot(a.rows, a.data, b.data);
}

#ifdef __GNUG__

#include <Classlib/LockingHandle.cc>

template class LockingHandle<ColumnMatrix>;

#endif
