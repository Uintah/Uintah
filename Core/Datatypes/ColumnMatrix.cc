//static char *id="@(#) $Id$";

/*
 *  ColumnMatrix.cc: for RHS and LHS
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCICore/CoreDatatypes/ColumnMatrix.h>
#include <SCICore/Util/Assert.h>
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Math/Expon.h>
#include <SCICore/Math/LinAlg.h>
#include <iostream.h>

namespace SCICore {
namespace CoreDatatypes {

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

void ColumnMatrix::resize(int new_rows)
{
    if(data)
	delete[] data;
    if(new_rows)
	data=new double[new_rows];
    else
	data=0;
    rows=new_rows;
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

double ColumnMatrix::vector_norm(int& flops, int& memrefs, int beg, int end)
{
    ASSERTRANGE(end, 0, rows+1);
    ASSERTRANGE(beg, 0, end);
    flops+=(end-beg)*2;
    memrefs+=(end-beg)*sizeof(double);
    return linalg_norm2((end-beg), data+beg);
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
    ASSERTEQ(result.rows, a.rows);
    ASSERTEQ(result.rows, b.rows);
    linalg_mult(result.rows, result.data, a.data, b.data);
}

void Mult(ColumnMatrix& result, const ColumnMatrix& a, const ColumnMatrix& b,
	  int& flops, int& memrefs)
{
    ASSERTEQ(result.rows, a.rows);
    ASSERTEQ(result.rows, b.rows);
    linalg_mult(result.rows, result.data, a.data, b.data);
    flops+=result.rows;
    memrefs+=result.rows*3*sizeof(double);
}

void Mult(ColumnMatrix& result, const ColumnMatrix& a, const ColumnMatrix& b,
	  int& flops, int& memrefs, int beg, int end)
{
    ASSERTEQ(result.rows, a.rows);
    ASSERTEQ(result.rows, b.rows);
    ASSERTRANGE(end, 0, result.rows+1);
    ASSERTRANGE(beg, 0, end);
    linalg_mult(end-beg, result.data+beg, a.data+beg, b.data+beg);
    flops+=(end-beg);
    memrefs+=(end-beg)*3*sizeof(double);
}

void Sub(ColumnMatrix& result, const ColumnMatrix& a, const ColumnMatrix& b)
{
    ASSERTEQ(result.rows, a.rows);
    ASSERTEQ(result.rows, b.rows);
    linalg_sub(result.rows, result.data, a.data, b.data);
}

void Sub(ColumnMatrix& result, const ColumnMatrix& a, const ColumnMatrix& b,
	  int& flops, int& memrefs)
{
    ASSERTEQ(result.rows, a.rows);
    ASSERTEQ(result.rows, b.rows);
    linalg_sub(result.rows, result.data, a.data, b.data);
    flops+=result.rows;
    memrefs+=result.rows*3*sizeof(double);
}

void ScMult_Add(ColumnMatrix& result, double s, const ColumnMatrix& a,
		const ColumnMatrix& b)
{
    ASSERTEQ(result.rows, a.rows);
    ASSERTEQ(result.rows, b.rows);
    linalg_smadd(result.rows, result.data, s, a.data, b.data);
}

void ScMult_Add(ColumnMatrix& result, double s, const ColumnMatrix& a,
		const ColumnMatrix& b, int& flops, int& memrefs)
{
    ASSERTEQ(result.rows, a.rows);
    ASSERTEQ(result.rows, b.rows);
    linalg_smadd(result.rows, result.data, s, a.data, b.data);
    flops+=result.rows*2;
    memrefs+=result.rows*3*sizeof(double);
}

void ScMult_Add(ColumnMatrix& result, double s, const ColumnMatrix& a,
		const ColumnMatrix& b, int& flops, int& memrefs,
		int beg, int end)
{
    ASSERTEQ(result.rows, a.rows);
    ASSERTEQ(result.rows, b.rows);
    ASSERTRANGE(end, 0, result.rows+1);
    ASSERTRANGE(beg, 0, end);
    linalg_smadd(end-beg, result.data+beg, s, a.data+beg, b.data+beg);
    flops+=(end-beg)*2;
    memrefs+=(end-beg)*3*sizeof(double);
}

double Dot(const ColumnMatrix& a, const ColumnMatrix& b)
{
    ASSERTEQ(a.rows, b.rows);
    return linalg_dot(a.rows, a.data, b.data);
}

double Dot(const ColumnMatrix& a, const ColumnMatrix& b,
	   int& flops, int& memrefs)
{
    ASSERTEQ(a.rows, b.rows);
    flops+=a.rows*2;
    memrefs+=2*sizeof(double)*a.rows;
    return linalg_dot(a.rows, a.data, b.data);
}

double Dot(const ColumnMatrix& a, const ColumnMatrix& b,
	   int& flops, int& memrefs, int beg, int end)
{
    ASSERTEQ(a.rows, b.rows);
    ASSERTRANGE(end, 0, a.rows+1);
    ASSERTRANGE(beg, 0, end);
    flops+=(end-beg)*2;
    memrefs+=2*sizeof(double)*(end-beg);
    return linalg_dot((end-beg), a.data+beg, b.data+beg);
}

void Copy(ColumnMatrix& out, const ColumnMatrix& in)
{
    ASSERTEQ(out.rows, in.rows);
    for(int i=0;i<out.rows;i++)
	out.data[i]=in.data[i];
}

void Copy(ColumnMatrix& out, const ColumnMatrix& in, int&, int& refs,
	  int beg, int end)
{
    ASSERTEQ(out.rows, in.rows);
    ASSERTRANGE(end, 0, out.rows+1);
    ASSERTRANGE(beg, 0, end);
    for(int i=beg;i<end;i++)
	out.data[i]=in.data[i];
    refs+=sizeof(double)*(end-beg);
}

void AddScMult(ColumnMatrix& result, const ColumnMatrix& a,
	       double s, const ColumnMatrix& b)
{
    ASSERTEQ(result.rows, a.rows);
    ASSERTEQ(result.rows, b.rows);
    linalg_smadd(result.rows, result.data, s, b.data, a.data);
}

void Add(ColumnMatrix& result, const ColumnMatrix& a, const ColumnMatrix& b)
{
    ASSERTEQ(result.rows, a.rows);
    ASSERTEQ(result.rows, b.rows);
    linalg_add(result.rows, result.data, a.data, b.data);
}

void Add(ColumnMatrix& result, const ColumnMatrix& a, const ColumnMatrix& b,
	 const ColumnMatrix& c)
{
    ASSERTEQ(result.rows, a.rows);
    ASSERTEQ(result.rows, b.rows);
    ASSERTEQ(result.rows, c.rows);
    for(int i=0;i<result.rows;i++)
	result.data[i]=a.data[i]+b.data[i]+c.data[i];
}

void Mult(ColumnMatrix& result, const ColumnMatrix& a, double s)
{
    ASSERTEQ(result.rows, a.rows);
    for(int i=0;i<result.rows;i++)
	result.data[i]=a.data[i]*s;
}

} // End namespace CoreDatatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:38:44  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:19  mcq
// Initial commit
//
// Revision 1.1  1999/04/25 04:07:05  dav
// Moved files into CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:50  dav
// Import sources
//
//

