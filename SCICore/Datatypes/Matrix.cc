//static char *id="@(#) $Id$";

/*
 *  Matrix.cc: Matrix definitions
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <CoreDatatypes/Matrix.h>
#include <CoreDatatypes/ColumnMatrix.h>
#include <Util/Assert.h>
#include <Util/NotFinished.h>
#include <Containers/String.h>

namespace SCICore {
namespace CoreDatatypes {

PersistentTypeID Matrix::type_id("Matrix", "Datatype", 0);

int Matrix::is_symmetric() {
    return sym==symmetric;
}

void Matrix::is_symmetric(int symm) {
    if (symm)
	sym=symmetric;
    else 
	sym=non_symmetric;
}

Matrix::Matrix(Sym sym, Representation rep)
: sym(sym), rep(rep), extremaCurrent(0)
{
}

Matrix::~Matrix()
{
}

clString Matrix::getType()
{
    if (rep==symsparse)
	return ("symsparse");
    else if (rep==dense)
	return ("dense");
    else if (rep==sparse)
	return ("sparse");
    else
	return ("unknown");
}

SparseRowMatrix* Matrix::getSparseRow()
{
    if (rep==sparse)
	return (SparseRowMatrix*)this;
    else
	return 0;
}

SymSparseRowMatrix* Matrix::getSymSparseRow()
{
    if (rep==symsparse)
	return (SymSparseRowMatrix*)this;
    else
	return 0;
}

DenseMatrix* Matrix::getDense()
{
    if (rep==dense)
	return (DenseMatrix*)this;
    else
	return 0;
}

Matrix* Matrix::clone()
{
    return 0;
}

#define MATRIX_VERSION 1

void Matrix::io(Piostream& stream)
{
    stream.begin_class("Matrix", MATRIX_VERSION);
    int* tmpsym=(int*)&sym;
    stream.io(*tmpsym);
    stream.end_class();
}

void Mult(ColumnMatrix& result, const Matrix& mat, const ColumnMatrix& v)
{
    int flops, memrefs;
    mat.mult(v, result, flops, memrefs);
}

} // End namespace CoreDatatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:23  mcq
// Initial commit
//
// Revision 1.1  1999/04/25 04:07:09  dav
// Moved files into CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:51  dav
// Import sources
//
//
