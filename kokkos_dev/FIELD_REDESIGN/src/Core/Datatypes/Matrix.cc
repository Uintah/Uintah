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

#include <SCICore/Datatypes/Matrix.h>
#include <SCICore/Datatypes/ColumnMatrix.h>
#include <SCICore/Util/Assert.h>
#include <SCICore/Containers/String.h>

namespace SCICore {
namespace Datatypes {

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
: sym(sym), rep(rep), extremaCurrent(0), separate_raw(0), raw_filename("")
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

} // End namespace Datatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.4.2.5  2000/11/01 23:03:01  mcole
// Fix for previous merge from trunk
//
// Revision 1.4.2.3  2000/10/26 17:30:45  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.5  2000/07/12 15:45:08  dmw
// Added Yarden's raw output thing to matrices, added neighborhood accessors to meshes, added ScalarFieldRGushort
//
// Revision 1.4  1999/08/25 03:48:34  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.3  1999/08/19 23:18:05  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.2  1999/08/17 06:38:47  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:23  mcq
// Initial commit
//
// Revision 1.1  1999/04/25 04:07:09  dav
// Moved files into Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:51  dav
// Import sources
//
//
