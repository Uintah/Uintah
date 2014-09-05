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

#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Util/Assert.h>
#include <Core/Containers/String.h>

namespace SCIRun {

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
  : separate_raw(0), raw_filename(""), sym(sym), extremaCurrent(0), rep(rep)
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

ColumnMatrix* Matrix::getColumn()
{
  if (rep==column)
    return (ColumnMatrix*)this;
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

} // End namespace SCIRun
