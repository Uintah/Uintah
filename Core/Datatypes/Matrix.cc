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

namespace SCIRun {

PersistentTypeID Matrix::type_id("Matrix", "Datatype", 0);

bool
Matrix::is_symmetric()
{
  return sym_==SYMMETRIC;
}


void
Matrix::is_symmetric(bool sym)
{
  sym_ = sym?SYMMETRIC:NON_SYMMETRIC;
}


Matrix::Matrix(Sym sym, Representation rep)
  : separate_raw_(0),
    raw_filename_(""),
    sym_(sym),
    extremaCurrent_(false),
    rep_(rep)
{
}


Matrix::~Matrix()
{
}


const string Matrix::getType() const
{
  switch(rep_)
  {
  case SPARSE:
    return "sparse";
  case SYMSPARSE:
    return "symsparse";
  case DENSE:
    return "dense";
  case TRIDIAGONAL:
    return "tridiagonal";
  case COLUMN:
    return "column";
  case OTHER:
  default:
    return "unknown";
  }
}


SparseRowMatrix *
Matrix::getSparseRow()
{
  if (rep_ == SPARSE)
    return (SparseRowMatrix*)this;
  else
    return 0;
}


SymSparseRowMatrix *
Matrix::getSymSparseRow()
{
  if (rep_ == SYMSPARSE)
    return (SymSparseRowMatrix*)this;
  else
    return 0;
}


DenseMatrix *
Matrix::getDense()
{
  if (rep_ == DENSE)
    return (DenseMatrix*)this;
  else
    return 0;
}


ColumnMatrix *
Matrix::getColumn()
{
  if (rep_ == COLUMN)
    return (ColumnMatrix*)this;
  else
    return 0;
}


Matrix *
Matrix::clone()
{
  return 0;
}

#define MATRIX_VERSION 1

void
Matrix::io(Piostream& stream)
{
  stream.begin_class("Matrix", MATRIX_VERSION);
  int *tmpsym = (int *)&sym_;
  stream.io(*tmpsym);
  stream.end_class();
}

void
Mult(ColumnMatrix& result, const Matrix& mat, const ColumnMatrix& v)
{
  int flops, memrefs;
  mat.mult(v, result, flops, memrefs);
}

} // End namespace SCIRun
