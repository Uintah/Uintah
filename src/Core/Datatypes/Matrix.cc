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

#define MATRIX_VERSION 2

void
Matrix::io(Piostream& stream)
{
  int version = stream.begin_class("Matrix", MATRIX_VERSION);
  if (version < 2) {
    int tmpsym;
    stream.io(tmpsym);
  }
  stream.end_class();
}

SCICORESHARE Matrix *Add(Matrix *A, Matrix *B) {
  ASSERT(A->nrows() == B->nrows());
  ASSERT(A->ncols() == B->ncols());
  ASSERTFAIL("Matrix addition not yet implemented.");
  return A;
  /*
  SparseMatrix *s;
  DenseMatrix *d;
  ColumnMatrix *c;
  if (d=dynamic_cast<DenseMatrix *>(A)) return d->add(B->dense());
  else if (d=dynamic_cast<DenseMatrix *>(B)) return d->add(A->dense());
  else if (c=dynamic_cast<ColumnMatrix *>(A)) return c->add(B->column());
  else if (c=dynamic_cast<ColumnMatrix *>(B)) return c->add(A->column());
  else if (s=dynamic_cast<SparseRowMatrix *>(A)) return s->add(B->sparse());
  else (s=dynamic_cast<SparseRowMatrix *>(B)) return s->add(A->sparse());
  */
}

SCICORESHARE Matrix *Mult(Matrix *A, Matrix *B) {
  ASSERT(A->ncols() == B->nrows());
  ASSERTFAIL("Matrix multiplication not yet implemented.");
  return A;
}

void
Mult(ColumnMatrix& result, const Matrix& mat, const ColumnMatrix& v)
{
  int flops, memrefs;
  mat.mult(v, result, flops, memrefs);
}

} // End namespace SCIRun
