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
 *  MatrixOperations.cc: Matrix Operations
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   August 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/MatrixOperations.h>


namespace SCIRun {


MatrixHandle
operator+(MatrixHandle A, MatrixHandle B)
{
  ASSERT(A.get_rep());
  ASSERT(B.get_rep());

  ASSERTEQ(A->ncols(), B->ncols());
  ASSERTEQ(A->nrows(), B->nrows());

  if (A->ncols() == 1)
  {
    ColumnMatrix *ac = A->column();
    ColumnMatrix *bc = B->column();
    ColumnMatrix *cc = scinew ColumnMatrix(ac->nrows());
    Add(*cc, *ac, *bc);
    // TODO: clean up leaked objects here and in LinAlgBinary
    return cc;
  }
  else if (dynamic_cast<SparseRowMatrix *>(A.get_rep()) &&
	   dynamic_cast<SparseRowMatrix *>(B.get_rep()))
  {
    SparseRowMatrix *as = A->sparse();
    SparseRowMatrix *bs = B->sparse();
    SparseRowMatrix *cs = AddSparse(*as, *bs);
    // TODO: clean up leaked objects here and in LinAlgBinary
    return cs;
  }
  else
  {
    DenseMatrix *ad = A->dense();
    DenseMatrix *bd = B->dense();
    DenseMatrix *cd = scinew DenseMatrix(ad->nrows(), bd->ncols());
    Add(*cd, *ad, *bd);
    // TODO: clean up leaked objects here and in LinAlgBinary
    return cd;
  }
}



MatrixHandle
operator*(MatrixHandle A, MatrixHandle B)
{
  ASSERT(A.get_rep());
  ASSERT(B.get_rep());

  ASSERTEQ(A->ncols(), B->nrows());

  DenseMatrix *ad = A->dense();
  DenseMatrix *bd = B->dense();
  DenseMatrix *cd = scinew DenseMatrix(ad->nrows(), bd->ncols());
    // TODO: clean up leaked objects here and in LinAlgBinary
  Mult(*cd, *ad, *bd);
  return cd;
}



} // End namespace SCIRun
