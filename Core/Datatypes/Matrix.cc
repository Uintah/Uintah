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
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/Assert.h>

namespace SCIRun {

PersistentTypeID Matrix::type_id("Matrix", "PropertyManager", 0);

#define MATRIX_VERSION 3


Matrix::~Matrix()
{
}

void
Matrix::io(Piostream& stream)
{
  int version = stream.begin_class("Matrix", MATRIX_VERSION);
  if (version < 2) {
    int tmpsym;
    stream.io(tmpsym);
  }
  if (version > 2) {
    PropertyManager::io(stream);
  }
  stream.end_class();
}

Transform Matrix::toTransform() {
  Transform t;
  if (nrows() != 4 || ncols() != 4) {
    cerr << "Error - can't make a transform from this matrix.\n";
    return t;
  }
  double dummy[16];
  int cnt=0;
  for (int i=0; i<4; i++) 
    for (int j=0; j<4; j++, cnt++)
      dummy[cnt] = get(i,j);
  t.set(dummy);
  return t;
}
  

void
Mult(ColumnMatrix& result, const Matrix& mat, const ColumnMatrix& v)
{
  int flops, memrefs;
  mat.mult(v, result, flops, memrefs);
}

int 
Matrix::cg_solve(const ColumnMatrix& rhs, ColumnMatrix& lhs) const
{
  double err;
  int niter, flops, memrefs;
  return cg_solve(rhs, lhs, err, niter, flops, memrefs);
}

int
Matrix::cg_solve(const ColumnMatrix& rhs, ColumnMatrix& lhs,
		 double &err, int &niter, 
		 int& flops, int& memrefs,
		 double max_error, int toomany) const
{
  int size=nrows();  
  niter=0;
  flops=0;
  memrefs=0;

  if(toomany == 0) toomany=2*size;

  if (rhs.vector_norm(flops, memrefs) < 0.0000001) {
    lhs=rhs;
    err=0;
    return 1;
  }
        
  ColumnMatrix diag(size), R(size), Z(size), P(size);

  int i;
  for(i=0;i<size;i++) {
    if (Abs(get(i,i)>0.000001)) diag[i]=1./get(i,i);
    else diag[i]=1;
  }
  flops+=size;
  memrefs+=2*size*sizeof(double);

  mult(lhs, R, flops, memrefs);    
  Sub(R, rhs, R, flops, memrefs);
  mult(R, Z, flops, memrefs);

  double bnorm=rhs.vector_norm(flops, memrefs);
  err=R.vector_norm(flops, memrefs)/bnorm;

  if(err == 0) {
    lhs=rhs;
    memrefs+=2*size*sizeof(double);
    return 1;
  } else if (err>1000000) return 0;

  double bkden=0;
  while(niter < toomany){
    if(err < max_error)
      return 1;

    niter++;
    
    // Simple Preconditioning...
    Mult(Z, R, diag, flops, memrefs);
    
    // Calculate coefficient bk and direction vectors p and pp
    double bknum=Dot(Z, R, flops, memrefs);

    if(niter==1){
      Copy(P, Z, flops, memrefs);
    } else {
      double bk=bknum/bkden;
      ScMult_Add(P, bk, P, Z, flops, memrefs);
    }

    // Calculate coefficient ak, new iterate x and new residuals r and rr
    mult(P, Z, flops, memrefs);
    bkden=bknum;
    double akden=Dot(Z, P, flops, memrefs);
    
    double ak=bknum/akden;
    ScMult_Add(lhs, ak, P, lhs, flops, memrefs);
    ScMult_Add(R, -ak, Z, R, flops, memrefs);
    
    err=R.vector_norm(flops, memrefs)/bnorm;
    if (err>1000000) return 0;
  }
  return 0;
}

int 
Matrix::bicg_solve(const ColumnMatrix& rhs, ColumnMatrix& lhs) const
{
  double err;
  int niter, flops, memrefs;
  return bicg_solve(rhs, lhs, err, niter, flops, memrefs);
}

int
Matrix::bicg_solve(const ColumnMatrix& rhs, ColumnMatrix& lhs,
		   double &err, int &niter, 
		   int& flops, int& memrefs,
		   double max_error, int toomany) const
{
  int size=nrows();  
  niter=0;
  flops=0;
  memrefs=0;

  if(toomany == 0) toomany=2*size;

  if (rhs.vector_norm(flops, memrefs) < 0.0000001) {
    lhs=rhs;
    err=0;
    return 1;
  }

  ColumnMatrix diag(size), R(size), R1(size), Z(size), Z1(size), 
    P(size), P1(size);

  int i;
  for(i=0;i<size;i++) {
    if (Abs(get(i,i)>0.000001)) diag[i]=1./get(i,i);
    else diag[i]=1;
  }
  
  flops+=size;
  memrefs+=2*size*sizeof(double);

  mult(lhs, R, flops, memrefs);
  Sub(R, rhs, R, flops, memrefs);

  double bnorm=rhs.vector_norm(flops, memrefs);
  err=R.vector_norm(flops, memrefs)/bnorm;
    
  if(err == 0){
    lhs=rhs;
    memrefs+=2*size*sizeof(double);
    return 1;
  } else {
    if (err>1000000) return 0;
  }

  // BiCG
  Copy(R1, R, flops, memrefs);

  double bkden=0;
  while(niter < toomany){
    if(err < max_error)
      return 1;

    niter++;

    // Simple Preconditioning...
    Mult(Z, R, diag, flops, memrefs);
    // BiCG
    Mult(Z1, R1, diag, flops, memrefs);
    
    // Calculate coefficient bk and direction vectors p and pp
    // BiCG - change R->R1
    double bknum=Dot(Z, R1, flops, memrefs);
    
    // BiCG
    if ( bknum == 0 ) {
      return 1;
    }
    
    if(niter==1){
      Copy(P, Z, flops, memrefs);
      // BiCG
      Copy(P1, Z1, flops, memrefs);
    } else {
      double bk=bknum/bkden;
      ScMult_Add(P, bk, P, Z, flops, memrefs);
      // BiCG
      ScMult_Add(P1, bk, P1, Z1, flops, memrefs);
    }

    // Calculate coefficient ak, new iterate x and new residuals r and rr
    mult(P, Z, flops, memrefs);
    bkden=bknum;

    // BiCG
    mult_transpose(P1, Z1, flops, memrefs);

    // BiCG = change P -> P1
    double akden=Dot(Z, P1, flops, memrefs);

    double ak=bknum/akden;
    ScMult_Add(lhs, ak, P, lhs, flops, memrefs);
    ScMult_Add(R, -ak, Z, R, flops, memrefs);
    // BiCG
    ScMult_Add(R1, -ak, Z1, R1, flops, memrefs);
    
    err=R.vector_norm(flops, memrefs)/bnorm;

    if (err>1000000) return 0;
  }
  return 0;
}

bool
Matrix::is_dense()
{
  return dynamic_cast<DenseMatrix *>(this) != NULL;
}

bool
Matrix::is_sparse()
{
  return dynamic_cast<SparseRowMatrix *>(this) != NULL;
}

bool
Matrix::is_column()
{
  return dynamic_cast<ColumnMatrix *>(this) != NULL;
}

DenseMatrix *
Matrix::as_dense()
{
  return dynamic_cast<DenseMatrix *>(this);
}

SparseRowMatrix *
Matrix::as_sparse()
{
  return dynamic_cast<SparseRowMatrix *>(this);
}

ColumnMatrix *
Matrix::as_column()
{
  return dynamic_cast<ColumnMatrix *>(this);
}



} // End namespace SCIRun
