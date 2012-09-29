/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
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
 */

#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/DenseColMajMatrix.h>
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


void
Matrix::scalar_multiply(double s)
{
  double *ptr = get_data_pointer();
  const size_t sz = get_data_size();
  for (size_t i = 0; i < sz; i++)
  {
    ptr[i] *= s;
  }
}


Transform Matrix::toTransform() {
  Transform t;
  if (nrows() != 4 || ncols() != 4) {
    std::cerr << "Error - can't make a transform from this matrix.\n";
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

DenseMatrix *
Matrix::direct_inverse()
{
  if (nrows() != ncols()) return 0;
  DenseMatrix *A=dense();
  if (is_dense()) A=scinew DenseMatrix(*A);
  A->invert();
  return A;
}

DenseMatrix *
Matrix::iterative_inverse()
{
  if (nrows() != ncols()) return 0;
  int n=nrows();
  SparseRowMatrix* B(SparseRowMatrix::identity(n));
  DenseMatrix *D = B->dense();
  DenseMatrix *X = scinew DenseMatrix(n,n);
  bicg_solve(*D, *X);
  delete D;
  delete B;
  return X;
}

int 
Matrix::cg_solve(const ColumnMatrix& rhs, ColumnMatrix& lhs) const
{
  double err;
  int niter, flops, memrefs;
  return cg_solve(rhs, lhs, err, niter, flops, memrefs);
}

int 
Matrix::cg_solve(const DenseMatrix& rhs, DenseMatrix& lhs) const
{
  double err;
  int niter, flops, memrefs;
  return cg_solve(rhs, lhs, err, niter, flops, memrefs);
}

int 
Matrix::cg_solve(const DenseMatrix& rhs, DenseMatrix& lhs,
		 double &err, int &niter,
		 int &flops, int &memrefs,
		 double max_error, int toomany, int useLhsAsGuess) const
{
  if (rhs.ncols() != lhs.ncols()) return 0;
  for (int i=0; i<rhs.ncols(); i++) {
    ColumnMatrix rh(rhs.nrows()), lh(lhs.nrows());
    int j;
    for (j=0; j<rh.nrows(); j++)
      rh[j]=rhs[i][j];
    if (!cg_solve(rh, lh, err, niter, flops, memrefs, max_error, 
		  toomany, useLhsAsGuess)) return 0;
    for (j=0; j<rh.nrows(); j++)
      lhs[i][j]=lh[j];
  }
  return 1;
}

int
Matrix::cg_solve(const ColumnMatrix& rhs, ColumnMatrix& lhs,
		 double &err, int &niter, 
		 int& flops, int& memrefs,
		 double max_error, int toomany, int useLhsAsGuess) const
{
  int size=nrows();  
  niter=0;
  flops=0;
  memrefs=0;
  if (!useLhsAsGuess) lhs.zero();

  if(toomany == 0) toomany=100*size;

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
Matrix::bicg_solve(const DenseMatrix& rhs, DenseMatrix& lhs) const
{
  double err;
  int niter, flops, memrefs;
  return bicg_solve(rhs, lhs, err, niter, flops, memrefs);
}

int 
Matrix::bicg_solve(const DenseMatrix& rhs, DenseMatrix& lhs,
		   double &err, int &niter,
		   int &flops, int &memrefs,
		   double max_error, int /*toomany*/, int useLhsAsGuess) const
{
  if (rhs.ncols() != lhs.ncols()) return 0;
  for (int i=0; i<rhs.ncols(); i++) {
    ColumnMatrix rh(rhs.nrows()), lh(lhs.nrows());
    int j;
    for (j=0; j<rh.nrows(); j++)
      rh[j]=rhs[i][j];
    if (!bicg_solve(rh, lh, err, niter, flops, memrefs, 
		    max_error, useLhsAsGuess)) return 0;
    for (j=0; j<rh.nrows(); j++)
      lhs[i][j]=lh[j];
  }
  return 1;
}

int
Matrix::bicg_solve(const ColumnMatrix& rhs, ColumnMatrix& lhs,
		   double &err, int &niter, 
		   int& flops, int& memrefs,
		   double max_error, int toomany, int useLhsAsGuess) const
{
  int size=nrows();  
  niter=0;
  flops=0;
  memrefs=0;
  if (!useLhsAsGuess) lhs.zero();
  
  if(toomany == 0) toomany=100*size;

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

DenseColMajMatrix *
Matrix::as_dense_col_maj()
{
  return dynamic_cast<DenseColMajMatrix *>(this);
}


} // End namespace SCIRun
