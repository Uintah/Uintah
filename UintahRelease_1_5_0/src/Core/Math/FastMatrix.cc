/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */



/*
 *  FastMatrix.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

/*
 * TODO
 *   Make non-destructive inverts
 *     would still need to make a copy, so make a version that
 *     gives the temp space.  We could still avoid the copy for
 *     small matrices
 */

#include <Core/Math/FastMatrix.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/Assert.h>
#include <Core/Util/FancyAssert.h>
#include <iostream>

using std::vector;
using namespace Uintah;
using namespace SCIRun;

FastMatrix::FastMatrix(int rows, int cols)
  : rows(rows), cols(cols)
{
  ASSERT(rows>0);
  ASSERT(cols>0);
  ASSERT(rows <= MaxSize);
  ASSERT(cols <= MaxSize);
}

FastMatrix::~FastMatrix()
{
}

void FastMatrix::destructiveInvert(FastMatrix& a)
{
  ASSERTEQ(rows, cols);
  ASSERTEQ(rows, a.rows);
  ASSERTEQ(cols, a.cols);
  switch(rows){
  case 1:
    mat[0][0] = 1./a.mat[0][0];
    break;
  case 2:
    {
      double one_over_denom = 1./(a.mat[0][0] * a.mat[1][1] - a.mat[0][1] * a.mat[1][0]);
      mat[0][0] =  a.mat[1][1]*one_over_denom;
      mat[0][1] = -a.mat[0][1]*one_over_denom;
      mat[1][0] = -a.mat[1][0]*one_over_denom;
      mat[1][1] =  a.mat[0][0]*one_over_denom;
    }
    break;
#if 0
  case 3:
    {
      double a00 = a.mat[0][0], a01 = a.mat[0][1], a02 = a.mat[0][2];
      double a10 = a.mat[1][0], a11 = a.mat[1][1], a12 = a.mat[1][2];
      double a20 = a.mat[2][0], a21 = a.mat[2][1], a22 = a.mat[2][2];
      double one_over_denom = 1./(-(a02*a11*a20) + a01*a12*a20 + a02*a10*a21 - 
				  a00*a12*a21 - a01*a10*a22 + a00*a11*a22);
      mat[0][0] =  (-(a12*a21) + a11*a22) * one_over_denom;
      mat[0][1] =  (a02*a21 - a01*a22) * one_over_denom;
      mat[0][2] =  (-(a02*a11) + a01*a12) * one_over_denom;
      mat[1][0] =  (a12*a20 - a10*a22) * one_over_denom;
      mat[1][1] =  (-(a02*a20) + a00*a22) * one_over_denom;
      mat[1][2] =  (a02*a10 - a00*a12) * one_over_denom;
      mat[2][0] =  (-(a11*a20) + a10*a21) * one_over_denom;
      mat[2][1] =  (a01*a20 - a00*a21) * one_over_denom;
      mat[2][2] =  (-(a01*a10) + a00*a11) * one_over_denom;
    }
#endif
    break;
  default:
    // Bigger than 4x4:
    big_destructiveInvert(a);
    break;
  }
}

void FastMatrix::transpose(const FastMatrix& a)
{
  ASSERTEQ(cols, a.rows);
  ASSERTEQ(rows, a.cols);
  // Cache-coherent on writes.  We could do better by blocking, but
  // This should be good enough for small matrices, which is what this
  // class is intended for
  for(int row=0;row<rows; row++){
    for(int col=0;col<cols; col++){
      mat[row][col]=a.mat[col][row];
    }
  }
}

/*---------------------------------------------------------------------
 Function~  MatrixMultiplication--
 Reference~  This multiplies matrix (this) and vector (b) and returns X
 ---------------------------------------------------------------------  */
void FastMatrix::multiply(const vector<double>& b, vector<double>& X) const
{
  ASSERTEQ(cols, (int)b.size());
  ASSERTEQ(rows, (int)X.size());
  for (int row=0; row<rows; row++) {
    double sum=0;
    for (int col=0; col<cols; col++) {
      sum += mat[row][col]*b[col];
    }
    X[row]=sum;
  }
}

void FastMatrix::multiply(const double* b, double* X) const
{
  for (int row=0; row<rows; row++) {
    double sum=0;
    for (int col=0; col<cols; col++) {
      sum += mat[row][col]*b[col];
    }
    X[row]=sum;
  }
}

// this = a*b;
void FastMatrix::multiply(const FastMatrix& a, const FastMatrix& b)
{
  ASSERTEQ(rows, a.rows);
  ASSERTEQ(cols, b.cols);
  ASSERTEQ(a.cols, b.rows);
  int s=a.cols;
  for(int i=0;i<rows;i++){
    for(int j=0;j<cols;j++){
      double sum=0;
      for(int k=0;k<s;k++){
	sum+=a.mat[i][k]*b.mat[k][j];
      }
      mat[i][j]=sum;
    }
  }
}

void FastMatrix::multiply(double s)
{
  for(int i=0;i<rows;i++){
    for(int j=0;j<cols;j++)
      mat[i][j] *= s;
  }
}

/*---------------------------------------------------------------------
 Function~  conditionNumber--
 Reference~  Computes the condition number of a matrix
 ---------------------------------------------------------------------  */
double FastMatrix::conditionNumber() const
{
  ASSERTEQ(rows, cols);
  //   Check for ill-conditioned system
  // - calculate the inverse
  // - compute the max row sum norm for (a) and the inverse
  // - condition_number = max_row_sum_a * max_row_sum_a_inverse
  FastMatrix a_invert(rows, cols);
  FastMatrix a_copy(rows, cols);
  a_copy.copy(*this);
  a_invert.destructiveInvert(a_copy);
  double max_row_sum_a         = 0.0;
  double max_row_sum_a_invert  = 0.0;

  for(int m = 0; m < rows; m++) {
    double row_sum_a = 0.0;
    double row_sum_a_invert = 0.0;
    for(int n = 0; n < cols; n++)  {  
      row_sum_a += Abs(mat[m][n]);
      row_sum_a_invert += Abs(a_invert.mat[m][n]);
    }
    max_row_sum_a = Max(max_row_sum_a, row_sum_a);
    max_row_sum_a_invert = Max(max_row_sum_a_invert, row_sum_a_invert);
  }
  return  max_row_sum_a * max_row_sum_a_invert;
}

void FastMatrix::identity()
{
  ASSERTEQ(rows, cols);
  zero();
  for(int i=0;i<rows;i++)
    mat[i][i]=1;
}

void FastMatrix::zero()
{
  for(int i=0;i<rows;i++){
    for(int j=0;j<cols;j++)
      mat[i][j] = 0;
  }
}

void FastMatrix::copy(const FastMatrix& a)
{
  ASSERTEQ(rows, a.rows);
  ASSERTEQ(cols, a.cols);
  //int size=rows*cols;
  for(int i=0;i<rows;i++){
    for(int j=0;j<cols;j++)
      mat[i][j] = a.mat[i][j];
  }
}

void FastMatrix::print(std::ostream& out)
{
  for(int i=0;i<rows;i++){
    out << i << ":";
    for(int j=0;j<cols;j++){
      out << '\t' << mat[i][j];
    }
    out << '\n';
  }
}

// This could be made a little faster
// - better backsolve so that original doesn't need to be updated
// - only work on part of matrix that is non-zero (careful)
void FastMatrix::big_destructiveInvert(FastMatrix& a)
{
  ASSERTEQ(rows, cols);
  ASSERTEQ(rows, a.rows);
  ASSERTEQ(cols, a.cols);
  identity();

  // Gauss-Jordan with partial pivoting
  for(int i=0;i<rows;i++){
    double max=Abs(a.mat[i][i]);
    int row=i;
    for(int j=i+1;j<rows;j++){
      if(Abs(a.mat[j][i]) > max){
	max=Abs(a.mat[j][i]);
	row=j;
      }
    }
    ASSERT(max > 1.e-12);
    if(row != i){
      // Switch rows
      for(int j=0;j<rows;j++){
	double tmp = mat[i][j];
	mat[i][j] = mat[row][j];
	mat[row][j] = tmp;
	double tmp2 = a.mat[i][j];
	a.mat[i][j] = a.mat[row][j];
	a.mat[row][j] = tmp2;
      }
    }
    double denom=1./a.mat[i][i];
    double* r1=a.mat[i];
    double* n1=mat[i];
    for(int j=0;j<rows;j++){
      r1[j]*=denom;
      n1[j]*=denom;
    }
    for(int j=i+1;j<rows;j++){
      double factor=a.mat[j][i];
      double* r2=a.mat[j];
      double* n2=mat[j];
      for(int k=0;k<rows;k++){
	r2[k]-=factor*r1[k];
	n2[k]-=factor*n1[k];
      }
    }
  }

  // Back-substitution
  for(int i=1;i<rows;i++){
    double* r1=a.mat[i];
    double* n1=mat[i];
    for(int j=0;j<i;j++){
      double factor=a.mat[j][i];
      double* r2=a.mat[j];
      double* n2=mat[j];
      for(int k=0;k<rows;k++){
	r2[k]-=factor*r1[k];
	n2[k]-=factor*n1[k];
      }
    }
  }
}

void FastMatrix::big_destructiveSolve(double* b)
{
  ASSERTEQ(rows, cols);

  // Gauss-Jordan with partial pivoting
  for(int i=0;i<rows;i++){
    double max=Abs(mat[i][i]);
    int row=i;
    for(int j=i+1;j<rows;j++){
      if(Abs(mat[j][i]) > max){
	max=Abs(mat[j][i]);
	row=j;
      }
    }
    ASSERT(max > 1.e-12);
    if(row != i){
      // Switch rows
      for(int j=i;j<rows;j++){
	double tmp = mat[i][j];
	mat[i][j] = mat[row][j];
	mat[row][j] = tmp;
      }
      double tmp2 = b[i];
      b[i] = b[row];
      b[row] = tmp2;
    }
    double scale=1./mat[i][i];
    b[i]*=scale;
    for(int j=i;j<rows;j++){
      mat[i][j]*=scale;
    }
    for(int j=i+1;j<rows;j++){
      double factor=mat[j][i];
      b[j]-=factor*b[i];
      for(int k=i;k<rows;k++){
	mat[j][k]-=factor*mat[i][k];
      }
    }
  }

  // Back-substitution
  for(int i=rows-1;i>=0;i--){
    for(int j=i-1;j>=0;j--){
      double factor=mat[j][i];
      b[j]-=factor*b[i];
    }
  }
}

template<int size> void med_destructiveSolve(double mat[FastMatrix::MaxSize][FastMatrix::MaxSize], double* b)
{
  // Gauss-Jordan with partial pivoting
  for(int i=0;i<size;i++){
    double max=Abs(mat[i][i]);
    int row=i;
    for(int j=i+1;j<size;j++){
      if(Abs(mat[j][i]) > max){
	max=Abs(mat[j][i]);
	row=j;
      }
    }
    ASSERT(max > 1.e-12);
    if(row != i){
      // Switch rows
      for(int j=i;j<size;j++){
	double tmp = mat[i][j];
	mat[i][j] = mat[row][j];
	mat[row][j] = tmp;
      }
      double tmp2 = b[i];
      b[i] = b[row];
      b[row] = tmp2;
    }
    double scale=1./mat[i][i];
    b[i]*=scale;
    for(int j=i;j<size;j++){
      mat[i][j]*=scale;
    }
    for(int j=i+1;j<size;j++){
      double factor=mat[j][i];
      b[j]-=factor*b[i];
      for(int k=i;k<size;k++){
	mat[j][k]-=factor*mat[i][k];
      }
    }
  }

  // Back-substitution
  for(int i=size-1;i>=0;i--){
    for(int j=i-1;j>=0;j--){
      double factor=mat[j][i];
      b[j]-=factor*b[i];
    }
  }
}

/*---------------------------------------------------------------------
 Function~  matrixSolver--
 Reference~  Mathematica provided the code
 ---------------------------------------------------------------------  */
void FastMatrix::destructiveSolve(double* b)
{
  ASSERTEQ(rows, cols);
  switch(rows){
  case 1:
    b[0] /= mat[0][0];
    break;
  case 2:
    {
      // 2 X 2 Matrix
      //__________________________________
      // Example Problem Hilbert Matrix
      //  Exact solution is 1,1
      /*
	mat[0][0] = 1.0;   mat[0][1] = 1/2.0;
	mat[1][0] = 1/2.0; mat[1][1] = 1/3.0;
	b[0] = 3.0/2.0;   b[1] = 5.0/6.0;*/
    
      double a00 = mat[0][0], a01 = mat[0][1];
      double a10 = mat[1][0], a11 = mat[1][1];
      double b0 = b[0], b1 = b[1];

      double one_over_denom = 1./(a00*a11 - a01*a10);
      b[0] = (a11*b0 - a01*b1)*one_over_denom;
      b[1] = (a00*b1 - a10*b0)*one_over_denom;
    } 
    break;
  case 3:
    {
      // 3 X 3 Matrix
      double a00 = mat[0][0], a01 = mat[0][1], a02 = mat[0][2];
      double a10 = mat[1][0], a11 = mat[1][1], a12 = mat[1][2];
      double a20 = mat[2][0], a21 = mat[2][1], a22 = mat[2][2];
      double b0 = b[0], b1 = b[1], b2 = b[2];

      //__________________________________
      // Example Problem Hilbert matrix
      // Exact Solution is 1,1,1
      /*
	double a00 = 1.0,       a01 = 1.0/2.0,  a02 = 1.0/3.0;
	double a10 = 1.0/2.0,   a11 = 1.0/3.0,  a12 = 1.0/4.0;
	double a20 = 1.0/3.0,   a21 = 1.0/4.0,  a22 = 1.0/5.0;
	double b0 = 11.0/6.0, b1 = 13.0/12.0, b2 = 47.0/60.0; */

      double one_over_denom = 1./(-(a02*a11*a20) + a01*a12*a20 + a02*a10*a21 
				  - a00*a12*a21 -  a01*a10*a22 + a00*a11*a22);

      b[0] = ( (-(a12*a21) + a11*a22)*b0 + (a02*a21 - a01*a22)*b1 +
	       (-(a02*a11) + a01*a12)*b2 )*one_over_denom;


      b[1] = ( (a12*a20 - a10*a22)*b0 +  (-(a02*a20) + a00*a22)*b1 +
	       (a02*a10 - a00*a12)*b2 ) * one_over_denom;

      b[2] =  ( (-(a11*a20) + a10*a21)*b0 +  (a01*a20 - a00*a21)*b1 +
		(-(a01*a10) + a00*a11)*b2) * one_over_denom;
    }
    break;
  case 4:
    med_destructiveSolve<4>(mat, b);
    break;
  case 5:
    med_destructiveSolve<5>(mat, b);
    break;
  case 6:
    med_destructiveSolve<6>(mat, b);
    break;
  default:
    big_destructiveSolve(b);
    break;
  }
}

void FastMatrix::big_destructiveSolve(double* b1, double* b2)
{
  ASSERTEQ(rows, cols);

  // Gauss-Jordan with partial pivoting
  for(int i=0;i<rows;i++){
    double max=Abs(mat[i][i]);
    int row=i;
    for(int j=i+1;j<rows;j++){
      if(Abs(mat[j][i]) > max){
	max=Abs(mat[j][i]);
	row=j;
      }
    }
    ASSERT(max > 1.e-12);
    if(row != i){
      // Switch rows
      for(int j=i;j<rows;j++){
	double tmp = mat[i][j];
	mat[i][j] = mat[row][j];
	mat[row][j] = tmp;
      }
      double tmp2 = b1[i];
      b1[i] = b1[row];
      b1[row] = tmp2;
      double tmp3 = b2[i];
      b2[i] = b2[row];
      b2[row] = tmp3;
    }
    double scale=1./mat[i][i];
    b1[i]*=scale;
    b2[i]*=scale;
    for(int j=i;j<rows;j++){
      mat[i][j]*=scale;
    }
    for(int j=i+1;j<rows;j++){
      double factor=mat[j][i];
      b1[j]-=factor*b1[i];
      b2[j]-=factor*b2[i];
      for(int k=i;k<rows;k++){
	mat[j][k]-=factor*mat[i][k];
      }
    }
  }

  // Back-substitution
  for(int i=rows-1;i>=0;i--){
    for(int j=i-1;j>=0;j--){
      double factor=mat[j][i];
      b1[j]-=factor*b1[i];
      b2[j]-=factor*b2[i];
    }
  }
}

template<int size> void med_destructiveSolve(double mat[FastMatrix::MaxSize][FastMatrix::MaxSize],
                                             double* b1, double* b2)
{
  // Gauss-Jordan with partial pivoting
  for(int i=0;i<size;i++){
    double max=Abs(mat[i][i]);
    int row=i;
    for(int j=i+1;j<size;j++){
      if(Abs(mat[j][i]) > max){
	max=Abs(mat[j][i]);
	row=j;
      }
    }
    ASSERT(max > 1.e-12);
    if(row != i){
      // Switch rows
      for(int j=i;j<size;j++){
	double tmp = mat[i][j];
	mat[i][j] = mat[row][j];
	mat[row][j] = tmp;
      }
      double tmp2 = b1[i];
      b1[i] = b1[row];
      b1[row] = tmp2;
      double tmp3 = b2[i];
      b2[i] = b2[row];
      b2[row] = tmp3;
    }
    double scale=1./mat[i][i];
    b1[i]*=scale;
    b2[i]*=scale;
    for(int j=i;j<size;j++){
      mat[i][j]*=scale;
    }
    for(int j=i+1;j<size;j++){
      double factor=mat[j][i];
      b1[j]-=factor*b1[i];
      b2[j]-=factor*b2[i];
      for(int k=i;k<size;k++){
	mat[j][k]-=factor*mat[i][k];
      }
    }
  }

  // Back-substitution
  for(int i=size-1;i>=0;i--){
    for(int j=i-1;j>=0;j--){
      double factor=mat[j][i];
      b1[j]-=factor*b1[i];
      b2[j]-=factor*b2[i];
    }
  }
}

/*---------------------------------------------------------------------
 Function~  matrixSolver--
 Reference~  Mathematica provided the code
 ---------------------------------------------------------------------  */
void FastMatrix::destructiveSolve(double* b1, double* b2)
{
  // Can ruin the matrix (this) and replaces the vector b
  ASSERTEQ(rows, cols);
  switch(rows){
  case 1:
    b1[0] /= mat[0][0];
    b2[0] /= mat[0][0];
    break;
  case 2:
    {
      // 2 X 2 Matrix
      //__________________________________
      // Example Problem Hilbert Matrix
      //  Exact solution is 1,1
      /*
	mat[0][0] = 1.0;   mat[0][1] = 1/2.0;
	mat[1][0] = 1/2.0; mat[1][1] = 1/3.0;
	b[0] = 3.0/2.0;   b[1] = 5.0/6.0;*/
    
      double a00 = mat[0][0], a01 = mat[0][1];
      double a10 = mat[1][0], a11 = mat[1][1];
      double one_over_denom = 1./(a00*a11 - a01*a10);
      double b10 = b1[0], b11 = b1[1];
      b1[0] = (a11*b10 - a01*b11)*one_over_denom;
      b1[1] = (a00*b11 - a10*b10)*one_over_denom;
      double b20 = b2[0], b21 = b2[1];
      b2[0] = (a11*b20 - a01*b21)*one_over_denom;
      b2[1] = (a00*b21 - a10*b20)*one_over_denom;
    } 
    break;
  case 3:
    {
      // 3 X 3 Matrix
      double a00 = mat[0][0], a01 = mat[0][1], a02 = mat[0][2];
      double a10 = mat[1][0], a11 = mat[1][1], a12 = mat[1][2];
      double a20 = mat[2][0], a21 = mat[2][1], a22 = mat[2][2];
      double b10 = b1[0], b11 = b1[1], b12 = b1[2];
      double b20 = b2[0], b21 = b2[1], b22 = b2[2];

      //__________________________________
      // Example Problem Hilbert matrix
      // Exact Solution is 1,1,1
      /*
	double a00 = 1.0,       a01 = 1.0/2.0,  a02 = 1.0/3.0;
	double a10 = 1.0/2.0,   a11 = 1.0/3.0,  a12 = 1.0/4.0;
	double a20 = 1.0/3.0,   a21 = 1.0/4.0,  a22 = 1.0/5.0;
	double b0 = 11.0/6.0, b1 = 13.0/12.0, b2 = 47.0/60.0; */

      double one_over_denom = 1./(-(a02*a11*a20) + a01*a12*a20 + a02*a10*a21 
				  - a00*a12*a21 -  a01*a10*a22 + a00*a11*a22);

      b1[0] = ( (-(a12*a21) + a11*a22)*b10 + (a02*a21 - a01*a22)*b11 +
	       (-(a02*a11) + a01*a12)*b12 )*one_over_denom;
      b2[0] = ( (-(a12*a21) + a11*a22)*b20 + (a02*a21 - a01*a22)*b21 +
	       (-(a02*a11) + a01*a12)*b22 )*one_over_denom;


      b1[1] = ( (a12*a20 - a10*a22)*b10 +  (-(a02*a20) + a00*a22)*b11 +
	       (a02*a10 - a00*a12)*b12 ) * one_over_denom;
      b2[1] = ( (a12*a20 - a10*a22)*b20 +  (-(a02*a20) + a00*a22)*b21 +
	       (a02*a10 - a00*a12)*b22 ) * one_over_denom;

      b1[2] =  ( (-(a11*a20) + a10*a21)*b10 +  (a01*a20 - a00*a21)*b11 +
		(-(a01*a10) + a00*a11)*b12) * one_over_denom;
      b2[2] =  ( (-(a11*a20) + a10*a21)*b20 +  (a01*a20 - a00*a21)*b21 +
		(-(a01*a10) + a00*a11)*b22) * one_over_denom;
    }
    break;
  case 4:
    med_destructiveSolve<4>(mat, b1, b2);
    break;
  case 5:
    med_destructiveSolve<5>(mat, b1, b2);
    break;
  case 6:
    med_destructiveSolve<6>(mat, b1, b2);
    break;
  default:
    big_destructiveSolve(b1, b2);
    break;
  }
}


// Vector RHS

void FastMatrix::big_destructiveSolve(Vector* b)
{
  ASSERTEQ(rows, cols);

  // Gauss-Jordan with partial pivoting
  for(int i=0;i<rows;i++){
    double max=Abs(mat[i][i]);
    int row=i;
    for(int j=i+1;j<rows;j++){
      if(Abs(mat[j][i]) > max){
	max=Abs(mat[j][i]);
	row=j;
      }
    }
    ASSERT(max > 1.e-12);
    if(row != i){
      // Switch rows
      for(int j=i;j<rows;j++){
	double tmp = mat[i][j];
	mat[i][j] = mat[row][j];
	mat[row][j] = tmp;
      }
      Vector tmp2 = b[i];
      b[i] = b[row];
      b[row] = tmp2;
    }

    double scale=1./mat[i][i];
    b[i]*=scale;
    for(int j=i;j<rows;j++){
      mat[i][j]*=scale;
    }
    for(int j=i+1;j<rows;j++){
      double factor=mat[j][i];
      b[j]-=factor*b[i];
      for(int k=i;k<rows;k++)
	mat[j][k]-=factor*mat[i][k];
    }
  }

  // Back-substitution
  for(int i=rows-1;i>=0;i--){
    for(int j=i-1;j>=0;j--){
      double factor=mat[j][i];
      b[j]-=factor*b[i];
    }
  }
}

template<int size> void med_destructiveSolve(double mat[FastMatrix::MaxSize][FastMatrix::MaxSize],
                                             Vector* b)
{
  // Gauss-Jordan with partial pivoting
  for(int i=0;i<size;i++){
    double max=Abs(mat[i][i]);
    int row=i;
    for(int j=i+1;j<size;j++){
      if(Abs(mat[j][i]) > max){
	max=Abs(mat[j][i]);
	row=j;
      }
    }
    ASSERT(max > 1.e-12);
    if(row != i){
      // Switch rows
      for(int j=i;j<size;j++){
	double tmp = mat[i][j];
	mat[i][j] = mat[row][j];
	mat[row][j] = tmp;
      }
      Vector tmp2 = b[i];
      b[i] = b[row];
      b[row] = tmp2;
    }

    double scale=1./mat[i][i];
    b[i]*=scale;
    for(int j=i;j<size;j++){
      mat[i][j]*=scale;
    }
    for(int j=i+1;j<size;j++){
      double factor=mat[j][i];
      b[j]-=factor*b[i];
      for(int k=i;k<size;k++)
	mat[j][k]-=factor*mat[i][k];
    }
  }

  // Back-substitution
  for(int i=size-1;i>=0;i--){
    for(int j=i-1;j>=0;j--){
      double factor=mat[j][i];
      b[j]-=factor*b[i];
    }
  }
}

/*---------------------------------------------------------------------
 Function~  matrixSolver--
 Reference~  Mathematica provided the code
 ---------------------------------------------------------------------  */
void FastMatrix::destructiveSolve(Vector* b)
{
  // Can ruin the matrix (this) and replaces the vector b
  ASSERTEQ(rows, cols);
  switch(rows){
  case 1:
    {
      double scale = 1./mat[0][0];
      b[0] *= scale;
    }
    break;
  case 2:
    {
      // 2 X 2 Matrix
      //__________________________________
      // Example Problem Hilbert Matrix
      //  Exact solution is 1,1
      /*
	mat[0][0] = 1.0;   mat[0][1] = 1/2.0;
	mat[1][0] = 1/2.0; mat[1][1] = 1/3.0;
	b[0] = 3.0/2.0;   b[1] = 5.0/6.0;*/
    
      double a00 = mat[0][0], a01 = mat[0][1];
      double a10 = mat[1][0], a11 = mat[1][1];
      double one_over_denom = 1./(a00*a11 - a01*a10);
      Vector b0 = b[0], b1 = b[1];
      b[0] = (a11*b0 - a01*b1)*one_over_denom;
      b[1] = (a00*b1 - a10*b0)*one_over_denom;
    } 
    break;
  case 3:
    {
      // 3 X 3 Matrix
      double a00 = mat[0][0], a01 = mat[0][1], a02 = mat[0][2];
      double a10 = mat[1][0], a11 = mat[1][1], a12 = mat[1][2];
      double a20 = mat[2][0], a21 = mat[2][1], a22 = mat[2][2];
      Vector b0 = b[0], b1 = b[1], b2 = b[2];

      //__________________________________
      // Example Problem Hilbert matrix
      // Exact Solution is 1,1,1
      /*
	double a00 = 1.0,       a01 = 1.0/2.0,  a02 = 1.0/3.0;
	double a10 = 1.0/2.0,   a11 = 1.0/3.0,  a12 = 1.0/4.0;
	double a20 = 1.0/3.0,   a21 = 1.0/4.0,  a22 = 1.0/5.0;
	double b0 = 11.0/6.0, b1 = 13.0/12.0, b2 = 47.0/60.0; */

      double one_over_denom = 1./(-(a02*a11*a20) + a01*a12*a20 + a02*a10*a21 
				  - a00*a12*a21 -  a01*a10*a22 + a00*a11*a22);

      b[0] = ( (-(a12*a21) + a11*a22)*b0 + (a02*a21 - a01*a22)*b1 +
	       (-(a02*a11) + a01*a12)*b2 )*one_over_denom;

      b[1] = ( (a12*a20 - a10*a22)*b0 +  (-(a02*a20) + a00*a22)*b1 +
	       (a02*a10 - a00*a12)*b2 ) * one_over_denom;

      b[2] =  ( (-(a11*a20) + a10*a21)*b0 +  (a01*a20 - a00*a21)*b1 +
		(-(a01*a10) + a00*a11)*b2) * one_over_denom;
    }
    break;
  case 4:
    med_destructiveSolve<4>(mat, b);
    break;
  case 5:
    med_destructiveSolve<5>(mat, b);
    break;
  case 6:
    med_destructiveSolve<6>(mat, b);
    break;
  default:
    big_destructiveSolve(b);
    break;
  }
}
