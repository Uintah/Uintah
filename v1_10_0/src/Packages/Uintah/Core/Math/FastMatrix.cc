
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

#include <Packages/Uintah/Core/Math/FastMatrix.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/Assert.h>
#include <Core/Util/FancyAssert.h>
#include <iostream>
using namespace std;
using namespace Uintah;
using namespace SCIRun;

const int FastMatrix::smallSize = 16; // Make sure smallMat and smallMatPtr
const int FastMatrix::smallRows = 4;  // are updated (in .h file) if
                                      // you change these.
FastMatrix::FastMatrix(int rows, int cols)
  : rows(rows), cols(cols)
{
  ASSERT(rows>0);
  ASSERT(cols>0);
  int size=rows*cols;
  if(size<=smallSize && rows <= smallRows){
    mat=&smallMatPtr[0];
    for(int i=0;i<rows;i++)
      mat[i]=&smallMat[i*cols];
  } else {
    // Require allocation...
    mat = new double*[rows];
    double* tmp=new double[size];
    for(int i=0;i<rows;i++){
      mat[i]=tmp;
      tmp+=cols;
    }
  }
}

FastMatrix::~FastMatrix()
{
  if(mat != &smallMatPtr[0]){
    delete[] mat[0];
    delete[] mat;
  }
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
    break;
  case 4:
    {
      double a00 = a.mat[0][0], a01 = a.mat[0][1], a02 = a.mat[0][2], a03 = a.mat[0][3];
      double a10 = a.mat[1][0], a11 = a.mat[1][1], a12 = a.mat[1][2], a13 = a.mat[1][3];
      double a20 = a.mat[2][0], a21 = a.mat[2][1], a22 = a.mat[2][2], a23 = a.mat[2][3];
      double a30 = a.mat[3][0], a31 = a.mat[3][1], a32 = a.mat[3][2], a33 = a.mat[3][3];
      double one_over_denom = 1./(a03*a12*a21*a30 - a02*a13*a21*a30 - 
				  a03*a11*a22*a30 + a01*a13*a22*a30 + 
				  a02*a11*a23*a30 - a01*a12*a23*a30 - 
				  a03*a12*a20*a31 + a02*a13*a20*a31 + 
				  a03*a10*a22*a31 - a00*a13*a22*a31 - 
				  a02*a10*a23*a31 + a00*a12*a23*a31 + 
				  a03*a11*a20*a32 - a01*a13*a20*a32 - 
				  a03*a10*a21*a32 + a00*a13*a21*a32 + 
				  a01*a10*a23*a32 - a00*a11*a23*a32 - 
				  a02*a11*a20*a33 + a01*a12*a20*a33 + 
				  a02*a10*a21*a33 - a00*a12*a21*a33 - 
				  a01*a10*a22*a33 + a00*a11*a22*a33);

      mat[0][0] = (-(a13*a22*a31) + a12*a23*a31 + a13*a21*a32 - 
		   a11*a23*a32 - a12*a21*a33 + a11*a22*a33)*one_over_denom;

      mat[0][1] = (a03*a22*a31 - a02*a23*a31 - a03*a21*a32 + a01*a23*a32 + 
		   a02*a21*a33 - a01*a22*a33) * one_over_denom;
    
      mat[0][2] = (-(a03*a12*a31) + a02*a13*a31 + a03*a11*a32 - a01*a13*a32
		   - a02*a11*a33 +  a01*a12*a33) * one_over_denom;
    
      mat[0][3] = (a03*a12*a21 - a02*a13*a21 - a03*a11*a22 + a01*a13*a22 + 
		   a02*a11*a23 -  a01*a12*a23) * one_over_denom;
      
      mat[1][0] = (a13*a22*a30 - a12*a23*a30 - a13*a20*a32 + a10*a23*a32 + 
		   a12*a20*a33 - a10*a22*a33) * one_over_denom;
      
      mat[1][1] = (-(a03*a22*a30) + a02*a23*a30 + a03*a20*a32 - 
		   a00*a23*a32 - a02*a20*a33 + a00*a22*a33)*one_over_denom;

      mat[1][2] = (a03*a12*a30 - a02*a13*a30 - a03*a10*a32 + a00*a13*a32 +
		   a02*a10*a33 -  a00*a12*a33) * one_over_denom;
     
      mat[1][3] = (-(a03*a12*a20) + a02*a13*a20 + a03*a10*a22 - 
		   a00*a13*a22 - a02*a10*a23 + a00*a12*a23)*one_over_denom;

      mat[2][0] = (-(a13*a21*a30) + a11*a23*a30 + a13*a20*a31 - 
		   a10*a23*a31 - a11*a20*a33 + a10*a21*a33)*one_over_denom;
     
      mat[2][1] = (a03*a21*a30 - a01*a23*a30 - a03*a20*a31 + a00*a23*a31 +
		   a01*a20*a33 - a00*a21*a33) * one_over_denom;
     
      mat[2][2] = (-(a03*a11*a30) + a01*a13*a30 + a03*a10*a31 - 
		   a00*a13*a31 - a01*a10*a33 + a00*a11*a33)*one_over_denom;
     
      mat[2][3] = (a03*a11*a20 - a01*a13*a20 - a03*a10*a21 + a00*a13*a21 +
		   a01*a10*a23 - a00*a11*a23) * one_over_denom;

      mat[3][0] = (a12*a21*a30 - a11*a22*a30 - a12*a20*a31 + a10*a22*a31 +
		   a11*a20*a32 - a10*a21*a32) * one_over_denom;
     
      mat[3][1] = (-(a02*a21*a30) + a01*a22*a30 + a02*a20*a31 - 
		   a00*a22*a31 - a01*a20*a32 + a00*a21*a32)*one_over_denom;
     
      mat[3][2] = (a02*a11*a30 - a01*a12*a30 - a02*a10*a31 + a00*a12*a31 +
		   a01*a10*a32 -  a00*a11*a32) * one_over_denom;

      mat[3][3] = (-(a02*a11*a20) + a01*a12*a20 + a02*a10*a21 - 
		   a00*a12*a21 - a01*a10*a22 + a00*a11*a22)*one_over_denom;
    }
    break;
  default:
    // Bigger than 4x4:
    big_destructiveInvert(a);
    break;
  }
}

/*---------------------------------------------------------------------
 Function~  matrixSolver--
 Reference~  Mathematica provided the code
 ---------------------------------------------------------------------  */
void FastMatrix::destructiveSolve(const vector<double>& b,
				  vector<double>& X)
{
  // Can ruin the matrix (this) and the vector b.  Currently only does so for
  // matrices > 4x4
  ASSERTEQ(rows, cols);
  ASSERTEQ(rows, (int)b.size());
  ASSERTEQ(cols, (int)X.size());
  switch(rows){
  case 1:
    X[0] = b[0]/mat[0][0];
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
      X[0] = (a11*b0 - a01*b1)*one_over_denom;
      X[1] = (a00*b1-a10*b0)*one_over_denom;
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

      X[0] = ( (-(a12*a21) + a11*a22)*b0 + (a02*a21 - a01*a22)*b1 +
	       (-(a02*a11) + a01*a12)*b2 )*one_over_denom;


      X[1] = ( (a12*a20 - a10*a22)*b0 +  (-(a02*a20) + a00*a22)*b1 +
	       (a02*a10 - a00*a12)*b2 ) * one_over_denom;

      X[2] =  ( (-(a11*a20) + a10*a21)*b0 +  (a01*a20 - a00*a21)*b1 +
		(-(a01*a10) + a00*a11)*b2) * one_over_denom;
    }
    break;
  case 4:
    {
      // 4 X 4 matrix
      double a00 = mat[0][0], a01 = mat[0][1], a02 = mat[0][2], a03 = mat[0][3];
      double a10 = mat[1][0], a11 = mat[1][1], a12 = mat[1][2], a13 = mat[1][3];
      double a20 = mat[2][0], a21 = mat[2][1], a22 = mat[2][2], a23 = mat[2][3];
      double a30 = mat[3][0], a31 = mat[3][1], a32 = mat[3][2], a33 = mat[3][3];
      double b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];

      //__________________________________
      //  Test Problem
      // exact solution is 1,1,1,1
      /*
	double a00 = 1.1348, a01 = 3.8326, a02 = 1.1651, a03 = 3.4017;
	double a10 = 0.5301, a11 = 1.7875, a12 = 2.5330, a13 = 1.5435;
	double a20 = 3.4129, a21 = 4.9317, a22 = 8.7643, a23 = 1.3142;
	double a30 = 1.2371, a31 = 4.9998, a32 = 10.6721, a33 = 0.0147;
	double b0 = 9.5342,  b1 = 6.3941,  b2 = 18.4231,  b3 = 16.9237;*/

      double one_over_denom = 1./(a03*a12*a21*a30 - a02*a13*a21*a30 -
				  a03*a11*a22*a30 + a01*a13*a22*a30 + 
				  a02*a11*a23*a30 - a01*a12*a23*a30 - 
				  a03*a12*a20*a31 + a02*a13*a20*a31 + 
				  a03*a10*a22*a31 - a00*a13*a22*a31 - 
				  a02*a10*a23*a31 + a00*a12*a23*a31 + 
				  a03*a11*a20*a32 - a01*a13*a20*a32 - 
				  a03*a10*a21*a32 + a00*a13*a21*a32 + 
				  a01*a10*a23*a32 - a00*a11*a23*a32 -
				  a02*a11*a20*a33 + a01*a12*a20*a33 + 
				  a02*a10*a21*a33 - a00*a12*a21*a33 - 
				  a01*a10*a22*a33 + a00*a11*a22*a33);

      X[0] = ((-(a13*a22*a31) + a12*a23*a31 + a13*a21*a32 - a11*a23*a32 - 
	       a12*a21*a33 + a11*a22*a33)*b0 +   (a03*a22*a31 - a02*a23*a31 - 
          a03*a21*a32 + a01*a23*a32 + a02*a21*a33 -  a01*a22*a33)*b1 + 
         (-(a03*a12*a31) + a02*a13*a31 + a03*a11*a32 - a01*a13*a32 - 
          a02*a11*a33 + a01*a12*a33)*b2 +  (a03*a12*a21 - a02*a13*a21 - 
         a03*a11*a22 + a01*a13*a22 + a02*a11*a23 -  a01*a12*a23)*b3) * 
	one_over_denom;

      X[1] = ( (a13*a22*a30 - a12*a23*a30 - a13*a20*a32 + a10*a23*a32 + 
	    a12*a20*a33 - a10*a22*a33)*b0 + (-(a03*a22*a30) + a02*a23*a30 + 
           a03*a20*a32 - a00*a23*a32 - a02*a20*a33 +  a00*a22*a33)*b1 + 
          (a03*a12*a30 - a02*a13*a30 - a03*a10*a32 + a00*a13*a32 + 
           a02*a10*a33 - a00*a12*a33)*b2 + (-(a03*a12*a20) + a02*a13*a20 + 
           a03*a10*a22 - a00*a13*a22 - a02*a10*a23 + a00*a12*a23)*b3 ) * 
	one_over_denom;

      X[2] = ((-(a13*a21*a30) + a11*a23*a30 + a13*a20*a31 - a10*a23*a31 - 
          a11*a20*a33 +  a10*a21*a33)*b0 +  (a03*a21*a30 - a01*a23*a30 - 
           a03*a20*a31 + a00*a23*a31 + a01*a20*a33 - a00*a21*a33)*b1 + 
         (-(a03*a11*a30) + a01*a13*a30 + a03*a10*a31 - a00*a13*a31 - 
          a01*a10*a33 + a00*a11*a33)*b2 + (a03*a11*a20 - a01*a13*a20 - 
          a03*a10*a21 + a00*a13*a21 + a01*a10*a23 - a00*a11*a23)*b3 ) * 
	one_over_denom;

      X[3] = ((a12*a21*a30 - a11*a22*a30 - a12*a20*a31 + a10*a22*a31 + 
          a11*a20*a32 - a10*a21*a32)*b0 + (-(a02*a21*a30) + a01*a22*a30 + 
           a02*a20*a31 - a00*a22*a31 - a01*a20*a32 + a00*a21*a32)*b1 + 
         (a02*a11*a30 - a01*a12*a30 - a02*a10*a31 + a00*a12*a31 + 
          a01*a10*a32 - a00*a11*a32)*b2 +  (-(a02*a11*a20) + a01*a12*a20 + 
          a02*a10*a21 - a00*a12*a21 - a01*a10*a22 + a00*a11*a22)*b3) * 
	one_over_denom;
    }
    break;
  default:
    big_destructiveSolve(b, X);
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
  int size=rows*cols;
  double* ptr=&mat[0][0];
  for(int i=0;i<size;i++)
    *ptr++*=s;
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
  int size=rows*cols;
  double* ptr=&mat[0][0];
  for(int i=0;i<size;i++)
    *ptr++=0;
}

void FastMatrix::copy(const FastMatrix& a)
{
  ASSERTEQ(rows, a.rows);
  ASSERTEQ(cols, a.cols);
  int size=rows*cols;
  double* ptr1=&mat[0][0];
  double* ptr2=&a.mat[0][0];
  for(int i=0;i<size;i++)
    *ptr1++=*ptr2++;
}

void FastMatrix::print(ostream& out)
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

// This could also be made faster...
void FastMatrix::big_destructiveSolve(const vector<double>& b, vector<double>& X)
{
  ASSERTEQ(rows, cols);
  ASSERTEQ(rows, (int)b.size());
  ASSERTEQ(cols, (int)X.size());
  for(int i=0;i<rows;i++)
    X[i]=b[i];

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
      double tmp2 = X[i];
      X[i] = X[row];
      X[row] = tmp2;
      for(int j=0;j<rows;j++){
	double tmp = mat[i][j];
	mat[i][j] = mat[row][j];
	mat[row][j] = tmp;
      }
    }
    double denom=1./mat[i][i];
    double* r1=mat[i];
    X[i]*=denom;
    for(int j=0;j<rows;j++){
      r1[j]*=denom;
    }
    for(int j=i+1;j<rows;j++){
      double factor=mat[j][i];
      double* r2=mat[j];
      X[j]-=factor*X[i];
      for(int k=0;k<rows;k++){
	r2[k]-=factor*r1[k];
      }
    }
  }

  // Back-substitution
  for(int i=1;i<rows;i++){
    double* r1=mat[i];
    for(int j=0;j<i;j++){
      double factor=mat[j][i];
      double* r2=mat[j];
      X[j]-=factor*X[i];
      for(int k=0;k<rows;k++){
	r2[k]-=factor*r1[k];
      }
    }
  }
}
