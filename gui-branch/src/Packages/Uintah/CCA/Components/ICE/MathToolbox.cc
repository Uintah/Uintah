#include <Packages/Uintah/CCA/Components/ICE/ICE.h>

using namespace SCIRun;
using namespace Uintah;
/*---------------------------------------------------------------------
 Function~  ICE::matrixInverse--
 Reference~  Computes the inverse of a matrix
 ---------------------------------------------------------------------  */
void ICE::matrixInverse( int numMatls,
                         DenseMatrix& a,
                         DenseMatrix& a_inverse)
{
  if (numMatls == 1) {
    a_inverse[0][0] = 1./a[0][0];
    return;
  }
  if (numMatls == 2) {
    double one_over_denom = 1./(a[0][0] * a[1][1] - a[0][1] * a[1][0]);
    a_inverse[0][0] =  a[1][1]*one_over_denom;
    a_inverse[0][1] = -a[0][1]*one_over_denom;
    a_inverse[1][0] = -a[1][0]*one_over_denom;
    a_inverse[1][1] =  a[0][0]*one_over_denom;
    return;
  }
  if (numMatls == 3 ) {
    double a00 = a[0][0], a01 = a[0][1], a02 = a[0][2];
    double a10 = a[1][0], a11 = a[1][1], a12 = a[1][2];
    double a20 = a[2][0], a21 = a[2][1], a22 = a[2][2];
    double one_over_denom = 1./(-(a02*a11*a20) + a01*a12*a20 + a02*a10*a21 - 
				a00*a12*a21 - a01*a10*a22 + a00*a11*a22);
    a_inverse[0][0] =  (-(a12*a21) + a11*a22) * one_over_denom;
    a_inverse[0][1] =  (a02*a21 - a01*a22) * one_over_denom;
    a_inverse[0][2] =  (-(a02*a11) + a01*a12) * one_over_denom;
    a_inverse[1][0] =  (a12*a20 - a10*a22) * one_over_denom;
    a_inverse[1][1] =  (-(a02*a20) + a00*a22) * one_over_denom;
    a_inverse[1][2] =  (a02*a10 - a00*a12) * one_over_denom;
    a_inverse[2][0] =  (-(a11*a20) + a10*a21) * one_over_denom;
    a_inverse[2][1] =  (a01*a20 - a00*a21) * one_over_denom;
    a_inverse[2][2] =  (-(a01*a10) + a00*a11) * one_over_denom;
    return;
  }
  if (numMatls == 4) {   // 4 X 4 matrix
    double a00 = a[0][0], a01 = a[0][1], a02 = a[0][2], a03 = a[0][3];
    double a10 = a[1][0], a11 = a[1][1], a12 = a[1][2], a13 = a[1][3];
    double a20 = a[2][0], a21 = a[2][1], a22 = a[2][2], a23 = a[2][3];
    double a30 = a[3][0], a31 = a[3][1], a32 = a[3][2], a33 = a[3][3];
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

    a_inverse[0][0] = (-(a13*a22*a31) + a12*a23*a31 + a13*a21*a32 - 
		       a11*a23*a32 - a12*a21*a33 + a11*a22*a33)*one_over_denom;

    a_inverse[0][1] = (a03*a22*a31 - a02*a23*a31 - a03*a21*a32 + a01*a23*a32 + 
		       a02*a21*a33 - a01*a22*a33) * one_over_denom;
    
    a_inverse[0][2] = (-(a03*a12*a31) + a02*a13*a31 + a03*a11*a32 - a01*a13*a32
		       - a02*a11*a33 +  a01*a12*a33) * one_over_denom;
    
    a_inverse[0][3] = (a03*a12*a21 - a02*a13*a21 - a03*a11*a22 + a01*a13*a22 + 
		       a02*a11*a23 -  a01*a12*a23) * one_over_denom;
    
    a_inverse[1][0] = (a13*a22*a30 - a12*a23*a30 - a13*a20*a32 + a10*a23*a32 + 
		       a12*a20*a33 - a10*a22*a33) * one_over_denom;
    
    a_inverse[1][1] = (-(a03*a22*a30) + a02*a23*a30 + a03*a20*a32 - 
		       a00*a23*a32 - a02*a20*a33 + a00*a22*a33)*one_over_denom;

    a_inverse[1][2] = (a03*a12*a30 - a02*a13*a30 - a03*a10*a32 + a00*a13*a32 +
		       a02*a10*a33 -  a00*a12*a33) * one_over_denom;
     
    a_inverse[1][3] = (-(a03*a12*a20) + a02*a13*a20 + a03*a10*a22 - 
		       a00*a13*a22 - a02*a10*a23 + a00*a12*a23)*one_over_denom;

    a_inverse[2][0] = (-(a13*a21*a30) + a11*a23*a30 + a13*a20*a31 - 
		       a10*a23*a31 - a11*a20*a33 + a10*a21*a33)*one_over_denom;
     
    a_inverse[2][1] = (a03*a21*a30 - a01*a23*a30 - a03*a20*a31 + a00*a23*a31 +
		       a01*a20*a33 - a00*a21*a33) * one_over_denom;
     
    a_inverse[2][2] = (-(a03*a11*a30) + a01*a13*a30 + a03*a10*a31 - 
		       a00*a13*a31 - a01*a10*a33 + a00*a11*a33)*one_over_denom;
     
    a_inverse[2][3] = (a03*a11*a20 - a01*a13*a20 - a03*a10*a21 + a00*a13*a21 +
		       a01*a10*a23 - a00*a11*a23) * one_over_denom;

    a_inverse[3][0] = (a12*a21*a30 - a11*a22*a30 - a12*a20*a31 + a10*a22*a31 +
		       a11*a20*a32 - a10*a21*a32) * one_over_denom;
     
    a_inverse[3][1] = (-(a02*a21*a30) + a01*a22*a30 + a02*a20*a31 - 
		       a00*a22*a31 - a01*a20*a32 + a00*a21*a32)*one_over_denom;
     
    a_inverse[3][2] = (a02*a11*a30 - a01*a12*a30 - a02*a10*a31 + a00*a12*a31 +
		       a01*a10*a32 -  a00*a11*a32) * one_over_denom;

    a_inverse[3][3] = (-(a02*a11*a20) + a01*a12*a20 + a02*a10*a21 - 
		       a00*a12*a21 - a01*a10*a22 + a00*a11*a22)*one_over_denom;
    return;
  }
  if (numMatls > 4) {    // numMatls X numMatls; with numMatls > 4
    a_inverse= a;
    a_inverse.invert();
  }
}
/*---------------------------------------------------------------------
 Function~  ICE::MatrixMultiplication--
 Reference~  This multiplies matrix (a) and vector (b) and returns X
 ---------------------------------------------------------------------  */
 void ICE::multiplyMatrixAndVector( int numMatls,
                                    DenseMatrix& a, 
                                    vector<double>& b, 
                                    vector<double>& X  )
{
  for (int row=0; row<numMatls; row++) {
    X[row] = 0.0;
    for (int col=0; col<numMatls; col++) {
      X[row] += a[row][col]*b[col];
    }
  }
}
/*---------------------------------------------------------------------
 Function~  ICE::conditionNumber--
 Reference~  Computes the condition number of a matrix
 ---------------------------------------------------------------------  */
double ICE::conditionNumber( const int numMatls,
                           const DenseMatrix& a)
{
                         
   //   Check for ill-conditioned system
   // - calculate the inverse
   // - compute the max row sum norm for (a) and the inverse
   // - condition_number = max_row_sum_a * max_row_sum_a_inverse
   DenseMatrix acopy(numMatls, numMatls), a_invert(numMatls, numMatls);
   acopy   = a;
   a_invert= acopy;
   a_invert.invert();
   double max_row_sum_a         = 0.0;
   double max_row_sum_a_invert  = 0.0;

   for(int m = 0; m < numMatls; m++) {
     double row_sum_a = 0.0;
     double row_sum_a_invert = 0.0;
     for(int n = 0; n < numMatls; n++)  {  
       row_sum_a += fabs(acopy[m][n]);
       row_sum_a_invert += fabs(a_invert[m][n]);
     }
     max_row_sum_a = max(max_row_sum_a, row_sum_a);
     max_row_sum_a_invert = max(max_row_sum_a_invert, row_sum_a_invert);
   }
   return  max_row_sum_a * max_row_sum_a_invert;
 }   
/*---------------------------------------------------------------------
 Function~  ICE::matrixSolver--
 Reference~  Mathematica provided the code
 ---------------------------------------------------------------------  */
void ICE::matrixSolver( int numMatls,
                        DenseMatrix& a, 
                        vector<double>& b, 
                        vector<double>& X  )
{
  if (numMatls == 1) {
    X[0] = b[0]/a[0][0];
    return;
  }   

  if (numMatls == 2) {   // 2 X 2 Matrix
    //__________________________________
    // Example Problem Hilbert Matrix
    //  Exact solution is 1,1
    /*
    a[0][0] = 1.0;   a[0][1] = 1/2.0;
    a[1][0] = 1/2.0; a[1][1] = 1/3.0;
    b[0] = 3.0/2.0;   b[1] = 5.0/6.0;*/
    
    double a00 = a[0][0], a01 = a[0][1];
    double a10 = a[1][0], a11 = a[1][1];
    double b0 = b[0], b1 = b[1];

    double one_over_denom = 1./(a00*a11 - a01*a10);
    X[0] = (a11*b0 - a01*b1)*one_over_denom;
    X[1] = (a00*b1-a10*b0)*one_over_denom;
    return; 
  } 

  if (numMatls == 3) {   // 3 X 3 Matrix
    double a00 = a[0][0], a01 = a[0][1], a02 = a[0][2];
    double a10 = a[1][0], a11 = a[1][1], a12 = a[1][2];
    double a20 = a[2][0], a21 = a[2][1], a22 = a[2][2];
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
    return;
  }      

  if (numMatls == 4) {   // 4 X 4 matrix
    double a00 = a[0][0], a01 = a[0][1], a02 = a[0][2], a03 = a[0][3];
    double a10 = a[1][0], a11 = a[1][1], a12 = a[1][2], a13 = a[1][3];
    double a20 = a[2][0], a21 = a[2][1], a22 = a[2][2], a23 = a[2][3];
    double a30 = a[3][0], a31 = a[3][1], a32 = a[3][2], a33 = a[3][3];
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
  return;
  }

  if (numMatls > 4) {    // numMatls X numMatls; with numMatls > 4
    a.solve(b);
    X = b;
  }
}
