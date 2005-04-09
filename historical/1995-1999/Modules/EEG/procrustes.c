///////1///////2////////3////////4/////////5////////6/////
//
//
//
// calculation of displacement transformation          /| |\
// matrix for rigid body motion from position          \|_|/
// one to position two as described by the global       / \
// locations of N landmarks.                           <   >
//
// Input:
// N   = number of landmarks (N >= 3)
// ax  = landmark x location vector of length N at position 1
// ay  = landmark y location vector of length N at position 1
// az  = landmark z location vector of length N at position 1 
// px  = landmark x location vector of length N at position 2
// py  = landmark y location vector of length N at position 2
// pz  = landmark z location vector of length N at position 2
//
// OUTPUTS
// TT - 4x4 transformation matrix
//  	| 1 |     |      |   | 1 |
//  	| x |  =  |  TT  | * | x |
//  	| y |     |      |   | y |
//  	| z |     |      |   | z |
//  	pos 2                pos 1
//
// References: A procedure for Determining Rigid Body Transformation 
//		Parameters.
//		John H. Challis, J. Biomech. 28(6), pp 733-737
//   Last update: Sat Apr 13 16:11:06 1996
//     - created from first version of the routine, but with a
//	 new algorithm
//////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cutil.h"
//////////// Prototype /////////////////////////////////
void MatrixMult3( double A[3][3], double B[3][3], double P[3][3] );
void PrintMatrix(char *name, double A[3][3] );
void PrintMatrix4(char *name, double A[4][4] );
void TestDeterminant( double tmatrix[4][4] );
void TestDeterminant3( double rot[3][3] );
void ToLin( int nrows, int ncols, double *amatrix, double *linmatrix );
void ToMat( int nrows, int ncols, double *linmatrix, double *amatrix );
double Det3Val( double matrix[3][3] );
extern "C" int dsyev_(char *jobz, char *uplo, int *n, double *a,
		      int *lda, double *w, double *work, int *lwork, 
		      int *info);
extern "C" int dgesvd_(char *jobu, char *jobvt, int *m, int *n, 
		       double *a, int *lda, double *s, double *u, int *
		       ldu, double *vt, int *ldvt, double *work, int *lwork, 
		       int *info);

//
void Procrustes(int npts, double ax[], double ay[], double az[], double px[], 
	    double py[], double pz[], double TT[4][4])
{
    double M[3][3]; // Matrix and its transpose
    double R[3][3]; // Rotation matrix;
    double chiral[3][3], prod1[3][3];
    double fnpts = (double)npts;
    double axmean = 0, aymean = 0, azmean = 0;
    double pxmean = 0, pymean = 0, pzmean = 0;
    int i, j;
    long reportlevel=0;
/**********************************************************************/  
    
 // Initialize matrices.
    
    for (i = 0; i < 3; i++)	
    {
	for (j=0; j<3; j++)
	{
	    chiral[i][j] = 0.;
	    M[i][j] = 0.;
	}
	chiral[i][i] = 1.0;
    }
    TT[0][0] = 1.0; TT[0][1] = 0.0; TT[0][2] = 0.0; TT[0][3] = 0.0;
    
//	Make up the means we need and the M matrix. First as sums,
    for (i = 0; i < npts; i++)	
    {
	axmean = axmean + ax[i];
	aymean = aymean + ay[i];
	azmean = azmean + az[i];
	pxmean = pxmean + px[i];
	pymean = pymean + py[i];
	pzmean = pzmean + pz[i];
	M[0][0] = M[0][0] + px[i] * ax[i];
	M[1][0] = M[1][0] + py[i] * ax[i];
	M[2][0] = M[2][0] + pz[i] * ax[i];
	M[0][1] = M[0][1] + px[i] * ay[i];
	M[1][1] = M[1][1] + py[i] * ay[i];
	M[2][1] = M[2][1] + pz[i] * ay[i];
	M[0][2] = M[0][2] + px[i] * az[i];
	M[1][2] = M[1][2] + py[i] * az[i];
	M[2][2] = M[2][2] + pz[i] * az[i];
    }
  
//	and now as means
    axmean = axmean / fnpts;
    aymean = aymean / fnpts;
    azmean = azmean / fnpts;
    pxmean = pxmean / fnpts;
    pymean = pymean / fnpts;
    pzmean = pzmean / fnpts;
    M[0][0] = M[0][0] / fnpts - pxmean * axmean;
    M[1][0] = M[1][0] / fnpts - pymean * axmean;
    M[2][0] = M[2][0] / fnpts - pzmean * axmean;
    M[0][1] = M[0][1] / fnpts - pxmean * aymean;
    M[1][1] = M[1][1] / fnpts - pymean * aymean;
    M[2][1] = M[2][1] / fnpts - pzmean * aymean;
    M[0][2] = M[0][2] / fnpts - pxmean * azmean;
    M[1][2] = M[1][2] / fnpts - pymean * azmean;
    M[2][2] = M[2][2] / fnpts - pzmean * azmean;

    if ( reportlevel ) 
	PrintMatrix("M", M);
    
    
 // And send it off to be decomposed by SVD.
// First load it into a linear array for processing.

    char jobu = 'A';
    int n = 3;
    int info;
    int lda = 3;
    int lwork = 16;
    double singvalues[3], mlin[9], umatrix[3][3], ulin[9];
    double vtmatrix[3][3], vlin[9], work[16];
    double *m_p, *um_p, *vtm_p;
    m_p = &M[0][0];
    um_p = &umatrix[0][0];
    vtm_p = &vtmatrix[0][0];
    
    ToLin( lda, lda, m_p, mlin );
    if ( reportlevel ) 
    {
	printf(" Linear version of M is \n");
	for (i=0; i<9; i++)
	{
	    printf(" %f ", mlin[i]);
	}
	printf("\n");
    }

    dgesvd_(&jobu, &jobu, &n, &n, mlin, &lda, singvalues, ulin, &lda,
	   vlin, &lda, work, &lwork, &info);
    if ( info != 0) {
	printf(" Failed return from dsyev - info = %d\n", info);
	exit ((int)info);
    }
    ToMat( lda, lda, ulin, um_p );
    ToMat( lda, lda, vlin, vtm_p );
    if ( reportlevel ) 
    {
	PrintMatrix("umatrix", umatrix);
	PrintMatrix("vtmatrix", vtmatrix);
    
	printf(" Singular values are %f, %f, %f\n",
	       singvalues[0], singvalues[1], singvalues[2] );
    }

 //  Find the determinant of the umatrix * vtmatrix product

    chiral[2][2] = Det3Val( umatrix ) * Det3Val( vtmatrix );
    
 // Now make up the R matrix

    MatrixMult3(umatrix, chiral, prod1);
    MatrixMult3(prod1, vtmatrix, R);
    
    // Test the rotational part of the matrix and make sure that the 
    // determinant is equal to 1 and not -1.  This may mean reversing the
    /// sign in all the coefficients.

    if ( reportlevel ) 
	PrintMatrix("R before correction", R);

    TestDeterminant3( R );

    if ( reportlevel ) 
	PrintMatrix("R after correction", R);

//  Set up the full transform matrix.

    TT[1][1] = R[0][0];  TT[1][2] = R[0][1];  TT[1][3] = R[0][2];
    TT[2][1] = R[1][0];  TT[2][2] = R[1][1];  TT[2][3] = R[1][2];
    TT[3][1] = R[2][0];  TT[3][2] = R[2][1];  TT[3][3] = R[2][2];
    
//	displacement vector
    TT[1][0] = pxmean - R[0][0] * axmean - R[0][1] * aymean - R[0][2] * azmean;
    TT[2][0] = pymean - R[1][0] * axmean - R[1][1] * aymean - R[1][2] * azmean;
    TT[3][0] = pzmean - R[2][0] * axmean - R[2][1] * aymean - R[2][2] * azmean;

    if ( reportlevel ) 
	PrintMatrix4(" At end of procrustes", TT);
   
}


/*======================================================================*/
void TestDeterminant( double tmatrix[4][4] )
{
    
 /*** Test this matrix and make sure the rotational part of it as a 
      determinant equal to one.  Sometimes it comes out -1 and this
      flips things inside out. ***/

    double determ;
    double rot[3][3];
/**********************************************************************/

    for (int i=0; i<3; i++)
    {
	for (int j=0; j<3; j++ )
	{
	    rot[i][j] = tmatrix[i+1][j+1];
	}
    }
    
    determ = rot[0][0] * (rot[1][1] * rot[2][2] - rot[1][2] * rot[2][1] )
	- rot[0][1] * (rot[1][0] * rot[2][2] - rot[1][2] * rot[2][0] ) 
	+ rot[0][2] * (rot[1][0] * rot[2][1] - rot[1][1] * rot[2][0] );
	
    if ( (fabs(determ) - 1.0 ) > 1e-6 )
    {
	printf(" Error in testDeterminant because determ = %f\n",
	       determ);
    }
    else if ( determ < 0.0 )
    {
	printf(" In TestDetermin we have determ = %f"
	       " so we have to flip sign of all the matrix elements\n",
	       determ);
	for (i=1; i<4; i++)
	{
	    for (int j=1; j<4; j++)
	    {
		tmatrix[i][j] *= -1.0;
	    }
	}
    }else
    {
	printf("\n In TestDeterminant, everything is fine\n");
    }
}

    
/*======================================================================*/
void TestDeterminant3( double rot[3][3] )
{
    
 /*** Test this matrix and make sure the rotational part of it as a 
      determinant equal to one.  Sometimes it comes out -1 and this
      flips things inside out. ***/

    double determ;
/**********************************************************************/

    determ = rot[0][0] * (rot[1][1] * rot[2][2] - rot[1][2] * rot[2][1] )
	- rot[0][1] * (rot[1][0] * rot[2][2] - rot[1][2] * rot[2][0] ) 
	+ rot[0][2] * (rot[1][0] * rot[2][1] - rot[1][1] * rot[2][0] );
	
    if ( (fabs(determ) - 1.0 ) > 1e-6 )
    {
	printf(" Error in testDeterminant because determ = %f\n",
	       determ);
    }
    else if ( determ < 0.0 )
    {
	printf(" In TestDetermin we have determ = %f"
	       " so we have to flip sign of all the matrix elements\n",
	       determ);
	for (int i=0; i<3; i++)
	{
	    for (int j=0; j<3; j++)
	    {
		rot[i][j] *= -1.0;
	    }
	}
    }else
    {
	printf("\n In TestDeterminant3, everything is fine\n");
    }
}

    
