/*************************************************************
Copyright (C) 1990, 1991, 1993 Andy C. Hung, all rights reserved.
PUBLIC DOMAIN LICENSE: Stanford University Portable Video Research
Group. If you use this software, you agree to the following: This
program package is purely experimental, and is licensed "as is".
Permission is granted to use, modify, and distribute this program
without charge for any purpose, provided this license/ disclaimer
notice appears in the copies.  No warranty or maintenance is given,
either expressed or implied.  In no event shall the author(s) be
liable to you or a third party for any special, incidental,
consequential, or other damages, arising out of the use or inability
to use the program for any purpose (or the loss of data), even if we
have been advised of such possibilities.  Any public reference or
advertisement of this source code should refer to it as the Portable
Video Research Group (PVRG) code, and not by any author(s) (or
Stanford University) name.
*************************************************************/
/*
************************************************************
transform.c

This file contains the reference DCT, the zig-zag and quantization
algorithms.

************************************************************
*/

/*LABEL transform.c */

#include "globals.h"
#include "dct.h"
#include <math.h>

/*PUBLIC*/

extern void ReferenceDct();
extern void ReferenceIDct();
extern void TransposeMatrix();
extern void MPEGIntraQuantize();
extern void MPEGIntraIQuantize();
extern void MPEGNonIntraQuantize();
extern void MPEGNonIntraIQuantize();
extern void BoundIntegerMatrix();
extern void BoundQuantizeMatrix();
extern void BoundIQuantizeMatrix();
extern void ZigzagMatrix();
extern void IZigzagMatrix();
extern void PrintMatrix();
extern void ClearMatrix();

static void DoubleReferenceDct1D();
static void DoubleReferenceIDct1D();
static void DoubleTransposeMatrix();

/*PRIVATE*/

static int transpose_index[] =
{0,  8, 16, 24, 32, 40, 48, 56,
 1,  9, 17, 25, 33, 41, 49, 57,
 2, 10, 18, 26, 34, 42, 50, 58,
 3, 11, 19, 27, 35, 43, 51, 59,
 4, 12, 20, 28, 36, 44, 52, 60, 
 5, 13, 21, 29, 37, 45, 53, 61,
 6, 14, 22, 30, 38, 46, 54, 62,
 7, 15, 23, 31, 39, 47, 55, 63};

static int zigzag_index[] =
{0,  1,  5,  6, 14, 15, 27, 28,
 2,  4,  7, 13, 16, 26, 29, 42,
 3,  8, 12, 17, 25, 30, 41, 43,
 9, 11, 18, 24, 31, 40, 44, 53,
10, 19, 23, 32, 39, 45, 52, 54,
20, 22, 33, 38, 46, 51, 55, 60,
21, 34, 37, 47, 50, 56, 59, 61,
35, 36, 48, 49, 57, 58, 62, 63};

#define MakeMatrix() (int *) calloc(BLOCKSIZE,sizeof(int))
#define FixedMultiply(s,x,y)  x = ((x * y) >> s);
#define DCT_OFFSET 128


/*START*/

/*BFUNC

ReferenceDct() does a reference DCT on the input (matrix) and output
(new matrix).

EFUNC*/

void ReferenceDct(matrix,newmatrix)
     int *matrix;
     int *newmatrix;
{
  BEGIN("ReferenceDct");
  int *mptr;
  double *sptr,*dptr;
  double sourcematrix[BLOCKSIZE],destmatrix[BLOCKSIZE];

  for(sptr=sourcematrix,mptr=matrix;mptr<matrix+BLOCKSIZE;mptr++)
    {
      *(sptr++) = (double) *mptr;
    }
  for(dptr = destmatrix,sptr=sourcematrix;
      sptr<sourcematrix+BLOCKSIZE;sptr+=BLOCKWIDTH)

    {
      DoubleReferenceDct1D(sptr,dptr);
      dptr+=BLOCKWIDTH;
    }
  DoubleTransposeMatrix(destmatrix,sourcematrix);
  for(dptr = destmatrix,sptr=sourcematrix;
      sptr<sourcematrix+BLOCKSIZE;sptr+=BLOCKWIDTH)
    {
      DoubleReferenceDct1D(sptr,dptr);
      dptr+=BLOCKWIDTH;
    }
  DoubleTransposeMatrix(destmatrix,sourcematrix);
  for(sptr = sourcematrix,mptr=newmatrix;
      mptr<newmatrix+BLOCKSIZE;sptr++)
    {    /* NB: Inversion on counter */
      *(mptr++) = (int) (*sptr > 0 ? (*(sptr)+0.5):(*(sptr)-0.5));
    }
}

/*BFUNC

DoubleReferenceDCT1D() does a 8 point dct on an array of double
input and places the result in a double output.

EFUNC*/

static void DoubleReferenceDct1D(ivect,ovect)
     double *ivect;
     double *ovect;
{
  BEGIN("DoubleReferenceDct1D");
  double *mptr,*iptr,*optr;

  for(mptr=DctMatrix,optr=ovect;optr<ovect+BLOCKWIDTH;optr++)
    {
      for(*optr=0,iptr=ivect;iptr<ivect+BLOCKWIDTH;iptr++)
	{
	  *optr += *iptr*(*(mptr++));
	}
    }
}

/*BFUNC

ReferenceIDct() is used to perform a reference 8x8 inverse dct.  It is
a balanced IDCT. It takes the input (matrix) and puts it into the
output (newmatrix).

EFUNC*/

void ReferenceIDct(matrix,newmatrix)
     int *matrix;
     int *newmatrix;
{
  BEGIN("ReferenceIDct");
  int *mptr;
  double *sptr,*dptr;
  double sourcematrix[BLOCKSIZE],destmatrix[BLOCKSIZE];

  for(sptr = sourcematrix,mptr=matrix;mptr<matrix+BLOCKSIZE;mptr++)
    {
      *(sptr++) = (double) *mptr;
    }
  for(dptr = destmatrix,sptr=sourcematrix;
      sptr<sourcematrix+BLOCKSIZE;sptr+=BLOCKWIDTH)
    {
      DoubleReferenceIDct1D(sptr,dptr);
      dptr+=BLOCKWIDTH;
    }
  DoubleTransposeMatrix(destmatrix,sourcematrix);
  for(dptr = destmatrix,sptr=sourcematrix;
      sptr<sourcematrix+BLOCKSIZE;sptr+=BLOCKWIDTH)
    {
      DoubleReferenceIDct1D(sptr,dptr);
      dptr+=BLOCKWIDTH;
    }
  DoubleTransposeMatrix(destmatrix,sourcematrix);
  for(sptr = sourcematrix,mptr=newmatrix;mptr<newmatrix+BLOCKSIZE;sptr++)
    {    /* NB: Inversion on counter */
      *(mptr++) = (int) (*sptr > 0 ? (*(sptr)+0.5):(*(sptr)-0.5));
    }
}

/*BFUNC

DoubleReferenceIDct1D() does an 8 point inverse dct on ivect and
puts the output in ovect.

EFUNC*/

static void DoubleReferenceIDct1D(ivect,ovect)
     double *ivect;
     double *ovect;
{
  BEGIN("DoubleReferenceIDct1D");
  double *mptr,*iptr,*optr;

  for(mptr = IDctMatrix,optr=ovect;optr<ovect+BLOCKWIDTH;optr++)
    {
      for(*optr=0,iptr=ivect;iptr<ivect+BLOCKWIDTH;iptr++)
	{
	  *optr += *iptr*(*(mptr++));
	}
    }
}

/*BFUNC

TransposeMatrix transposes an input matrix and puts the output in
newmatrix.

EFUNC*/

void TransposeMatrix(matrix,newmatrix)
     int *matrix;
     int *newmatrix;
{
  BEGIN("TransposeMatrix");
  int *tptr;

  for(tptr=transpose_index;tptr<transpose_index+BLOCKSIZE;tptr++)
    {
      *(newmatrix++) = matrix[*tptr];
    }
}

/*BFUNC

DoubleTransposeMatrix transposes a double input matrix and puts the
double output in newmatrix.

EFUNC*/

static void DoubleTransposeMatrix(matrix,newmatrix)
     double *matrix;
     double *newmatrix;
{
  BEGIN("DoubleTransposeMatrix");
  int *tptr;

  for(tptr=transpose_index;tptr<transpose_index+BLOCKSIZE;tptr++)
    {
      *(newmatrix++) = matrix[*tptr];
    }
}  


/*BFUNC

MPEGIntraQuantize() quantizes the input matrix with a fixed DC
quantize step and an AC quantize step; along with a variable
quantization factor.

EFUNC*/

void MPEGIntraQuantize(matrix,qptr,qfact)
     int *matrix;
     int *qptr;
     int qfact;
{
  BEGIN("MPEGIntraQuantize");
  int *mptr;
  int qp;

  qp = qfact << 1;
  mptr = matrix;
  if (*mptr>0)
    *mptr=(*mptr + 4)/8;
  else
    *mptr=(*mptr - 4)/8;
  for(qptr++,mptr++;mptr<matrix+BLOCKSIZE;mptr++,qptr++)
    {
      if (*mptr>0)
	{
	  *mptr = ((*mptr << 4) + (*qptr >> 1)) / *qptr;
	  *mptr = (*mptr + qfact) / qp;
	}
      else if (*mptr < 0)
	{
	  *mptr = ((*mptr << 4) - (*qptr >> 1)) / *qptr;
	  *mptr = (*mptr - qfact) / qp;
	}
    }
}

/*BFUNC

MPEGIntraIQuantize() inverse quantizes the input matrix with a fixed
DC quantize step and an AC quantize step; along with a variable
quantization factor.

EFUNC*/

void MPEGIntraIQuantize(matrix,qptr,qfact)
     int *matrix;
     int *qptr;
     int qfact;
{
  BEGIN("MPEGIntraIQuantize");
  int *mptr;
  
  mptr=matrix;
  *mptr= *mptr * 8;
  for(qptr++,mptr++;mptr<matrix+BLOCKSIZE;mptr++,qptr++)
    {
      *mptr = (*mptr * qfact * (*qptr))/8; /* Factor of 2 disappears */
      if (!(*mptr & 1))
	{
	  if (*mptr>0) (*mptr)--;
	  else if (*mptr<0) (*mptr)++;
	}
    }
}

/*BFUNC

MPEGNonIntraQuantize() quantizes the input matrix with a quantization
matrix and a quantization factor.

EFUNC*/

void MPEGNonIntraQuantize(matrix,qptr,qfact)
     int *matrix;
     int *qptr;
     int qfact;
{
  BEGIN("MPEGNonIntraQuantize");
  int *mptr;

#ifdef VERSION1
  for(mptr=matrix;mptr<matrix+BLOCKSIZE;mptr++,qptr++)
    {
      *mptr = (8 * (*mptr)) / (*qptr * qfact);
    }
#else
  int qp = qfact<<1;
  for(mptr=matrix;mptr<matrix+BLOCKSIZE;mptr++,qptr++)
    {
      if (*mptr>0)
	{
	  *mptr = ((*mptr << 4) + (*qptr >> 1)) / *qptr;
	  if (qfact&1)
	    *mptr = (*mptr)/qp;
	  else
	    *mptr = (*mptr+1)/qp;
	}
      else if (*mptr < 0)
	{
	  *mptr = ((*mptr << 4) - (*qptr >> 1)) / *qptr;
	  if (qfact&1)
	    *mptr = (*mptr)/qp;
	  else
	    *mptr = (*mptr-1)/qp;
	}
    }
#endif
}


/*BFUNC

MPEGNonIntraIQuantize() inverse quantizes the input matrix with a
quantization matrix and a quantization factor.

EFUNC*/

void MPEGNonIntraIQuantize(matrix,qptr,qfact)
     int *matrix;
     int *qptr;
     int qfact;
{
  BEGIN("MPEGNonIntraIQuantize");
  int *mptr;
  
  for(mptr=matrix;mptr<matrix+BLOCKSIZE;mptr++,qptr++)
    {
      if (*mptr>0)
	{
	  *mptr = ((((*mptr)<<1) + 1) * qfact * (*qptr))/16;
	  if (!(*mptr & 1)) (*mptr)--;
	}
      else if (*mptr<0)
	{
	  *mptr = ((2*(*mptr) - 1) * qfact * (*qptr))/16;
	  if (!(*mptr & 1)) (*mptr)++;
	}
      /* By default, the "else" it equals 0 */
    }
}

/*BFUNC

BoundIntegerMatrix bounds the output matrix so that no pixel has a
value greater than 255 or less than 0.

EFUNC*/

void BoundIntegerMatrix(matrix)
     int *matrix;
{
  BEGIN("BoundIntegerMatrix");
  int *mptr;

  for(mptr=matrix;mptr<matrix+BLOCKSIZE;mptr++)
    {
      if (*mptr < 0) {*mptr = 0;}
      else if (*mptr > 255) {*mptr = 255;}
    }
}

/*BFUNC

BoundQuantizeMatrix() bounds the coefficients of a quantized matrix.

EFUNC*/

void BoundQuantizeMatrix(matrix)
     int *matrix;
{
  BEGIN("BoundQuantizeMatrix");
  int *mptr;

  /* Ensure that is within 255,-255   */
  /* for Huffman coding purposes. */

  for(mptr=matrix;mptr<matrix+BLOCKSIZE;mptr++)
    {
      if (*mptr < -255) {*mptr = -255;}
      else if (*mptr > 255) {*mptr = 255;}
    }
}

/*BFUNC

BoundIQuantizeMatrix() bounds an inverse quantized matrix.

EFUNC*/

void BoundIQuantizeMatrix(matrix)
     int *matrix;
{
  BEGIN("BoundIQuantizeMatrix");
  int *mptr;

  /* Ensure that is within -2048 to 2047 for limited sig-bit   */
  /* transforms. */

  for(mptr=matrix+1;mptr<matrix+BLOCKSIZE;mptr++)
    {
      if (*mptr < -2048) {*mptr = -2048;}
      else if (*mptr > 2047) {*mptr = 2047;}
    }
}

/*BFUNC

IZigzagMatrix() performs an inverse zig-zag translation on the
input imatrix and places the output in omatrix.

EFUNC*/

void IZigzagMatrix(imatrix,omatrix)
     int *imatrix;
     int *omatrix;
{
  BEGIN("IZigzagMatrix");
  int *tptr;

  for(tptr=zigzag_index;tptr<zigzag_index+BLOCKSIZE;tptr++)
    {
      *(omatrix++) = imatrix[*tptr];
    }
}

/*BFUNC

ZigzagMatrix() performs a zig-zag translation on the input imatrix
and puts the output in omatrix.

EFUNC*/

void ZigzagMatrix(imatrix,omatrix)
     int *imatrix;
     int *omatrix;
{
  BEGIN("ZigzagMatrix");
  int *tptr;

  for(tptr=zigzag_index;tptr<zigzag_index+BLOCKSIZE;tptr++)
    {
      omatrix[*tptr] = *(imatrix++);
    }
}

/*BFUNC

PrintMatrix() prints an 8x8 matrix in row/column form. 

EFUNC*/

void PrintMatrix(matrix)
     int *matrix;
{
  BEGIN("PrintMatrix");
  int i,j;

  if (matrix)
    {
      for(i=0;i<BLOCKHEIGHT;i++)
	{
	  for(j=0;j<BLOCKWIDTH;j++) {printf("%6d ",*(matrix++));}
	  printf("\n");
	}
    }
  else {printf("Null\n");}
}


/*BFUNC

ClearMatrix() sets all the elements of a matrix to be zero.

EFUNC*/

void ClearMatrix(matrix)
     int *matrix;
{
  BEGIN("ClearMatrix");
  int *mptr;

  for(mptr=matrix;mptr<matrix+BLOCKSIZE;mptr++) {*mptr = 0;}
}

/*END*/
