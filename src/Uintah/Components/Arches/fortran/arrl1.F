c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE arrl1(indexLow, indexHigh, 
     &     A,ALPHA,ALPAVE)
C-----------------------------------------------------------------------
C     WRITTEN BY SESHADRI KUMAR, OCTOBER 14, 1996
C
C     CALCULATES THE L1 VECTOR NORM OF THE MATRIX A AND STORES IT IN
C     ALPHA
C
C     I.E., ALPHA = SUMMATION(ABS(A(I,J,K))
C
C     ALSO CALCULATES THE L1 MEAN VALUE OF A, ALPAVE.
C
C     THIS SUBROUTINE ASSUMES THAT A IS DIMENSIONED (NX,NY,NZ).
C     (IST,JST,KST) AND (IEND,JEND,KEND) ARE THE STARTING AND ENDING 
C     LOCATIONS WITHIN WHICH THE OPERATION IS TO BE PERFORMED.
C     
C-----------------------------------------------------------------------
      implicit none

#include "param4.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer indexLow(3)
      integer indexHigh(3)
      double precision A(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision ALPHA, ALPAVE

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer IST,JST,KST,IEND,JEND,KEND
      integer i,j,k,NTOTAL

c*********************************************************************
c     Start :
c*********************************************************************
      IST = indexLow(1)+1
      JST = indexLow(2)+1
      KST = indexLow(3)+1
      IEND = indexHigh(1)-1
      JEND = indexHigh(2)-1
      KEND = indexHigh(3)-1

C-----------------------------------------------------------------------
C     EXECUTABLE STATEMENTS
C-----------------------------------------------------------------------
      ALPHA = ZERO
      DO 300 K=KST,KEND
         DO 200 J=JST,JEND
            DO 100 I=IST,IEND
               ALPHA = ALPHA + ABS(A(I,J,K))
 100        CONTINUE
 200     CONTINUE
 300  CONTINUE
      NTOTAL = (KEND-KST+1)*(JEND-JST+1)*(IEND-IST+1)
      ALPAVE = ALPHA/NTOTAL
      RETURN
      END

c*********************************************************************
c
c $Log$
c Revision 1.1  2000/04/12 20:31:42  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************

