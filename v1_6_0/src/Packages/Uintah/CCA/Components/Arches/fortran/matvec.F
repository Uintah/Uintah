c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/matvec_fort.h>
#else
      SUBROUTINE matvec(indexLow, indexHigh,
     &     PHI,APHI,
     &     AE,AW,AN,AS,AT,AB,AP)
C-----------------------------------------------------------------------
C     WRITTEN BY: SESHADRI KUMAR, DECEMBER 11, 1996
C
C     This routine calculates the matrix-vector product when a field
C     vector PHI (say, U,V,etc.) is multiplied by the finite difference
C     coefficient matrix A, defined by its diagonals, AE,AW,AN,AS,AT,AB,
C     and AP (the principal diagonal).  The product is calculated between
C     the limits (IST,JST,KST) and (IEND,JEND,KEND).  The resulting 
C     vector is stored in the array APHI.
C
C-----------------------------------------------------------------------
      implicit none

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer indexLow(3)
      integer indexHigh(3)
      double precision PHI(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision APHI(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision AE(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision AW(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision AN(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision AS(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision AT(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision AB(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision AP(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer IST,JST,KST,IEND,JEND,KEND
      integer i,j,k

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
      DO 300 K = KST,KEND
         DO 200 J = JST,JEND
            DO 100 I = IST,IEND

C-----CALCULATE RESIDUAL------------------------------------------------

               APHI(I,J,K) = AP(I,J,K)*PHI(I,J,K) - (
     $              AE(I,J,K)*PHI(I+1,J,K)+
     &              AW(I,J,K)*PHI(I-1,J,K)+
     &              AN(I,J,K)*PHI(I,J+1,K)+
     &              AS(I,J,K)*PHI(I,J-1,K)+
     &              AT(I,J,K)*PHI(I,J,K+1)+
     &              AB(I,J,K)*PHI(I,J,K-1))

 100        CONTINUE
 200     CONTINUE
 300  CONTINUE
C-----------------------------------------------------------------------
      RETURN
      END

c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:48  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
