c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE apcal(indexLow, indexHigh,
     &     AP,AE,AW,AN,AS,AT,AB,SP)
C-----------------------------------------------------------------------
C     WRITTEN BY SESHADRI KUMAR, OCTOBER 14, 1996
C
C     APCAL CALCULATES THE DIAGONAL TERM AP OF THE SEVEN-DIAGONAL
C     COEFFICIENT MATRIX BY SUMMING THE OFF-DIAGONAL TERMS (AE,AW,AN,AS,
C     AT, AND AB) AND ADDING THE POSITIVE PART OF THE LINEAR SOURCE 
C     TERMS, SP.  THE CALCULATION IS ONLY DONE FOR THE INTERIOR NODES
C     WHICH ARE SOLVED FOR IN THE CALCULATION, BETWEEN (IST,JST,KST) AND
C     (IEND,JEND,KEND).
*-----------------------------------------------------------------------
      implicit none

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer indexLow(3)
      integer indexHigh(3)
      double precision AP(indexLow(1):indexHigh(1), 
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
      double precision SP(indexLow(1):indexHigh(1), 
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
               AP(I,J,K) = AE(I,J,K)+AW(I,J,K)
     $              +AN(I,J,K)+AS(I,J,K)
     $              +AT(I,J,K)+AB(I,J,K)
     $              -SP(I,J,K)
 100        CONTINUE
 200     CONTINUE
 300  CONTINUE
      RETURN
      END

c*********************************************************************
c
c $Log$
c Revision 1.1  2000/04/12 20:31:41  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
