c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE fixval(indexLow, indexHigh,
     &     VAL,
     &     AE,AW,AN,AS,AT,AB,SU,SP,
     &     CONST,
     &     I,J,K)
C-----------------------------------------------------------------------
C     WRITTEN BY SESHADRI KUMAR, OCTOBER 14, 1996
C
C     fixval FIXES THE VALUE OF A VARIABLE AT THE VALUE VAL AT THE
C     POSITION (I,J,K) BY SETTING THE OFF-DIAGONALS, AE,AW,AN,AS,AT,AB,
C     OF THAT EQUATION TO ZERO AND THE DIAGONAL, AP, TO THAT VALUE.
C     
C-----------------------------------------------------------------------
      implicit none

#include "param4.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer indexLow(3)
      integer indexHigh(3)
      double precision VAL
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
      double precision SU(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision SP(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision CONST
      integer I,J,K

c*********************************************************************
c     Local Variables :
c*********************************************************************

c*********************************************************************
c     Start
c*********************************************************************
C-----------------------------------------------------------------------
      SU(I,J,K) = CONST*VAL
      SP(I,J,K) = -CONST
      AE(I,J,K) = ZERO
      AW(I,J,K) = ZERO
      AN(I,J,K) = ZERO
      AS(I,J,K) = ZERO
      AT(I,J,K) = ZERO
      AB(I,J,K) = ZERO
C-----------------------------------------------------------------------
      RETURN
      END

c*********************************************************************
c
c $Log$
c Revision 1.1  2000/04/12 20:31:45  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
