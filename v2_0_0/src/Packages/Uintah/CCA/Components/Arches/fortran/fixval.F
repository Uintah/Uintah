c*********************************************************************
c
c
c*********************************************************************

      SUBROUTINE fixval(VAL,
     &     AE,AW,AN,AS,AT,AB,SU,SP,
     &     CONST)
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
      double precision VAL
      double precision AE
      double precision AW
      double precision AN
      double precision AS
      double precision AT
      double precision AB
      double precision SU
      double precision SP
      double precision CONST

c*********************************************************************
c     Local Variables :
c*********************************************************************

c*********************************************************************
c     Start
c*********************************************************************
C-----------------------------------------------------------------------
      SU = CONST*VAL
      SP = -CONST
      AE = ZERO
      AW = ZERO
      AN = ZERO
      AS = ZERO
      AT = ZERO
      AB = ZERO
C-----------------------------------------------------------------------
      RETURN
      END

c*********************************************************************
c
c Revision 1.2  2000/07/14 03:45:48  rawat
c completed velocity bc and fixed some bugs
c
c Revision 1.1  2000/04/12 20:31:45  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
