      subroutine fixradval(val,
     $     aw, as, ab, ap, su, const)


c*********************************************************************
c
c
c*********************************************************************
C-----------------------------------------------------------------------
C     WRITTEN BY Gautham, OCTOBER 14, 2002
C
C     fixval FIXES THE VALUE OF A VARIABLE AT THE VALUE VAL AT THE
C     POSITION (I,J,K) BY SETTING THE OFF-DIAGONALS, AE,AW,AN,AS,AT,AB,
C     OF THAT EQUATION TO ZERO AND THE DIAGONAL, AP, TO THAT VALUE.
C     
C-----------------------------------------------------------------------
      implicit none


c*********************************************************************
c     Arguments :
c*********************************************************************
      double precision VAL
      double precision AW
      double precision AS
      double precision AB
      double precision AP
      double precision SU
      double precision CONST
c     Local Variables :
c*********************************************************************

c*********************************************************************
c     Start
c*********************************************************************
C-----------------------------------------------------------------------
      SU = CONST*VAL
      AP = CONST
      AW = 0.0d0
      AS = 0.0d0
      AB = 0.0d0
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
