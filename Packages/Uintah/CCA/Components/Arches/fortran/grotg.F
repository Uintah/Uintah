c*********************************************************************
c
c
c*********************************************************************

      SUBROUTINE grotg(AA,BB,CC,SS)

      implicit none

      double precision AA, BB, CC, SS

      EXTERNAL SROTG,DROTG
C***********************************************************************
      CALL DROTG(AA,BB,CC,SS)
      RETURN
      END

c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:46  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
