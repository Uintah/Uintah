c*********************************************************************
c
c
c*********************************************************************

      SUBROUTINE grot(NVAR,XX,INCX,YY,INCY,CC,SS)

      implicit none

      integer NVAR, INCX, INCY
      double precision XX(*), YY(*)
      double precision CC, SS

      EXTERNAL SROT,DROT
C***********************************************************************
      CALL DROT(NVAR,XX,INCX,YY,INCY,CC,SS)
      RETURN
      END

c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:46  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
