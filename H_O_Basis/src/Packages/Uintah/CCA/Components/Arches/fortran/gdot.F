c*********************************************************************
c
c
c*********************************************************************

      double precision function gdot(NVAR,XX,INCX,YY,INCY)

      implicit none
      integer NVAR
      double precision XX(*),YY(*)
      integer INCX, INCY

      double precision DDOT
C***********************************************************************
      gdot = DDOT(NVAR,XX,INCX,YY,INCY)
      RETURN
      END

c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:45  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
