c*********************************************************************
c
c $Id$
c
c*********************************************************************

      double precision function gnrm2(NVAR,XX,INCX)

      implicit none

      integer NVAR, INCX
      double precision XX(*)

      double precision DNRM2
C***********************************************************************
      gnrm2 = DNRM2(NVAR,XX,INCX)

      RETURN
      END

c*********************************************************************
c
c $Log$
c Revision 1.1  2000/04/12 20:31:46  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************

