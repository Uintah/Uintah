c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/gcopy_fort.h>
#else
      SUBROUTINE gcopy(NVAR,XX,INCX,YY,INCY)

      implicit none

      integer NVAR
      double precision XX(*),YY(*)
      integer INCX, INCY
      EXTERNAL SCOPY,DCOPY

C***********************************************************************
      CALL DCOPY(NVAR,XX,INCX,YY,INCY)
      RETURN
      END

c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:45  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
