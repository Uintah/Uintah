c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/gscal_fort.h>
#else
      SUBROUTINE gscal(NVAR,ALPHA,XX,INCX)

      implicit none
      
      integer NVAR, INCX
      double precision ALPHA
      double precision XX(*)

      EXTERNAL SSCAL,DSCAL
C***********************************************************************
      CALL DSCAL(NVAR,ALPHA,XX,INCX)
      RETURN
      END

c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:46  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
