c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/gtrsv_fort.h>
#else
      SUBROUTINE gtrsv(UPLO,TRANS,DIAG,NN,AA,LDA,XX,INCX)

      implicit none

      CHARACTER UPLO,DIAG,TRANS
      integer NN, INCX, LDA
      double precision XX(*),AA(LDA,*)

      EXTERNAL STRSV,DTRSV
C***********************************************************************
      CALL DTRSV(UPLO,TRANS,DIAG,NN,AA,LDA,XX,INCX)
      RETURN
      END

c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:47  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
