c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/ggemv_fort.h>
#else
      SUBROUTINE ggemv(TRANS,MM,NN,ALPHA,AA,LDA,XX,INCX,BETA,YY,INCY)

      implicit none

      CHARACTER*7 TRANS
      integer MM, NN, INCX, INCY, LDA
      double precision ALPHA, BETA
      double precision XX(*),YY(*),AA(LDA,*)

      EXTERNAL SGEMV,DGEMV
C***********************************************************************
      CALL DGEMV(TRANS,MM,NN,ALPHA,AA,LDA,XX,INCX,BETA,YY,INCY)
      RETURN
      END

c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:46  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************

