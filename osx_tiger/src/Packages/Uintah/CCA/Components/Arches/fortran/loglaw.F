c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/loglaw_fort.h>
#else
      SUBROUTINE loglaw(I,J,K,DY,U,AKVIS,UFRIC)

      implicit none

#include "param4.h"
#include "pconst.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer I, J, K
      double precision DY, U, AKVIS, UFRIC

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer NMAX, N
      double precision CONST, F, FPRIME

c*********************************************************************
c     Start :
c*********************************************************************

c     Solves the law of the wall to get the friction velocity
c     (ufric) from inputs dy and u (u = velocity at dy from wall)

      NMAX = 100

      IF (ABS(U).LE.CENT*SMALL) THEN
        UFRIC = SMALL*ABS(U)
        RETURN
      ENDIF

      CONST = ELOG*DY/AKVIS
      N = 0

  1   N = N+1
      F = U/UFRIC-LOG(CONST*UFRIC)/CAPPA
      FPRIME = -U/(UFRIC**2)-ONE/UFRIC/CAPPA
      UFRIC = MAX(SMALL,UFRIC-F/FPRIME)
      IF ((ABS(F).GT.1.0D-3).AND.(N.LT.NMAX)) THEN
        GOTO 1
      ELSEIF (N.GE.NMAX) THEN
        PRINT *,' '
        PRINT *,'NO CONVERGENCE IN LOGLAW,I,J,K,N,UFRIC',
     1               I,J,K,N,UFRIC
        PRINT *,'F,U,DY',F,U,DY
        CALL FLUSH(6)
      ENDIF

      RETURN
      END

c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:48  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
