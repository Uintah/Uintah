c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/assign_fort.h>
#else
      SUBROUTINE assign(X,IST,JST,KST,IEND,JEND,KEND,LTEST,PCELL)
C-----------------------------------------------------------------------
C     clips the values of X if they are negative
*-----------------------------------------------------------------------
      implicit none

#include "param4.h"
#include "pcell.h"

*-----------------------------------------------------------------------
*     passed variables

      integer ist,jst,kst,iend,jend,kend
      double precision X(IST:IEND,JST:JEND,KST:KEND)
      LOGICAL LTEST
      integer PCELL(IST:IEND,JST:JEND,KST:KEND)
*-----------------------------------------------------------------------
*     local variables

      double precision dummax
      integer icount,i,j,k
C-----------------------------------------------------------------------
      ICOUNT = 0
      dummax = zero
C-----------------------------------------------------------------------
      DO 820 K = KST,KEND
         DO 810 J = JST,JEND
            DO 800 I = IST,IEND
               IF (X(I,J,K).LT.SMALL.AND.PCELL(I,J,K).NE.WALL
     &            .AND..NOT.LTEST) THEN
                  ICOUNT = ICOUNT + 1
                  dummax = max(dummax,abs(x(i,j,k)))
                  X(I,J,K) = SMALL
               ENDIF
 800        CONTINUE
 810     CONTINUE
 820  CONTINUE
C-----------------------------------------------------------------------
      if (icount.gt.0) then
      print*,'number of negatives encountered in tke/f = ',
     &     icount
         print*,'maximum negative deviation =',dummax
      endif
C-----------------------------------------------------------------------
      RETURN
      END

c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:42  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************

