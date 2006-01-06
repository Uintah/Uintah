c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/props_fort.h>
#else
      SUBROUTINE props(indexLow, indexHigh, PCELL,
     &     FF,
     &     DEN, VIS,
     &     TED, TKE, FMU,
     &     DENP,
     &     RR,
     &     VISCOS,
     &     PDEN,
     &     DENSF0, DENSF1,
     &     INITPR,
     &     NITER,
     &     LMODBC, LWRAP, LKETM, LMLTM, LCALF, LCALP,
     &     LPRDAV, LPRDJK, LPRDKK,
     &     ierr)
C~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
C
C    This subroutine calculates gas phase properties from the cardinal
C    variables, including turbulence effects when appropriate.  This
C    routine actually just calls the routines that do the calculations
C    depending on the turbulence options. 
C
C~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      implicit none

#include "param4.h"
#include "pconst.h"
#include "pcell.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer indexLow(3), indexHigh(3)
      integer PCELL(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision FF(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision DEN(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision VIS(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision TED(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision TKE(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision FMU(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision DENP(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision RR(indexLow(2):indexHigh(2))
      double precision VISCOS
      double precision PDEN
      double precision DENSF0, DENSF1
      integer INITPR
      integer NITER
      logical LMODBC, LWRAP, LKETM, LMLTM, LCALF, LCALP
      logical LPRDKK, LPRDAV, LPRDJK
      integer ierr
      
c*********************************************************************
c     Locals :
c*********************************************************************
      integer i, j, k
      integer IST, JST, KST, IEND, JEND, KEND
      double precision EPSG

C-----------------------------------------------------------------------
C       Cold flow.
C-----------------------------------------------------------------------
      IF ((NITER.GE.INITPR).AND.LCALF) THEN
         CALL PRCF(indexLow, indexHigh, PCELL,
     &     FF,
     &     DEN, DENP,
     &     PDEN,
     &     DENSF0, DENSF1,
     &     LMODBC, LCALP,
     &     ierr)
      ENDIF
C-----------------------------------------------------------------------
C
C     NOW DO THE VISCOSITY FOR THE K-EPSILON MODEL
C
C-----------------------------------------------------------------------
      IF (.NOT.LMLTM) THEN
         IF (LMODBC) THEN
            IST = indexLow(1)
            JST = indexLow(2)
            KST = indexLow(3)
            IEND = indexHigh(1)
            JEND = indexHigh(2)
            KEND = indexHigh(3)
         ELSE
            IST = indexLow(1) + 1
            JST = indexLow(2) + 1
            KST = indexLow(3) + 1
            IEND = indexHigh(1) - 1
            JEND = indexHigh(2) - 1
            KEND = indexHigh(3) - 1
         ENDIF
         DO 190 K = KST,KEND
            DO 180 J = JST,JEND
               DO 170 I = IST,IEND
                  IF (PCELL(I,J,K).NE.WALL) THEN
                     IF (LCALP) THEN
                        CALL CALEPS(indexLow, indexHigh,
     &                       DENP,PDEN,I,J,K,EPSG)
                     ELSE
                        EPSG = ONE
                     ENDIF
                     IF (.NOT.LKETM .OR. TED(I,J,K).LE.ZERO)THEN
                        VIS(I,J,K) = VISCOS*EPSG
                     ELSE IF (LKETM) THEN
                        VIS(I,J,K) = VISCOS*EPSG + FMU(I,J,K)*CMU*
     &                       DEN(I,J,K)*TKE(I,J,K)**2/TED(I,J,K) 
                     END IF
                  END IF
 170           CONTINUE
 180        CONTINUE
 190     CONTINUE
         IF (LWRAP.AND.LMODBC) THEN
            CALL PRDBC1(indexLow, indexHigh, NITER, PCELL,
     &           VIS,RR,
     &           LPRDAV, LPRDJK, LPRDKK)
         ENDIF
      END IF
C-----------------------------------------------------------------------
C     Debug and timing statements.
C-----------------------------------------------------------------------
      RETURN
C-----------------------------------------------------------------------
C     Format statements.
C-----------------------------------------------------------------------
      END

c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:49  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************

