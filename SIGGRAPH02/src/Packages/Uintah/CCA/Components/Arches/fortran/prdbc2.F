c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/prdbc2_fort.h>
#else
      SUBROUTINE prdbc2(domLo, domHi, idxLo, idxHi,
     &     PHI,
     &     AS,AT,AB,
c    &     NITER, 
     &     PCELL, SYMTRY, FFIELD,
     &     LPRDAV, LPRDJK, LPRDKK)
C--------------------------------------------------------------------
C     This subroutine handles periodic boundary conditions for
C          all scalar variables except nj (velocities must be 
C          done in modu,v,w because of directional transformations).
C  CURRENTLY ONLY CODED FOR K=1 PLANE PERIODIC WITH K=NK PLANE
C                   AND FOR J=1 PLANE PERIODIC WITH K=1 PLANE
C--------------------------------------------------------------------
      implicit none

#include "param4.h"
c #include "pcell.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer domLo(3), domHi(3), idxLo(3), idxHi(3)
      double precision PHI(domLo(1):domHi(1),
     &     domLo(2):domHi(2), domLo(3):domHi(3))
      double precision AS(domLo(1):domHi(1),
     &     domLo(2):domHi(2), domLo(3):domHi(3))
      double precision AT(domLo(1):domHi(1),
     &     domLo(2):domHi(2), domLo(3):domHi(3))
      double precision AB(domLo(1):domHi(1),
     &     domLo(2):domHi(2), domLo(3):domHi(3))
c     integer NITER
      integer PCELL(domLo(1):domHi(1), domLo(2):domHi(2), 
     &     domLo(3):domHi(3))
      integer SYMTRY, FFIELD
      logical LPRDAV,LPRDKK,LPRDJK

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer iBegGhost, iEndGhost, IST, IEND
      integer jBegGhost, jEndGhost, JST, JEND
      integer kBegGhost, kEndGhost, KST, KEND

c     integer IPRDUP

      integer i, j, k, JK
      integer NAVG

      double precision PHIAVG

c*********************************************************************
c     Get the indices of interest
c*********************************************************************
      iBegGhost = domLo(1)
      jBegGhost = domLo(2)
      kBegGhost = domLo(3)
      IST = idxLo(1)
      JST = idxLo(2)
      KST = idxLo(3)
      iEndGhost = domHi(1)
      jEndGhost = domHi(2)
      kEndGhost = domHi(3)
      IEND = idxHi(1)
      JEND = idxHi(2)
      KEND = idxHi(3)

c*********************************************************************
c     Start :
c*********************************************************************
c     IPRDUP = 1
c     IF (LPRDKK .AND. (MOD(NITER,IPRDUP).EQ.0 .OR. NITER.EQ.1)) THEN
      IF (LPRDKK) THEN
         DO 200 J = jBegGhost,jEndGhost
            DO 100 I = iBegGhost,iEndGhost
               IF (PCELL(I,J,kBegGhost) .EQ.SYMTRY .AND.
     &              PCELL(I,J,kEndGhost).EQ.SYMTRY) THEN
                  PHIAVG = ZERO
                  NAVG = 0
                  DO 50 K=KST+1,KEND-1
                     IF (PCELL(I,J,K).EQ.FFIELD) THEN
                        PHIAVG = PHIAVG+PHI(I,J,K)
                        NAVG = NAVG + 1
                     END IF
 50               CONTINUE
                  IF(LPRDAV
C     &          .OR. MAX(ABS(PHI(I,J,2)),ABS(PHI(I,J,NKM1)))
C     &          .GT.5.*ABS(PHIAVG)/(NK-5)
C     &          .OR. R(J) .LT. 0.1*R(NJ))
C     &          .AND. R(J) .LT. 0.1*R(NJ))
     &          .AND. NAVG.NE.0) THEN
               ELSE
C                  AT(I,J,NKM1) = -AB(I,J,2)
               END IF
            END IF
 100     CONTINUE
 200  CONTINUE
      ELSE IF (LPRDJK) THEN
         DO 400 JK = jBegGhost,jEndGhost
            DO 300 I = iBegGhost,iEndGhost
C               AB(I,JK,2) = AS(I,2,JK)
 300        CONTINUE
 400     CONTINUE
      END IF
      RETURN
      END 

c*********************************************************************
c
c Revision 1.2  2000/07/13 04:51:35  bbanerje
c Added pressureBC (bcp) .. now called bcpress.F (bcp.F removed)
c
c Revision 1.1  2000/04/12 20:31:49  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
