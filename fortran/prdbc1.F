c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/prdbc1_fort.h>
#else
      SUBROUTINE prdbc1(domLo, domHi, 
     &     PHI,
     &     PCELL, SYMTRY, FFIELD,
     &     LPRDAV, LPRDJK, LPRDKK)

C--------------------------------------------------------------------
C     This subroutine handles periodic boundary conditions for
C          all scalar variables except nj (velocities must be 
C          done in modu,v,w because of directional transformations).
C
C  CURRENTLY ONLY CODED FOR K=1 PLANE PERIODIC WITH K=NK PLANE
C                   AND FOR J=1 PLANE PERIODIC WITH K=1 PLANE
C
C     MODIFIED, OCTOBER 14, 1996, BY SESHADRI KUMAR, TO MAKE
C     THE ROUTINE MORE MODULAR.
C     NOTE: I have split the prdbcs.f routine into two parts, one which
C     deals with the variable PHI alone (the current routine), and
C     another which deals with the coefficients (prdbc2.f)
C     This routine uses only one common block, which I have
C     retained since eliminating it causes too many hassles for me.
C--------------------------------------------------------------------

      implicit none

#include "param4.h"
c#include "pcell.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer domLo(3), domHi(3), idxLo(3), idxHi(3)
      double precision PHI(domLo(1):domHi(1), 
     &     domLo(2):domHi(2), domLo(3):domHi(3))
      integer PCELL(domLo(1):domHi(1), domLo(2):domHi(2), 
     &     domLo(3):domHi(3))
      integer SYMTRY, FFIELD
      logical LPRDKK, LPRDAV, LPRDJK

c*********************************************************************
c     Locals
c*********************************************************************
      integer iBegGhost, iEndGhost, IST, IEND
      integer jBegGhost, jEndGhost, JST, JEND
      integer kBegGhost, kEndGhost, KST, KEND

      integer NAVG, i, j, k, JK
      double precision PHIAVG

c*********************************************************************
c     Get the indices of interest
c*********************************************************************
      iBegGhost = domLo(1)
      jBegGhost = domLo(2)
      kBegGhost = domLo(3)
      iEndGhost = domHi(1)
      jEndGhost = domHi(2)
      kEndGhost = domHi(3)
      IST = idxLo(1)
      JST = idxLo(2)
      KST = idxLo(3)
      IEND = idxHi(1)
      JEND = idxHi(2)
      KEND = idxHi(3)

c*********************************************************************
c     Start
c*********************************************************************
c     IF (LPRDKK .AND. (NITER.EQ.1))
      IF (LPRDKK) THEN
        DO 200 J = jBegGhost,jEndGhost
           DO 100 I = iBegGhost,iEndGhost
              IF (PCELL(I,J,kBegGhost) .EQ.SYMTRY .AND.
     &             PCELL(I,J,kEndGhost).EQ.SYMTRY) THEN
                 PHIAVG = ZERO
                 NAVG = 0
                 DO 50 K=KST+1,KEND-1
                    IF (PCELL(I,J,K).EQ.FFIELD) THEN
                       PHIAVG = PHIAVG+PHI(I,J,K)
                       NAVG = NAVG + 1
                    END IF
 50              CONTINUE
                 IF(LPRDAV .AND. NAVG.NE.0) THEN
C     &          .OR. MAX(ABS(PHI(I,J,2)),ABS(PHI(I,J,NKM1)))
C     &          .GT.5.*ABS(PHIAVG)/(NK-5)
C     &          .OR. R(J) .LT. 0.1*R(NJ))
C     &          .AND. R(J) .LT. 0.1*R(NJ))
                    PHIAVG = PHIAVG/NAVG
                    PHI(I,J,kBegGhost) = PHIAVG
                    PHI(I,J,kEndGhost) = PHIAVG
                 ELSE
                    PHI(I,J,kBegGhost) = PHI(I,J,KEND)
                    PHI(I,J,kEndGhost) = PHI(I,J,KST)
                 END IF
              END IF
 100       CONTINUE
 200    CONTINUE
      ELSE IF (LPRDJK) THEN
         DO 400 JK = jBegGhost,jEndGhost
            DO 300 I = iBegGhost,iEndGhost
               IF (PCELL(I,JK,kBegGhost).EQ.SYMTRY) 
     &              PHI(I,JK,kBegGhost) = PHI(I,JST,JK)
               IF (PCELL(I,jBegGhost,JK).EQ.SYMTRY) 
     &              PHI(I,jBegGhost,JK) = PHI(I,JK,KST)
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
c*********************************************************************
