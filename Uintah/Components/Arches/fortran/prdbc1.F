c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE prdbc1(indexLow, indexHigh, NITER, PCELL,
     &     PHI,RR,
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
#include "pcell.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer indexLow(3)
      integer indexHigh(3)
      integer NITER
      integer PCELL(indexLow(1):indexHigh(1), indexLow(2):indexHigh(2), 
     &     indexLow(3):indexHigh(3))
      double precision PHI(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision RR(*)
      logical LPRDKK, LPRDAV, LPRDJK

c*********************************************************************
c     Locals
c*********************************************************************
      integer iBegGhost, iEndGhost, iBegActual, iEndActual
      integer jBegGhost, jEndGhost, jBegActual, jEndActual
      integer kBegGhost, kEndGhost, kBegActual, kEndActual

      integer IPRDUP
      
      integer NAVG, i, j, k, JK
      double precision PHIAVG

c*********************************************************************
c     Get the indices of interest
c*********************************************************************
      iBegGhost = indexLow(1)
      jBegGhost = indexLow(2)
      kBegGhost = indexLow(3)
      iBegActual = iBegGhost + 1
      jBegActual = jBegGhost + 1
      kBegActual = kBegGhost + 1
      iEndGhost = indexHigh(1)
      jEndGhost = indexHigh(2)
      kEndGhost = indexHigh(3)
      iEndActual = iEndGhost - 1
      jEndActual = jEndGhost - 1
      kEndActual = kEndGhost - 1

c*********************************************************************
c     Start
c*********************************************************************
      IPRDUP = 1
      IF (LPRDKK .AND. (MOD(NITER,IPRDUP).EQ.0 .OR. NITER.EQ.1))
     &  THEN
        DO 200 J = jBegGhost,jEndGhost
           DO 100 I = iBegGhost,iEndGhost
              IF (PCELL(I,J,kBegGhost) .EQ.SYMTRY .AND.
     &             PCELL(I,J,kEndGhost).EQ.SYMTRY) THEN
                 PHIAVG = ZERO
                 NAVG = 0
                 DO 50 K=kBegActual+1,kEndActual-1
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
                    PHI(I,J,kBegGhost) = PHI(I,J,kEndActual)
                    PHI(I,J,kEndGhost) = PHI(I,J,kBegActual)
                 END IF
              END IF
 100       CONTINUE
 200    CONTINUE
      ELSE IF (LPRDJK) THEN
         DO 400 JK = jBegGhost,jEndGhost
            DO 300 I = iBegGhost,iEndGhost
               IF (PCELL(I,JK,kBegGhost).EQ.SYMTRY) 
     &              PHI(I,JK,kBegGhost) = PHI(I,jBegActual,JK)
               IF (PCELL(I,jBegGhost,JK).EQ.SYMTRY) 
     &              PHI(I,jBegGhost,JK) = PHI(I,JK,kBegActual)
 300        CONTINUE
 400     CONTINUE
      END IF
      RETURN
      END 

c*********************************************************************
c
c $Log$
c Revision 1.1  2000/04/12 20:31:49  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
