c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE symbcs(indexLow, indexHigh, PCELL,
     &     PHI,
     &     AE,AW,AN,AS,AT,AB,
     &     LPRDJK, LPRDKK)
C--------------------------------------------------------------------
C     Subroutine provides symmtry boundary conditions for the 
C         following variables: p, pp, u, v, w, tke, and ted.
C-------------------------------------------------------------------- 

      implicit none

#include "param4.h"
#include "pcell.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer indexLow(3)
      integer indexHigh(3)
      integer PCELL(indexLow(1):indexHigh(1), indexLow(2):indexHigh(2), 
     &     indexLow(3):indexHigh(3))
      double precision PHI(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AE(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AW(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AN(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AS(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AT(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AB(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      logical LPRDKK,LPRDJK

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer i, j, k
      integer iBegGhost, iEndGhost, iBegActual, iEndActual
      integer jBegGhost, jEndGhost, jBegActual, jEndActual
      integer kBegGhost, kEndGhost, kBegActual, kEndActual
      
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
c     Start :
c*********************************************************************
C$DIR SCALAR
      DO 420 K = kBegGhost,kEndGhost
C$DIR SCALAR
         DO 410 J = jBegGhost,jEndGhost
C$DIR SCALAR
            DO 400 I = iBegGhost,iEndGhost
C--------------------------------------------------------------------
C           SYMMETRY BOUNDARIES
C--------------------------------------------------------------------
               IF (PCELL(I,J,K).EQ.SYMTRY) THEN
                  IF (K.EQ.kBegGhost) THEN
                     IF (.NOT.(LPRDKK.OR.LPRDJK)) THEN
                        PHI(I,J,kBegGHost) = PHI(I,J,kBegActual)
                        AB(I,J,kBegActual) = ZERO
                     END IF
                  ELSE IF (K.EQ.kEndGhost) THEN
                     IF (.NOT.LPRDKK) THEN
                        PHI(I,J,kEndGhost) = PHI(I,J,kEndActual)
                        AT(I,J,kEndActual) = ZERO
                     END IF
                  ELSE IF (J.EQ.jBegGhost) THEN
                     IF (.NOT.LPRDJK) THEN
                        PHI(I,jBegGhost,K) = PHI(I,jBegActual,K)
                        AS(I,jBegActual,K) = ZERO
                     END IF
                  ELSE IF (J.EQ.jEndGhost) THEN
                     PHI(I,jEndGhost,K) = PHI(I,jEndActual,K)
                     AN(I,jEndActual,K) = ZERO
                  ELSE IF (I.EQ.iBegGhost) THEN
                     PHI(iBegGhost,J,K) = PHI(iBegActual,J,K)
                     AW(iBegActual,J,K) = ZERO
                  ELSE IF (I.EQ.iEndGhost) THEN
                     PHI(iEndGhost,J,K) = PHI(iEndActual,J,K)
                     AE(iEndActual,J,K) = ZERO
                  ELSE 
                     WRITE(6,*)' ERROR: SYMMETRY CONDITION 
     &                    SPECIFIED ON INTERIOR CELL '
                     WRITE(3,*)' ERROR: SYMMETRY CONDITION 
     &                    SPECIFIED ON INTERIOR CELL '
                  END IF
               END IF
 400        CONTINUE
 410     CONTINUE
 420  CONTINUE
      RETURN
      END

c*********************************************************************
c
c $Log$
c Revision 1.1  2000/04/12 20:31:51  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
