c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/intgrt_fort.h>
#else
      SUBROUTINE intgrt(indexLow, indexHigh, PCELL,
     &     UU,VV,WW,
     &     DIR,PHI,H2O,AVGING,DRYING,XXX,NEND,
     &     DEN,
     &     XX, YY, ZZ,
     &     SEW, SNS, STB,
     &     RA, RV,
     &     WTM,
     &     ierr)

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
      double precision UU(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision VV(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision WW(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      integer DIR
      double precision PHI(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision H2O(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision AVGING(indexLow(1):indexHigh(1))
      double precision DRYING(indexLow(1):indexHigh(1))
      double precision XXX(indexLow(1):indexHigh(1))
      integer NEND
      double precision DEN(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision XX(indexLow(1):indexHigh(1))
      double precision YY(indexLow(2):indexHigh(2))
      double precision ZZ(indexLow(3):indexHigh(3))
      double precision SEW(indexLow(1):indexHigh(1))
      double precision SNS(indexLow(2):indexHigh(2))
      double precision STB(indexLow(3):indexHigh(3))
      double precision RV(indexLow(2):indexHigh(2))
      double precision RA(indexLow(2):indexHigh(2))
      double precision WTM(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      integer ierr

c*********************************************************************
c     Local :
c*********************************************************************
      integer iBegGhost, iEndGhost, iBegActual, iEndActual
      integer jBegGhost, jEndGhost, jBegActual, jEndActual
      integer kBegGhost, kEndGhost, kBegActual, kEndActual

      integer i, j, k
      double precision SUMFLW, ARFLOW

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
      IF (DIR.EQ.IDIR) THEN
         DO 20 I = iBegActual,iEndActual
            SUMFLW       = ZERO
            AVGING(I)    = ZERO
            DRYING(I)    = ZERO
            XXX(I) = XX(I)
            DO 10 J = jBegActual,jEndActual
               DO 5 K = kBegActual,kEndActual
                  IF(PCELL(I,J,K).NE.WALL) THEN
                     ARFLOW    = RA(J)*SNS(J)*STB(K)*DEN(I,J,K)*
     &                    ABS(UU(I,J,K)+UU(I+1,J,K))*PT5*WTM(I,J,K)
                     SUMFLW    = SUMFLW + ARFLOW 
                     AVGING(I) = AVGING(I)  + PHI(I,J,K)*ARFLOW
                     DRYING(I) = DRYING(I)  + PHI(I,J,K)*ARFLOW/
     &                    (ONE-H2O(I,J,K))
                  ENDIF
 5             CONTINUE
 10         CONTINUE
            AVGING(I)    = AVGING(I)/SUMFLW
            DRYING(I)    = DRYING(I)/SUMFLW
 20      CONTINUE
         NEND = iEndActual
      ELSE IF (DIR.EQ.JDIR) THEN
         DO 40 J = jBegActual,jEndActual
            SUMFLW       = ZERO
            AVGING(J)    = ZERO
            DRYING(J)    = ZERO
            XXX(J) = YY(J)
            DO 30 I = iBegActual,iEndActual
               DO 25 K = kBegActual,kEndActual
                  IF(PCELL(I,J,K).NE.WALL) THEN
                     ARFLOW    = (RV(J+1)+RV(J))*PT5*SEW(I)*STB(K)
     &                    *DEN(I,J,K)*ABS(VV(I,J,K)+VV(I,J+1,K))
     &                    *PT5*WTM(I,J,K)
                     SUMFLW    = SUMFLW + ARFLOW 
                     AVGING(J) = AVGING(J)  + PHI(I,J,K)*ARFLOW
                     DRYING(J) = DRYING(J)  + PHI(I,J,K)*ARFLOW/
     &                    (ONE-H2O(I,J,K))
                  ENDIF
 25            CONTINUE
 30         CONTINUE
            AVGING(J)    = AVGING(J)/SUMFLW
            DRYING(J)    = DRYING(J)/SUMFLW
 40      CONTINUE
         NEND = jEndActual
      ELSE IF (DIR.EQ.KDIR) THEN
         DO 60 K = kBegActual,kEndActual
            SUMFLW       = ZERO
            AVGING(K)    = ZERO
            DRYING(K)    = ZERO
            XXX(K) = ZZ(K)
            DO 50 J = jBegActual,jEndActual
               DO 45 I = iBegActual,iEndActual
                  IF(PCELL(I,J,K).NE.WALL) THEN
                     ARFLOW    = SEW(I)*SNS(J)*DEN(I,J,K)*
     &                    ABS(WW(I,J,K)+WW(I,J,K+1))*PT5*WTM(I,J,K)
                     SUMFLW    = SUMFLW + ARFLOW 
                     AVGING(K) = AVGING(K)  + PHI(I,J,K)*ARFLOW
                     DRYING(K) = DRYING(K)  + PHI(I,J,K)*ARFLOW/
     &                    (ONE-H2O(I,J,K))
                  ENDIF
 45            CONTINUE
 50         CONTINUE
            AVGING(K)    = AVGING(K)/SUMFLW
            DRYING(K)    = DRYING(K)/SUMFLW
 60      CONTINUE
         NEND = kEndActual
      END IF
C
      RETURN
      END

c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:47  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
