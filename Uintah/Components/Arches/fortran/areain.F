c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE areain(domainLow, domainHigh,
     &     indexLow, indexHigh, PCELL, PFIELD, 
     &     SEW, SNS, STB, 
c     &     RA, RV, 
     &     AREAPR,
c     &     LCORD, LPRIN,
     &     ierr)
C---------------------------------------------------------------------
C     THIS SUBROUTINE WILL CALCULATE INLET
C     AREAS FOR THE PRIMARY (AREAPR) AND SECONDARY (AREASC)
C---------------------------------------------------------------------

      implicit none

#include "param4.h"


c*********************************************************************
c     Arguments :
c*********************************************************************
      integer domainLow(3)
      integer domainHigh(3)
      integer indexLow(3)
      integer indexHigh(3)
      integer PCELL(domainLow(1):domainHigh(1), 
     &     domainLow(2):domainHigh(2), 
     &     domainLow(3):domainHigh(3))
      double precision SEW(domainLow(1):domainHigh(1)), 
     &     SNS(domainLow(2):domainHigh(2)), 
     &     STB(domainLow(3):domainHigh(3))
c      double precision RA(domainLow(2): domainHigh(2)), 
c     &     RV(domainLow(2): domainHigh(2))
      double precision AREAPR
      integer PFIELD
c      logical LCORD, LPRIN
      integer ierr

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer iBegGhost, jBegGhost, kBegGhost
      integer iEndGhost, jEndGhost, kEndGhost
      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k
      double precision AREAT

c*********************************************************************
c     Start :
c*********************************************************************
      iBegGhost = indexLow(1)
      jBegGhost = indexLow(2)
      kBegGhost = indexLow(3)
      iEndGhost = indexHigh(1)
      jEndGhost = indexHigh(2)
      kEndGhost = indexHigh(3)
      IST = iBegGhost+1
      JST = iBegGhost+1
      KST = iBegGhost+1
      IEND = iEndGhost-1
      JEND = iEndGhost-1
      KEND = iEndGhost-1

C-------------------------------------------------------------------- 
C     CALCULATION OF THE AREA OF TOTAL PRIMARY STREAMS
C--------------------------------------------------------------------
        AREAT = ZERO
        DO 110 K = KST,KEND
          DO 100 J = JST,JEND
            IF (PCELL(iBegGhost,J,K).EQ.PFIELD) THEN
              AREAT = AREAT + SNS(J)*STB(K)
            END IF
            IF (PCELL(iEndGhost,J,K).EQ.PFIELD) THEN
              AREAT = AREAT + SNS(J)*STB(K)
            END IF
  100     CONTINUE
  110   CONTINUE
        DO 140 K = KST,KEND
          DO 130 I = IST,IEND
            IF (PCELL(I,jBegGhost,K).EQ.PFIELD) THEN
              AREAT = AREAT + SEW(I)*STB(K)
c              IF(LCORD.AND.LPRIN)WRITE(3,*)' ERROR ON INLET CONDITION'
            END IF
            IF (PCELL(I,jEndGhost,K).EQ.PFIELD) THEN
              AREAT = AREAT + SEW(I)*STB(K)
            END IF
  130     CONTINUE
  140   CONTINUE
        DO 170 J = JST,JEND
          DO 160 I = IST,IEND
            IF (PCELL(I,J,kBegGhost).EQ.PFIELD) THEN
              AREAT = AREAT + SEW(I)*SNS(J)
c              IF(LCORD.AND.LPRIN)WRITE(3,*)' ERROR ON INLET CONDITION'
            END IF
            IF (PCELL(I,J,kEndGhost).EQ.PFIELD) THEN
              AREAT = AREAT + SEW(I)*SNS(J)
c              IF(LCORD.AND.LPRIN)WRITE(3,*)' ERROR ON INLET CONDITION'
            END IF
  160     CONTINUE
  170   CONTINUE
        IF (AREAT.EQ.ZERO) THEN
c           IF(LPRIN) WRITE(3,*)
c     &              ' NO PRIMARY INLET AREA IN INPUT GEOMETRY '
           WRITE(6,*) ' NO PRIMARY INLET AREA IN INPUT GEOMETRY '
        ELSE
          AREAPR = AREAT
        END IF

      RETURN
      END

c*********************************************************************
c
c $Log$
c Revision 1.2  2000/06/15 22:13:24  rawat
c modified boundary stuff
c
c Revision 1.1  2000/04/12 20:31:41  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
