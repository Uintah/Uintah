c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE areain(indexLow, indexHigh, PCELL, 
     &     SEW, SNS, STB, 
     &     RA, RV, 
     &     AREAPR, AREASC, 
     &     LCORD, LPRIN,
     &     ierr)
C---------------------------------------------------------------------
C     THIS SUBROUTINE WILL CALCULATE INLET
C     AREAS FOR THE PRIMARY (AREAPR) AND SECONDARY (AREASC)
C---------------------------------------------------------------------

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
      double precision SEW(indexLow(1):indexHigh(1)), 
     &     SNS(indexLow(2):indexHigh(2)), 
     &     STB(indexLow(3):indexHigh(3))
      double precision RA(indexLow(2): indexHigh(2)), 
     &     RV(indexLow(2): indexHigh(2))
      double precision AREAPR, AREASC
      logical LCORD, LPRIN
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
              AREAT = AREAT + SNS(J)*RA(J)*STB(K)
            END IF
            IF (PCELL(iEndGhost,J,K).EQ.PFIELD) THEN
              AREAT = AREAT + SNS(J)*RA(J)*STB(K)
            END IF
  100     CONTINUE
  110   CONTINUE
        DO 140 K = KST,KEND
          DO 130 I = IST,IEND
            IF (PCELL(I,jBegGhost,K).EQ.PFIELD) THEN
              AREAT = AREAT + SEW(I)*RV(jBegGhost)*STB(K)
              IF(LCORD.AND.LPRIN)WRITE(3,*)' ERROR ON INLET CONDITION'
            END IF
            IF (PCELL(I,jEndGhost,K).EQ.PFIELD) THEN
              AREAT = AREAT + SEW(I)*RV(jEndGhost)*STB(K)
            END IF
  130     CONTINUE
  140   CONTINUE
        DO 170 J = JST,JEND
          DO 160 I = IST,IEND
            IF (PCELL(I,J,kBegGhost).EQ.PFIELD) THEN
              AREAT = AREAT + SEW(I)*SNS(J)
              IF(LCORD.AND.LPRIN)WRITE(3,*)' ERROR ON INLET CONDITION'
            END IF
            IF (PCELL(I,J,kEndGhost).EQ.PFIELD) THEN
              AREAT = AREAT + SEW(I)*SNS(J)
              IF(LCORD.AND.LPRIN)WRITE(3,*)' ERROR ON INLET CONDITION'
            END IF
  160     CONTINUE
  170   CONTINUE
        IF (AREAT.EQ.ZERO) THEN
           IF(LPRIN) WRITE(3,*)
     &              ' NO PRIMARY INLET AREA IN INPUT GEOMETRY '
           WRITE(6,*) ' NO PRIMARY INLET AREA IN INPUT GEOMETRY '
        ELSE
          AREAPR = AREAT
        END IF
C--------------------------------------------------------------------
C     CALCULATE THE AREA OF THE TOTAL SECONDARY STREAMS 
C--------------------------------------------------------------------
        AREAT = ZERO
        DO 310 K = KST,KEND
          DO 300 J = JST,JEND
            IF (PCELL(iBegGhost,J,K).EQ.SFIELD) THEN
              AREAT = AREAT + SNS(J)*RA(J)*STB(K)
            END IF
            IF (PCELL(iEndGhost,J,K).EQ.SFIELD) THEN
              AREAT = AREAT + SNS(J)*RA(J)*STB(K)
            END IF
  300     CONTINUE
  310   CONTINUE
        DO 340 K = KST,KEND
          DO 330 I = IST,IEND
            IF (PCELL(I,jBegGhost,K).EQ.SFIELD) THEN
              AREAT = AREAT + SEW(I)*RV(J)*STB(K)
              IF(LCORD.AND.LPRIN)WRITE(3,*)' ERROR ON INLET CONDITION'
            END IF
            IF (PCELL(I,jEndGhost,K).EQ.SFIELD) THEN
              AREAT = AREAT + SEW(I)*RV(J)*STB(K)
            END IF
  330     CONTINUE
  340   CONTINUE
        DO 370 J = JST,JEND
          DO 360 I = IST,IEND
            IF (PCELL(I,J,kBegGhost).EQ.SFIELD) THEN
              AREAT = AREAT + SEW(I)*SNS(J)
              IF(LCORD.AND.LPRIN)WRITE(3,*)' ERROR ON INLET CONDITION'
            END IF
            IF (PCELL(I,J,kEndGhost).EQ.SFIELD) THEN
              AREAT = AREAT + SEW(I)*SNS(J)
              IF(LCORD.AND.LPRIN)WRITE(3,*)' ERROR ON INLET CONDITION'
            END IF
  360     CONTINUE
  370   CONTINUE
        IF (AREAT.EQ.ZERO) THEN
          IF (LPRIN)
     &    WRITE(3,*) ' NO SECONDARY INLET AREA IN INPUT GEOMETRY'
          WRITE(6,*) ' NO SECONDARY INLET AREA IN INPUT GEOMETRY'
        ELSE
          AREASC = AREAT
        END IF

      RETURN
      END

c*********************************************************************
c
c $Log$
c Revision 1.1  2000/04/12 20:31:41  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
