c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/grid_fort.h>
#else
      SUBROUTINE grid(indexLow, indexHigh, PCELL,
     &     XX, YY, ZZ, 
     &     DIACH, DIAPR, DIASC,
     &     CHHGH, CHLGH, CHWID,
     &     NSUBD, NDDMN, IDDMAP,
     &     LDDI1, LDDJ1, LDDK1, LDDNI, LDDNJ, LDDNK,
     &     LRDGD, LCORD, LWRAP,
     &     ierr)
C-----------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     SUBROUTINE CALCULATES PCELL ARRAY AND THE DISTANCE IN ALL
C     THREE DIRECTIONS BETWEEN GRID POINTS
C-----------------------------------------------------------------------

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
      double precision XX(indexLow(1):indexHigh(1))
      double precision YY(indexLow(2):indexHigh(2))
      double precision ZZ(indexLow(3):indexHigh(3))
      double precision DIACH, DIAPR, DIASC
      double precision CHLGH, CHHGH, CHWID
      integer NSUBD, NDDMN
      integer IDDMAP(6)
      logical LDDI1, LDDJ1, LDDK1, LDDNI, LDDNJ, LDDNK
      logical LRDGD, LCORD, LWRAP
      integer ierr

C
c*********************************************************************
c     Locals :
c*********************************************************************
      integer i, j, k, JI, JJ, JK, KI, KJ, KK
      integer iBegGhost, iEndGhost, iBegActual, iEndActual
      integer jBegGhost, jEndGhost, jBegActual, jEndActual
      integer kBegGhost, kEndGhost, kBegActual, kEndActual
      integer NJSTR, NJMID, NJLEFT
      integer JPST,JPED,JSST,JSED
      integer NKSTR, NKMID, NKLEFT
      integer KPST,KPED,KSST,KSED
      integer JU, JD, KU, KD
      double precision CENTER, CALCLGH
      double precision DX, DYPR, DYSC, DYCH
      double precision DZ, DZPR, DZSC, DZCH
      double precision EDGEPR, EDGESC
      double precision EPSI
      double precision RADCH, RADPR, RADSC

C
c*********************************************************************
c        Get the indices of interest
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
C     PI = FOUR*ATAN(ONE)
      IF (LRDGD) THEN
         CALL GEOMIN(indexLow, indexHigh, PCELL, 
     &        XX, YY, ZZ, 
     &        NSUBD, NDDMN, IDDMAP,
     &        LDDI1, LDDJ1, LDDK1, LDDNI, LDDNJ, LDDNK,
     &        ierr)

c*********************************************************************
c        Get the indices of interest
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

         CHLGH = (XX(iEndGhost)+XX(iEndActual))/TWO
         CHWID = (YY(jEndGhost)+YY(jEndActual))/TWO
         IF (LCORD) THEN
            CHHGH = TWO*CHWID
            CHWID = CHHGH
         ELSE
            CHHGH = (ZZ(kEndGhost)+ZZ(kEndActual))/TWO
         END IF
         DO 100 I = iBegActual,iEndGhost
            IF (XX(I) .LE. XX(I-1)) STOP 'ERROR IN INPUT X S'
 100     CONTINUE
         DO 120 J = jBegActual,jEndGhost
            IF (YY(J) .LE. YY(J-1)) STOP 'ERROR IN INPUT Y S'
 120     CONTINUE
         DO 140 K = kBegActual,kEndGhost
            IF (ZZ(K) .LE. ZZ(K-1)) STOP 'ERROR IN INPUT Z S'
 140     CONTINUE
      ELSE
        RADPR = DIAPR/TWO
        RADSC = DIASC/TWO
        RADCH = MAX(DIACH,CHHGH,CHWID)/TWO
        DIACH = TWO*RADCH
        CHHGH = DIACH
        CHWID = DIACH
        DO 220 K = kBegGhost,kEndGhost
          DO 210 J = jBegGhost,jEndGhost
            DO 200 I = iBegGhost,iEndGhost
              PCELL(I,J,K) = FFIELD
              IF (J.EQ.jEndGhost) PCELL(I,J,K) = WALL
              IF (LCORD) THEN
                IF (J.EQ.jBegGhost) PCELL(I,J,K) = SYMTRY
              ELSE
                IF (J.EQ.jBegGhost) PCELL(I,J,K) = WALL
                IF (K.EQ.kBegGhost) PCELL(I,J,K) = WALL
                IF (K.EQ.kEndGhost) PCELL(I,J,K) = WALL
              END IF
              IF (I.EQ.iBegGhost) THEN
                IF (LCORD) THEN
                  NJSTR = INT(jEndGhost/4)
                  JPED = NJSTR
                  JSED = 2*NJSTR
                  IF (J.GE.jBegActual.AND.J.LE.JPED) 
     &                 PCELL(I,J,K) = PFIELD
                  IF (J.GT.JPED.AND.J.LE.JSED) PCELL(I,J,K) = SFIELD
                ELSE
                  NJMID = INT(jEndGhost/2)
                  NJSTR = INT(jEndGhost/8)
                  IF (NJSTR.LE.jBegGhost) NJSTR = jBegActual 
                  JPST = NJMID-NJSTR+2
                  JPED = NJMID+NJSTR-1
                  JSST = JPST-NJSTR
                  JSED = JPED+NJSTR
                  NKMID = INT(kEndGhost/2)
                  NKSTR = INT(kEndGhost/8)
                  IF (NKSTR.LE.kBegGhost) NKSTR = kBegActual
                  KPST = NKMID-NKSTR+2
                  KPED = NKMID+NKSTR-1
                  KSST = KPST-NKSTR
                  KSED = KPED+NKSTR
                  IF ((J.GE.JSST.AND.J.LE.JSED).AND.(K.GE.KSST.AND.
     &                 K.LE.KSED))  PCELL(I,J,K) = SFIELD
                  IF ((J.GE.JPST.AND.J.LE.JPED).AND.(K.GE.KPST.AND.
     &                 K.LE.KPED))  PCELL(I,J,K) = PFIELD
                END IF
                IF (PCELL(iBegGhost,J,K).NE.PFIELD.AND.
     &               PCELL(iBegGhost,J,K).NE.SFIELD.
     &               AND.PCELL(I,J,K).NE.SYMTRY) PCELL(I,J,K) = WALL    
              END IF
  200       CONTINUE
  210     CONTINUE
  220   CONTINUE
        IF (LCORD.AND.LWRAP) THEN
          DO 260 I = iBegGhost,iEndGhost
            DO 250 J = jBegGhost,jEndGhost
              PCELL(I,J,kBegGhost) = SYMTRY
              PCELL(I,J,kEndGhost) = SYMTRY
  250       CONTINUE
  260     CONTINUE
        END IF
        EPSI = 1.1D0
C------------------------------------------------------------------------
C     CALCULATION OF X'S
C------------------------------------------------------------------------
        DX = ONE
        XX(iBegActual) = PT5
        DO 300 I = iBegActual+1,iEndActual
           DX = EPSI*DX
           XX(I) = DX+XX(I-1)
 300    CONTINUE
        XX(iEndGhost) = XX(iEndActual)+DX
        CALCLGH = (XX(iEndGhost)+XX(iEndActual))/TWO
        DX = CHLGH/CALCLGH
        XX(1) = -DX/TWO
        DO 350 I = iBegActual,iEndActual
           XX(I) = XX(I-1)+DX
           DX = EPSI*DX
 350    CONTINUE
        DX = DX/EPSI
        XX(iEndGhost) = XX(iEndActual)+DX
C------------------------------------------------------------------------
C     CALCULATION OF Y'S
C------------------------------------------------------------------------
        IF (LCORD) THEN
           DYPR = RADPR/(NJSTR-1)
           YY(1) = -DYPR/TWO
           DO 400 JI = jBegActual,JPED+1
              YY(JI) = YY(JI-1)+DYPR
 400       CONTINUE
           DYSC = TWO*(RADSC-YY(JPED+1))/(2*NJSTR-1)
           DO 420 JJ = JPED+2,JSED+1
              YY(JJ) = YY(JJ-1)+DYSC
 420       CONTINUE
           DYCH = TWO*(RADCH-YY(JSED+1))/(2*(jEndGhost-2*NJSTR)-3)
           DO 440 JK = JSED+2,jEndGhost
              YY(JK) = YY(JK-1)+DYCH
 440       CONTINUE
        ELSE
           CENTER = DIACH/TWO
           DYPR = DIAPR/(TWO*NJSTR-TWO)
           YY(jEndGhost/2) = CENTER - DYPR/TWO
           YY(jEndGhost/2+1) = CENTER + DYPR/TWO
           DO 460 JI = jBegActual,NJSTR
              JU = (jEndGhost/2+JI)
              JD = (jEndGhost/2+1-JI)
              YY(JU) = YY(JU-1)+DYPR
              YY(JD) = YY(JD+1)-DYPR
 460       CONTINUE
          EDGEPR = YY(jEndGhost/2+NJSTR)-RADCH
          DYSC = TWO*(RADSC-EDGEPR)/(TWO*NJSTR-ONE)
          DO 480 JJ = jBegGhost,NJSTR
             JU = jEndGhost/2+NJSTR+JJ
             JD = jEndGhost/2+1-NJSTR-JJ
             YY(JU) = YY(JU-1)+DYSC
             YY(JD) = YY(JD+1)-DYSC
 480      CONTINUE
          EDGESC = YY(jEndGhost/2+2*NJSTR)-RADCH
          NJLEFT = (jEndGhost-4*NJSTR)/2
          DYCH = TWO*(RADCH-EDGESC)/(TWO*NJLEFT-ONE)
          DO 500 JK = jBegGhost,NJLEFT
             JU = jEndGhost/2+2*NJSTR+JK
             JD = jEndGhost/2+1-2*NJSTR-JK
             YY(JU) = YY(JU-1)+DYCH
             YY(JD) = YY(JD+1)-DYCH
 500      CONTINUE
        END IF
C-----------------------------------------------------------------------
C     CALCULATIONS OF Z'S
C------------------------------------------------------------------------
        IF (LCORD) THEN
           DZ = TWO*PI/(kEndGhost-2)
           ZZ(1) = -DZ/TWO
           DO 600 K = kBegActual,kEndGhost
              ZZ(K) = ZZ(K-1)+DZ
 600       CONTINUE
        ELSE
          CENTER = DIACH/TWO
          DZPR = DIAPR/(TWO*NKSTR-TWO)
          ZZ(kEndGhost/2) = CENTER - DZPR/TWO
          ZZ(kEndGhost/2+1) = CENTER + DZPR/TWO
          DO 650 KI = kBegActual,NKSTR
            KU = (kEndGhost/2+KI)
            KD = (kEndGhost/2+1-KI)
            ZZ(KU) = ZZ(KU-1)+DZPR
            ZZ(KD) = ZZ(KD+1)-DZPR
  650     CONTINUE
          EDGEPR = ZZ(kEndGhost/2+NKSTR)-RADCH
          DZSC = TWO*(RADSC-EDGEPR)/(TWO*NKSTR-ONE)
          DO 700 KJ = kBegGhost,NKSTR
            KU = kEndGhost/2+NKSTR+KJ
            KD = kEndGhost/2+1-NKSTR-KJ
            ZZ(KU) = ZZ(KU-1)+DZSC
            ZZ(KD) = ZZ(KD+1)-DZSC
  700     CONTINUE
          EDGESC = ZZ(kEndGhost/2+2*NKSTR)-RADCH
          NKLEFT = (kEndGhost-4*NKSTR)/2
          DZCH = TWO*(RADCH-EDGESC)/(TWO*NKLEFT-ONE)
          DO 750 KK = kBegGhost,NKLEFT
            KU = kEndGhost/2+2*NKSTR+KK
            KD = kEndGhost/2+1-2*NKSTR-KK
            ZZ(KU) = ZZ(KU-1)+DZCH
            ZZ(KD) = ZZ(KD+1)-DZCH
  750     CONTINUE
        END IF
      END IF
C
      RETURN
      END

c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:46  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
