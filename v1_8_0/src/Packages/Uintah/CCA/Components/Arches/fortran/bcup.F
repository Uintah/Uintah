c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/bcup_fort.h>
#else
      SUBROUTINE bcup(indexLow, indexHigh, NITER, PCELL,
     &     UU, VV, WW,
     &     AE, AW, AN, AS, AT, AB,
     &     SU, SP,
     &     SUL, 
     &     VIS,
     &     TAUYX, TAUZX,
     &     UINLT, VINLT, WINLT,
     &     SEW, SNS, STB, 
     &     SEWU,
     &     RR, RA, RV, 
     &     YY, YV, ZZ, ZW,
     &     UVWINC,
     &     LZERO, LSYMB, LWRAP, LPRDAV, LPRDJK, LPRDKK, LOUTS,
     &     ierr)
C-----------------------------------------------------------------------
C     THIS SUBROUTINE CALCULATES THE BOUNDARY CONDITIONS FOR UP
C-----------------------------------------------------------------------
c
c     This subroutine at present uses the no-slip
c     boundary condition - we can change it later when necessary.
c
C-----------------------------------------------------------------------

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
      double precision AE(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision AW(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision AN(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision AS(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision AT(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision AB(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision SU(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision SP(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision SUL(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision UU(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision VV(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision WW(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision VIS(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision TAUYX(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision TAUZX(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision UINLT(indexLow(2):indexHigh(2), 
     &     indexLow(3):indexHigh(3))
      double precision VINLT(indexLow(1):indexHigh(1), 
     &     indexLow(3):indexHigh(3))
      double precision WINLT(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2))
      double precision SEW(indexLow(1):indexHigh(1))
      double precision SNS(indexLow(2):indexHigh(2))
      double precision STB(indexLow(3):indexHigh(3))
      double precision SEWU(indexLow(1):indexHigh(1))
      double precision RR(indexLow(2):indexHigh(2))
      double precision RA(indexLow(2):indexHigh(2))
      double precision RV(indexLow(2):indexHigh(2))
      double precision YY(indexLow(2):indexHigh(2))
      double precision YV(indexLow(2):indexHigh(2))
      double precision ZZ(indexLow(3):indexHigh(3))
      double precision ZW(indexLow(3):indexHigh(3))
      double precision UVWINC
      logical LZERO, LSYMB, LWRAP, LPRDAV, LPRDJK, LPRDKK, LOUTS
      integer ierr

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer iBegGhost, jBegGhost, kBegGhost
      integer iEndGhost, jEndGhost, kEndGhost
      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k, JK
      double precision TMULT
      double precision VISC
      double precision YP, ZP

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
      JST = jBegGhost+1
      KST = kBegGhost+1
      IEND = iEndGhost-1
      JEND = jEndGhost-1
      KEND = kEndGhost-1


c*********************************************************************
c     Start computation
c*********************************************************************
      LZERO = .FALSE.
      DO 120 K = kBegGhost,kEndGhost
        DO 110 J = jBegGhost,jEndGhost
          DO 100 I = iBegGhost,iEndGhost
             IF (PCELL(I,J,K).EQ.WALL) THEN
                UU(I,J,K) = ZERO
C--------------------------------------------------------------------
C             SOUTHWEST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
                IF (J.NE.jBegGhost.AND.I.NE.iBegGhost) THEN
                   IF (PCELL(I,J-1,K).EQ.FFIELD.AND.
     &                  PCELL(I-1,J-1,K).EQ.FFIELD) THEN
                      VISC = PT5*(VIS(I,J-1,K)+VIS(I-1,J-1,K))
                      YP = YV(J)-YY(J-1)
                      TMULT = VISC/YP
                      TAUYX(I,J-1,K) = -TMULT*UU(I,J-1,K)
                      SU(I,J-1,K) = SUL(I,J-1,K)
                      SP(I,J-1,K) =SP(I,J-1,K)-TMULT*SEWU(I)*RV(J)
     &                     *STB(K)
                      AN(I,J-1,K) = ZERO
                   END IF
                END IF
C--------------------------------------------------------------------
C             WEST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
                IF (I.NE.iBegGhost) THEN
                   IF(PCELL(I-1,J,K).EQ.FFIELD) AE(I-1,J,K) = ZERO
                END IF
C--------------------------------------------------------------------
C             NORTHWEST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
                IF (J.NE.jEndGhost.AND.I.NE.iBegGhost) THEN
                   IF (PCELL(I,J+1,K).EQ.FFIELD.AND.
     &                  PCELL(I-1,J+1,K).EQ.FFIELD) THEN
                      VISC = PT5*(VIS(I,J+1,K) + VIS(I-1,J+1,K))
                      YP = YY(J+1)-YV(J+1)
                      TMULT = VISC/YP
                      TAUYX(I,J+1,K) = -TMULT*UU(I,J+1,K)
                      SU(I,J+1,K) = SUL(I,J+1,K)
                      SP(I,J+1,K) =SP(I,J+1,K)-TMULT*SEWU(I)*RV(J+1)
     &                     *STB(K)
                      AS(I,J+1,K) = ZERO
                   END IF
                END IF
C--------------------------------------------------------------------
C             BOTTOM WEST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
                IF (K.NE.kBegGhost.AND.I.NE.iBegGhost) THEN
                   IF (PCELL(I-1,J,K-1).EQ.FFIELD.AND.
     &                  PCELL(I,J,K-1).EQ.FFIELD) THEN
                      VISC = PT5*(VIS(I-1,J,K-1) + VIS(I,J,K-1))
                      ZP = RA(J)*(ZW(K)-ZZ(K-1))
                      TMULT = VISC/ZP
                      TAUZX(I,J,K-1) = -TMULT*UU(I,J,K-1)
                      SU(I,J,K-1) = SUL(I,J,K-1)
                      SP(I,J,K-1) = SP(I,J,K-1)-TMULT*SEWU(I)*SNS(J)
                      AT(I,J,K-1) = ZERO
                   END IF
                END IF
C--------------------------------------------------------------------
C             TOP WEST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
                IF (K.NE.kEndGhost.AND.I.NE.iBegGhost) THEN
                   IF (PCELL(I-1,J,K+1).EQ.FFIELD.AND.
     &                  PCELL(I,J,K+1).EQ.FFIELD) THEN
                      VISC = PT5*(VIS(I-1,J,K+1) + VIS(I,J,K+1))
                      ZP = RA(J)*(ZZ(K+1)-ZW(K+1))
                      TMULT = VISC/ZP
                      TAUZX(I,J,K+1) = -TMULT*UU(I,J,K+1)
                      SU(I,J,K+1) = SUL(I,J,K+1)
                      SP(I,J,K+1) = SP(I,J,K+1)-TMULT*SEWU(I)*SNS(J)
                      AB(I,J,K+1) = ZERO
                   END IF
                END IF
C--------------------------------------------------------------------
C             IF CELL EAST OF INTRUSION IS AN INTRUSION SKIP
C             MODIFICATIONS FOR EASTERN FACES OF FIRST INTRUSION.
C--------------------------------------------------------------------
                IF (I.NE.iEndGhost) THEN
                   IF (PCELL(I+1,J,K).NE.WALL) THEN
C--------------------------------------------------------------------
C               SOUTHEAST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
                      IF (J.NE.jBegGhost.AND.I.NE.iEndGhost) THEN
                         IF (PCELL(I+1,J-1,K).EQ.FFIELD.AND.
     &                        PCELL(I,J-1,K).EQ.FFIELD)  THEN
                            VISC = PT5*(VIS(I+1,J-1,K)
     &                           +VIS(I,J-1,K))
                            YP = YV(J)-YY(J-1)
                            TMULT = VISC/YP
                            TAUYX(I+1,J-1,K) = -TMULT*UU(I+1,J-1,K)
                            SU(I+1,J-1,K) = SUL(I+1,J-1,K)
                            SP(I+1,J-1,K) = SP(I+1,J-1,K) - 
     &                           TMULT*SEWU(I+1)*RV(J)*STB(K)
                            AN(I+1,J-1,K) = ZERO
                         END IF
                      END IF
C--------------------------------------------------------------------
C               EAST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
                      IF (I.NE.IEND) AW(I+2,J,K) = ZERO
C--------------------------------------------------------------------
C               NORTHEAST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
                      IF (J.NE.jEndGhost.AND.I.NE.iEndGhost) THEN
                         IF (PCELL(I+1,J+1,K).EQ.FFIELD.AND.
     &                        PCELL(I,J+1,K).EQ.FFIELD) THEN
                            VISC = PT5*(VIS(I+1,J+1,K)
     &                           +VIS(I,J+1,K))
                            YP = YY(J+1)-YV(J+1)
                            TMULT = VISC/YP
                            TAUYX(I+1,J+1,K) = -TMULT*UU(I+1,J+1,K)
                            SU(I+1,J+1,K) = SUL(I+1,J+1,K)
                            SP(I+1,J+1,K) = SP(I+1,J+1,K)-
     &                           TMULT*SEWU(I+1)*RV(J+1)*STB(K)
                            AS(I+1,J+1,K) = ZERO
                         END IF
                      END IF
C--------------------------------------------------------------------
C               BOTTOM EAST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
                      IF (K.NE.kBegGhost.AND.I.NE.iEndGhost) THEN
                         IF (PCELL(I+1,J,K-1).EQ.FFIELD.AND.
     &                        PCELL(I,J,K-1).EQ.FFIELD) THEN
                            VISC = PT5*(VIS(I+1,J,K-1) + 
     &                           VIS(I,J,K-1))
                            ZP = RA(J)*(ZW(K)-ZZ(K-1))
                            TMULT = VISC/ZP
                            TAUZX(I+1,J,K-1) = -TMULT*UU(I+1,J,K-1)
                            SU(I+1,J,K-1) = SUL(I+1,J,K-1)
                            SP(I+1,J,K-1) =SP(I+1,J,K-1)-TMULT*
     $                           SEWU(I+1)*SNS(J)
                            AT(I+1,J,K-1) = ZERO
                         END IF
                      END IF
C--------------------------------------------------------------------
C               TOP EAST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
                      IF (K.NE.kEndGhost.AND.I.NE.iEndGhost) THEN
                         IF (PCELL(I+1,J,K+1).EQ.FFIELD.AND.
     &                        PCELL(I,J,K+1).EQ.FFIELD) THEN
                            VISC = PT5*(VIS(I+1,J,K+1) 
     &                           + VIS(I,J,K+1))
                            ZP = RA(J)*(ZZ(K+1)-ZW(K+1))
                            TMULT = VISC/ZP
                            TAUZX(I+1,J,K+1) = -TMULT*UU(I+1,J,K+1)
                            SU(I+1,J,K+1) = SUL(I+1,J,K+1)
                            SP(I+1,J,K+1) =SP(I+1,J,K+1)-
     &                           TMULT*SEWU(I+1)*SNS(J)
                            AB(I+1,J,K+1) = ZERO
                         END IF
                      END IF
                   END IF
                END IF
             END IF
 100      CONTINUE
 110   CONTINUE
 120  CONTINUE
C--------------------------------------------------------------------
C     SYMMETRY BOUNDARY CONDITIONS
C--------------------------------------------------------------------
      IF (LSYMB) THEN
         CALL SYMBCS(indexLow, indexHigh, PCELL,
     &        UU, AE ,AW ,AN ,AS ,AT ,AB, LPRDJK, LPRDKK)
         DO 140 K = KST,KEND
            DO 130 J = JST,JEND
               IF (PCELL(iBegGhost,J,K).EQ.SYMTRY) THEN
                  UU(IST,J,K) = ZERO
                  AW(IST+1,J,K) = ZERO
               END IF
               IF (PCELL(iEndGhost,J,K).EQ.SYMTRY) 
     &              UU(iEndGhost,J,K) = ZERO
 130        CONTINUE
 140     CONTINUE
      END IF 
      IF (LWRAP) THEN
         IF (LPRDKK) THEN
            CALL PRDBC1(indexLow, indexHigh, NITER, PCELL,
     &           UU, RR, LPRDAV, LPRDJK, LPRDKK)
            CALL PRDBC2(indexLow, indexHigh, NITER, PCELL,
     &           UU, AS, AT, AB, RR, LPRDAV, LPRDJK, LPRDKK)
         ELSE IF (LPRDJK) THEN
c
c  *** WARNING *** Not sure what this will do ! (bb)
c
            DO 160 JK = jBegGhost,jEndGhost
               DO 150 I = iBegGhost,iEndGhost
                  IF (PCELL(I,JK,kBegGhost).EQ.SYMTRY) 
     &                 UU(I,JK,kBegGhost) = UU(I,JST,JK)
                  IF (PCELL(I,jBegGhost,JK).EQ.SYMTRY) 
     &                 UU(I,jBegGhost,JK) = UU(I,JK,KST)
 150           CONTINUE
 160        CONTINUE 
         END IF
      END IF
C--------------------------------------------------------------------
C     OUTLET BOUNDARY CONDITIONS 
C       EAST AND/OR WEST WALLS
C--------------------------------------------------------------------
C$DIR SCALAR
      DO 210 K = KST,KEND
C$DIR SCALAR
         DO 200 J = JST,JEND
            IF ((PCELL(iEndGhost,J,K).EQ.FFIELD)
     &           .OR. (PCELL(iEndGhost,J,K).EQ.DECOMP.AND.UINLT(J,K)
     &           .GT.ZERO)) 
     &           THEN
               IF (LOUTS) THEN
                  AE(IEND,J,K) = TWO*TINY
                  UU(iEndGhost,J,K) = UU(IEND,J,K)
               ELSE
                  UU(iEndGhost,J,K) = 
     &                 MAX(ZERO,(UU(IEND,J,K)+UVWINC))
               END IF
            END IF
            IF ((PCELL(iBegGhost,J,K).EQ.FFIELD)
     &           .OR.(PCELL(iBegGhost,J,K).EQ.DECOMP.AND.
     &           UU(iBegGhost,J,K).LT.ZERO)) THEN
               IF (LOUTS) THEN
                  AW(IST+1,J,K) = TWO*TINY
                  UU(IST,J,K) = UU(IST+1,J,K)
               ELSE
                  UU(IST,J,K) = 
     &                 MIN(ZERO,(UU(IST+1,J,K)-UVWINC))
               END IF
               UU(iBegGhost,J,K) = UU(IST,J,K)
            END IF
 200     CONTINUE
 210  CONTINUE
C--------------------------------------------------------------------
C     NORTH AND/OR SOUTH WALLS
C--------------------------------------------------------------------
C$DIR SCALAR
      DO 310 K = KST,KEND
C$DIR SCALAR
         DO 300 I = IST,IEND
            IF ((PCELL(I,jEndGhost,K).EQ.FFIELD)
     &           .OR.(PCELL(I,jEndGhost,K).EQ.DECOMP.AND.VINLT(I,K)
     &           .GT.ZERO)) 
     &           THEN
            UU(I,jEndGhost,K) = UU(I,JEND,K)
            IF (LZERO) UU(I,jEndGhost,K) = ZERO
            AN(I,JEND,K) = TWO*TINY
         END IF
         IF ((PCELL(I,jBegGhost,K).EQ.FFIELD)
     &        .OR.(PCELL(I,jBegGhost,K).EQ.DECOMP.AND.
     &        VV(I,jBegGhost,K).LT.ZERO)) THEN
            UU(I,jBegGhost,K) = UU(I,JST,K)
            IF (LZERO) UU(I,jBegGhost,K) = ZERO
            AS(I,JST,K) = TWO*TINY
         END IF
 300  CONTINUE
 310  CONTINUE
C--------------------------------------------------------------------
C     TOP AND/OR BOTTOM WALLS
C--------------------------------------------------------------------
C$DIR SCALAR
      DO 410 J = JST,JEND
C$DIR SCALAR
         DO 400 I = IST,IEND
            IF ((PCELL(I,J,kEndGhost).EQ.FFIELD)
     &           .OR.(PCELL(I,J,kEndGhost).EQ.DECOMP.AND.WINLT(I,J)
     &           .GT.ZERO)) 
     &           THEN
               UU(I,J,kEndGhost) = UU(I,J,KEND)
               IF (LZERO) UU(I,J,kEndGhost) = ZERO
               AT(I,J,KEND) = TWO*TINY
            END IF
            IF ((PCELL(I,J,kBegGhost).EQ.FFIELD)
     &           .OR.(PCELL(I,J,kBegGhost).EQ.DECOMP.AND.
     &           WW(I,J,kBegGhost).LT.ZERO)) THEN
               UU(I,J,kBegGhost) = UU(I,J,KST)
               IF (LZERO) UU(I,J,kBegGhost) = ZERO
               AB(I,J,KST) = TWO*TINY
            END IF
 400     CONTINUE
 410  CONTINUE
C----------------------------------------------------------------------
C     INLET BOUNDARY CONDITIONS FOR SURFACES NORMAL TO U
C     IE.  EAST AND/OR WEST WALLS IF NOT LTEST
C     (note: need to declare and pass ltest if using this section!)
C----------------------------------------------------------------------
C      IF (.NOT. LTEST) THEN
CC$DIR SCALAR
C        DO 510 K = 2,indexHigh(3)-1
CC$DIR SCALAR
C          DO 500 J = 2,indexHigh(2)-1
C            IF (PCELL(indexHigh(1),J,K).EQ.PFIELD.OR.PCELL(indexHigh(1),J,K).EQ.SFIELD) 
C     &        THEN
C              AREAEW = RA(J)*SNS(J)*STB(K)
C              AE(indexHigh(1)-1,J,K) = ZERO
C              SU(indexHigh(1)-1,J,K) = SU(indexHigh(1)-1,J,K) 
C     &                   - UINLT(J,K)*DEN(indexHigh(1),J,K)*UINLT(J,K)*AREAEW
C            END IF
C            IF (PCELL(1,J,K).EQ.PFIELD.OR.PCELL(1,J,K).EQ.
C     $        SFIELD) THEN
C              AREAEW = RA(J)*SNS(J)*STB(K)
C              AW(3,J,K) = ZERO
C              SU(3,J,K) = SU(3,J,K) 
C     &                  + U(1,J,K)*DEN(1,J,K)*U(1,J,K)*AREAEW
C            END IF
C 500      CONTINUE
C 510    CONTINUE
C      END IF
C
c      print*,'stmult =',stmult
      RETURN
      END

c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:43  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
