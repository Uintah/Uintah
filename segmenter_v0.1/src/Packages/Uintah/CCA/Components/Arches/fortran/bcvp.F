c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/bcvp_fort.h>
#else
      SUBROUTINE bcvp(indexLow, indexHigh, NITER, PCELL,
     &     UU, VV, WW,
     &     AE, AW, AN, AS, AT, AB,
     &     SU, SP,
     &     SUL, 
     &     VIS,
     &     TAUXY, TAUZY,
     &     UINLT, VINLT, WINLT,
     &     SEW, SNS, STB, 
     &     SNSV,
     &     RR, RA, RCV, 
     &     XX, XU, ZZ, ZW,
     &     UVWINC,
     &     LZERO, LSYMB, LPRDAV, LPRDJK, LPRDKK, LOUTS,
     &     ierr)

C-----------------------------------------------------------------------
C    SUBROUTINE CALCULATES BOUNDARY CONDITIONS FOR V.  
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
      double precision TAUXY(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision TAUZY(indexLow(1):indexHigh(1), 
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
      double precision SNSV(indexLow(2):indexHigh(2))
      double precision RR(indexLow(2):indexHigh(2))
      double precision RA(indexLow(2):indexHigh(2))
      double precision RCV(indexLow(2):indexHigh(2))
      double precision XX(indexLow(1):indexHigh(1))
      double precision XU(indexLow(1):indexHigh(1))
      double precision ZZ(indexLow(3):indexHigh(3))
      double precision ZW(indexLow(3):indexHigh(3))
      double precision UVWINC
      logical LZERO, LSYMB, LPRDAV, LPRDJK, LPRDKK, LOUTS
      integer ierr

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer i, j, k, JK
      integer iBegGhost, jBegGhost, kBegGhost
      integer iEndGhost, jEndGhost, kEndGhost
      integer IST, JST, KST, IEND, JEND, KEND
      double precision TMULT
      double precision VISC
      double precision XP, ZP

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
C--------------------------------------------------------------------
      LZERO = .FALSE.
C--------------------------------------------------------------------
      DO 120 K = kBegGhost,kEndGhost
         DO 110 J = jBegGhost,jEndGhost
            DO 100 I = iBegGhost,iEndGhost
               IF (PCELL(I,J,K).EQ.WALL) THEN
                  VV(I,J,K) = ZERO
C--------------------------------------------------------------------
C     SOUTHEAST FACE OF INTRUSION/WALL
C--------------------------------------------------------------------
                  IF (I.NE.iEndGhost.AND.J.NE.jBegGhost) THEN
                     IF (PCELL(I+1,J,K).EQ.FFIELD.AND.
     &                    PCELL(I+1,J-1,K).EQ.FFIELD)  THEN
                        VISC = PT5*(VIS(I+1,J,K)+VIS(I+1,J-1,K))
                        XP = XX(I+1)-XU(I+1)
                        TMULT = VISC/XP
                        TAUXY(I+1,J,K) = -TMULT*VV(I+1,J,K)
                        SU(I+1,J,K) = SUL(I+1,J,K)
                        SP(I+1,J,K) = SP(I+1,J,K) - 
     &                       TMULT*SNSV(J)*STB(K)*RCV(J)
                        AW(I+1,J,K) = ZERO
                     END IF
                  END IF
C--------------------------------------------------------------------
C             SOUTH FACE OF INTRUSION/WALL
C--------------------------------------------------------------------
                  IF (J.NE.jBegGhost) THEN
                     IF (PCELL(I,J-1,K).EQ.FFIELD) 
     $                    AN(I,J-1,K) = ZERO 
                  END IF
C--------------------------------------------------------------------
C             SOUTHWEST FACE OF INTRUSION/WALL
C--------------------------------------------------------------------
                  IF (I.NE.iBegGhost.AND.J.NE.jBegGhost) THEN
                     IF (PCELL(I-1,J,K).EQ.FFIELD.AND.
     &                    PCELL(I-1,J-1,K).EQ.FFIELD) THEN
                        VISC = PT5*(VIS(I-1,J,K)+VIS(I-1,J-1,K))
                        XP = XU(I)-XX(I-1)
                        TMULT = VISC/XP
                        TAUXY(I-1,J,K) = -TMULT*VV(I-1,J,K)
                        SU(I-1,J,K) = SUL(I-1,J,K)
                        SP(I-1,J,K) = SP(I-1,J,K) - 
     &                        TMULT*SNSV(J)*STB(K)*RCV(J)
                        AE(I-1,J,K) = ZERO
                     END IF
                  END IF
C-------------------------------------------------------------------=
C             TOP SOUTH FACE OF INTRUSION/WALL
C--------------------------------------------------------------------
                  IF (K.NE.kEndGhost.AND.J.NE.jBegGhost) THEN
                     IF (PCELL(I,J-1,K+1).EQ.FFIELD.AND.
     &                    PCELL(I,J,K+1).EQ.FFIELD) THEN
                        VISC = PT5*(VIS(I,J-1,K+1)+VIS(I,J,K+1))
                        ZP = RA(J)*(ZZ(K+1)-ZW(K+1))
                        TMULT = VISC/ZP
                        TAUZY(I,J,K+1) = -TMULT*VV(I,J,K+1)
                        SU(I,J,K+1) = SUL(I,J,K+1)
                        SP(I,J,K+1) = SP(I,J,K+1)-TMULT*
     $                       SNSV(J)*SEW(I)
                        AB(I,J,K+1) = ZERO
                     END IF
                  END IF
C--------------------------------------------------------------------
C             BOTTOM SOUTH FACE OF INTRUSION/WALL
C--------------------------------------------------------------------
                  IF (K.NE.kBegGhost.AND.J.NE.jBegGhost) THEN
                     IF (PCELL(I,J-1,K-1).EQ.FFIELD.AND.
     &                    PCELL(I,J,K-1).EQ.FFIELD) THEN
                        VISC = PT5*(VIS(I,J-1,K-1) + VIS(I,J,K-1))
                        ZP = RA(J)*(ZW(K)-ZZ(K-1))
                        TMULT = VISC/ZP
                        TAUZY(I,J,K-1) = -TMULT*VV(I,J,K-1)
                        SU(I,J,K-1) = SUL(I,J,K-1)
                        SP(I,J,K-1) = SP(I,J,K-1)-TMULT*
     $                       SNSV(J)*SEW(I)
                        AT(I,J,K-1) = ZERO
                     END IF
                  END IF
C--------------------------------------------------------------------
C             IF CELL NORTH OF INTRUSION IS AN INTRUSION, SKIP
C             MODIFICATIONS FOR NORTHERN FACES OF FIRST INTRUSION.
C--------------------------------------------------------------------
                  IF (J.NE.jEndGhost) THEN
                     IF (PCELL(I,J+1,K).NE.WALL) THEN
C--------------------------------------------------------------------
C             NORTHEAST FACE OF INTRUSION/WALL
C--------------------------------------------------------------------
                        IF (I.NE.iEndGhost.AND.J.NE.jEndGhost)THEN
                           IF (PCELL(I+1,J+1,K).EQ.FFIELD.AND.
     &                          PCELL(I+1,J,K).EQ.FFIELD) THEN
                              VISC = PT5*(VIS(I+1,J+1,K) 
     &                             + VIS(I+1,J,K))
                              XP = XX(I+1)-XU(I+1)
                              TMULT = VISC/XP
                              TAUXY(I+1,J+1,K) = -TMULT*VV(I+1,J+1,K)
                              SU(I+1,J+1,K) = SUL(I+1,J+1,K)
                              SP(I+1,J+1,K) = SP(I+1,J+1,K)-
     &                             TMULT*SNSV(J+1)*STB(K)*RCV(J+1)
                              AW(I+1,J+1,K) = ZERO
                           END IF
                        END IF
C--------------------------------------------------------------------
C             NORTH FACE OF INTRUSION/WALL
C--------------------------------------------------------------------
                        IF (J.NE.JEND) AS(I,J+2,K) = ZERO
C--------------------------------------------------------------
C               NORTHWEST FACE OF INTRUSION/WALL
C--------------------------------------------------------------------
                        IF (I.NE.iBegGhost.AND.J.NE.jEndGhost) THEN
                           IF (PCELL(I-1,J+1,K).EQ.FFIELD.AND.
     &                          PCELL(I-1,J,K).EQ.FFIELD) THEN
                              VISC = PT5*(VIS(I-1,J+1,K) 
     &                             + VIS(I-1,J,K))
                              XP = XU(I)-XX(I-1)
                              TMULT = VISC/XP
                              TAUXY(I-1,J+1,K) = -TMULT*VV(I-1,J+1,K)
                              SU(I-1,J+1,K) = SUL(I-1,J+1,K)
                              SP(I-1,J+1,K) = SP(I-1,J+1,K)-
     &                             TMULT*SNSV(J+1)*STB(K)*RCV(J+1)
                              AE(I-1,J+1,K) = ZERO
                           END IF
                        END IF
C--------------------------------------------------------------------
C               TOP NORTH FACE OF INTRUSION/WALL
C--------------------------------------------------------------------
                        IF (K.NE.kEndGhost.AND.J.NE.jEndGhost)THEN
                           IF (PCELL(I,J+1,K+1).EQ.FFIELD.AND.
     &                          PCELL(I,J,K+1).EQ.FFIELD) THEN
                              VISC = PT5*(VIS(I,J+1,K+1)
     &                             +VIS(I,J,K+1))
                              ZP = RA(J)*(ZZ(K+1)-ZW(K+1))
                              TMULT = VISC/ZP
                              TAUZY(I,J+1,K+1) = -TMULT*VV(I,J+1,K+1)
                              SU(I,J+1,K+1) = SUL(I,J+1,K+1)
                              SP(I,J+1,K+1) =SP(I,J+1,K+1)
     &                             -TMULT*SNSV(J+1)*SEW(I)
                              AB(I,J+1,K+1) = ZERO
                           END IF
                        END IF
C---------------------------------------------------------------------
C               BOTTOM NORTH FACE OF INTRUSION/WALL
C--------------------------------------------------------------------
                        IF (K.NE.kBegGhost.AND.J.NE.jEndGhost) THEN
                           IF (PCELL(I,J+1,K-1).EQ.FFIELD.AND.
     &                          PCELL(I,J,K-1).EQ.FFIELD) THEN
                              VISC = PT5*(VIS(I,J+1,K-1) 
     &                             + VIS(I,J,K-1))
                              ZP = RA(J)*(ZW(K)-ZZ(K-1))
                              TMULT = VISC/ZP
                              TAUZY(I,J+1,K-1) = -TMULT*VV(I,J+1,K-1)
                              SU(I,J+1,K-1) = SUL(I,J+1,K-1)
                              SP(I,J+1,K-1) =SP(I,J+1,K-1)-
     &                             TMULT*SNSV(J+1)*SEW(I)
                              AT(I,J+1,K-1) = ZERO
                           END IF
                        END IF
                     END IF
                  END IF
               END IF
 100        CONTINUE
 110     CONTINUE
 120  CONTINUE
C--------------------------------------------------------------------
C     SYMMETRY AND PERIODIC BOUNDARY CONDITIONS
C--------------------------------------------------------------------
      IF (LSYMB) CALL SYMBCS(indexLow, indexHigh, PCELL,
     &     VV, AE, AW, AN, AS, AT, AB, LPRDJK, LPRDKK)
      IF (LPRDKK) THEN
         CALL PRDBC1(indexLow, indexHigh, NITER, PCELL,
     &        VV, RR, LPRDAV, LPRDJK, LPRDKK)
         CALL PRDBC2(indexLow, indexHigh, NITER, PCELL,
     &        VV, AS, AT, AB, RR, LPRDAV, LPRDJK, LPRDKK)
      ENDIF
      IF (LPRDJK) THEN
c
c  *** WARNING *** Not sure what this will do ! (bb)
c
         DO 160 JK = jBegGhost,jEndGhost
            DO 150 I = iBegGhost,iEndGhost
               IF (PCELL(I,JK,kBegGhost).EQ.SYMTRY) 
     &              VV(I,JK,kBegGhost) = WW(I,JST,JK)
               IF (PCELL(I,jBegGhost,JK).EQ.SYMTRY) 
     &              VV(I,jBegGhost,JK) = -WW(I,JK,KST+1)
 150        CONTINUE
 160     CONTINUE 
      ELSE
         DO 180 K = KST,KEND
            DO 170 I = IST,IEND
               IF (PCELL(I,jBegGhost,K).EQ.SYMTRY) THEN
                  VV(I,JST,K) = ZERO
                  AS(I,JST+1,K) = ZERO
               END IF
               IF (PCELL(I,jEndGhost,K).EQ.SYMTRY) 
     &              VV(I,jEndGhost,K) = ZERO
 170        CONTINUE
 180     CONTINUE 
      END IF
C--------------------------------------------------------------------
C     OUTLET BOUNDARY CONDITONS
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
               VV(iEndGhost,J,K) = VV(IEND,J,K)
               IF (LZERO) VV(iEndGhost,J,K) = ZERO
               AE(IEND,J,K) = TWO*TINY
            END IF
            IF ((PCELL(iBegGhost,J,K).EQ.FFIELD)
     &           .OR.(PCELL(iBegGhost,J,K).EQ.DECOMP.AND.
     &           UU(iBegGhost,J,K).LT.ZERO)) THEN
               VV(iBegGhost,J,K) = VV(IST,J,K)
               IF (LZERO) VV(iBegGhost,J,K) = ZERO
               AW(IST,J,K) = TWO*TINY
            END IF
 200     CONTINUE
 210  CONTINUE
C--------------------------------------------------------------------
C     NORTH AND/OR SOUTH WALL OUTLET
C--------------------------------------------------------------------
C$DIR SCALAR
      DO 310 I = IST,IEND
C$DIR SCALAR
         DO 300 K = KST,KEND
            IF ((PCELL(I,jEndGhost,K).EQ.FFIELD).OR.
     &           (PCELL(I,jEndGhost,K).EQ.DECOMP.AND.VINLT(I,K)
     &           .GT.ZERO))THEN
               IF (LOUTS) THEN
                  AN(I,JEND,K) = TWO*TINY
                  VV(I,jEndGhost,K) = VV(I,JEND,K)
               ELSE
                  VV(I,jEndGhost,K) = 
     &                 MAX(ZERO,(VV(I,JEND,K)+UVWINC))
               END IF
            END IF
            IF ((PCELL(I,jBegGhost,K).EQ.FFIELD)
     &           .OR.(PCELL(I,jBegGhost,K).EQ.DECOMP.AND.
     &           VV(I,jBegGhost,K).LT.ZERO)) THEN
               IF (LOUTS) THEN
                  AS(I,JST+1,K) = TWO*TINY
                  VV(I,JST,K) = VV(I,JST+1,K)
               ELSE
                  VV(I,JST,K) = 
     &                 MIN(ZERO,(VV(I,JST+1,K)-UVWINC))
               END IF
               VV(I,jBegGhost,K) = VV(I,JST,K)
            END IF
 300     CONTINUE
         IF (PCELL(I,jBegGhost,kBegGhost).EQ.FFIELD) 
     &        VV(I,jBegGhost,kBegGhost) = 
     &        VV(I,JST,kBegGhost)
         IF (PCELL(I,jEndGhost,kEndGhost).EQ.FFIELD)
     &        VV(I,jEndGhost,kEndGhost) = 
     &        VV(I,JEND,kEndGhost)
 310  CONTINUE
C--------------------------------------------------------------------
C     TOP AND/OR BOTTOM WALL OUTLET
C--------------------------------------------------------------------
C$DIR SCALAR
      DO 410 J = JST,JEND
C$DIR SCALAR
         DO 400 I = IST,IEND
            IF ((PCELL(I,J,kEndGhost).EQ.FFIELD).OR.
     &           (PCELL(I,J,kEndGhost).EQ.DECOMP.AND.WINLT(I,J)
     &           .GT.ZERO))THEN
               VV(I,J,kEndGhost) = VV(I,J,KEND)
               IF (LZERO) VV(I,J,kEndGhost) = ZERO
               AT(I,J,KEND) = TWO*TINY
            END IF
            IF ((PCELL(I,J,kBegGhost).EQ.FFIELD)
     &           .OR.(PCELL(I,J,kBegGhost).EQ.DECOMP.AND.
     &           WW(I,J,kBegGhost).LT.ZERO)) THEN
               VV(I,J,kBegGhost) = VV(I,J,KST)
               IF (LZERO) VV(I,J,kBegGhost) = ZERO
               AB(I,J,KST) = TWO*TINY
            END IF
 400     CONTINUE
 410  CONTINUE
C--------------------------------------------------------------------
C     INLET BOUNDARY CONDITONS FOR SURFACES NORMAL TO V
C     IE.  NORTH AND/OR SOUTH WALL OUTLET
C     Note: need to declare and pass ltest if using this section!
C--------------------------------------------------------------------
C      IF (.NOT. LTEST) THEN
CC$DIR SCALAR
C        DO 510 K = 2,KEND
CC$DIR SCALAR
C          DO 500 I = 2,IEND
C            IF (PCELL(I,jEndGhost,K).EQ.PFIELD.OR.PCELL(I,jEndGhost,K).EQ.SFIELD)
C     &        THEN
C              AREAN = RV(jEndGhost)*SEW(I)*STB(K)
C              AN(I,JEND,K) = ZERO
C              SU(I,JEND,K) = SU(I,JEND,K)
C     &                     + VINLT(I,K)*DEN(I,jEndGhost,K)*VINLT(I,K)*AREAN
C            END IF
C            IF (PCELL(I,1,K).EQ.PFIELD.OR.PCELL(I,1,K).EQ.SFIELD) THEN
C              AREAS = RV(3)*SEW(I)*STB(K)
C              AS(I,3,K) = ZERO
C              SU(I,3,K) = SU(I,3,K)
C     &                  + V(I,1,K)*DEN(I,1,K)*V(I,1,K)*AREAS
C            END IF
C  500     CONTINUE
C  510   CONTINUE
C      END IF
C
      RETURN
      END

c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:43  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
