c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/bcwp_fort.h>
#else
      SUBROUTINE bcwp(indexLow, indexHigh, NITER, PCELL,
     &     UU, VV, WW,
     &     AE, AW, AN, AS, AT, AB,
     &     SU, SP,
     &     SUL, 
     &     VIS,
     &     TAUXZ, TAUYZ,
     &     UINLT, VINLT, WINLT,
     &     SEW, SNS, STB, 
     &     STBW,
     &     RR, RV, 
     &     XX, XU, YY, YV,
     &     UVWINC,
     &     LZERO, LSYMB, LPRDAV, LPRDJK, LPRDKK, LOUTS,
     &     ierr)

C-----------------------------------------------------------------------
C     SUBROUTINE CALCULATES BOUNDARY CONDITIONS FOR W 
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
      double precision TAUXZ(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision TAUYZ(indexLow(1):indexHigh(1), 
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
      double precision STBW(indexLow(3):indexHigh(3))
      double precision RR(indexLow(2):indexHigh(2))
      double precision RV(indexLow(2):indexHigh(2))
      double precision XX(indexLow(1):indexHigh(1))
      double precision XU(indexLow(1):indexHigh(1))
      double precision YY(indexLow(2):indexHigh(2))
      double precision YV(indexLow(2):indexHigh(2))
      double precision UVWINC
      logical LZERO, LSYMB, LPRDAV, LPRDJK, LPRDKK, LOUTS
      integer ierr

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer iBegGhost, jBegGhost, kBegGhost
      integer iEndGhost, jEndGhost, kEndGhost
      integer IST, JST, KST, IEND, JEND, KEND
      logical LPRDKK0
      integer i, j, k, JK
      double precision TMULT
      double precision VISC
      double precision XP, YP

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
C$DIR SCALAR
      DO 120 K = kBegGhost,kEndGhost
C$DIR SCALAR
         DO 110 J = jBegGhost,jEndGhost
C$DIR SCALAR
            DO 100 I = iBegGhost,iEndGhost	
               IF (PCELL(I,J,K).EQ.WALL) THEN
                  WW(I,J,K) = ZERO
C--------------------------------------------------------------------
C             BOTTOM EAST FACE OF INTRUSION/WALL
C--------------------------------------------------------------------
                  IF (I.NE.iEndGhost.AND.K.NE.kBegGhost) THEN
                     IF (PCELL(I+1,J,K-1).EQ.FFIELD.AND.
     &                    PCELL(I+1,J,K).EQ.FFIELD) THEN
                        VISC = PT5*(VIS(I+1,J,K-1) + VIS(I+1,J,K))
                        XP = XX(I+1)-XU(I+1)
                        TMULT = VISC/XP
                        TAUXZ(I+1,J,K) = -TMULT*WW(I+1,J,K)
                        SU(I+1,J,K) = SUL(I+1,J,K)
                        SP(I+1,J,K) = SP(I+1,J,K)-TMULT*SNS(J)*
     &                       (RV(J)+RV(J+1))*PT5*STBW(K)
                        AW(I+1,J,K) = ZERO
                     END IF
                  END IF
C--------------------------------------------------------------------
C             BOTTOM FACE OF INTRUSION/WALL
C--------------------------------------------------------------------
                  IF (K.NE.kBegGhost) THEN
                     IF (PCELL(I,J,K-1).EQ.FFIELD) 
     $                    AT(I,J,K-1) = ZERO
                  END IF
C--------------------------------------------------------------------
C             BOTTOM WEST FACE OF INTRUSION/WALL
C--------------------------------------------------------------------
                  IF (I.NE.iBegGhost.AND.K.NE.kBegGhost) THEN
                     IF (PCELL(I-1,J,K-1).EQ.FFIELD.AND.
     &                    PCELL(I-1,J,K).EQ.FFIELD) THEN
                        VISC = PT5*(VIS(I-1,J,K-1) + VIS(I-1,J,K))
                        XP = XU(I)-XX(I-1)
                        TMULT = VISC/XP
                        TAUXZ(I-1,J,K) = -TMULT*WW(I-1,J,K)
                        SU(I-1,J,K) = SUL(I-1,J,K)
                        SP(I-1,J,K) = SP(I-1,J,K)-TMULT*SNS(J)*
     &                       (RV(J)+RV(J+1))*PT5*STBW(K)
                        AE(I-1,J,K) = ZERO
                     END IF
                  END IF
C--------------------------------------------------------------------
C             BOTTOM NORTH FACE OF INTRUSION/WALL
C--------------------------------------------------------------------
                  IF (J.NE.jEndGhost.AND.K.NE.kBegGhost) THEN
                     IF (PCELL(I,J+1,K-1).EQ.FFIELD.AND.
     &                    PCELL(I,J+1,K).EQ.FFIELD) THEN
                        VISC = PT5*(VIS(I,J+1,K-1) + VIS(I,J+1,K))
                        YP = YY(J+1)-YV(J+1)
                        TMULT = VISC/YP
                        TAUYZ(I,J+1,K) = -TMULT*WW(I,J+1,K)
                        SU(I,J+1,K) = SUL(I,J+1,K)
                        SP(I,J+1,K) = SP(I,J+1,K)-TMULT*SEW(I)*
     &                       RV(J+1)*STBW(K)
                        AS(I,J+1,K) = ZERO
                     END IF
                  END IF
C--------------------------------------------------------------------
C             BOTTOM SOUTH FACE OF INTRUSION/WALL
C--------------------------------------------------------------------
                  IF (J.NE.jBegGhost.AND.K.NE.kBegGhost) THEN
                     IF (PCELL(I,J-1,K-1).EQ.FFIELD.AND.
     &                    PCELL(I,J-1,K).EQ.FFIELD) THEN
                        VISC = PT5*(VIS(I,J-1,K-1) + VIS(I,J-1,K))
                        YP = YV(J)-YY(J-1)
                        TMULT = VISC/YP
                        TAUYZ(I,J-1,K) = -TMULT*WW(I,J-1,K)
                        SU(I,J-1,K) = SUL(I,J-1,K)
                        SP(I,J-1,K) = SP(I,J-1,K)-TMULT*SEW(I)*
     &                       RV(J)*STBW(K)
                        AN(I,J-1,K) = ZERO
                     END IF
                  END IF
C--------------------------------------------------------------------
C             IF CELL ABOVE INTRUSION IS ALSO AN INTRUSION, SKIP
C             MODIFICATIONS FOR TOP FACES OF FIRST INTRUSION
C--------------------------------------------------------------------
                  IF (K.NE.kEndGhost) THEN
                     IF (PCELL(I,J,K+1).NE.WALL) THEN
C--------------------------------------------------------------------
C               TOP EAST FACE OF INTRUSION/WALL
C--------------------------------------------------------------------
                        IF (I.NE.iEndGhost.AND.K.NE.kEndGhost)THEN
                           IF (PCELL(I+1,J,K+1).EQ.FFIELD.AND.
     &                          PCELL(I+1,J,K).EQ.FFIELD) THEN
                              VISC = PT5*(VIS(I+1,J,K+1) 
     &                             + VIS(I+1,J,K))
                              XP = XX(I+1)-XU(I+1)
                              TMULT = VISC/XP
                              TAUXZ(I+1,J,K+1) = -TMULT*
     $                             WW(I+1,J,K+1)
                              SU(I+1,J,K+1) = SUL(I+1,J,K+1)
                              SP(I+1,J,K+1) = SP(I+1,J,K+1)-TMULT*
     &                             SNS(J)*(RV(J+1)+RV(J))*
     $                             PT5*STBW(K+1)
                              AW(I+1,J,K+1) = ZERO
                           END IF
                        END IF
C--------------------------------------------------------------------
C               TOP FACE OF INTRUSION/WALL
C--------------------------------------------------------------------
                        IF (K.LT.KEND) AB(I,J,K+2) = ZERO
C--------------------------------------------------------------------
C               TOP WEST FACE OF INTRUSION/WALL
C--------------------------------------------------------------------
                        IF (I.NE.iBegGhost.AND.K.NE.kEndGhost) THEN
                           IF (PCELL(I-1,J,K+1).EQ.FFIELD.AND.
     &                          PCELL(I-1,J,K).EQ.FFIELD) THEN
                              VISC = PT5*(VIS(I-1,J,K+1) 
     &                             + VIS(I-1,J,K))
                              XP = XU(I)-XX(I-1)
                              TMULT = VISC/XP
                              TAUXZ(I-1,J,K+1) = -TMULT*
     $                             WW(I-1,J,K+1)
                              SU(I-1,J,K+1) = SUL(I-1,J,K+1)
                              SP(I-1,J,K+1) = SP(I-1,J,K+1)-TMULT*
     &                             SNS(J)*(RV(J+1)+RV(J))*
     $                             PT5*STBW(K+1)
                              AE(I-1,J,K+1) = ZERO
                           END IF
                        END IF
C--------------------------------------------------------------------
C               TOP NORTH FACE OF INTRUSION/WALL
C--------------------------------------------------------------------
                        IF (J.NE.jEndGhost.AND.K.NE.kEndGhost)THEN
                           IF (PCELL(I,J+1,K+1).EQ.FFIELD.AND.
     &                          PCELL(I,J+1,K).EQ.FFIELD) THEN
                              VISC = PT5*(VIS(I,J+1,K+1) 
     &                             + VIS(I,J+1,K))
                              YP = YY(J+1)-YV(J+1)
                              TMULT = VISC/YP
                              TAUYZ(I,J+1,K+1) = -TMULT*
     $                             WW(I,J+1,K+1)
                              SU(I,J+1,K+1) = SUL(I,J+1,K+1)
                              SP(I,J+1,K+1) = SP(I,J+1,K+1)-TMULT*
     &                             SEW(I)*RV(J+1)*STBW(K+1)
                              AS(I,J+1,K+1) = ZERO
                           END IF
                        END IF
C--------------------------------------------------------------------
C               TOP SOUTH FACE OF INTRUSION/WALL
C--------------------------------------------------------------------
                        IF (J.NE.jBegGhost.AND.K.NE.kEndGhost) THEN
                           IF (PCELL(I,J-1,K+1).EQ.FFIELD.AND.
     &                          PCELL(I,J-1,K).EQ.FFIELD) THEN
                              VISC = PT5*(VIS(I,J-1,K+1) 
     &                             + VIS(I,J-1,K))
                              YP = YV(J)-YY(J-1)
                              TMULT = VISC/YP
                              TAUYZ(I,J-1,K+1) = -TMULT*WW(I,J-1,K+1)
                              SU(I,J-1,K+1) = SUL(I,J-1,K+1)
                              SP(I,J-1,K+1) = SP(I,J-1,K+1)-TMULT*
     &                             SEW(I)*RV(J)*STBW(K+1)
                              AN(I,J-1,K+1) = ZERO
                           END IF
                        END IF
                     END IF
                  END IF
               END IF
 100        CONTINUE
 110     CONTINUE
 120  CONTINUE
C--------------------------------------------------------------------
C     SYMMETRY BOUNDARY CONDITIONS
C--------------------------------------------------------------------
      LPRDKK0 = LPRDKK
      LPRDKK = .TRUE.
      IF (LSYMB) CALL SYMBCS(indexLow, indexHigh, PCELL,
     &     WW,AE,AW,AN,AS,AT,AB,LPRDJK,LPRDKK)
      LPRDKK = LPRDKK0
      IF (LPRDKK) THEN
         CALL PRDBC1(indexLow, indexHigh, NITER, PCELL,
     &        WW,RR,LPRDAV, LPRDJK, LPRDKK)
         CALL PRDBC2(indexLow, indexHigh, NITER, PCELL,
     &        WW,AS,AT,AB,RR,LPRDAV, LPRDJK, LPRDKK)
      END IF
      IF (LPRDJK) THEN
         DO 160 JK = jBegGhost,jEndGhost
            DO 150 I = iBegGhost,iEndGhost
               IF (PCELL(I,JK,kBegGhost).EQ.SYMTRY) 
     &              WW(I,JK,kBegGhost) = -VV(I,JST+1,JK)
               IF (PCELL(I,jBegGhost,JK).EQ.SYMTRY) 
     &              WW(I,jBegGhost,JK) = VV(I,JK,KST)
 150        CONTINUE
 160     CONTINUE 
      END IF
      IF (.NOT.(LPRDKK.OR.LPRDJK)) THEN
         DO 180 J = JST,JEND
            DO 170 I = IST,IEND
               IF (PCELL(I,J,kBegGhost).EQ.SYMTRY) 
     &              WW(I,J,KST) = ZERO
               IF (PCELL(I,J,kEndGhost).EQ.SYMTRY) 
     &              WW(I,J,kEndGhost) = ZERO
 170        CONTINUE
 180     CONTINUE
      END IF
C--------------------------------------------------------------------
C     OUTLET BOUNDARY CONDITOINS 
C       EAST AND/OR WEST WALLS
C--------------------------------------------------------------------
C$DIR SCALAR
      DO 210 K = KST,KEND
C$DIR SCALAR
         DO 200 J = JST,JEND
            IF ((PCELL(iEndGhost,J,K).EQ.FFIELD).OR.
     &           (PCELL(iEndGhost,J,K).EQ.DECOMP.AND.UINLT(J,K)
     $           .GT.ZERO))THEN
               WW(iEndGhost,J,K) = WW(IEND,J,K)
               IF (LZERO) WW(iEndGhost,J,K) = ZERO
               AE(IEND,J,K) = TWO*TINY
            END IF
            IF ((PCELL(iBegGhost,J,K).EQ.FFIELD).OR.
     &           (PCELL(iBegGhost,J,K).EQ.DECOMP.AND.
     &           UU(iBegGhost,J,K).LT.ZERO)) THEN
               WW(iBegGhost,J,K) = WW(IST,J,K)
               IF (LZERO) WW(iBegGhost,J,K) = ZERO
               AW(IST,J,K) = TWO*TINY
            END IF
 200     CONTINUE
 210  CONTINUE
C-------------------------------------------------------------------
C      NORTH AND/OR SOUTH WALLS
C-------------------------------------------------------------------
C$DIR SCALAR
      DO 310 K = KST,KEND
C$DIR SCALAR
         DO 300 I = IST,IEND
            IF ((PCELL(I,jEndGhost,K).EQ.FFIELD).OR.
     &           (PCELL(I,jEndGhost,K).EQ.DECOMP.AND.VINLT(I,K)
     $           .GT.ZERO))THEN
               WW(I,jEndGhost,K) = WW(I,JEND,K)
               IF (LZERO) WW(I,jEndGhost,K) = ZERO
               AN(I,JEND,K) = TWO*TINY
            END IF
            IF ((PCELL(I,jBegGhost,K).EQ.FFIELD).OR.
     &           (PCELL(I,jBegGhost,K).EQ.DECOMP.AND.
     &           VV(I,jBegGhost,K).LT.ZERO)) THEN
               WW(I,jBegGhost,K) = WW(I,JST,K)
               IF (LZERO) WW(I,jBegGhost,K) = ZERO
               AS(I,JST,K) = TWO*TINY
            END IF
 300     CONTINUE
 310  CONTINUE
C--------------------------------------------------------------------
C      TOP AND BOTTOM WALLS
C--------------------------------------------------------------------
C$DIR SCALAR
      DO 410 I = IST,IEND
C$DIR SCALAR
         DO 400 J = JST,JEND
            IF ((PCELL(I,J,kEndGhost).EQ.FFIELD).OR.
     &           (PCELL(I,J,kEndGhost).EQ.DECOMP.AND.WINLT(I,J)
     $           .GT.ZERO))THEN
               IF (LOUTS) THEN
                  AT(I,J,KEND) = TWO*TINY
                  WW(I,J,kEndGhost) = WW(I,J,KEND)
               ELSE
                  WW(I,J,kEndGhost) = 
     &                 MAX(ZERO,(WW(I,J,KEND)+UVWINC))
               END IF
            END IF
            IF ((PCELL(I,J,kBegGhost).EQ.FFIELD).OR.
     &           (PCELL(I,J,kBegGhost).EQ.DECOMP.AND.
     &           WW(I,J,kBegGhost).LT.ZERO)) THEN
               IF (LOUTS) THEN
                  AB(I,J,KST+1) = TWO*TINY
                  WW(I,J,KST) = WW(I,J,KST+1)
               ELSE
                  WW(I,J,KST) = 
     &                 MIN(ZERO,(WW(I,J,KST+1)-UVWINC))
               END IF
               WW(I,J,kBegGhost) = WW(I,J,KST)
            END IF
 400     CONTINUE
         IF (PCELL(I,jBegGhost,kBegGhost).EQ.FFIELD) 
     &        WW(I,jBegGhost,kBegGhost) = 
     &        WW(I,jBegGhost,KST)
         IF (PCELL(I,jEndGhost,kEndGhost).EQ.FFIELD) 
     &        WW(I,jEndGhost,kEndGhost) = 
     &        WW(I,jEndGhost,KEND)
 410  CONTINUE
C--------------------------------------------------------------------
C     INLET BOUNDARY CONDITIONS FOR SURFACES NORMAL TO W
C     IE.  TOP AND/OR BOTTOM WALLS
C     Note: need to declare and pass ltest if using this section!
C--------------------------------------------------------------------
C      IF (.NOT. LTEST) THEN
CC$DIR SCALAR
C        DO 510 J = 2,JEND
CC$DIR SCALAR
C          DO 500 I = 2,IEND
C            IF (PCELL(I,J,kEndGhost).EQ.PFIELD.OR.PCELL(I,J,kEndGhost).EQ.SFIELD) 
C     &        THEN
C              AREATB = SEW(I)*SNS(J)
C              AT(I,J,KEND) = ZERO
C              SU(I,J,KEND) = SU(I,J,KEND) 
C     &                 - WINLT(I,J)*DEN(I,J,indexHigh(3))*WINLT(I,J)*AREATB
C            END IF
C            IF (PCELL(I,J,1).EQ.PFIELD.OR.PCELL(I,J,1).EQ.SFIELD) THEN
C              AREATB = SEW(I)*SNS(J)
C              AB(I,J,3) = ZERO
C              SU(I,J,3) = SU(I,J,3) 
C     &                  + W(I,J,1)*DEN(I,J,1)*W(I,J,1)*AREATB
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
