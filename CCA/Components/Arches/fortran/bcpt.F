c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/bcpt_fort.h>
#else
      SUBROUTINE bcpt(indexLow, indexHigh, PCELL,
     &     UP, VP, WP, 
     &     PT, 
     &     AE, AW, AN, AS, AT, AB, 
     &     SU, SP, 
     &     DENP, VISP,
     &     UPINLT, VPINLT, WPINLT,
     &     SEW, SNS, STB,
     &     RR, RA, RV,
     &     XX, XU, YY, YV, ZZ, ZW, 
     &     PTPR, PTSC, 
     &     LSYMB, LWRAP, LPRDAV, LPRDJK, LPRDKK, LIFLX,
     &     ierr)
C-----------------------------------------------------------------------
C     THIS SUBROUTINE CALCULATES THE BOUNDARY CONDITIONS FOR ETA
C-----------------------------------------------------------------------

#include "param4.h"
#include "pcell.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer indexLow(3)
      integer indexHigh(3)
      integer PCELL(indexLow(1):indexHigh(1), indexLow(2):indexHigh(2), 
     &     indexLow(3):indexHigh(3))
      double precision PT(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
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
      double precision DENP(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision VISP(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision UP(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision VP(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision WP(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision UPINLT(indexLow(2):indexHigh(2), 
     &     indexLow(3):indexHigh(3))
      double precision VPINLT(indexLow(1):indexHigh(1), 
     &     indexLow(3):indexHigh(3))
      double precision WPINLT(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2))
      double precision SEW(indexLow(1):indexHigh(1))
      double precision SNS(indexLow(2):indexHigh(2))
      double precision STB(indexLow(3):indexHigh(3))
      double precision RR(indexLow(2):indexHigh(2))
      double precision RA(indexLow(2):indexHigh(2))
      double precision RV(indexLow(2):indexHigh(2))
      double precision XX(indexLow(1):indexHigh(1))
      double precision XU(indexLow(1):indexHigh(1))
      double precision YY(indexLow(2):indexHigh(2))
      double precision YV(indexLow(2):indexHigh(2))
      double precision ZZ(indexLow(3):indexHigh(3))
      double precision ZW(indexLow(3):indexHigh(3))
      double precision PTPR, PTSC
      logical LSYMB, LWRAP, LPRDKK, LPRDJK, LPRDAV, LIFLX
      integer ierr

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer iBegGhost, jBegGhost, kBegGhost
      integer iEndGhost, jEndGhost, kEndGhost
      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k
      double precision AREAEW, AREAN, AREAS, AREATB
      double precision PTBC
      double precision TMULT
      double precision VISC
      double precision XP, YP, ZP

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

C--------------------------------------------------------------------
C           INLET CONDITIONS AT NODE POINTS ARE SPECIFIED IN 
C           INVAR HOWEVER THE FLUX BC's ARE SPECIFIED HERE!
C
C           SYMMETRY BOUNDARY CONDITIONS
C---------------------------------------------------------------------
      LIFLX = .FALSE.
      IF (LSYMB) CALL SYMBCS(indexLow, indexHigh, PCELL, 
     &     PT, 
     &     AE, AW, AN, AS, AT, AB,
     &     LPRDJK, LPRDKK)
      IF (LWRAP) THEN
         CALL PRDBC1(indexLow, indexHigh, NITER, PCELL,
     &        PT, RR,
     &        LPRDAV, LPRDJK, LPRDKK)
         CALL PRDBC2(indexLow, indexHigh, NITER, PCELL,
     &        PT, 
     &        AS, AT, AB, 
     &        RR,
     &        LPRDAV, LPRDJK, LPRDKK)
      ENDIF
C---------------------------------------------------------------------
C     INTRUSION/WALL BOUNDARY CONDITIONS
C     currently we use the pt=0 boundary condition;
C     if we later switch to a more complicated mixed boundary condition
C     we will have to re-code this routine.
C---------------------------------------------------------------------
C$DIR SCALAR
      DO 120 K = kBegGhost, kEndGhost
C$DIR SCALAR
         DO 110 J = jBegGhost, jEndGhost
C$DIR SCALAR
            DO 100 I = iBegGhost, iEndGhost
               IF (PCELL(I,J,K).EQ.WALL) THEN
                  PT(I,J,K) = ZERO
C--------------------------------------------------------------------
C             SOUTH FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
                  IF (J.NE.jBegGhost) THEN
                     IF (PCELL(I,J-1,K).EQ.FFIELD) THEN
                        VISC = VISP(I,J-1,K)
                        YP = YV(J)-YY(J-1)
                        TMULT = VISC/YP
                        SP(I,J-1,K) = SP(I,J-1,K)-TMULT*SEW(I)*
     &                       RV(J)*STB(K)
                        AN(I,J-1,K) = ZERO
                     END IF
                  END IF
C--------------------------------------------------------------------
C             WEST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
                  IF (I.NE.iBegGhost) THEN
                     IF(PCELL(I-1,J,K).EQ.FFIELD) THEN
                        VISC = VISP(I-1,J,K)
                        XP = XU(I)-XX(I-1)
                        TMULT = VISC/XP
                        SP(I-1,J,K) = SP(I-1,J,K)-TMULT*SNS(J)*
     &                       STB(K)*RA(J)
                        AE(I-1,J,K) = ZERO
                     ENDIF
                  END IF
C--------------------------------------------------------------------
C             NORTH FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
                  IF (J.NE.jEndGhost) THEN
                     IF (PCELL(I,J+1,K).EQ.FFIELD) THEN
                        VISC = VISP(I,J+1,K)
                        YP = YY(J+1)-YV(J+1)
                        TMULT = VISC/YP
                        SP(I,J+1,K) = SP(I,J+1,K)-TMULT*SEW(I)*
     &                       RV(J+1)*STB(K)
                        AS(I,J+1,K) = ZERO
                     END IF
                  END IF
C--------------------------------------------------------------------
C             BOTTOM FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
                  IF (K.NE.kBegGhost) THEN
                     IF (PCELL(I,J,K-1).EQ.FFIELD) THEN
                        VISC = VISP(I,J,K-1)
                        ZP = RA(J)*(ZW(K)-ZZ(K-1))
                        TMULT = VISC/ZP
                        SP(I,J,K-1) = SP(I,J,K-1)-TMULT*SEW(I)*
     $                       SNS(J)
                        AT(I,J,K-1) = ZERO
                     END IF
                  END IF
C--------------------------------------------------------------------
C             TOP FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
                  IF (K.NE.kEndGhost) THEN
                     IF (PCELL(I,J,K+1).EQ.FFIELD) THEN
                        VISC = VISP(I,J,K+1)
                        ZP = RA(J)*(ZZ(K+1)-ZW(K+1))
                        TMULT = VISC/ZP
                        SP(I,J,K+1) = SP(I,J,K+1)-TMULT*SEW(I)*
     $                       SNS(J)
                        AB(I,J,K+1) = ZERO
                     END IF
                  END IF
C--------------------------------------------------------------------
C             EAST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
                  IF (I.NE.iEndGhost) THEN
                     IF (PCELL(I+1,J,K).EQ.FFIELD) THEN
                        VISC = VISP(I+1,J,K)
                        XP = XX(I+1)-XU(I+1)
                        TMULT = VISC/XP
                        SP(I+1,J,K) = SP(I+1,J,K)-TMULT*SNS(J)*
     &                       STB(K)*RA(J)
                        AW(I+1,J,K) = ZERO
                     ENDIF
                  ENDIF
               ENDIF
 100        CONTINUE
 110     CONTINUE
 120  CONTINUE
C
      IF (LIFLX) THEN
C----------------------------------------------------------------------
C     INLET BOUNDARY CONDITIONS 
C       EAST AND/OR WEST WALLS
C----------------------------------------------------------------------
C$DIR SCALAR
         DO 210 K = KST,KEND
C$DIR SCALAR
            DO 200 J = JST,JEND
               IF (PCELL(iEndGhost,J,K).EQ.PFIELD.OR.
     $              PCELL(iEndGhost,J,K).EQ.SFIELD) 
     &              THEN
                  PTBC = PTPR
                  IF (PCELL(iEndGhost,J,K).EQ.SFIELD) PTBC = PTSC
                  AREAEW = RA(J)*SNS(J)*STB(K)
                  AE(IEND,J,K) = ZERO
                  SU(IEND,J,K) = SU(IEND,J,K) 
     &                 - UPINLT(J,K)*DENP(iEndGhost,J,K)*PTBC*AREAEW
               END IF
               IF (PCELL(iBegGhost,J,K).EQ.PFIELD.OR.
     $              PCELL(iBegGhost,J,K).EQ.SFIELD)THEN
                  PTBC = PTPR
                  IF (PCELL(iBegGhost,J,K).EQ.SFIELD) PTBC = PTSC
                  AREAEW = RA(J)*SNS(J)*STB(K)
                  AW(IST,J,K) = ZERO
                  SU(IST,J,K) = SU(IST,J,K) 
     &                 + UP(iBegGhost,J,K)*DENP(iBegGhost,J,K)
     &                 *PTBC*AREAEW
               END IF
 200        CONTINUE
 210     CONTINUE
C--------------------------------------------------------------------
C     NORTH AND/OR SOUTH WALLS
C--------------------------------------------------------------------
C$DIR SCALAR
        DO 310 K = KST,KEND
C$DIR SCALAR
           DO 300 I = IST,IEND
              IF (PCELL(I,jEndGhost,K).EQ.PFIELD.OR.
     $             PCELL(I,jEndGhost,K).EQ.SFIELD) 
     &             THEN
                 PTBC = PTPR
                 IF (PCELL(I,jEndGhost,K).EQ.SFIELD) PTBC = PTSC
                 AREAN = RV(jEndGhost)*SEW(I)*STB(K)
                 AN(I,JEND,K) = ZERO
                 SU(I,JEND,K) = SU(I,JEND,K) 
     &                - VPINLT(I,K)*DENP(I,jEndGhost,K)*PTBC*AREAN
              END IF
              IF (PCELL(I,jBegGhost,K).EQ.PFIELD.OR.
     $             PCELL(I,jBegGhost,K).EQ.SFIELD)THEN
                 PTBC = PTPR
                 IF (PCELL(I,jBegGhost,K).EQ.SFIELD) PTBC = PTSC
                 AREAS = RV(JST)*SEW(I)*STB(K)
                 AS(I,JST,K) = ZERO
                 SU(I,JST,K) = SU(I,JST,K) 
     &                + VP(I,jBegGhost,K)*DENP(I,jBegGhost,K)*
     &                PTBC*AREAS
              END IF
 300       CONTINUE
 310    CONTINUE
C--------------------------------------------------------------------
C     TOP AND/OR BOTTOM WALLS
C--------------------------------------------------------------------
C$DIR SCALAR
        DO 410 J = JST,JEND
C$DIR SCALAR
           DO 400 I = IST,IEND
              IF (PCELL(I,J,kEndGhost).EQ.PFIELD.OR.
     $             PCELL(I,J,kEndGhost).EQ.SFIELD) 
     &             THEN
                 PTBC = PTPR
                 IF (PCELL(I,J,kEndGhost).EQ.SFIELD) PTBC = PTSC
                 AREATB = SEW(I)*SNS(J)
                 AT(I,J,KEND) = ZERO
                 SU(I,J,KEND) = SU(I,J,KEND) 
     &                - WPINLT(I,J)*DENP(I,J,kEndGhost)*PTBC*AREATB
              END IF
              IF (PCELL(I,J,kBegGhost).EQ.PFIELD.OR.
     $             PCELL(I,J,kBegGhost).EQ.SFIELD)THEN
                 AREATB = SEW(I)*SNS(J)
                 PTBC = PTPR
                 IF (PCELL(I,J,kBegGhost).EQ.SFIELD) PTBC = PTSC
                 AB(I,J,KST) = ZERO
                 SU(I,J,KST) = SU(I,J,KST) 
     &                + WP(I,J,kBegGhost)*DENP(I,J,kBegGhost)*
     &                PTBC*AREATB
              END IF
 400       CONTINUE
 410    CONTINUE
      END IF
C----------------------------------------------------------------------
C     OUTLET BOUNDARY CONDITIONS 
C       EAST AND/OR WEST WALLS
C----------------------------------------------------------------------
C$DIR SCALAR
      DO 610 K = KST,KEND
C$DIR SCALAR
         DO 600 J = JST,JEND
            IF ((PCELL(iEndGhost,J,K).EQ.FFIELD.AND.
     $           PCELL(IEND,J,K).EQ.FFIELD)
     &           .OR. (PCELL(iEndGhost,J,K).EQ.DECOMP.AND.
     $           UPINLT(J,K).GT.ZERO))THEN
               PT(iEndGhost,J,K) = PT(IEND,J,K)
               AE(IEND,J,K) = TWO*TINY
            END IF
            IF ((PCELL(iBegGhost,J,K).EQ.FFIELD.AND.
     $           PCELL(IST,J,K).EQ.FFIELD)
     &           .OR.(PCELL(iBegGhost,J,K).EQ.DECOMP.AND.
     $           UP(iBegGhost,J,K).LT.ZERO))THEN
               PT(iBegGhost,J,K) = PT(IST,J,K)
               AW(IST,J,K) = TWO*TINY
            END IF
 600     CONTINUE
 610  CONTINUE
C--------------------------------------------------------------------
C     NORTH AND/OR SOUTH WALLS
C--------------------------------------------------------------------
C$DIR SCALAR
      DO 710 K = KST,KEND
C$DIR SCALAR
         DO 700 I = IST,IEND
            IF ((PCELL(I,jEndGhost,K).EQ.FFIELD.AND.
     $           PCELL(I,JEND,K).EQ.FFIELD)
     &           .OR.(PCELL(I,jEndGhost,K).EQ.DECOMP.AND.
     $           VPINLT(I,K).GT.ZERO))THEN
               PT(I,jEndGhost,K) = PT(I,JEND,K)
               AN(I,JEND,K) = TWO*TINY
            END IF
            IF ((PCELL(I,jBegGhost,K).EQ.FFIELD.AND.
     $           PCELL(I,JST,K).EQ.FFIELD)
     &           .OR.(PCELL(I,jBegGhost,K).EQ.DECOMP.AND.
     $           VP(I,jBegGhost,K).LT.ZERO))THEN
               PT(I,jBegGhost,K) = PT(I,JST,K)
               AS(I,JST,K) = TWO*TINY
            END IF
 700     CONTINUE
 710  CONTINUE
C--------------------------------------------------------------------
C     TOP AND/OR BOTTOM WALLS
C--------------------------------------------------------------------
C$DIR SCALAR
      DO 810 J = JST,JEND
C$DIR SCALAR
         DO 800 I = IST,IEND
            IF ((PCELL(I,J,kEndGhost).EQ.FFIELD.AND.
     $           PCELL(I,J,KEND).EQ.FFIELD)
     &           .OR.(PCELL(I,J,kEndGhost).EQ.DECOMP.AND.
     $           WPINLT(I,J).GT.ZERO))THEN
               PT(I,J,kEndGhost) = PT(I,J,KEND)
               AT(I,J,KEND) = TWO*TINY
            END IF
            IF ((PCELL(I,J,kBegGhost).EQ.FFIELD.AND.
     $           PCELL(I,J,KST).EQ.FFIELD)
     &           .OR.(PCELL(I,J,kBegGhost).EQ.DECOMP.AND.
     $           WP(I,J,kBegGhost).LT.ZERO))THEN
               PT(I,J,kBegGhost) = PT(I,J,KST)
               AB(I,J,KST) = TWO*TINY
            END IF
 800     CONTINUE
 810  CONTINUE
      RETURN
      END

c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:43  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
