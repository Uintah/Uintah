c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/bcted_fort.h>
#else
      SUBROUTINE bcted(indexLow, indexHigh, NITER, PCELL,
     &     UU, VV, WW, 
     &     AE, AW, AN, AS, AT, AB,
     &     SU, SP,
     &     TED, TKE,
     &     TEMP, TEDSAV, 
     &     UINLT, VINLT, WINLT,
     &     RR, RA,
     &     XX, XU, YY, YV, ZZ, ZW,
     &     LMODBC, LSYMB, LWRAP, LPRDAV, LPRDJK, LPRDKK,
     &     ierr)
C-----------------------------------------------------------------------
C     THIS SUBROUTINE CALCULATES THE BOUNDARY CONDITIONS FOR TED
C-----------------------------------------------------------------------
      implicit none

#include "param4.h"
#include "pconst.h"
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
      double precision UU(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision VV(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision WW(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision TED(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision TKE(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision TEMP(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision TEDSAV(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision UINLT(indexLow(2):indexHigh(2), 
     &     indexLow(3):indexHigh(3))
      double precision VINLT(indexLow(1):indexHigh(1), 
     &     indexLow(3):indexHigh(3))
      double precision WINLT(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2))
      double precision RR(indexLow(2):indexHigh(2))
      double precision RA(indexLow(2):indexHigh(2))
      double precision XX(indexLow(1):indexHigh(1))
      double precision XU(indexLow(1):indexHigh(1))
      double precision YY(indexLow(2):indexHigh(2))
      double precision YV(indexLow(2):indexHigh(2))
      double precision ZZ(indexLow(3):indexHigh(3))
      double precision ZW(indexLow(3):indexHigh(3))
      logical LMODBC, LSYMB, LWRAP, LPRDAV, LPRDJK, LPRDKK
      integer ierr

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer i, j, k
      integer iBegGhost, jBegGhost, kBegGhost
      integer iEndGhost, jEndGhost, kEndGhost
      integer IST, JST, KST, IEND, JEND, KEND
      double precision CONST
      double precision TED0, TERM
      double precision XP, YP, ZP

c*********************************************************************
c     Start
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
C           SET UP A TEMPORARY ARRAY WITH IindexHigh(1)TIALLY LARGE VALUES
C           WE WILL USE THIS LATER TO DECIDE WHERE IS CLOSEST WALL
C     Note:  I use the GEN array for the temp storage because I 
C            have already combined it with the rest of SU by the
C            call to bcted and thus no longer need it.
C--------------------------------------------------------------------
      TERM = EXP(LOG(CMU)*PT75)/CAPPA
      CALL ARRASS(indexLow, indexHigh,
     &     TEMP,GREAT)

      CONST = ONE
C$DIR SCALAR
      DO 420 K = kBegGhost,kEndGhost
C$DIR SCALAR
         DO 410 J = jBegGhost,jEndGhost
C$DIR SCALAR
            DO 400 I = iBegGhost,iEndGhost
C--------------------------------------------------------------------
C           INLET CONDITIONS ARE SPECIFIED IN PROFV.F & PROFT.F
C
C           INTRUSIONS AND WALLS 
C--------------------------------------------------------------------  
               IF (PCELL(I,J,K).EQ.WALL) THEN
C--------------------------------------------------------------------
C             WEST FACE OF INTRUSION/WALL
C--------------------------------------------------------------------
                  IF (I.NE.iBegGhost) THEN
                     IF (PCELL(I-1,J,K).EQ.FFIELD) THEN
                        XP = XU(I)-XX(I-1)
                        IF (XP .LT. TEMP(I-1,J,K)) THEN
                           TEMP(I-1,J,K) = XP
                           TED0 = TERM*EXP(LOG(TKE(I-1,J,K))*
     $                          ONEPT5)/XP
                           IF (LMODBC) THEN
                              TEDSAV(I-1,J,K) = TED0
                           ENDIF
                           TED(I-1,J,K) = TEDSAV(I-1,J,K)
                           CALL FIXVAL(indexLow, indexHigh,
     &                          TEDSAV(I-1,J,K),
     $                          AE,AW,AN,AS,AT,AB,SU,SP,
     &                          CONST,
     &                          I-1,J,K)
                        END IF
                     END IF
                  END IF
C--------------------------------------------------------------------
C             EAST FACE OF INTRUSION/WALL
C--------------------------------------------------------------------
                  IF (I.NE.iEndGhost) THEN
                     IF (PCELL(I+1,J,K).EQ.FFIELD) THEN
                        XP = XX(I+1)-XU(I+1)
                        IF (XP .LT. TEMP(I+1,J,K)) THEN
                           TEMP(I+1,J,K) = XP
                           TED0 = TERM*EXP(LOG(TKE(I+1,J,K))*
     $                          ONEPT5)/XP
                           IF (LMODBC) THEN
                              TEDSAV(I+1,J,K) = TED0
                           ENDIF
                           TED(I+1,J,K) = TEDSAV(I+1,J,K)
                           CALL FIXVAL(indexLow, indexHigh,
     &                          TEDSAV(I+1,J,K),
     $                          AE,AW,AN,AS,AT,AB,SU,SP,
     &                          CONST,
     &                          I+1,J,K)
                        END IF
                     END IF
                  END IF
C--------------------------------------------------------------------
C             SOUTH FACE OF INTRUSION/WALL
C--------------------------------------------------------------------
                  IF (J.NE.jBegGhost) THEN
                     IF (PCELL(I,J-1,K).EQ.FFIELD) THEN
                        YP = YV(J)-YY(J-1)
                        IF (YP .LT. TEMP(I,J-1,K)) THEN
                           TEMP(I,J-1,K) = YP
                           TED0 = TERM*EXP(LOG(TKE(I,J-1,K))*
     $                          ONEPT5)/YP
                           IF (LMODBC) THEN
                              TEDSAV(I,J-1,K) = TED0
                           ENDIF
                           TED(I,J-1,K) = TEDSAV(I,J-1,K)
                           CALL FIXVAL(indexLow, indexHigh,
     &                          TEDSAV(I,J-1,K),
     $                          AE,AW,AN,AS,AT,AB,SU,SP,
     &                          CONST,
     &                          I,J-1,K)
                        END IF
                     END IF
                  END IF
C--------------------------------------------------------------------
C             NORTH FACE OF INTRUSION/WALL
C--------------------------------------------------------------------
                  IF (J.NE.jEndGhost) THEN
                     IF (PCELL(I,J+1,K).EQ.FFIELD) THEN
                        YP = YY(J+1)-YV(J+1)
                        IF (YP .LT. TEMP(I,J+1,K)) THEN
                           TEMP(I,J+1,K) = YP
                           TED0 = TERM*EXP(LOG(TKE(I,J+1,K))*
     $                          ONEPT5)/YP
                           IF (LMODBC) THEN
                              TEDSAV(I,J+1,K) = TED0
                           ENDIF
                           TED(I,J+1,K) = TEDSAV(I,J+1,K)
                           CALL FIXVAL(indexLow, indexHigh,
     &                          TEDSAV(I,J+1,K),
     $                          AE,AW,AN,AS,AT,AB,SU,SP,
     &                          CONST,
     &                          I,J+1,K)
                        END IF
                     END IF
                  END IF
C--------------------------------------------------------------------
C             BOTTOM FACE OF INTRUSION/WALL
C--------------------------------------------------------------------
                  IF (K.NE.kBegGhost) THEN
                     IF (PCELL(I,J,K-1).EQ.FFIELD) THEN
                        ZP = RA(J)*(ZW(K)-ZZ(K-1))
                        IF (ZP .LT. TEMP(I,J,K-1)) THEN
                           TEMP(I,J,K-1) = ZP
                           TED0 = TERM*EXP(LOG(TKE(I,J,K-1))*
     $                          ONEPT5)/ZP
                           IF (LMODBC) THEN
                              TEDSAV(I,J,K-1) = TED0
                           ENDIF
                           TED(I,J,K-1) = TEDSAV(I,J,K-1)
                           CALL FIXVAL(indexLow, indexHigh,
     &                          TEDSAV(I,J,K-1),
     $                          AE,AW,AN,AS,AT,AB,SU,SP,
     &                          CONST,
     &                          I,J,K-1)
                        END IF
                     END IF
                  END IF
C--------------------------------------------------------------------
C             TOP FACE OF INTRUSION/WALL
C--------------------------------------------------------------------
                  IF (K.NE.kEndGhost) THEN
                     IF (PCELL(I,J,K+1).EQ.FFIELD) THEN
                        ZP = RA(J)*(ZZ(K+1)-ZW(K+1))
                        IF (ZP .LT. TEMP(I,J,K+1)) THEN
                           TEMP(I,J,K+1) = ZP
                           TED0 = TERM*EXP(LOG(TKE(I,J,K+1))*
     $                          ONEPT5)/ZP
                           IF (LMODBC) THEN
                              TEDSAV(I,J,K+1) = TED0
                           ENDIF
                           TED(I,J,K+1) = TEDSAV(I,J,K+1)
                           CALL FIXVAL(indexLow, indexHigh,
     &                          TEDSAV(I,J,K+1),
     $                          AE,AW,AN,AS,AT,AB,SU,SP,
     &                          CONST,
     &                          I,J,K+1)
                        END IF  
                     END IF
                  END IF
               END IF
 400        CONTINUE
 410     CONTINUE
 420  CONTINUE
C--------------------------------------------------------------------
C     SYMMETRY BOUNDARY CONDITIONS
C--------------------------------------------------------------------
      IF (LSYMB) CALL SYMBCS(indexLow, indexHigh, PCELL,
     &     TED, 
     &     AE, AW, AN, AS, AT, AB,
     &     LPRDJK, LPRDKK)
      IF (LWRAP) THEN
         CALL PRDBC1(indexLow, indexHigh, NITER, PCELL,
     &        TED, RR,
     &        LPRDAV, LPRDJK, LPRDKK)
         CALL PRDBC2(indexLow, indexHigh, NITER, PCELL,
     &        TED, 
     &        AS, AT, AB, 
     &        RR,
     &        LPRDAV, LPRDJK, LPRDKK)
      ENDIF
C--------------------------------------------------------------------
C     OUTLET BOUNDARY CONDITIONS 
C       EAST AND/OR WEST WALLS
C----------------------------------------------------------------------
C$DIR SCALAR
      DO 610 K = KST,KEND
C$DIR SCALAR
         DO 600 J = JST,JEND
            IF ( (PCELL(iEndGhost,J,K).EQ.FFIELD .AND.
     $           PCELL(IEND,J,K).EQ.FFIELD) .OR. 
     &           (PCELL(iEndGhost,J,K).EQ.DECOMP .AND.
     $           UINLT(J,K).GT.ZERO) ) THEN
               IF (LMODBC) THEN
                  TEDSAV(IEND,J,K) = TED(IEND,J,K)
               ENDIF
               TED(iEndGhost,J,K) = TEDSAV(IEND,J,K)
               AE(IEND,J,K) = TWO*TINY
            END IF
            IF ( (PCELL(iBegGhost,J,K).EQ.FFIELD .AND.
     $           PCELL(IST,J,K).EQ.FFIELD) .OR.
     &           (PCELL(iBegGhost,J,K).EQ.DECOMP .AND.
     $           UU(iBegGhost,J,K).LT.ZERO) ) THEN
               IF (LMODBC) THEN
                  TEDSAV(IST,J,K) = TED(IST,J,K)
               ENDIF
               TED(iBegGhost,J,K) = TEDSAV(IST,J,K)
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
            IF ( (PCELL(I,jEndGhost,K).EQ.FFIELD .AND.
     $           PCELL(I,JEND,K).EQ.FFIELD) .OR. 
     &           (PCELL(I,jEndGhost,K).EQ.DECOMP .AND.
     $           VINLT(I,K).GT.ZERO) ) THEN
               IF (LMODBC) THEN
                  TEDSAV(I,JEND,K) = TED(I,JEND,K)
               ENDIF
               TED(I,jEndGhost,K) = TEDSAV(I,JEND,K)
               AN(I,JEND,K) = TWO*TINY
            END IF
            IF ( (PCELL(I,jBegGhost,K).EQ.FFIELD .AND.
     $           PCELL(I,JST,K).EQ.FFIELD) .OR.
     &           (PCELL(I,jBegGhost,K).EQ.DECOMP .AND.
     $           VV(I,jBegGhost,K).LT.ZERO) ) THEN
               IF (LMODBC) THEN
                  TEDSAV(I,JST,K) = TED(I,JST,K)
               ENDIF
               TED(I,jBegGhost,K) = TEDSAV(I,JST,K)
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
            IF ( (PCELL(I,J,kEndGhost).EQ.FFIELD .AND.
     $           PCELL(I,J,KEND).EQ.FFIELD) .OR. 
     &           (PCELL(I,J,kEndGhost).EQ.DECOMP .AND.
     $           WINLT(I,J).GT.ZERO) ) THEN
               IF (LMODBC) THEN
                  TEDSAV(I,J,KEND) = TED(I,J,KEND)
               ENDIF
               TED(I,J,kEndGhost) = TEDSAV(I,J,KEND)
               AT(I,J,KEND) = TWO*TINY
            END IF
            IF ( (PCELL(I,J,kBegGhost).EQ.FFIELD .AND.
     $           PCELL(I,J,kST).EQ.FFIELD) .OR. 
     &           (PCELL(I,J,kBegGhost).EQ.DECOMP .AND. 
     $           WW(I,J,kBegGhost).LT.ZERO) ) THEN
               IF (LMODBC) THEN
                  TEDSAV(I,J,kST) = TED(I,J,kST)
               ENDIF
               TED(I,J,kBegGhost) = TEDSAV(I,J,kST)
               AB(I,J,kST) = TWO*TINY
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
