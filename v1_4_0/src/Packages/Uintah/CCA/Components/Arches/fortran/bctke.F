c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/bctke_fort.h>
#else
      SUBROUTINE bctke(indexLow, indexHigh, NITER, PCELL,
     &     UU, VV, WW,
     &     AE, AW, AN, AS, AT, AB,
     &     SU, SP,
     &     TKE, TKESAV, 
     &     DEN, 
     &     UINLT, VINLT, WINLT,
     &     RA, RR,
     &     XX, XU, YY, YV, ZZ, ZW,
     &     EFAC, WFAC, NFAC, SFAC, TFAC, BFAC,
     &     VISCOS, 
     &     LKWBC, LMODBC,LSYMB,LWRAP,LPRDAV,LPRDJK,LPRDKK,
     &     ierr)

C-----------------------------------------------------------------------
C     THIS SUBROUTINE CALCULATES THE BOUNDARY CONDTIONS FOR TKE
C-----------------------------------------------------------------------
C     This file contains Packages/Phil's new boundary condition routine for TKE,
C     except that I have taken out his extruded primaries, etc., stuff 
C     out as all that stuff is not compatible with my earlier version of
C     Packages/Phil's code.  The old version of bctke is in the same directory - 
C     I've renamed it bctke_backup.
C
C     The only changes I've made in transporting this subroutine from 
C     ~/codes/banff/source to the $cwd are those dealing with the
C     conversion to double precision and the necessary changes in the
C     common blocks.
C
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
      double precision TKE(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision TKESAV(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision DEN(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision UINLT(indexLow(2):indexHigh(2), 
     &     indexLow(3):indexHigh(3))
      double precision VINLT(indexLow(1):indexHigh(1), 
     &     indexLow(3):indexHigh(3))
      double precision WINLT(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2))
      double precision RA(indexLow(2):indexHigh(2))
      double precision RR(indexLow(2):indexHigh(2))
      double precision XX(indexLow(1):indexHigh(1))
      double precision XU(indexLow(1):indexHigh(1))
      double precision YY(indexLow(2):indexHigh(2))
      double precision YV(indexLow(2):indexHigh(2))
      double precision ZZ(indexLow(3):indexHigh(3))
      double precision ZW(indexLow(3):indexHigh(3))
      double precision EFAC(indexLow(1):indexHigh(1))
      double precision WFAC(indexLow(1):indexHigh(1))
      double precision NFAC(indexLow(2):indexHigh(2))
      double precision SFAC(indexLow(2):indexHigh(2))
      double precision TFAC(indexLow(3):indexHigh(3))
      double precision BFAC(indexLow(3):indexHigh(3))
      double precision VISCOS
      logical LKWBC, LMODBC, LSYMB,LWRAP, LPRDAV,LPRDJK,LPRDKK
      integer ierr

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer i, j, k
      integer iBegGhost, jBegGhost, kBegGhost
      integer iEndGhost, jEndGhost, kEndGhost
      integer IST, JST, KST, IEND, JEND, KEND
      double precision AKVIS
      double precision DENW, DENE, DENS, DENN, DENT, DENB
      double precision CONST, CDTERM, CTTERM
      double precision TKE0
      double precision UFRIC
      double precision U0, V0, VEL0, W0
      double precision XP, YP, ZP
      double precision YPLUS

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
C     If LKWBC = .TRUE. Newton-Raphson method is used to solve the 
C                       non-linear law of the wall for the friction 
C                       velocity 
C     If LKWBC = .FALSE. the law of the wall is linearized by using
C                       terms from the previous time step in the RHS
C--------------------------------------------------------------------
C           INTRUSIONS AND WALLS 
C--------------------------------------------------------------------  
      CONST = ONE
      CDTERM = EXP(LOG(CMU)*PT25)
      CTTERM = EXP(LOG(CMU)*PT5)
      UFRIC = ONE
C$DIR SCALAR
      DO 420 K=kBegGhost,kEndGhost
C$DIR SCALAR
         DO 410 J=jBegGhost,jEndGhost
C$DIR SCALAR
            DO 400 I=iBegGhost,iEndGhost
               IF (PCELL(I,J,K).EQ.WALL) THEN
C--------------------------------------------------------------------
C             WEST FACE OF INTRUSION/WALL
C--------------------------------------------------------------------
                  IF (I.NE.iBegGhost.AND.J.NE.jBegGhost.AND.
     &                 J.NE.jEndGhost.AND.K.NE.kBegGhost.AND.
     &                 K.NE.kEndGhost) THEN
                     IF (PCELL(I-1,J,K).NE.WALL) THEN
                        XP = XU(I)-XX(I-1)
                        DENW = DEN(I-1,J,K)
                        AKVIS = VISCOS/DENW
                        V0 = SFAC(J)*VV(I-1,J,K)+
     $                       NFAC(J)*VV(I-1,J+1,K)
                        W0 = BFAC(K)*WW(I-1,J,K)+
     $                       TFAC(K)*WW(I-1,J,K+1)
                        VEL0 = SQRT(V0**2+W0**2)
                        IF (LKWBC) THEN
                           CALL LOGLAW(I-1,J,K,XP,VEL0,AKVIS,UFRIC)
                        ELSE 
                           UFRIC = SQRT(TKE(I-1,J,K))*CDTERM
                           YPLUS = XP*UFRIC/AKVIS
                           UFRIC = CAPPA*VEL0/LOG(ELOG*YPLUS)
                        ENDIF
                        TKE0 = (UFRIC**2)/CTTERM
                        IF (LMODBC) THEN
                           TKESAV(I-1,J,K) = TKE0
                        ENDIF
                        TKE(I-1,J,K) = TKESAV(I-1,J,K)
                        CALL FIXVAL(indexLow, indexHigh,
     &                       TKESAV(I-1,J,K),
     $                       AE,AW,AN,AS,AT,AB,
     $                       SU,SP,CONST,I-1,J,K)
                     END IF
                  END IF
C--------------------------------------------------------------------
C             EAST FACE OF INTRUSION/WALL
C--------------------------------------------------------------------
                  IF (I.NE.iEndGhost.AND.J.NE.jBegGhost.AND.
     &                 J.NE.jEndGhost.AND.K.NE.kBegGhost.AND.
     &                 K.NE.kEndGhost) THEN
                     IF (PCELL(I+1,J,K).NE.WALL) THEN
                        XP = XX(I+1)-XU(I+1)
                        DENE = DEN(I+1,J,K)
                        AKVIS = VISCOS/DENE
                        V0 = SFAC(J)*VV(I+1,J,K)+
     $                       NFAC(J)*VV(I+1,J+1,K)
                        W0 = BFAC(K)*WW(I+1,J,K)+
     $                       TFAC(K)*WW(I+1,J,K+1)
                        VEL0 = SQRT(V0**2+W0**2)
                        IF (LKWBC) THEN
                           CALL LOGLAW(I+1,J,K,XP,VEL0,AKVIS,UFRIC)
                        ELSE
                           UFRIC = SQRT(TKE(I+1,J,K))*CDTERM
                           YPLUS = XP*UFRIC/AKVIS
                           UFRIC = CAPPA*VEL0/LOG(ELOG*YPLUS)
                        ENDIF
                        TKE0 = (UFRIC**2)/CTTERM
                        IF (LMODBC) THEN
                           TKESAV(I+1,J,K) = TKE0
                        ENDIF
                        TKE(I+1,J,K) = TKESAV(I+1,J,K)
                        CALL FIXVAL(indexLow, indexHigh,
     &                       TKESAV(I+1,J,K),
     $                       AE,AW,AN,AS,AT,AB,
     $                       SU,SP,CONST,I+1,J,K)
                     END IF
                  END IF
C--------------------------------------------------------------------
C             SOUTH FACE OF INTRUSION/WALL
C--------------------------------------------------------------------
                  IF (I.NE.iBegGhost.AND.I.NE.iEndGhost.AND.
     &                 J.NE.jBegGhost.AND.K.NE.kBegGhost.AND.
     &                 K.NE.kEndGhost) THEN
                     IF (PCELL(I,J-1,K).NE.WALL) THEN
                        YP = YV(J)-YY(J-1)
                        DENS = DEN(I,J-1,K)
                        AKVIS = VISCOS/DENS
                        U0 = WFAC(I)*UU(I,J-1,K)+
     $                       EFAC(I)*UU(I+1,J-1,K)
                        W0 = BFAC(K)*WW(I,J-1,K)+
     $                       TFAC(K)*WW(I,J-1,K+1)
                        VEL0 = SQRT(U0**2+W0**2)
                        IF (LKWBC) THEN
                           CALL LOGLAW(I,J-1,K,YP,VEL0,AKVIS,UFRIC)
                        ELSE
                           UFRIC = SQRT(TKE(I,J-1,K))*CDTERM
                           YPLUS = YP*UFRIC/AKVIS
                           UFRIC = CAPPA*VEL0/LOG(ELOG*YPLUS)
                        ENDIF
                        TKE0 = (UFRIC**2)/CTTERM
                        IF (LMODBC) THEN
                           TKESAV(I,J-1,K) = TKE0
                        ENDIF
                        TKE(I,J-1,K) = TKESAV(I,J-1,K)
                        CALL FIXVAL(indexLow, indexHigh,
     &                       TKESAV(I,J-1,K),
     $                       AE,AW,AN,AS,AT,AB,
     $                       SU,SP,CONST,I,J-1,K)
                     END IF
                  END IF
C--------------------------------------------------------------------
C             NORTH FACE OF INTRUSION/WALL
C--------------------------------------------------------------------
                  IF (I.NE.iBegGhost.AND.I.NE.iEndGhost.AND.
     &                 J.NE.jEndGhost.AND.K.NE.kBegGhost.AND.
     &                 K.NE.kEndGhost) THEN
                     IF (PCELL(I,J+1,K).NE.WALL) THEN
                        YP = YY(J+1)-YV(J+1)
                        DENN = DEN(I,J+1,K)
                        AKVIS = VISCOS/DENN
                        U0 = WFAC(I)*UU(I,J+1,K)+
     $                       EFAC(I)*UU(I+1,J+1,K)
                        W0 = BFAC(K)*WW(I,J+1,K)+
     $                       TFAC(K)*WW(I,J+1,K+1)
                        VEL0 = SQRT(U0**2+W0**2)
                        IF (LKWBC) THEN
                           CALL LOGLAW(I,J+1,K,YP,VEL0,AKVIS,UFRIC)
                        ELSE
                           UFRIC = SQRT(TKE(I,J+1,K))*CDTERM
                           YPLUS = YP*UFRIC/AKVIS
                           UFRIC = CAPPA*VEL0/LOG(ELOG*YPLUS)
                        ENDIF
                        TKE0 = (UFRIC**2)/CTTERM
                        IF (LMODBC) THEN
                           TKESAV(I,J+1,K) = TKE0
                        ENDIF
                        TKE(I,J+1,K) = TKESAV(I,J+1,K)
                        CALL FIXVAL(indexLow, indexHigh,
     &                       TKESAV(I,J+1,K),
     $                       AE,AW,AN,AS,AT,AB,
     $                       SU,SP,CONST,I,J+1,K)
                     END IF
                  END IF
C--------------------------------------------------------------------
C             BOTTOM FACE OF INTRUSION/WALL
C--------------------------------------------------------------------
                  IF (I.NE.iBegGhost.AND.I.NE.iEndGhost.AND.
     &                 J.NE.jBegGhost.AND.J.NE.jEndGhost.AND.
     &                 K.NE.kBegGhost) THEN
                     IF (PCELL(I,J,K-1).NE.WALL) THEN
                        ZP = RA(J)*(ZW(K) - ZZ(K-1))
                        DENB = DEN(I,J,K-1)
                        AKVIS = VISCOS/DENB
                        U0 = WFAC(I)*UU(I,J,K-1)+
     $                       EFAC(I)*UU(I+1,J,K-1)
                        V0 = SFAC(J)*VV(I,J,K-1)+
     $                       NFAC(J)*VV(I,J+1,K-1)
                        VEL0 = SQRT(U0**2+V0**2)
                        IF (LKWBC) THEN
                           CALL LOGLAW(I,J,K-1,ZP,VEL0,AKVIS,UFRIC)
                        ELSE
                           UFRIC = SQRT(TKE(I,J,K-1))*CDTERM
                           YPLUS = ZP*UFRIC/AKVIS
                           UFRIC = CAPPA*VEL0/LOG(ELOG*YPLUS)
                        ENDIF
                        TKE0 = (UFRIC**2)/CTTERM
                        IF (LMODBC) THEN
                           TKESAV(I,J,K-1) = TKE0
                        ENDIF
                        TKE(I,J,K-1) = TKESAV(I,J,K-1)
                        CALL FIXVAL(indexLow, indexHigh,
     &                       TKESAV(I,J,K-1),
     $                       AE,AW,AN,AS,AT,AB,
     $                       SU,SP,CONST,I,J,K-1)
                     END IF 
                  END IF 
C-------------------------------------------------------------------- 
C             TOP FACE OF INTRUSION/WALL
C--------------------------------------------------------------------
                  IF (I.NE.iBegGhost.AND.I.NE.iEndGhost.AND.
     &                 J.NE.jBegGhost.AND.J.NE.jEndGhost.AND.
     &                 K.NE.kEndGhost) THEN
                     IF (PCELL(I,J,K+1).NE.WALL) THEN
                        ZP = RA(J)*(ZZ(K+1) - ZW(K+1))
                        DENT = DEN(I,J,K+1)
                        AKVIS = VISCOS/DENT
                        U0 = WFAC(I)*UU(I,J,K+1)+
     $                       EFAC(I)*UU(I+1,J,K+1)
                        V0 = SFAC(J)*VV(I,J,K+1)+
     $                       NFAC(J)*VV(I,J+1,K+1)
                        W0 = ZERO
                        VEL0 = SQRT(U0**2+V0**2+W0**2)
                        IF (LKWBC) THEN
                           CALL LOGLAW(I,J,K+1,ZP,VEL0,AKVIS,UFRIC)
                        ELSE
                           UFRIC = SQRT(TKE(I,J,K+1))*CDTERM
                           YPLUS = ZP*UFRIC/AKVIS
                           UFRIC = CAPPA*VEL0/LOG(ELOG*YPLUS)
                        ENDIF
                        TKE0 = (UFRIC**2)/CTTERM
                        IF (LMODBC) THEN
                           TKESAV(I,J,K+1) = TKE0
                        ENDIF
                        TKE(I,J,K+1) = TKESAV(I,J,K+1)
                        CALL FIXVAL(indexLow, indexHigh,
     &                       TKESAV(I,J,K+1),
     $                       AE,AW,AN,AS,AT,AB,
     $                       SU,SP,CONST,I,J,K+1)
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
     &     TKE,AE,AW,AN,AS,AT,AB,LPRDJK,LPRDKK)
      IF (LWRAP) THEN
         CALL PRDBC1(indexLow, indexHigh, NITER, PCELL,
     &        TKE,RR, LPRDAV,LPRDJK,LPRDKK)
         CALL PRDBC2(indexLow, indexHigh, NITER, PCELL,
     &        TKE,AS,AT,AB,RR,LPRDAV,LPRDJK,LPRDKK)
      ENDIF
C--------------------------------------------------------------------
C     OUTLET BOUNDARY CONDITIONS 
C       EAST AND/OR WEST WALLS
C--------------------------------------------------------------------
C$DIR SCALAR
      DO 610 K = KST,KEND
C$DIR SCALAR
         DO 600 J = JST,JEND
            IF ((PCELL(iEndGhost,J,K).EQ.FFIELD.AND.
     $           PCELL(IEND,J,K).EQ.FFIELD)
     &           .OR. (PCELL(iEndGhost,J,K).EQ.DECOMP.AND.
     $           UINLT(J,K).GT.ZERO)) THEN
               IF (LMODBC) THEN
                  TKESAV(IEND,J,K) = TKE(IEND,J,K)
               ENDIF
               TKE(iEndGhost,J,K) = TKESAV(IEND,J,K)
               AE(IEND,J,K) = TWO*TINY
            END IF
            IF ((PCELL(iBegGhost,J,K).EQ.FFIELD.AND.
     $           PCELL(IST,J,K).EQ.FFIELD)
     &           .OR.(PCELL(iBegGhost,J,K).EQ.DECOMP.AND.
     $           UU(iBegGhost,J,K).LT.ZERO)) THEN
               IF (LMODBC) THEN
                  TKESAV(IST,J,K) = TKE(IST,J,K)
               ENDIF
               TKE(iBegGhost,J,K) = TKESAV(IST,J,K)
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
     $           VINLT(I,K).GT.ZERO)) THEN
               IF (LMODBC) THEN
                  TKESAV(I,JEND,K) = TKE(I,JEND,K)
               ENDIF
               TKE(I,jEndGhost,K) = TKESAV(I,JEND,K)
               AN(I,JEND,K) = TWO*TINY
            END IF
            IF ((PCELL(I,jBegGhost,K).EQ.FFIELD.AND.
     $           PCELL(I,JST,K).EQ.FFIELD)
     &           .OR.(PCELL(I,jBegGhost,K).EQ.DECOMP.AND.
     $           VV(I,jBegGhost,K).LT.ZERO)) THEN
               IF (LMODBC) THEN
                  TKESAV(I,JST,K) = TKE(I,JST,K)
               ENDIF
               TKE(I,jBegGhost,K) = TKESAV(I,JST,K)
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
     $           WINLT(I,J).GT.ZERO)) THEN
               IF (LMODBC) THEN
                  TKESAV(I,J,KEND) = TKE(I,J,KEND)
               ENDIF
               TKE(I,J,kEndGhost) = TKESAV(I,J,KEND)
               AT(I,J,KEND) = TWO*TINY
            END IF
            IF ((PCELL(I,J,kBegGhost).EQ.FFIELD.AND.
     $           PCELL(I,J,KST).EQ.FFIELD)
     &           .OR.(PCELL(I,J,kBegGhost).EQ.DECOMP.AND.
     $           WW(I,J,kBegGhost).LT.ZERO)) THEN
               IF (LMODBC) THEN
                  TKESAV(I,J,KST) = TKE(I,J,KST)
               ENDIF
               TKE(I,J,kBegGhost) = TKESAV(I,J,KST)
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
