c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE bcuvel(domLoU, domHiU, domLoUng, domHiUng,
     $     idxLoU, idxHiU, UU,
     &     AP, AE, AW, AN, AS, AT, AB,
     &     SU, SP, 
     &     domLoV, domHiV, idxLoV, idxHiV, VV,
     &     domLoW, domHiW, idxLoW, idxHiW, WW,
     &     domLo, domHi, idxLo, idxHi, PCELL, WALL, FFIELD, PBCFLD,
     &     VISCOS, 
     &     SEWU, SNS, STB,
     &     Y, YV, Z, ZW, 
     &     ierr)
C---------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     THIS SUBROUTINE WILL compute matrix coefficients for uvelocity
C---------------------------------------------------------------------

      implicit none

#include "param4.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer domLoU(3), domHiU(3), idxLoU(3), idxHiU(3)
      integer domLoUng(3), domHiUng(3)
      double precision UU(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision AP(domLoUng(1):domHiUng(1), 
     $     domLoUng(2):domHiUng(2), 
     &     domLoUng(3):domHiUng(3))
      double precision AE(domLoUng(1):domHiUng(1),
     $     domLoUng(2):domHiUng(2), 
     &     domLoUng(3):domHiUng(3))
      double precision AW(domLoUng(1):domHiUng(1),
     $     domLoUng(2):domHiUng(2), 
     &     domLoUng(3):domHiUng(3))
      double precision AN(domLoUng(1):domHiUng(1),
     $     domLoUng(2):domHiUng(2), 
     &     domLoUng(3):domHiUng(3))
      double precision AS(domLoUng(1):domHiUng(1),
     $     domLoUng(2):domHiUng(2), 
     &     domLoUng(3):domHiUng(3))
      double precision AT(domLoUng(1):domHiUng(1),
     $     domLoUng(2):domHiUng(2), 
     &     domLoUng(3):domHiUng(3))
      double precision AB(domLoUng(1):domHiUng(1),
     $     domLoUng(2):domHiUng(2), 
     &     domLoUng(3):domHiUng(3))
      double precision SU(domLoUng(1):domHiUng(1), 
     $     domLoUng(2):domHiUng(2), 
     &     domLoUng(3):domHiUng(3))
      double precision SP(domLoUng(1):domHiUng(1), 
     $     domLoUng(2):domHiUng(2), 
     &     domLoUng(3):domHiUng(3))
      integer domLoV(3), domHiV(3), idxLoV(3), idxHiV(3)
      double precision VV(domLoV(1):domHiV(1), domLoV(2):domHiV(2), 
     &     domLoV(3):domHiV(3))
      integer domLoW(3), domHiW(3), idxLoW(3), idxHiW(3)
      double precision WW(domLoW(1):domHiW(1), domLoW(2):domHiW(2), 
     &     domLoW(3):domHiW(3))
      integer domLo(3), domHi(3), idxLo(3), idxHi(3)
      integer PCELL(domLo(1):domHi(1), domLo(2):domHi(2), 
     &     domLo(3):domHi(3))
      double precision SEWU(domLo(1):domHi(1))
      double precision SNS(domLo(2):domHi(2))
      double precision STB(domLo(3):domHi(3))
      double precision Y(domLo(2):domHi(2))
      double precision YV(domLo(2):domHi(2))
      double precision Z(domLo(3):domHi(3))
      double precision ZW(domLo(3):domHi(3))
      double precision viscos
      double precision const
      
      integer WALL
      integer FFIELD, PBCFLD
      integer ierr

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer IST, JST, KST, IEND, JEND, KEND
      integer IBegGhost, JBegGhost, KBegGhost
      integer IEndGhost, JEndGhost, KEndGhost
      integer idxUInlet, idxVInlet, idxWInlet
      integer i, j, k

      double precision YP, ZP, TMULT

c*******************u*************************************************
c     Start :
c*********************************************************************
      IST = idxLo(1)
      JST = idxLo(2)
      KST = idxLo(3)
      IEND = idxHi(1)
      JEND = idxHi(2)
      KEND = idxHi(3)
      iBegGhost = ist-1
      jBegGhost = jst-1
      kBegGhost = kst-1
      iEndGhost = iend+1
      jEndGhost = jend+1
      kEndGhost = kend+1
      idxUInlet = domHiU(1)
      idxVInlet = domHiV(2)
      idxWInlet = domHiW(3)
      
c------------------------------------------------------------------ 
c     interior intrusions
c------------------------------------------------------------------ 
      const = ONE
      do 820 k=kst, kend
         do 810 j=jst, jend
            do 800 i=ist, iend

               if (pcell(i-1,j,k).eq.wall.or. pcell(i,j,k).eq.wall)
     $              call fixval(zero,ae(i,j,k),aw(i,j,k),an(i,j,k),
     $                          as(i,j,k),at(i,j,k),ab(i,j,k),
     $                          su(i,j,k),sp(i,j,k),const)

 800        continue
 810     continue
 820  continue


c*********************************************************************
c     Go thru all the cells
c*********************************************************************
C$DIR SCALAR
      DO 120 K = kBegGhost, kEndGhost
C$DIR SCALAR
        DO 110 J = jBegGhost, jEndGhost
C$DIR SCALAR
          DO 100 I = iBegGhost, iEndGhost
            IF (PCELL(I,J,K).EQ.WALL) THEN
               UU(I,J,K) = 0.0
C--------------------------------------------------------------------
C             WEST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
               IF (I.NE.iBegGhost) THEN
                  IF(PCELL(I-1,J,K).EQ.FFIELD) AE(I-1,J,K) = 0.0
               END IF
C--------------------------------------------------------------------
C               EAST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
               IF (I.NE.iEndGhost) THEN
                  IF (PCELL(I+1, J, K).NE.WALL) THEN
                     IF (I.NE.iEnd) AW(I+2,J,K) = 0.0
                  end if
               end if
C--------------------------------------------------------------------
C             SOUTHWEST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
               IF (J.NE.jBegGhost.AND.I.NE.iBegGhost) THEN
                  IF (PCELL(I,J-1,K).NE.WALL.AND.
     &                 PCELL(I-1,J-1,K).NE.WALL) THEN
                     YP = YV(J)-Y(J-1)
                     TMULT = VISCOS/YP
                
c                  TAUYX(I,J-1,K) = -TMULT*U(I,J-1,K)
c                  SU(I,J-1,K) = SUL(I,J-1,K)
                     SP(I,J-1,K) =SP(I,J-1,K)-TMULT*SEWU(I)*STB(K)
                     AN(I,J-1,K) = 0.0
                  END IF
               END IF

C--------------------------------------------------------------------
C             NORTHWEST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
               IF (J.NE.jEndGhost.AND.I.NE.iBegGhost) THEN
                  IF (PCELL(I,J+1,K).NE.WALL.AND.
     &                 PCELL(I-1,J+1,K).NE.WALL) THEN
                     YP = Y(J+1)-YV(J+1)
                     TMULT = VISCOS/YP
c               TAUYX(I,J+1,K) = -TMULT*U(I,J+1,K)
c                SU(I,J+1,K) = SUL(I,J+1,K)
                     SP(I,J+1,K) =SP(I,J+1,K)-TMULT*SEWU(I)*STB(K)
                     AS(I,J+1,K) = 0.0
                  END IF
               END IF
C--------------------------------------------------------------------
C             BOTTOM WEST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
              IF (K.NE.kBegGhost.AND.I.NE.iBegGhost) THEN
                 IF (PCELL(I-1,J,K-1).NE.WALL.AND.
     &                PCELL(I,J,K-1).NE.WALL) THEN
                    ZP = ZW(K)-Z(K-1)
                    TMULT = VISCOS/ZP
               
c                 TAUZX(I,J,K-1) = -TMULT*U(I,J,K-1)
c                 SU(I,J,K-1) = SUL(I,J,K-1)
                    SP(I,J,K-1) = SP(I,J,K-1)-TMULT*SEWU(I)*SNS(J)
                    AT(I,J,K-1) = 0.0
                 END IF 
              END IF
C--------------------------------------------------------------------
C             TOP WEST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
              IF (K.NE.kEndGhost.AND.I.NE.iBegGhost) THEN
                 IF (PCELL(I-1,J,K+1).NE.WALL.AND.
     &                PCELL(I,J,K+1).NE.WALL) THEN
                    ZP = Z(K+1)-ZW(K+1)
                    TMULT = VISCOS/ZP
c                TAUZX(I,J,K+1) = -TMULT*U(I,J,K+1)
c                SU(I,J,K+1) = SUL(I,J,K+1)
                    SP(I,J,K+1) = SP(I,J,K+1)-TMULT*SEWU(I)*SNS(J)
                    AB(I,J,K+1) = 0.0
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
                       IF (PCELL(I+1,J-1,K).NE.WALL.AND.
     &                      PCELL(I,J-1,K).NE.WALL)  THEN
                          YP = YV(J)-Y(J-1)
                          TMULT = VISCOS/YP
c                       TAUYX(I+1,J-1,K) = -TMULT*U(I+1,J-1,K)
c                       SU(I+1,J-1,K) = SUL(I+1,J-1,K)
                          SP(I+1,J-1,K) = SP(I+1,J-1,K) - TMULT*
     &                         SEWU(I+1)*STB(K)
                          AN(I+1,J-1,K) = 0.0
                       END IF
                    END IF
C--------------------------------------------------------------------
C               NORTHEAST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
                    IF (J.NE.jEndGhost.AND.I.NE.iEndGhost) THEN
                       IF (PCELL(I+1,J+1,K).NE.WALL.AND.
     &                      PCELL(I,J+1,K).NE.WALL) THEN
                          YP = Y(J+1)-YV(J+1)
                          TMULT = VISCOS/YP
c                  TAUYX(I+1,J+1,K) = -TMULT*U(I+1,J+1,K)
c                  SU(I+1,J+1,K) = SUL(I+1,J+1,K)
                          SP(I+1,J+1,K) = SP(I+1,J+1,K) - TMULT*
     &                         SEWU(I+1)*STB(K)
                          AS(I+1,J+1,K) = 0.0
                       END IF
                    END IF
C--------------------------------------------------------------------
C               BOTTOM EAST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
                    IF (K.NE.kBegGhost.AND.I.NE.iEndGhost) THEN
                       IF (PCELL(I+1,J,K-1).NE.WALL.AND.
     &                      PCELL(I,J,K-1).NE.WALL) THEN
                          ZP = ZW(K)-Z(K-1)
                          TMULT = VISCOS/ZP
c                  TAUZX(I+1,J,K-1) = -TMULT*U(I+1,J,K-1)
c                  SU(I+1,J,K-1) = SUL(I+1,J,K-1)
                          SP(I+1,J,K-1) =SP(I+1,J,K-1)-TMULT*
     &                         SEWU(I+1)*SNS(J)
                          AT(I+1,J,K-1) = 0.0
                       END IF
                    END IF
C--------------------------------------------------------------------
C               TOP EAST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
                    IF (K.NE.kEndGhost.AND.I.NE.iEndGhost) THEN
                       IF (PCELL(I+1,J,K+1).NE.WALL.AND.
     &                      PCELL(I,J,K+1).NE.WALL) THEN
                          ZP = Z(K+1)-ZW(K+1)
                          TMULT = VISCOS/ZP
c                  TAUZX(I+1,J,K+1) = -TMULT*U(I+1,J,K+1)
c                  SU(I+1,J,K+1) = SUL(I+1,J,K+1)
                          SP(I+1,J,K+1) =SP(I+1,J,K+1)-TMULT*SEWU(I+1)
     &                         *SNS(J)
                          AB(I+1,J,K+1) = 0.0
                       END IF
                    END IF
                 END IF
              END IF
           END IF
 100    CONTINUE
 110  CONTINUE
 120  CONTINUE

C--------------------------------------------------------------------
C     PRESSURE BOUNDARY CONDITIONS 
C--------------------------------------------------------------------
C--------------------------------------------------------------------
C     East AND/OR West WALLS
C--------------------------------------------------------------------
      DO 310 K = kst, kend
        DO 300 J = jst, jend
           IF (PCELL(iEndGhost, J, K).EQ.PBCFLD) THEN
              UU(iEndGhost+1,J,K) = UU(iEndGhost,J,K)
          END IF
          IF (PCELL(iBegGhost,J,K).EQ.PBCFLD) THEN
             UU(iBegGhost,J,K) = UU(ist,J,K)
          END IF
  300   CONTINUE
  310 CONTINUE
C--------------------------------------------------------------------
C     North AND/OR South WALLS
C--------------------------------------------------------------------
      DO 410 K = kst, kend
        DO 400 I = ist, iend
          IF ((PCELL(I,jEndGhost, K).EQ.PBCFLD).AND.
     $        (VV(I,jEndGhost+1,K).GT.0.0)) THEN
             UU(I,jEndGhost,K) = UU(I,jend,K)
             AN(I,jend,K) = 2.0*TINY
          END IF
          IF (PCELL(I,jBegGhost,K).EQ.PBCFLD.AND.
     $         VV(I,jBegGhost,K).LT.0.0) THEN
            UU(I,jBegGhost, K) = UU(I,jst,K)
            AS(I,jst,K) = 2.0*TINY
            END IF
  400   CONTINUE
  410 CONTINUE


C--------------------------------------------------------------------
C     TOP AND/OR BOTTOM WALLS
C--------------------------------------------------------------------
      DO 510 J = jst, jend
        DO 500 I = ist, iend
          IF ((PCELL(I,J,kEndGhost).EQ.PBCFLD).AND.
     $        (WW(I,J,kEndGhost+1).GT.0.0)) THEN
             UU(I,J,kEndGhost) = UU(I,J,kend)
             AT(I,J,kend) = 2.0*TINY
          END IF
          IF (PCELL(I,J,kBegGhost).EQ.PBCFLD.AND.
     $         WW(I,J,kBegGhost).LT.0.0) THEN
            UU(I,J,kBegGhost) = UU(I,J,kst)
            AB(I,J,kst) = 2.0*TINY
            END IF
  500   CONTINUE
  510 CONTINUE

      RETURN
      END

c*********************************************************************
c
c $Log$
c Revision 1.7  2000/09/26 04:35:34  rawat
c added some more multi-patch support
c
c Revision 1.6  2000/09/07 23:07:21  rawat
c fixed some bugs in bc and added pressure solver using petsc
c
c Revision 1.5  2000/08/23 06:20:53  bbanerje
c 1) Results now correct for pressure solve.
c 2) Modified BCU, BCV, BCW to add stuff for pressure BC.
c 3) Removed some bugs in BCU, V, W.
c 4) Coefficients for MOM Solve not computed correctly yet.
c
c Revision 1.4  2000/08/17 20:32:02  rawat
c Fixed some bugs
c
c Revision 1.3  2000/08/10 00:56:38  rawat
c added pressure bc for scalar and changed discretization option for velocity
c
c Revision 1.2  2000/07/14 03:45:48  rawat
c completed velocity bc and fixed some bugs
c
c Revision 1.1  2000/07/12 23:59:23  rawat
c added wall bc for u-velocity
c
c*********************************************************************
