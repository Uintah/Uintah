c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE bcwvel(domLoW, domHiW, idxLoW, idxHiW, WW,
     &     AP, AE, AW, AN, AS, AT, AB,
     &     SU, SP, 
     &     domLoU, domHiU, idxLoU, idxHiU, UU,
     &     domLoV, domHiV, idxLoV, idxHiV, VV,
     &     domLo, domHi, idxLo, idxHi, PCELL, WALL, FFIELD, PBCFLD,
     &     VISCOS, 
     &     SEW, SNS, STBW,
     &     X, XU, Y, YV, 
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
      integer domLoW(3), domHiW(3), idxLoW(3), idxHiW(3)
      double precision WW(domLoW(1):domHiW(1), domLoW(2):domHiW(2), 
     &     domLoW(3):domHiW(3))
      double precision AP(domLoW(1):domHiW(1), domLoW(2):domHiW(2), 
     &     domLoW(3):domHiW(3))
      double precision AE(domLoW(1):domHiW(1), domLoW(2):domHiW(2), 
     &     domLoW(3):domHiW(3))
      double precision AW(domLoW(1):domHiW(1), domLoW(2):domHiW(2), 
     &     domLoW(3):domHiW(3))
      double precision AN(domLoW(1):domHiW(1), domLoW(2):domHiW(2), 
     &     domLoW(3):domHiW(3))
      double precision AS(domLoW(1):domHiW(1), domLoW(2):domHiW(2), 
     &     domLoW(3):domHiW(3))
      double precision AT(domLoW(1):domHiW(1), domLoW(2):domHiW(2), 
     &     domLoW(3):domHiW(3))
      double precision AB(domLoW(1):domHiW(1), domLoW(2):domHiW(2), 
     &     domLoW(3):domHiW(3))
      double precision SU(domLoW(1):domHiW(1), domLoW(2):domHiW(2), 
     &     domLoW(3):domHiW(3))
      double precision SP(domLoW(1):domHiW(1), domLoW(2):domHiW(2), 
     &     domLoW(3):domHiW(3))
      integer domLoU(3), domHiU(3), idxLoU(3), idxHiU(3)
      double precision UU(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      integer domLoV(3), domHiV(3), idxLoV(3), idxHiV(3)
      double precision VV(domLoV(1):domHiV(1), domLoV(2):domHiV(2), 
     &     domLoV(3):domHiV(3))
      integer domLo(3), domHi(3), idxLo(3), idxHi(3)
      integer PCELL(domLo(1):domHi(1), domLo(2):domHi(2), 
     &     domLo(3):domHi(3))
      double precision SEW(domLo(1):domHi(1))
      double precision SNS(domLo(2):domHi(2))
      double precision STBW(domLo(3):domHi(3))
      double precision X(domLo(1):domHi(1))
      double precision XU(domLo(1):domHi(1))
      double precision Y(domLo(2):domHi(2))
      double precision YV(domLo(2):domHi(2))
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
      integer idxUInlet, idxVInlet
      integer i, j, k


      double precision XP, YP, TMULT

c*********************************************************************
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
      idxUInlet = idxHiU(1)+1
      idxVInlet = idxHiV(2)+1
      
c------------------------------------------------------------------ 
c     interior intrusions
c------------------------------------------------------------------ 
      const = ONE
      do 820 k=kst, kend
         do 810 j=jst, jend
            do 800 i=ist, iend

               if (pcell(i,j,k).eq.wall.or.
     $              pcell(i,j,k-1).eq.wall)
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
               WW(I,J,K) = 0.0
C--------------------------------------------------------------------
C             BOTTOM FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
               IF (K.NE.kBegGhost) THEN
                  IF(PCELL(I,J,K-1).EQ.FFIELD) AT(I,J,K-1) = 0.0
               END IF
C--------------------------------------------------------------------
C               TOP FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
               IF (K.NE.kEndGhost) THEN
                  IF (PCELL(I, J, K+1).NE.WALL) THEN
                     IF (K.NE.kend) AB(I,J,K+2) = 0.0
                  end if
               end if
C--------------------------------------------------------------------
C             BOTTOMEAST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
               IF (I.NE.iEndGhost.AND.K.NE.kBegGhost) THEN
                  IF (PCELL(I+1,J,K-1).NE.WALL.AND.
     &                 PCELL(I+1,J,K).NE.WALL) THEN
                     XP = X(I+1) - XU(I+1)
                     TMULT = VISCOS/XP
                     SP(I+1,J,K) =SP(I+1,J,K)-TMULT*SNS(J)*STBW(K)
                     AW(I+1,J,K) = 0.0
                  END IF
               END IF

C--------------------------------------------------------------------
C             BOTTOMWEST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
               IF (I.NE.iBegGhost.AND.K.NE.kBegGhost) THEN
                  IF (PCELL(I-1,J,K-1).NE.WALL.AND.
     &                 PCELL(I-1,J,K).NE.WALL) THEN
                     XP = XU(I) - X(I-1)
                     TMULT = VISCOS/XP
                     SP(I-1,J,K) =SP(I-1,J,K)-TMULT*SNS(J)*STBW(K)
                     AE(I-1,J,K) = 0.0
                  END IF
               END IF
C--------------------------------------------------------------------
C             BOTTOM SOUTH FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
              IF (K.NE.kBegGhost.AND.J.NE.jBegGhost) THEN
                 IF (PCELL(I,J-1,K-1).NE.WALL.AND.
     &                PCELL(I,J-1,K).NE.WALL) THEN
                    YP = YV(J)-Y(J-1)
                    TMULT = VISCOS/YP
               
                    SP(I,J-1,K) = SP(I,J-1,K)-TMULT*STBW(K)*SEW(I)
                    AS(I,J-1,K) = 0.0
                 END IF 
              END IF
C--------------------------------------------------------------------
C             BOTTOM NORTH FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
              IF (K.NE.kBegGhost.AND.J.NE.jEndGhost) THEN
                 IF (PCELL(I,J+1,K-1).NE.WALL.AND.
     &                PCELL(I,J+1,K).NE.WALL) THEN
                    YP = Y(J+1)-YV(J+1)
                    TMULT = VISCOS/YP
c                TAUZX(I,J,K+1) = -TMULT*U(I,J,K+1)
c                SU(I,J,K+1) = SUL(I,J,K+1)
                    SP(I,J+1,K) = SP(I,J+1,K)-TMULT*SEW(I)*STBW(K)
                    AS(I,J+1,K) = 0.0
                 END IF
              END IF
C--------------------------------------------------------------------
C             IF CELL ABOVE INTRUSION IS ALSO AN INTRUSION, SKIP
C             MODIFICATIONS FOR TOP FACES OF FIRST INTRUSION
C--------------------------------------------------------------------
              IF (K.NE.kEndGhost) THEN
                 IF (PCELL(I,J,K+1).NE.WALL) THEN
C--------------------------------------------------------------------
C               TOPWEST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
                    IF (I.NE.iBegGhost.AND.K.NE.kEndGhost) THEN
                       IF (PCELL(I-1,J,K+1).NE.WALL.AND.
     &                      PCELL(I-1,J,K).NE.WALL)  THEN
                          XP = XU(I)-X(I-1)
                          TMULT = VISCOS/XP
c                       TAUYX(I+1,J-1,K) = -TMULT*U(I+1,J-1,K)
c                       SU(I+1,J-1,K) = SUL(I+1,J-1,K)
                          SP(I-1,J,K+1) = SP(I-1,J,K+1) - TMULT*
     &                         SNS(J)*STBW(K+1)
                          AE(I-1,J,K+1) = 0.0
                       END IF
                    END IF
C--------------------------------------------------------------------
C               TOPEAST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
                    IF (I.NE.iEndGhost.AND.K.NE.kEndGhost) THEN
                       IF (PCELL(I+1,J,K+1).NE.WALL.AND.
     &                      PCELL(I+1,J,K).NE.WALL) THEN
                          XP = X(I+1)-XU(I+1)
                          TMULT = VISCOS/XP
c                  TAUYX(I+1,J+1,K) = -TMULT*U(I+1,J+1,K)
c                  SU(I+1,J+1,K) = SUL(I+1,J+1,K)
                          SP(I+1,J,K+1) = SP(I+1,J,K+1) - TMULT*
     &                         SNS(J)*STBW(K+1)
                          AW(I+1,J,K+1) = 0.0
                       END IF
                    END IF
C--------------------------------------------------------------------
C               TOP SOUTH FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
                    IF (J.NE.jBegGhost.AND.K.NE.kEndGhost) THEN
                       IF (PCELL(I,J-1,K+1).NE.WALL.AND.
     &                      PCELL(I,J-1,K).NE.WALL) THEN
                          YP = YV(J)-Y(J-1)
                          TMULT = VISCOS/YP
c                  TAUZX(I+1,J,K-1) = -TMULT*U(I+1,J,K-1)
c                  SU(I+1,J,K-1) = SUL(I+1,J,K-1)
                          SP(I,J-1,K+1) =SP(I,J-1,K+1)-TMULT*
     &                         STBW(K+1)*SEW(I)
                          AN(I,J-1,K+1) = 0.0
                       END IF
                    END IF
C--------------------------------------------------------------------
C               TOP NORTH FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
                    IF (K.NE.kEndGhost.AND.J.NE.jEndGhost) THEN
                       IF (PCELL(I,J+1,K+1).NE.WALL.AND.
     &                      PCELL(I,J+1,K).NE.WALL) THEN
                          YP = Y(J+1)-YV(J+1)
                          TMULT = VISCOS/YP
c                  TAUZX(I+1,J,K+1) = -TMULT*U(I+1,J,K+1)
c                  SU(I+1,J,K+1) = SUL(I+1,J,K+1)
                          SP(I,J+1,K+1) =SP(I,J+1,K+1)-TMULT*SEW(I)
     &                         *STBW(K+1)
                          AS(I,J+1,K+1) = 0.0
                       END IF
                    END IF
                 END IF
              END IF
           END IF
 100    CONTINUE
 110  CONTINUE
 120  CONTINUE

C--------------------------------------------------------------------
C     PRESSURE BOUNDARY CONDITOINS 
C       EAST AND/OR WEST WALLS
C--------------------------------------------------------------------
      DO 210 K = kst,kend
         DO 200 J = jst,jend
            IF (PCELL(iEndGhost,J,K).EQ.PBCFLD
     &           .AND. UU(idxUInlet,J,K).GT.0.0) THEN
               WW(iEndGhost,J,K) = WW(iend,J,K)
               AE(iend,J,K) = 2.0*TINY
            END IF
            IF (PCELL(iBegGhost,J,K).EQ.PBCFLD
     &           .AND. UU(iBegGhost,J,K).LT.0.0) THEN
               WW(iBegGhost,J,K) = WW(ist,J,K)
               AW(ist,J,K) = 2.0*TINY
            END IF
 200     CONTINUE
 210  CONTINUE
C-------------------------------------------------------------------
C      NORTH AND/OR SOUTH WALLS
C-------------------------------------------------------------------
      DO 310 K = kst,kend
        DO 300 I = kst,iend
          IF (PCELL(I,jEndGhost,K).EQ.PBCFLD
     &          .AND. VV(I,idxVInlet,K).GT.0.0) THEN
             WW(I,jEndGhost,K) = WW(I,jend,K)
             AN(I,jend,K) = 2.0*TINY
          END IF
          IF (PCELL(I,jBegGhost,K).EQ.PBCFLD
     &         .AND. VV(I,jBegGhost,K).LT.0.0) THEN
            WW(I,jBegGhost,K) = WW(I,jst,K)
            AS(I,jst,K) = 2.0*TINY
          END IF
  300   CONTINUE
  310 CONTINUE
      RETURN
      END

c*********************************************************************
c
c $Log$
c Revision 1.3  2000/08/23 06:20:53  bbanerje
c 1) Results now correct for pressure solve.
c 2) Modified BCU, BCV, BCW to add stuff for pressure BC.
c 3) Removed some bugs in BCU, V, W.
c 4) Coefficients for MOM Solve not computed correctly yet.
c
c Revision 1.2  2000/08/17 20:32:02  rawat
c Fixed some bugs
c
c Revision 1.1  2000/07/14 03:58:43  bbanerje
c Added missed files.
c
c Revision 1.1  2000/07/12 23:59:23  rawat
c added wall bc for u-velocity
c
c*********************************************************************
