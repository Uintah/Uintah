c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE bcwvel(domLoW, domHiW, idxLoW, idxHiW, WW,
     &     AP, AE, AW, AN, AS, AT, AB,
     &     SU, SP, 
     &     domLo, domHi, PCELL, WALL, FFIELD, VISCOS, 
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
      integer domLo(3), domHi(3)
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
      integer FFIELD
      integer ierr

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer ISTU, JSTU, KSTU, IENDU, JENDU, KENDU
      integer i, j, k


      double precision XP, YP, TMULT

c*********************************************************************
c     Start :
c*********************************************************************
      ISTU = idxLoW(1)+1
      JSTU = idxLoW(2)+1
      KSTU = idxLoW(3)+1
      IENDU = idxHiW(1)-1
      JENDU = idxHiW(2)-1
      KENDU = idxHiW(3)-1
      
c------------------------------------------------------------------ 
c     interior intrusions
c------------------------------------------------------------------ 
      const = ONE
      do 820 k=kstu, kendu
         do 810 j=jstu, jendu
            do 800 i=istu, iendu

               if (pcell(i-1,j,k).eq.wall.or.
     $              pcell(i,j,k).eq.wall)
     $              call fixval(zero,ae(i,j,k),aw(i,j,k),an(i,j,k),
     $                          as(i,j,k),at(i,j,k),ab(i,j,k),
     $              su(i,j,k),sp(i,j,k),const)

 800        continue
 810     continue
 820  continue


c*********************************************************************
c     Go thru all the cells
c*********************************************************************
C$DIR SCALAR
      DO 120 K = idxLoW(3), idxHiW(3)
C$DIR SCALAR
        DO 110 J = idxLoW(2), idxHiW(2)
C$DIR SCALAR
          DO 100 I = idxLoW(1), idxLoW(1)
            IF (PCELL(I,J,K).EQ.WALL) THEN
               WW(I,J,K) = 0.0
C--------------------------------------------------------------------
C             BOTTOM FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
               IF (K.NE.idxLoW(3)) THEN
                  IF(PCELL(I,J,K-1).EQ.FFIELD) AT(I,J,K-1) = 0.0
               END IF
C--------------------------------------------------------------------
C               TOP FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
               IF (K.NE.idxHiW(3)) THEN
                  IF (PCELL(I, J, K+1).NE.WALL) THEN
                     IF (K.NE.(idxHiW(3)-1)) AB(I,J,K+2) = 0.0
                  end if
               end if
C--------------------------------------------------------------------
C             BOTTOMEAST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
               IF (I.NE.idxHiW(1).AND.K.NE.idxLoW(3)) THEN
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
               IF (I.NE.idxLoW(1).AND.K.NE.idxLoW(3)) THEN
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
              IF (K.NE.idxLoW(3).AND.J.NE.idxLoW(2)) THEN
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
              IF (K.NE.idxLoW(3).AND.J.NE.idxHiW(2)) THEN
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
              IF (K.NE.idxHiW(3)) THEN
                 IF (PCELL(I,J,K+1).NE.WALL) THEN
C--------------------------------------------------------------------
C               TOPWEST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
                    IF (I.NE.idxLoW(1).AND.K.NE.idxHiW(3)) THEN
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
                    IF (I.NE.idxHiW(1).AND.K.NE.idxHiW(3)) THEN
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
                    IF (J.NE.idxLoW(2).AND.K.NE.idxHiW(3)) THEN
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
                    IF (K.NE.idxHiW(3).AND.J.NE.idxHiW(2)) THEN
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

      RETURN
      END

c*********************************************************************
c
c $Log$
c Revision 1.1  2000/07/14 03:58:43  bbanerje
c Added missed files.
c
c Revision 1.1  2000/07/12 23:59:23  rawat
c added wall bc for u-velocity
c
c*********************************************************************
