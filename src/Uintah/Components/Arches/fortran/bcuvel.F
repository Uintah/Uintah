c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE bcuvel(domLoU, domHiU, idxLoU, idxHiU, UU,
     &     AP, AE, AW, AN, AS, AT, AB,
     &     SU, SP, 
     &     domLo, domHi, PCELL, WALL, FFIELD, VISCOS, 
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
      double precision UU(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision AP(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision AE(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision AW(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision AN(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision AS(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision AT(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision AB(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision SU(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision SP(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      integer domLo(3), domHi(3)
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
      integer FFIELD
      integer ierr

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer ISTU, JSTU, KSTU, IENDU, JENDU, KENDU
      integer i, j, k


      double precision YP, ZP, TMULT

c*********************************************************************
c     Start :
c*********************************************************************
      ISTU = idxLoU(1)+1
      JSTU = idxLoU(2)+1
      KSTU = idxLoU(3)+1
      IENDU = idxHiU(1)
      JENDU = idxHiU(2)
      KENDU = idxHiU(3)
      
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
      DO 120 K = idxLoU(3), idxHiU(3)
C$DIR SCALAR
        DO 110 J = idxLoU(2), idxHiU(2)
C$DIR SCALAR
          DO 100 I = idxLoU(1), idxLoU(1)
            IF (PCELL(I,J,K).EQ.WALL) THEN
               UU(I,J,K) = 0.0
C--------------------------------------------------------------------
C             WEST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
               IF (I.NE.idxLoU(1)) THEN
                  IF(PCELL(I-1,J,K).EQ.FFIELD) AE(I-1,J,K) = 0.0
               END IF
C--------------------------------------------------------------------
C               EAST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
               IF (I.NE.idxHiU(1)) THEN
                  IF (PCELL(I+1, J, K).NE.WALL) THEN
                     IF (I.NE.(idxHiU(1)-1)) AW(I+2,J,K) = 0.0
                  end if
               end if
C--------------------------------------------------------------------
C             SOUTHWEST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
               IF (J.NE.idxLoU(2).AND.I.NE.idxLoU(1)) THEN
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
               IF (J.NE.idxHiU(2).AND.I.NE.idxLoU(1)) THEN
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
              IF (K.NE.idxLoU(3).AND.I.NE.idxLoU(1)) THEN
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
              IF (K.NE.idxHiU(3).AND.I.NE.idxLoU(1)) THEN
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
              IF (I.NE.idxHiU(1)) THEN
                 IF (PCELL(I+1,J,K).NE.WALL) THEN
C--------------------------------------------------------------------
C               SOUTHEAST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
                    IF (J.NE.idxLoU(2).AND.I.NE.idxHiU(1)) THEN
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
                    IF (J.NE.idxHiU(2).AND.I.NE.idxHiU(1)) THEN
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
                    IF (K.NE.idxLoU(3).AND.I.NE.idxHiU(1)) THEN
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
                    IF (K.NE.idxHiU(3).AND.I.NE.idxHiU(1)) THEN
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

      RETURN
      END

c*********************************************************************
c
c $Log$
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
