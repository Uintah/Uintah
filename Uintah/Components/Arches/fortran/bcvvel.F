c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE bcvvel(domLoV, domHiV, idxLoV, idxHiV, VV,
     &     AP, AE, AW, AN, AS, AT, AB,
     &     SU, SP, 
     &     domLo, domHi, PCELL, WALL, FFIELD, VISCOS, 
     &     SEW, SNSV, STB,
     &     X, XU, Z, ZW, 
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
      integer domLoV(3), domHiV(3), idxLoV(3), idxHiV(3)
      double precision VV(domLoV(1):domHiV(1), domLoV(2):domHiV(2), 
     &     domLoV(3):domHiV(3))
      double precision AP(domLoV(1):domHiV(1), domLoV(2):domHiV(2), 
     &     domLoV(3):domHiV(3))
      double precision AE(domLoV(1):domHiV(1), domLoV(2):domHiV(2), 
     &     domLoV(3):domHiV(3))
      double precision AW(domLoV(1):domHiV(1), domLoV(2):domHiV(2), 
     &     domLoV(3):domHiV(3))
      double precision AN(domLoV(1):domHiV(1), domLoV(2):domHiV(2), 
     &     domLoV(3):domHiV(3))
      double precision AS(domLoV(1):domHiV(1), domLoV(2):domHiV(2), 
     &     domLoV(3):domHiV(3))
      double precision AT(domLoV(1):domHiV(1), domLoV(2):domHiV(2), 
     &     domLoV(3):domHiV(3))
      double precision AB(domLoV(1):domHiV(1), domLoV(2):domHiV(2), 
     &     domLoV(3):domHiV(3))
      double precision SU(domLoV(1):domHiV(1), domLoV(2):domHiV(2), 
     &     domLoV(3):domHiV(3))
      double precision SP(domLoV(1):domHiV(1), domLoV(2):domHiV(2), 
     &     domLoV(3):domHiV(3))
      integer domLo(3), domHi(3)
      integer PCELL(domLo(1):domHi(1), domLo(2):domHi(2), 
     &     domLo(3):domHi(3))
      double precision SEW(domLo(1):domHi(1))
      double precision SNSV(domLo(2):domHi(2))
      double precision STB(domLo(3):domHi(3))
      double precision X(domLo(1):domHi(1))
      double precision XU(domLo(1):domHi(1))
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


      double precision XP, ZP, TMULT

c*********************************************************************
c     Start :
c*********************************************************************
      ISTU = idxLoV(1)+1
      JSTU = idxLoV(2)+1
      KSTU = idxLoV(3)+1
      IENDU = idxHiV(1)-1
      JENDU = idxHiV(2)-1
      KENDU = idxHiV(3)-1
      
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
      DO 120 K = idxLoV(3), idxHiV(3)
C$DIR SCALAR
        DO 110 J = idxLoV(2), idxHiV(2)
C$DIR SCALAR
          DO 100 I = idxLoV(1), idxLoV(1)
            IF (PCELL(I,J,K).EQ.WALL) THEN
               VV(I,J,K) = 0.0
C--------------------------------------------------------------------
C             SOUTH FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
               IF (J.NE.idxLoV(2)) THEN
                  IF(PCELL(I,J-1,K).EQ.FFIELD) AN(I,J-1,K) = 0.0
               END IF
C--------------------------------------------------------------------
C               NORTH FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
               IF (J.NE.idxHiV(2)) THEN
                  IF (PCELL(I, J+1, K).NE.WALL) THEN
                     IF (J.NE.(idxHiV(2)-1)) AS(I,J+2,K) = 0.0
                  end if
               end if
C--------------------------------------------------------------------
C             SOUTHEAST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
               IF (I.NE.idxHiV(1).AND.J.NE.idxLoV(2)) THEN
                  IF (PCELL(I+1,J,K).NE.WALL.AND.
     &                 PCELL(I+1,J-1,K).NE.WALL) THEN
                     XP = X(I+1) - XU(I+1)
                     TMULT = VISCOS/XP
                     SP(I+1,J,K) =SP(I+1,J,K)-TMULT*SNSV(J)*STB(K)
                     AW(I+1,J,K) = 0.0
                  END IF
               END IF

C--------------------------------------------------------------------
C             SOUTHWEST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
               IF (I.NE.idxLoV(1).AND.J.NE.idxLoV(2)) THEN
                  IF (PCELL(I-1,J,K).NE.WALL.AND.
     &                 PCELL(I-1,J-1,K).NE.WALL) THEN
                     XP = XU(I) - X(I-1)
                     TMULT = VISCOS/XP
                     SP(I-1,J,K) =SP(I-1,J,K)-TMULT*SNSV(J)*STB(K)
                     AE(I-1,J,K) = 0.0
                  END IF
               END IF
C--------------------------------------------------------------------
C             BOTTOM SOUTH FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
              IF (K.NE.idxLoV(3).AND.J.NE.idxLoV(2)) THEN
                 IF (PCELL(I,J-1,K-1).NE.WALL.AND.
     &                PCELL(I,J,K-1).NE.WALL) THEN
                    ZP = ZW(K)-Z(K-1)
                    TMULT = VISCOS/ZP
               
                    SP(I,J,K-1) = SP(I,J,K-1)-TMULT*SNSV(J)*SEW(I)
                    AT(I,J,K-1) = 0.0
                 END IF 
              END IF
C--------------------------------------------------------------------
C             TOP SOUTH FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
              IF (K.NE.idxHiV(3).AND.J.NE.idxLoV(2)) THEN
                 IF (PCELL(I,J-1,K+1).NE.WALL.AND.
     &                PCELL(I,J,K+1).NE.WALL) THEN
                    ZP = Z(K+1)-ZW(K+1)
                    TMULT = VISCOS/ZP
c                TAUZX(I,J,K+1) = -TMULT*U(I,J,K+1)
c                SU(I,J,K+1) = SUL(I,J,K+1)
                    SP(I,J,K+1) = SP(I,J,K+1)-TMULT*SEW(I)*SNSV(J)
                    AB(I,J,K+1) = 0.0
                 END IF
              END IF
C--------------------------------------------------------------------
C             IF CELL NORTH OF INTRUSION IS AN INTRUSION, SKIP
C             MODIFICATIONS FOR NORTHERN FACES OF FIRST INTRUSION.
C--------------------------------------------------------------------
              IF (J.NE.idxHiV(2)) THEN
                 IF (PCELL(I,J+1,K).NE.WALL) THEN
C--------------------------------------------------------------------
C               NORTHWEST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
                    IF (I.NE.idxLoV(1).AND.J.NE.idxHiV(2)) THEN
                       IF (PCELL(I-1,J+1,K).NE.WALL.AND.
     &                      PCELL(I-1,J,K).NE.WALL)  THEN
                          XP = XU(I)-X(I-1)
                          TMULT = VISCOS/XP
c                       TAUYX(I+1,J-1,K) = -TMULT*U(I+1,J-1,K)
c                       SU(I+1,J-1,K) = SUL(I+1,J-1,K)
                          SP(I-1,J+1,K) = SP(I-1,J+1,K) - TMULT*
     &                         SNSV(J+1)*STB(K)
                          AE(I-1,J+1,K) = 0.0
                       END IF
                    END IF
C--------------------------------------------------------------------
C               NORTHEAST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
                    IF (I.NE.idxHiV(1).AND.J.NE.idxHiV(2)) THEN
                       IF (PCELL(I+1,J+1,K).NE.WALL.AND.
     &                      PCELL(I+1,J,K).NE.WALL) THEN
                          XP = X(I+1)-XU(I+1)
                          TMULT = VISCOS/XP
c                  TAUYX(I+1,J+1,K) = -TMULT*U(I+1,J+1,K)
c                  SU(I+1,J+1,K) = SUL(I+1,J+1,K)
                          SP(I+1,J+1,K) = SP(I+1,J+1,K) - TMULT*
     &                         SNSV(J+1)*STB(K)
                          AW(I+1,J+1,K) = 0.0
                       END IF
                    END IF
C--------------------------------------------------------------------
C               BOTTOM NORTH FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
                    IF (K.NE.idxLoV(3).AND.J.NE.idxHiV(2)) THEN
                       IF (PCELL(I,J+1,K-1).NE.WALL.AND.
     &                      PCELL(I,J,K-1).NE.WALL) THEN
                          ZP = ZW(K)-Z(K-1)
                          TMULT = VISCOS/ZP
c                  TAUZX(I+1,J,K-1) = -TMULT*U(I+1,J,K-1)
c                  SU(I+1,J,K-1) = SUL(I+1,J,K-1)
                          SP(I,J+1,K-1) =SP(I,J+1,K-1)-TMULT*
     &                         SNSV(J+1)*SEW(I)
                          AT(I,J+1,K-1) = 0.0
                       END IF
                    END IF
C--------------------------------------------------------------------
C               TOP NORTH FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
                    IF (K.NE.idxHiV(3).AND.J.NE.idxHiV(2)) THEN
                       IF (PCELL(I,J+1,K+1).NE.WALL.AND.
     &                      PCELL(I,J,K+1).NE.WALL) THEN
                          ZP = Z(K+1)-ZW(K+1)
                          TMULT = VISCOS/ZP
c                  TAUZX(I+1,J,K+1) = -TMULT*U(I+1,J,K+1)
c                  SU(I+1,J,K+1) = SUL(I+1,J,K+1)
                          SP(I,J+1,K+1) =SP(I,J+1,K+1)-TMULT*SEW(I)
     &                         *SNSV(J+1)
                          AB(I,J+1,K+1) = 0.0
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
