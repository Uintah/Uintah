c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE bcp(indexLow, indexHigh, NI, NJ, NK, NITER, PCELL,
     &     PP,
     &     AE,AW,AN,AS,AT,AB,
     &     RR,
     $     LWRAP,LSYMB,LPRDAV,LPRDJK,LPRDKK)
*-----------------------------------------------------------------------
*     This subroutine calculates the boundary
*     conditions for the pressure and pressure correction equations.
*-----------------------------------------------------------------------
      implicit none

#include "param4.h"
#include "pcell.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer indexLow(3)
      integer indexHigh(3)
      integer NI, NJ, NK
      integer NITER
      integer PCELL(indexLow(1):indexHigh(1), indexLow(2):indexHigh(2), 
     &     indexLow(3):indexHigh(3))
      double precision PP(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AE(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AW(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AN(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AS(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AT(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AB(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision RR(indexLow(2):indexHigh(2))
      LOGICAL LWRAP,LSYMB, LPRDKK, LPRDAV, LPRDJK

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer IST, JST, KST, IEND, JEND, KEND
      integer i,j,k,iendp1,jendp1,kendp1,istm1,jstm1,kstm1

c*********************************************************************
c     Start :
c*********************************************************************
      ISTM1 = indexLow(1)
      JSTM1 = indexLow(2)
      KSTM1 = indexLow(3)
      IENDP1 = indexHigh(1)
      JENDP1 = indexHigh(2)
      KENDP1 = indexHigh(3)
*----------------------------------------------------------------------
*     executable statements
*----------------------------------------------------------------------
      IEND = IENDP1 - 1
      JEND = JENDP1 - 1
      KEND = KENDP1 - 1
      IST  = ISTM1 + 1
      JST  = JSTM1 + 1
      KST  = KSTM1 + 1
*----------------------------------------------------------------------
*     INLET/OUTLET BOUNDARY CONDITIONS 
*       EAST AND/OR WEST WALLS
*----------------------------------------------------------------------
C$DIR SCALAR
      DO 610 K = KST,KEND
C$DIR SCALAR
         DO 600 J = JST,JEND
            IF (PCELL(IENDP1,J,K).NE.WALL .AND.
     &           PCELL(IEND,J,K).NE.WALL) THEN
               AE(IEND,J,K) = ZERO
            END IF
            IF (PCELL(ISTM1,J,K).NE.WALL .AND.
     &           PCELL(IST,J,K).NE.WALL) THEN
               AW(IST,J,K) = ZERO
            END IF
 600     CONTINUE
 610  CONTINUE
*--------------------------------------------------------------------
*     NORTH AND/OR SOUTH WALLS
*--------------------------------------------------------------------
C$DIR SCALAR
      DO 710 K = KST,KEND
C$DIR SCALAR
         DO 700 I = IST,IEND
            IF (PCELL(I,JENDP1,K).NE.WALL.AND.
     &           PCELL(I,JEND,K).NE.WALL) THEN
               AN(I,JEND,K) = ZERO
            END IF
            IF (PCELL(I,JSTM1,K).NE.WALL.AND.
     &           PCELL(I,JST,K).NE.WALL) THEN
               AS(I,JST,K) = ZERO
            END IF
 700     CONTINUE
 710  CONTINUE
*--------------------------------------------------------------------
*     TOP AND/OR BOTTOM WALLS
*--------------------------------------------------------------------
C$DIR SCALAR
      DO 810 J = JST,JEND
C$DIR SCALAR
         DO 800 I = IST,IEND
            IF (PCELL(I,J,KENDP1).NE.WALL .AND. 
     &           PCELL(I,J,KEND).NE.WALL) THEN
               AT(I,J,KEND) = ZERO
            END IF
            IF (PCELL(I,J,KSTM1).NE.WALL .AND. 
     &           PCELL(I,J,KST).NE.WALL) THEN
               AB(I,J,KST) = ZERO
            END IF
 800     CONTINUE
 810  CONTINUE
*--------------------------------------------------------------------
*     SYMMETRY AND INTRUSION BOUNDARY CONDITIONS
*--------------------------------------------------------------------
      IF (LSYMB) THEN
         CALL SYMBCS(indexLow, indexHigh, PCELL,
     $        PP,
     &        AE,AW,AN,AS,AT,AB,
     $        LPRDJK, LPRDKK)
      ENDIF
      IF (LWRAP) THEN
         CALL PRDBC1(indexLow, indexHigh, NITER, PCELL,
     $        PP,RR,
     &        LPRDAV, LPRDJK, LPRDKK)
         CALL PRDBC2(indexLow, indexHigh, NITER, PCELL,
     $        PP,
     &        AS,AT,AB,
     &        RR,
     $        LPRDAV, LPRDJK, LPRDKK)
      ENDIF
      CALL WALLBC(indexLow, indexHigh, PCELL,
     $     AE,AW,AN,AS,AT,AB)
      RETURN
      END 

c*********************************************************************
c
c $Log$
c Revision 1.1  2000/04/12 20:31:42  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
