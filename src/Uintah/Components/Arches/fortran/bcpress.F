c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE bcpress(domLo, domHi, idxLo, idxHi,
     &     PP,
     &     AE,AW,AN,AS,AT,AB,
     &     PCELL, WALL, SYMTRY, FFIELD)
*-----------------------------------------------------------------------
*     This subroutine calculates the boundary
*     conditions for the pressure and pressure correction equations.
*-----------------------------------------------------------------------
      implicit none

#include "param4.h"
c#include "pcell.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer domLo(3), domHi(3), idxLo(3), idxHi(3)
      double precision PP(domLo(1):domHi(1), domLo(2):domHi(2),
     &     domLo(3):domHi(3))
      double precision AE(domLo(1):domHi(1), domLo(2):domHi(2),
     &     domLo(3):domHi(3))
      double precision AW(domLo(1):domHi(1), domLo(2):domHi(2),
     &     domLo(3):domHi(3))
      double precision AN(domLo(1):domHi(1), domLo(2):domHi(2),
     &     domLo(3):domHi(3))
      double precision AS(domLo(1):domHi(1), domLo(2):domHi(2),
     &     domLo(3):domHi(3))
      double precision AT(domLo(1):domHi(1), domLo(2):domHi(2),
     &     domLo(3):domHi(3))
      double precision AB(domLo(1):domHi(1), domLo(2):domHi(2),
     &     domLo(3):domHi(3))
      integer PCELL(domLo(1):domHi(1), domLo(2):domHi(2), 
     &     domLo(3):domHi(3))
      integer WALL, SYMTRY, FFIELD

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer iBegGhost, jBegGhost, kBegGhost
      integer iEndGhost, jEndGhost, kEndGhost
      integer IST, JST, KST, IEND, JEND, KEND
      integer i,j,k

      LOGICAL LWRAP,LSYMB, LPRDKK, LPRDAV, LPRDJK

c*********************************************************************
c     Start :
c*********************************************************************
      iBegGhost = domLo(1)
      jBegGhost = domLo(2)
      kBegGhost = domLo(3)
      iEndGhost = domHi(1)
      jEndGhost = domHi(2)
      kEndGhost = domHi(3)
      IST = idxLo(1)
      JST = idxLo(2)
      KST = idxLo(3)
      IEND = idxHi(1)
      JEND = idxHi(2)
      KEND = idxHi(3)

c*********************************************************************
c     Set the logicals :
c*********************************************************************
      LSYMB = .false.
      LWRAP = .false.
      LPRDJK = .false.
      LPRDKK = .false.
      LPRDAV = .false.

*----------------------------------------------------------------------
*     INLET/OUTLET BOUNDARY CONDITIONS 
*       EAST AND/OR WEST WALLS
*----------------------------------------------------------------------
C$DIR SCALAR
      DO 610 K = KST,KEND
C$DIR SCALAR
         DO 600 J = JST,JEND
            IF (PCELL(iEndGhost,J,K).NE.WALL .AND.
     &           PCELL(IEND,J,K).NE.WALL) THEN
               AE(IEND,J,K) = ZERO
            END IF
            IF (PCELL(iBegGhost,J,K).NE.WALL .AND.
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
            IF (PCELL(I,jEndGhost,K).NE.WALL.AND.
     &           PCELL(I,JEND,K).NE.WALL) THEN
               AN(I,JEND,K) = ZERO
            END IF
            IF (PCELL(I,jBegGhost,K).NE.WALL.AND.
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
            IF (PCELL(I,J,kEndGhost).NE.WALL .AND. 
     &           PCELL(I,J,KEND).NE.WALL) THEN
               AT(I,J,KEND) = ZERO
            END IF
            IF (PCELL(I,J,kBegGhost).NE.WALL .AND. 
     &           PCELL(I,J,KST).NE.WALL) THEN
               AB(I,J,KST) = ZERO
            END IF
 800     CONTINUE
 810  CONTINUE
*--------------------------------------------------------------------
*     SYMMETRY AND INTRUSION BOUNDARY CONDITIONS
*--------------------------------------------------------------------
      IF (LSYMB) THEN
         CALL SYMBCS(domLo, domHi, idxLo, idxHi,
     $        PP,
     &        AE,AW,AN,AS,AT,AB,
     &        PCELL, SYMTRY,
     $        LPRDJK, LPRDKK)
      ENDIF
      IF (LWRAP) THEN
         CALL PRDBC1(domLo, domHi, idxLo, idxHi,
     $        PP,
     &        PCELL, SYMTRY, FFIELD,
     &        LPRDAV, LPRDJK, LPRDKK)
         CALL PRDBC2(domLo, domHi, idxLo, idxHi,
     $        PP,
     &        AS,AT,AB,
     &        PCELL, SYMTRY, FFIELD,
     $        LPRDAV, LPRDJK, LPRDKK)
      ENDIF
      CALL WALLBC(domLo, domHi, idxLo, idxHi,
     $     AE,AW,AN,AS,AT,AB,
     &     PCELL,WALL)
      RETURN
      END 

c*********************************************************************
c
c $Log$
c Revision 1.1  2000/07/13 04:51:34  bbanerje
c Added pressureBC (bcp) .. now called bcpress.F (bcp.F removed)
c
c
c*********************************************************************
