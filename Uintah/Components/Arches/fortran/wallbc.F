c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE wallbc(domLo, domHi, idxLo, idxHi,
     &     AE,AW,AN,AS,AT,AB,
     &     PCELL, WALL)
*--------------------------------------------------------------------
*     This subroutine handles intrusion boundary conditions
*          for ETA, F, GETA, GF, HCN, NO, P, PP, and YFU.
*          ALSO NOTE: some of the boundary condition for walls is
*                     handled in the calc routines themselves.  
*                     This happens in TWO places:
*                            1) CP at the boundary
*                            2) GREAT in the intrusion
*--------------------------------------------------------------------
      implicit none

#include "param4.h"
c #include "pcell.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer domLo(3), domHi(3), idxLo(3), idxHi(3)
      double precision AE(domLo(1):domHi(1), 
     &     domLo(2):domHi(2), domLo(3):domHi(3))
      double precision AW(domLo(1):domHi(1), 
     &     domLo(2):domHi(2), domLo(3):domHi(3))
      double precision AN(domLo(1):domHi(1), 
     &     domLo(2):domHi(2), domLo(3):domHi(3))
      double precision AS(domLo(1):domHi(1), 
     &     domLo(2):domHi(2), domLo(3):domHi(3))
      double precision AT(domLo(1):domHi(1), 
     &     domLo(2):domHi(2), domLo(3):domHi(3))
      double precision AB(domLo(1):domHi(1), 
     &     domLo(2):domHi(2), domLo(3):domHi(3))
      integer PCELL(domLo(1):domHi(1), domLo(2):domHi(2), 
     &     domLo(3):domHi(3))
      integer WALL

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer iBegGhost,jBegGhost,kBegGhost
      integer iEndGhost,jEndGhost,kEndGhost,i,j,k
      integer ist,jst,kst,iend,jend,kend

c*********************************************************************
c     Start :
c*********************************************************************
      iBegGhost = domLo(1)
      jBegGhost = domLo(2)
      kBegGhost = domLo(3)
      iEndGhost = domHi(1)
      jEndGhost = domHi(2)
      kEndGhost = domHi(3)
      ist = idxLo(1)
      jst = idxLo(2)
      kst = idxLo(3)
      iend = idxHi(1)
      jend = idxHi(2)
      kend = idxHi(3)
      
      DO 500 K = kBegGhost,kEndGhost
         DO 400 J = jBegGhost,jEndGhost
            DO 300 I = iBegGhost,iEndGhost
               IF (PCELL(I,J,K).EQ.WALL) THEN
                  IF (I.NE.iBegGhost) THEN
                     IF (PCELL(I-1,J,K).NE.WALL) 
     $                    AE(I-1,J,K) = ZERO
                  END IF
                  IF (I.NE.iEndGhost) THEN
                     IF (PCELL(I+1,J,K).NE.WALL) 
     $                    AW(I+1,J,K) = ZERO
                  END IF
                  IF (J.NE.jBegGhost) THEN
                     IF (PCELL(I,J-1,K).NE.WALL) 
     $                    AN(I,J-1,K) = ZERO
                  END IF
                  IF (J.NE.jEndGhost) THEN
                     IF (PCELL(I,J+1,K).NE.WALL) 
     $                    AS(I,J+1,K) = ZERO
                  END IF
                  IF (K.NE.kBegGhost) THEN
                     IF (PCELL(I,J,K-1).NE.WALL) 
     $                    AT(I,J,K-1) = ZERO
                  END IF
                  IF (K.NE.kEndGhost) THEN
                     IF (PCELL(I,J,K+1).NE.WALL) 
     $                    AB(I,J,K+1) = ZERO
                  END IF
               END IF
 300        CONTINUE
 400     CONTINUE
 500  CONTINUE
      RETURN
      END 

c*********************************************************************
c
c $Log$
c Revision 1.2  2000/07/13 04:51:35  bbanerje
c Added pressureBC (bcp) .. now called bcpress.F (bcp.F removed)
c
c Revision 1.1  2000/04/12 20:31:51  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
