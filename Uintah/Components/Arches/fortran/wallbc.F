c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE wallbc(domLo, domHi, domLong, domHing, idxLo, idxHi,
     &     AE,AW,AN,AS,AT,AB,
     &     PCELL, WALL,
     $     xminus, xplus, yminus, yplus, zminus, zplus)
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
      integer domLo(3), domHi(3), domLong(3), domHing(3)
      integer idxLo(3), idxHi(3)
      double precision AE(domLong(1):domHing(1), 
     &     domLong(2):domHing(2), domLong(3):domHing(3))
      double precision AW(domLong(1):domHing(1), 
     &     domLong(2):domHing(2), domLong(3):domHing(3))
      double precision AN(domLong(1):domHing(1), 
     &     domLong(2):domHing(2), domLong(3):domHing(3))
      double precision AS(domLong(1):domHing(1), 
     &     domLong(2):domHing(2), domLong(3):domHing(3))
      double precision AT(domLong(1):domHing(1), 
     &     domLong(2):domHing(2), domLong(3):domHing(3))
      double precision AB(domLong(1):domHing(1), 
     &     domLong(2):domHing(2), domLong(3):domHing(3))
      integer PCELL(domLo(1):domHi(1), domLo(2):domHi(2), 
     &     domLo(3):domHi(3))
      integer WALL
      integer xminus, xplus, yminus, yplus, zminus, zplus

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer iBegGhost,jBegGhost,kBegGhost
      integer iEndGhost,jEndGhost,kEndGhost,i,j,k
      integer ist,jst,kst,iend,jend,kend

c*********************************************************************
c     Start :
c*********************************************************************
      ist = idxLo(1)
      jst = idxLo(2)
      kst = idxLo(3)
      iend = idxHi(1)
      jend = idxHi(2)
      kend = idxHi(3)
      if (xminus) then
         iBegGhost = ist -1
      else
         iBegGhost = ist
      endif
      if (yminus) then
         jBegGhost = jst -1
      else
         jBegGhost = jst
      endif
      if (zminus) then
         kBegGhost = kst - 1
      else
         kBegGhost = kst
      endif
      if (xplus) then
         iEndGhost = iend +1
      else
         iEndGhost = iend
      endif
      if (yplus) then
         jEndGhost = jend + 1
      else
         jEndGhost = jend
      endif
      if (zplus) then
         kEndGhost = kend+1
      else
         kEndGhost = kend
      endif
      
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
c Revision 1.4  2000/10/06 23:07:53  rawat
c fixed some more bc routines for mulit-patch
c
c Revision 1.3  2000/09/26 19:59:19  sparker
c Work on MPI petsc
c
c Revision 1.2  2000/07/13 04:51:35  bbanerje
c Added pressureBC (bcp) .. now called bcpress.F (bcp.F removed)
c
c Revision 1.1  2000/04/12 20:31:51  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
