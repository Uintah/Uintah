c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE wallbc(indexLow, indexHigh, PCELL,
     &     AE,AW,AN,AS,AT,AB)
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
#include "pcell.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer indexLow(3)
      integer indexHigh(3)
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


c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer istm1,jstm1,kstm1,iendp1,jendp1,kendp1,i,j,k
      integer ist,jst,kst,iend,jend,kend

c*********************************************************************
c     Start :
c*********************************************************************
      istm1 = indexLow(1)
      jstm1 = indexLow(2)
      kstm1 = indexLow(3)
      iendp1 = indexHigh(1)
      jendp1 = indexHigh(2)
      kendp1 = indexHigh(3)
      ist = istm1 + 1
      jst = jstm1 + 1
      kst = kstm1 + 1
      iend = iendp1 - 1
      jend = jendp1 - 1
      kend = kendp1 - 1
      
      DO 500 K = kstm1,kendp1
         DO 400 J = jstm1,jendp1
            DO 300 I = istm1,iendp1
               IF (PCELL(I,J,K).EQ.WALL) THEN
                  IF (I.NE.istm1) THEN
                     IF (PCELL(I-1,J,K).NE.WALL) 
     $                    AE(I-1,J,K) = ZERO
                  END IF
                  IF (I.NE.iendp1) THEN
                     IF (PCELL(I+1,J,K).NE.WALL) 
     $                    AW(I+1,J,K) = ZERO
                  END IF
                  IF (J.NE.jstm1) THEN
                     IF (PCELL(I,J-1,K).NE.WALL) 
     $                    AN(I,J-1,K) = ZERO
                  END IF
                  IF (J.NE.jendp1) THEN
                     IF (PCELL(I,J+1,K).NE.WALL) 
     $                    AS(I,J+1,K) = ZERO
                  END IF
                  IF (K.NE.kstm1) THEN
                     IF (PCELL(I,J,K-1).NE.WALL) 
     $                    AT(I,J,K-1) = ZERO
                  END IF
                  IF (K.NE.kendp1) THEN
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
c Revision 1.1  2000/04/12 20:31:51  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
