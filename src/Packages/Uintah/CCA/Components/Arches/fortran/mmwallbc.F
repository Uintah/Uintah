c*********************************************************************
c
c
c*********************************************************************

      SUBROUTINE mmwallbc(domLo, domHi, domLong, domHing, 
     $     idxLo, idxHi,
     &     AE,AW,AN,AS,AT,AB, su, sp,
     &     PCELL, MMWALLID)
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
      double precision SU(domLong(1):domHing(1), 
     $     domLong(2):domHing(2), 
     &     domLong(3):domHing(3))
      double precision SP(domLong(1):domHing(1), 
     $     domLong(2):domHing(2), 
     &     domLong(3):domHing(3))

      integer PCELL(domLo(1):domHi(1), domLo(2):domHi(2), 
     &     domLo(3):domHi(3))
      integer MMWALLID

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer i,j,k
      integer ist,jst,kst,iend,jend,kend
      double precision const

c*********************************************************************
c     Start :
c*********************************************************************
      ist = idxLo(1)
      jst = idxLo(2)
      kst = idxLo(3)
      iend = idxHi(1)
      jend = idxHi(2)
      kend = idxHi(3)
      const = one
      DO 500 K = kst,kend
         DO 400 J = jst,jend
            DO 300 I = ist, iend
               IF (PCELL(I,J,K).EQ.MMWALLID) THEN
c intrusion boundary, fix coeffs
                  call fixval(zero,ae(i,j,k),aw(i,j,k),an(i,j,k),
     $                        as(i,j,k),at(i,j,k),ab(i,j,k),
     $                        su(i,j,k),sp(i,j,k),const)
               else
                  if (pcell(i-1,j,k).eq.mmwallid)
     $                 aw(i,j,k) = zero
                  if (pcell(i+1,j,k).eq.mmwallid)
     $                 ae(i,j,k) = zero
                  if (pcell(i,j-1,k).eq.mmwallid)
     $                 as(i,j,k) = zero
                  if (pcell(i,j+1,k).eq.mmwallid)
     $                 an(i,j,k) = zero
                  if (pcell(i,j,k-1).eq.mmwallid)
     $                 ab(i,j,k) = zero
                  if (pcell(i,j,k+1).eq.mmwallid)
     $                 at(i,j,k) = zero
               END IF
 300        continue
 400     CONTINUE
 500  CONTINUE
      RETURN
      END 

