c*********************************************************************
c
c
c*********************************************************************

      SUBROUTINE mmbcvelocity(domLoUng, domHiUng,
     $     idxLoU, idxHiU,
     &     AE, AW, AN, AS, AT, AB,
     &     SU, SP,
     &     domLo, domHi, PCELL, MMWALLID,
     &     ioff, joff, koff,
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
      integer domLoUng(3), domHiUng(3)
      integer idxLoU(3), idxHiU(3)
      double precision AE(domLoUng(1):domHiUng(1),
     $     domLoUng(2):domHiUng(2), 
     &     domLoUng(3):domHiUng(3))
      double precision AW(domLoUng(1):domHiUng(1),
     $     domLoUng(2):domHiUng(2), 
     &     domLoUng(3):domHiUng(3))
      double precision AN(domLoUng(1):domHiUng(1),
     $     domLoUng(2):domHiUng(2), 
     &     domLoUng(3):domHiUng(3))
      double precision AS(domLoUng(1):domHiUng(1),
     $     domLoUng(2):domHiUng(2), 
     &     domLoUng(3):domHiUng(3))
      double precision AT(domLoUng(1):domHiUng(1),
     $     domLoUng(2):domHiUng(2), 
     &     domLoUng(3):domHiUng(3))
      double precision AB(domLoUng(1):domHiUng(1),
     $     domLoUng(2):domHiUng(2), 
     &     domLoUng(3):domHiUng(3))
      double precision SU(domLoUng(1):domHiUng(1), 
     $     domLoUng(2):domHiUng(2), 
     &     domLoUng(3):domHiUng(3))
      double precision SP(domLoUng(1):domHiUng(1), 
     $     domLoUng(2):domHiUng(2), 
     &     domLoUng(3):domHiUng(3))
      integer domLo(3), domHi(3)
      integer PCELL(domLo(1):domHi(1), 
     $     domLo(2):domHi(2), 
     &     domLo(3):domHi(3))
      integer MMWALLID
      integer ioff, joff, koff
      integer ierr

c*********************************************************************
c     Local Variables :
c*********************************************************************
      double precision const
      integer IST, JST, KST, IEND, JEND, KEND
      integer idt1, jdt1, kdt1, idt2, jdt2, kdt2
      integer i, j, k


c*******************u*************************************************
c     Start :
c*********************************************************************
      IST = idxLoU(1)
      JST = idxLoU(2)
      KST = idxLoU(3)
      IEND = idxHiU(1)
      JEND = idxHiU(2)
      KEND = idxHiU(3)
      idt1 = koff
      jdt1 = ioff
      kdt1 = joff
      idt2 = joff
      jdt2 = koff
      kdt2 = ioff
c------------------------------------------------------------------ 
c     interior intrusions
c------------------------------------------------------------------ 
      const = ONE
      do 820 k=kst, kend
         do 810 j=jst, jend
            do 800 i=ist, iend

               if (pcell(i-ioff,j-joff,k-koff).eq.mmwallid.or. 
     $              pcell(i,j,k).eq.mmwallid) then
                  call fixval(zero,ae(i,j,k),aw(i,j,k),an(i,j,k),
     $                 as(i,j,k),at(i,j,k),ab(i,j,k),
     $                 su(i,j,k),sp(i,j,k),const)
               else
                  if (pcell(i-2*ioff,j-2*joff,k-2*koff).eq.mmwallid) 
     $                 aw(i,j,k) = 0.0
                  if (pcell(i+ioff,j+joff,k+koff).eq.mmwallid)
     $                 ae(i,j,k) = 0.0
                  if ((pcell(i-idt1,j-jdt1,k-kdt1).eq.mmwallid).or.
     $                (pcell(i-ioff-idt1,j-joff-jdt1,k-koff-kdt1)
     $                 .eq.mmwallid))
     $                 as(i,j,k) = 0.0
                  if ((pcell(i+idt1,j+jdt1,k+jdt1).eq.mmwallid).or.
     $                (pcell(i-ioff+idt1,j-joff+jdt1,k-koff+kdt1)
     $                 .eq.mmwallid))
     $                 an(i,j,k) = 0.0
                  if ((pcell(i-idt2,j-jdt2,k-kdt2).eq.mmwallid).or.
     $                (pcell(i-ioff-idt2,j-joff-jdt2,k-koff-kdt2)
     $                 .eq.mmwallid))
     $                 ab(i,j,k) = 0.0
                  if ((pcell(i+idt2,j+jdt2,k+kdt2).eq.mmwallid).or.
     $                (pcell(i-ioff+idt2,j-joff+jdt2,k-koff+kdt2)
     $                 .eq.mmwallid))
     $                 at(i,j,k) = 0.0
               endif
 800        continue
 810     continue
 820  continue



      RETURN
      END

