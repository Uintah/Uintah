C
C The MIT License
C
C Copyright (c) 1997-2012 The University of Utah
C
C Permission is hereby granted, free of charge, to any person obtaining a copy
C of this software and associated documentation files (the "Software"), to
C deal in the Software without restriction, including without limitation the
C rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
C sell copies of the Software, and to permit persons to whom the Software is
C furnished to do so, subject to the following conditions:
C
C The above copyright notice and this permission notice shall be included in
C all copies or substantial portions of the Software.
C
C THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
C IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
C FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
C AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
C LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
C FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
C IN THE SOFTWARE.
C
C 
C 
c*********************************************************************
c
c
c*********************************************************************

#include <CCA/Components/Arches/fortran/mmbcvelocity_fort.h>
#include "param4.h"

#undef normalbc
#undef tangentialbc
#define normalbc
#define tangentialbc

c     The "usual" normal and tangential bcs involve severing
c     connections to the cell under consideration.
c     These usual bcs are the ones that are true if normalbc
c     and tangentialbc are true in this routine.
c     If not, we utilize implicit bcs.
c     Currently (08/29) we utilize implicit bcs.

C---------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     THIS SUBROUTINE WILL compute matrix coefficients for uvelocity
C---------------------------------------------------------------------

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

c     CURRENT CELL OR WEST CELL IS A WALL: SET VALUE OF 
C     VELOCITY TO ZERO

               if (pcell(i-ioff,j-joff,k-koff).eq.mmwallid.or. 
     $              pcell(i,j,k).eq.mmwallid) then
                  call fixval(zero,ae(i,j,k),aw(i,j,k),an(i,j,k),
     $                 as(i,j,k),at(i,j,k),ab(i,j,k),
     $                 su(i,j,k),sp(i,j,k),const)
               else

#ifdef normalbc

C     CELL ON FAR WEST IS A WALL: SET COEFFICIENT IN THAT 
C     DIRECTION TO ZERO
                  if (pcell(i-2*ioff,j-2*joff,k-2*koff).eq.
     $                 mmwallid) then
                     aw(i,j,k) = 0.0D0
                  endif
C     CELL ON EAST IS A WALL: SET COEFICIENT IN THAT DIRECTION
C     TO ZERO
                  if (pcell(i+ioff,j+joff,k+koff).eq.mmwallid) then
                     ae(i,j,k) = 0.0D0
                  endif
#endif

#ifdef tangentialbc

C     WALL ON SOUTH
                  if ((pcell(i-idt1,j-jdt1,k-kdt1).eq.mmwallid).or.
     $                (pcell(i-ioff-idt1,j-joff-jdt1,k-koff-kdt1)
     $                 .eq.mmwallid)) then
                     as(i,j,k) = 0.0D0
                  endif
C     WALL ON NORTH
                  if ((pcell(i+idt1,j+jdt1,k+kdt1).eq.mmwallid).or.
     $                (pcell(i-ioff+idt1,j-joff+jdt1,k-koff+kdt1)
     $                 .eq.mmwallid)) then
                     an(i,j,k) = 0.0D0
                  endif
C     WALL ON BOTTOM
                  if ((pcell(i-idt2,j-jdt2,k-kdt2).eq.mmwallid).or.
     $                (pcell(i-ioff-idt2,j-joff-jdt2,k-koff-kdt2)
     $                 .eq.mmwallid)) then
                     ab(i,j,k) = 0.0D0
                  endif
C     WALL ON TOP
                  if ((pcell(i+idt2,j+jdt2,k+kdt2).eq.mmwallid).or.
     $                (pcell(i-ioff+idt2,j-joff+jdt2,k-koff+kdt2)
     $                 .eq.mmwallid)) then
                     at(i,j,k) = 0.0D0
                  endif
#endif

               endif
 800        continue
 810     continue
 820  continue



      RETURN
      END

