C
C The MIT License
C
C Copyright (c) 1997-2012 The University of Utah
C
C Permission is hereby granted, free of charge, to any person obtaining a copy
C of this software and associated documentation files (the "Software"), to
C deal in the Software without restriction, including without limitation the
C rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#include <CCA/Components/Arches/fortran/bcuvel_fort.h>
#include "param4.h"
C---------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     THIS SUBROUTINE WILL compute matrix coefficients for uvelocity
C---------------------------------------------------------------------

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k


c*********************************************************************
c     Start :
c*********************************************************************
      IST = idxLo(1)
      JST = idxLo(2)
      KST = idxLo(3)
      IEND = idxHi(1)
      JEND = idxHi(2)
      KEND = idxHi(3)
c------------------------------------------------------------------ 
c     interior intrusions
c------------------------------------------------------------------ 
      do 820 k=kst, kend
         do 810 j=jst, jend
            do 800 i=ist, iend

               if (pcell(i,j,k).eq.wall.or. pcell(i-1,j,k).eq.wall) then
                call fixval(zero,ae(i,j,k),aw(i,j,k),an(i,j,k),
     &                          as(i,j,k),at(i,j,k),ab(i,j,k),
     &                          su(i,j,k),sp(i,j,k),one)
                call fixval(zero,cesav(i,j,k),cwsav(i,j,k),cnsav(i,j,k),
     &                 cssav(i,j,k),ctsav(i,j,k),cbsav(i,j,k),su(i,j,k),
     &                 sp(i,j,k),one)
               end if

 800        continue
 810     continue
 820  continue


c*********************************************************************
c     Go thru all the cells
c*********************************************************************
      if (xminus) then
         I = ist - 2
         DO 120 K = kst, kend
            DO 110 J = jst, jend
C--------------------------------------------------------------------
C             East FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
               if ((pcell(I,J,K).eq.wall).and.
     &             (PCELL(I+1, J, K).NE.WALL).and.
     &             (PCELL(I+2, J, K).NE.WALL)) then
                call fixval(zero,ae(i+1,j,k),aw(i+1,j,k),an(i+1,j,k),
     &                          as(i+1,j,k),at(i+1,j,k),ab(i+1,j,k),
     &                          su(i+1,j,k),sp(i+1,j,k),one)
                call fixval(zero,cesav(i+1,j,k),cwsav(i+1,j,k),
     &                      cnsav(i+1,j,k), cssav(i+1,j,k),
     &                      ctsav(i+1,j,k), cbsav(i+1,j,k),su(i+1,j,k),
     &                      sp(i+1,j,k),one)
                     AW(I+2,J,K) = 0.0D0
c zeroing these out will break nonuniform discretization
c                     cwsav(i+2,j,k) = 0.0D0
               end if
 110        CONTINUE
 120     CONTINUE
      endif

      RETURN
      END

