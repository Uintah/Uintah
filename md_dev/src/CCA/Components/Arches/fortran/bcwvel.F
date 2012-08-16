C 
C 
C The MIT License
C 
C Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
C Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
C University of Utah.
C 
C License for the specific language governing rights and limitations under
C Permission is hereby granted, free of charge, to any person obtaining a 
C copy of this software and associated documentation files (the "Software"),
C to deal in the Software without restriction, including without limitation 
C the rights to use, copy, modify, merge, publish, distribute, sublicense, 
C and/or sell copies of the Software, and to permit persons to whom the 
C Software is furnished to do so, subject to the following conditions:
C 
C The above copyright notice and this permission notice shall be included 
C in all copies or substantial portions of the Software.
C 
C THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
C OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
C FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
C THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
C LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
C FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
C DEALINGS IN THE SOFTWARE.
C 
C
C 
C 
c*********************************************************************
c
c
c*********************************************************************

#include <CCA/Components/Arches/fortran/bcwvel_fort.h>
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
      double precision XP, TMULT

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

               if (pcell(i,j,k).eq.wall.or.pcell(i,j,k-1).eq.wall) then
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
         I = ist - 1
         DO 120 K = kst, kend
            DO 110 J = jst, jend
C--------------------------------------------------------------------
C             BOTTOMEAST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
               IF (((PCELL(I,J,K).EQ.WALL).or.
     &              (pcell(i,j,k-1).eq.wall)).and.
     &             (pcell(i+1,j,k).ne.wall).and.
     &             (pcell(i+1,j,k-1).ne.wall)) then
                  XP = X(I+1) - XU(I+1)
                  TMULT = 0.5D0*(VISCOS(I+1,j,k-1)+VISCOS(I+1,j,k))/XP
                  SP(I+1,J,K) = SP(I+1,J,K)-TMULT*SNS(J)*STBW(K)
                  AW(I+1,J,K) = 0.0D0
c zeroing these out will break nonuniform discretization
c                  cwsav(i+1,j,k) = 0.0D0
                END IF
 110        CONTINUE
 120     CONTINUE
      endif

      RETURN
      END
