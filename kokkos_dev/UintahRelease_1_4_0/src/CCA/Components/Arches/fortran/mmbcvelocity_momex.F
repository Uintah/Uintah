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

#include <CCA/Components/Arches/fortran/mmbcvelocity_momex_fort.h>
#include "param4.h"
C---------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     THIS SUBROUTINE WILL compute matrix coefficients for uvelocity
C---------------------------------------------------------------------

c*********************************************************************
c     Local Variables :
c*********************************************************************
      double precision const
      double precision yp, zp, tmult
      integer IST, JST, KST, IEND, JEND, KEND
      integer idt1, jdt1, kdt1, idt2, jdt2, kdt2, fdir, t1dir, t2dir
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
               if ((pcell(i,j,k).ne.mmwallid).and.
     $             (pcell(i-ioff,j-joff,k-koff).ne.mmwallid)) then
                  fdir = i*ioff+j*joff+k*koff
                  t1dir = i*idt1+j*jdt1+k*kdt1
                  t2dir = i*idt2+j*jdt2+k*kdt2
                  if ((pcell(i-idt1,j-jdt1,k-kdt1).eq.mmwallid).or.
     $                 (pcell(i-ioff-idt1,j-joff-jdt1,k-koff-kdt1)
     $                 .eq.mmwallid)) then
                     as(i,j,k) = 0.0D0
                     yp = y(t1dir)-yv(t1dir)
                     tmult = viscos/yp
                     sp(i,j,k)=sp(i,j,k)-tmult*sewu(fdir)*stb(t2dir)
                  endif
                  if ((pcell(i+idt1,j+jdt1,k+jdt1).eq.mmwallid)
     $              .or.(pcell(i-ioff+idt1,j-joff+jdt1,k-koff+kdt1)
     $                 .eq.mmwallid)) then
                     an(i,j,k) = 0.0D0
                     yp = yv(t1dir+1)-y(t1dir)
                     tmult = viscos/yp
                     sp(i,j,k)=sp(i,j,k)-tmult*sewu(fdir+1)*stb(t2dir)
                  endif
                  if ((pcell(i-idt2,j-jdt2,k-kdt2).eq.mmwallid).or.
     $                 (pcell(i-ioff-idt2,j-joff-jdt2,k-koff-kdt2)
     $                 .eq.mmwallid)) then
                     ab(i,j,k) = 0.0D0
                     zp = z(t2dir)-zw(t2dir)
                     tmult = viscos/zp
                     sp(i,j,k)=sp(i,j,k)-tmult*sewu(fdir)*sns(t1dir)
                  endif
                  if ((pcell(i+idt2,j+jdt2,k+kdt2).eq.mmwallid).or.
     $                 (pcell(i-ioff+idt2,j-joff+jdt2,k-koff+kdt2)
     $                 .eq.mmwallid)) then
                     at(i,j,k) = 0.0D0
                     zp = zw(t2dir+1)-z(t2dir)
                     tmult = viscos/zp
                     sp(i,j,k)=sp(i,j,k)-tmult*sewu(fdir+1)*sns(t1dir)
                  endif
               endif
 800        continue
 810     continue
 820  continue



      RETURN
      END

