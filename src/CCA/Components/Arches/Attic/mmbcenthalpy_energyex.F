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

#include <CCA/Components/Arches/fortran/mmbcenthalpy_energyex_fort.h>
#include "param4.h"


c*********************************************************************
c     Local Variables :
c*********************************************************************
      double precision const
      double precision xp, yp, zp, qmult, qflux, prh
      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k


c*********************************************************************
c     Start :
c*********************************************************************
      IST = idxLoU(1)
      JST = idxLoU(2)
      KST = idxLoU(3)
      IEND = idxHiU(1)
      JEND = idxHiU(2)
      KEND = idxHiU(3)

c------------------------------------------------------------------ 
c     interior intrusions
c------------------------------------------------------------------ 
      const = ONE
c prandtl number
      prh = 0.9d0
      do 820 k=kst, kend
         do 810 j=jst, jend
            do 800 i=ist, iend
               if (pcell(i,j,k).ne.mmwallid) then
                  if (pcell(i-1,j,k).eq.mmwallid) then
                     xp = x(i)-xu(i)
c     conduction
                     qmult = viscos/(xp*prh)
                     qflux = qmult*cp(i,j,k)*(tg(i,j,k)-tg(i-1,j,k))
                     su(i,j,k)=su(i,j,k)-qflux*sns(j)*stb(k)
                  endif
                  if (pcell(i+1,j,k).eq.mmwallid) then
                     xp = xu(i+1)-xu(i)
                     qmult = viscos/(xp*prh)
                     qflux = qmult*cp(i,j,k)*(tg(i,j,k)-tg(i+1,j,k))
                     su(i,j,k)=su(i,j,k)-qflux*sns(j)*stb(k)
                  endif
                  if (pcell(i,j-1,k).eq.mmwallid) then
                     yp = y(j)-yv(j)
                     qmult = viscos/(yp*prh)
                     qflux = qmult*cp(i,j,k)*(tg(i,j,k)-tg(i,j-1,k))
                     su(i,j,k)=su(i,j,k)-qflux*sew(i)*stb(k)
                  endif
                  if (pcell(i,j+1,k).eq.mmwallid) then
                     yp = yv(j+1)-y(j)
                     qmult = viscos/(yp*prh)
                     qflux = qmult*cp(i,j,k)*(tg(i,j,k)-tg(i,j+1,k))
                     su(i,j,k)=su(i,j,k)-qflux*sew(i)*stb(k)
                  endif
                  if (pcell(i,j,k-1).eq.mmwallid) then
                     zp = z(k)-zw(k)
                     qmult = viscos/(zp*prh)
                     qflux = qmult*cp(i,j,k)*(tg(i,j,k)-tg(i,j,k-1))
                     su(i,j,k)=su(i,j,k)-qflux*sew(i)*sns(j)
                  endif
                  if (pcell(i,j,k+1).eq.mmwallid) then
                     zp = zw(k+1)-z(k)
                     qmult = viscos/(zp*prh)
                     qflux = qmult*cp(i,j,k)*(tg(i,j,k)-tg(i,j,k+1))
                     su(i,j,k)=su(i,j,k)-qflux*sew(i)*sns(j)
                  endif
               endif
 800        continue
 810     continue
 820  continue



      RETURN
      END

