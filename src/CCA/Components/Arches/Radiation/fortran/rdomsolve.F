C
C The MIT License
C
C Copyright (c) 1997-2025 The University of Utah
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
#include <CCA/Components/Arches/Radiation/fortran/rdomsolve_fort.h>

      integer i,j,k,ii,jj,kk,iii,jjj,kkk

      double precision PI,PI1,SIG
      double precision val, factor
      double precision areaew, arean, areatb
      double precision volume
      double precision diffuse_reflection

      DATA SIG/5.67D-08/

      pi = dacos(-1.0D0)
      PI1 = 1.0d0 / PI

c     wall is a perfectly diffuse reflector
      diffuse_reflection = 1.0d0  / pi

      do 30 k=idxlo(3),idxhi(3)
      do 20 j=idxlo(2),idxhi(2)
      do 10 i=idxlo(1),idxhi(1)


      if (omu(l).gt.0) then
         
         ii = i
         iii=ii-1
         plusX = .true.

      else

         ii = idxlo(1)+idxhi(1)-i
         iii=ii+1
         plusX = .false.

      end if

      if (oeta(l).gt.0) then

         jj = j 
         jjj=jj-1
         plusY = .true.

      else

         jj = idxlo(2)+idxhi(2)-j
         jjj=jj+1
         plusY = .false.

      end if

      if (oxi(l).gt.0) then
         
         kk = k
         kkk=kk-1
         plusZ = .true.

      else

         kk = idxlo(3)+idxhi(3)-k
         kkk=kk+1
         plusZ = .false.
         
      end if

      
      if (pcell(ii,jj,kk).ne.ffield) then
c  if non-flow cell, then set LHS to 1*I and RHS to sigma*T^4/pi 

         val = abs(fraction(bands))*SIG*PI1*tg(ii,jj,kk)**4
         factor = 1.d0
         call fixradval(val,aw(ii,jj,kk),as(ii,jj,kk),ab(ii,jj,kk),
     $                  ap(ii,jj,kk),su(ii,jj,kk),factor)
      else

c  if flow cell, populate corresponding elements in arrays 
      volume = sew(ii) * sns(jj) * stb(kk)
      su(ii,jj,kk)=(esrct(ii,jj,kk) + scatSrc(ii,jj,kk)) *volume

      areaew = sns(jj) * stb(kk)
      arean  = sew(ii) * stb(kk)                 
      areatb = sew(ii) * sns(jj)

         if (plusX) then          
            if (pcell(ii-1,jj,kk).eq.ffield) then
               aw(ii,jj,kk) = abs(omu(l))*areaew
            else
               aw(ii,jj,kk) = 0.0d0
               su(ii,jj,kk) = su(ii,jj,kk) + (abs(omu(l))*areaew) *
     &         (PI1 * (SIG * abs(fraction(bands))*
c     Empirical model for emissivity at boundaries
     &           abskt(ii-1,jj,kk) *
     &              tg(ii-1,jj,kk)**4) + (1.0d0-abskt(ii-1,jj,kk))
     &    *IncidentFluxE(ii,jj,kk)*diffuse_reflection)
            end if
         else
            if (pcell(ii+1,jj,kk).eq.ffield) then
               aw(ii,jj,kk) = abs(omu(l))*areaew
            else
               aw(ii,jj,kk) = 0.0d0
               su(ii,jj,kk) = su(ii,jj,kk) + (abs(omu(l)) * areaew) * 
     &         (PI1 * (SIG * abs(fraction(bands))*
c     Empirical model for emissivity at boundaries
     &           abskt(ii+1,jj,kk) *
     &              tg(ii+1,jj,kk)**4) + (1.0d0-abskt(ii+1,jj,kk))
     &    *IncidentFluxW(ii,jj,kk)*diffuse_reflection)
            end if
         endif
         
         if (plusY) then
            if (pcell(ii,jj-1,kk).eq.ffield) then
               as(ii,jj,kk) = abs(oeta(l))*arean
            else
               as(ii,jj,kk) = 0.0d0
               su(ii,jj,kk) = su(ii,jj,kk) + 
     &              (abs(oeta(l))*arean) *
     &         (PI1 * (SIG * abs(fraction(bands)) *
c     Empirical model for emissivity at boundaries
     &           abskt(ii,jj-1,kk) *
     &              tg(ii,jj-1,kk)**4) + (1.0d0-abskt(ii,jj-1,kk))
     &    *IncidentFluxS(ii,jj,kk)*diffuse_reflection)
            end if
         else
            if (pcell(ii,jj+1,kk).eq.ffield) then
               as(ii,jj,kk) = abs(oeta(l))*arean
            else
               as(ii,jj,kk) = 0.0d0
               su(ii,jj,kk) = su(ii,jj,kk) + 
     &              (abs(oeta(l))*arean) *
     &        (PI1 * (SIG * abs(fraction(bands)) *
c     Empirical model for emissivity at boundaries
     &           abskt(ii,jj+1,kk) *
     &              tg(ii,jj+1,kk)**4) + (1.0d0-abskt(ii,jj+1,kk))
     &    *IncidentFluxN(ii,jj,kk)*diffuse_reflection)
            end if
         endif

         if (plusZ) then
            if (pcell(ii,jj,kk-1).eq.ffield) then
               ab(ii,jj,kk) = abs(oxi(l))*areatb
            else
               ab(ii,jj,kk) = 0.0d0
               su(ii,jj,kk) = su(ii,jj,kk) + 
     &              (abs(oxi(l))*areatb) *
     &         (PI1 * (SIG * abs(fraction(bands)) *
c     Empirical model for emissivity at boundaries
     &           abskt(ii,jj,kk-1) *
     &              tg(ii,jj,kk-1)**4) + (1.0d0-abskt(ii,jj,kk-1))
     &    *IncidentFluxB(ii,jj,kk)*diffuse_reflection)
            end if
         else
            if (pcell(ii,jj,kk+1).eq.ffield) then
               ab(ii,jj,kk) = abs(oxi(l))*areatb
            else
               ab(ii,jj,kk) = 0.0d0
               su(ii,jj,kk) = su(ii,jj,kk) + 
     &              (abs(oxi(l))*areatb) * 
     &         (PI1 * (SIG * abs(fraction(bands)) *
c     Empirical model for emissivity at boundaries
     &           abskt(ii,jj,kk+1) *
     &              tg(ii,jj,kk+1)**4) + (1.0d0-abskt(ii,jj,kk+1))
     &    *IncidentFluxT(ii,jj,kk)*diffuse_reflection)
            end if
         endif
         ap(ii,jj,kk) =  abs(omu(l))*areaew + abs(oeta(l))*arean + 
     &        abs(oxi(l))*areatb + abskt(ii,jj,kk) * volume
         
      endif
10    continue
20    continue
30    continue
      RETURN
      end










