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
c
c
#include <CCA/Components/Arches/fortran/bcscalar_fort.h>
#include "param4.h"
C-----------------------------------------------------------------------
C     THIS SUBROUTINE CALCULATES THE BOUNDARY CONDITONS FOR F
C-----------------------------------------------------------------------

*
*     declare local variables
*
      integer i, j, k
      

*-----------------------------------------------------------------------

*-----corrections for interior intrusions-------------------------------

      do 305 k=idxLo(3),idxHi(3)
         do 205 j=idxLo(2),idxHi(2)
            do 105 i=idxLo(1),idxHi(1)
               if (pcell(i,j,k).eq.wall) then
                call fixval(zero,ae(i,j,k),aw(i,j,k),an(i,j,k),
     &                 as(i,j,k),at(i,j,k),ab(i,j,k),su(i,j,k),
     &                 sp(i,j,k),one)
                call fixval(zero,cesav(i,j,k),cwsav(i,j,k),cnsav(i,j,k),
     &                 cssav(i,j,k),ctsav(i,j,k),cbsav(i,j,k),su(i,j,k),
     &                 sp(i,j,k),one)
                call fixval(zero,decoef(i,j,k),dwcoef(i,j,k),
     &                 dncoef(i,j,k), dscoef(i,j,k),dtcoef(i,j,k),
     &                 dbcoef(i,j,k),su(i,j,k),sp(i,j,k),one)
               end if
 105        continue
 205     continue
 305  continue

      call wallbc(idxLo, idxHi,
     & PASS3A(ae)
     & PASS3B(ae),
     & PASS3A(aw)
     & PASS3B(aw),
     & PASS3A(an)
     & PASS3B(an),
     & PASS3A(as)
     & PASS3B(as),
     & PASS3A(at)
     & PASS3B(at),
     & PASS3A(ab)
     & PASS3B(ab),
     & PASS3A(pcell)
     & PASS3B(pcell),
     & wall, xminus, xplus, yminus, yplus, zminus, zplus)
c zeroing these out will break nonuniform discretization
c      call wallbc(idxLo, idxHi,
c     & PASS3A(cesav)
c     & PASS3B(cesav),
c     & PASS3A(cwsav)
c     & PASS3B(cwsav),
c     & PASS3A(cnsav)
c     & PASS3B(cnsav),
c     & PASS3A(cssav)
c     & PASS3B(cssav),
c     & PASS3A(ctsav)
c     & PASS3B(ctsav),
c     & PASS3A(cbsav)
c     & PASS3B(cbsav),
c     & PASS3A(pcell)
c     & PASS3B(pcell),
c     & wall, xminus, xplus, yminus, yplus, zminus, zplus)
c set bc's for diffusion coeff's required for divergence constraint
      call wallbc(idxLo, idxHi, 
     & PASS3A(decoef)
     & PASS3B(decoef),
     & PASS3A(dwcoef)
     & PASS3B(dwcoef),
     & PASS3A(dncoef)
     & PASS3B(dncoef),
     & PASS3A(dscoef)
     & PASS3B(dscoef),
     & PASS3A(dtcoef)
     & PASS3B(dtcoef),
     & PASS3A(dbcoef)
     & PASS3B(dbcoef),
     & PASS3A(pcell)
     & PASS3B(pcell),
     & wall, xminus, xplus, yminus, yplus, zminus, zplus)

      return
      end
