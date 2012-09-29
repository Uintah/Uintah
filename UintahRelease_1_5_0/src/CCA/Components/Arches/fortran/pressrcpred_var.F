C
C Copyright (c) 1997-2012 The University of Utah
C
C Permission is hereby granted, free of charge, to any person obtaining a copy
C of this software and associated documentation files (the \"Software\"), to
C deal in the Software without restriction, including without limitation the
C rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
C sell copies of the Software, and to permit persons to whom the Software is
C furnished to do so, subject to the following conditions:
C
C The above copyright notice and this permission notice shall be included in
C all copies or substantial portions of the Software.
C
C THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
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
#include <CCA/Components/Arches/fortran/pressrcpred_var_fort.h>
#include "param4.h"

      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k
      double precision dene, denw, denn, dens, dent, denb
      double precision ce, cw, cn, cs, ct, cb
      double precision arean, areas, areaew, areatb, vol
      double precision apo, smp

      IST = idxLo(1)
      JST = idxLo(2)
      KST = idxLo(3)
      IEND = idxHi(1)
      JEND = idxHi(2)
      KEND = idxHi(3)
c-----------------------------------------------------------------------

      do 120 k = KST, KEND
         do 110 j = JST, JEND
            do 100 i = IST, IEND
               arean = sew(i)*stb(k)
               areas = sew(i)*stb(k)
               areaew = sns(j)*stb(k)
               areatb = sns(j)*sew(i)
               vol = sns(j)*sew(i)*stb(k)
               ce = areaew
               cw = areaew
               cn = arean
               cs = areas
               ct = areatb
               cb = areatb
               smp = (cn*vhat(i,j+1,k)-cs*vhat(i,j,k)+
     $                ce*uhat(i+1,j,k)-cw*uhat(i,j,k)+
     $                ct*what(i,j,k+1)-cb*what(i,j,k) -
     $                divU(i,j,k))/delta_t
               su(i,j,k)= -smp
 100           continue
 110        continue
 120     continue

         return
         end

