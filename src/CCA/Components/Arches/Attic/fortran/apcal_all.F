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
#include <CCA/Components/Arches/fortran/apcal_all_fort.h>
*-----------------------------------------------------------------------

*
*     local variables
*
      integer i
      integer j
      integer k
c-----------------------------------------------------------------------
c     executable statements
c-----------------------------------------------------------------------
      do 350 k = valid_lo(3), valid_hi(3)
         do 250 j = valid_lo(2), valid_hi(2)
            do 150 i = valid_lo(1), valid_hi(1)

               ap(i,j,k) = 0.0D0

 150        continue
 250     continue
 350  continue

      do 300 k = valid_lo(3), valid_hi(3)
         do 200 j = valid_lo(2), valid_hi(2)
            do 100 i = valid_lo(1), valid_hi(1)

               ap(i,j,k) = ae(i,j,k)+aw(i,j,k)
     $              +an(i,j,k)+as(i,j,k)
     $              +at(i,j,k)+ab(i,j,k)
     $              -sp(i,j,k)

 100        continue
 200     continue
 300  continue

      return

      end
