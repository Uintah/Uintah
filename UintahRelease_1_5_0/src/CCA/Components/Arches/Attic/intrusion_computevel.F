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
#include <CCA/Components/Arches/fortran/intrusion_computevel_fort.h>      

c---------------------------------------------------------------------
c     three-dimensional version
c     this subroutine will add pressure gradient term for momentum eqn
c     modification, Seshadri Kumar, July 30, 2001,
c     to take care of multimaterial wall intrusions
c---------------------------------------------------------------------
#include "param4.h"

c*********************************************************************
c     local variables :
c*********************************************************************
      integer i
      integer j
      integer k
      integer idf

      do 120 k =idxlo_u(3), idxhi_u(3)
         do 110 j = idxlo_u(2), idxhi_u(2)
            do 100 i = idxlo_u(1), idxhi_u(1)

               if ((pcell(i,j,k).eq.mmwallid) .or.
     $              (pcell(i-ioff, j-joff, k-koff)
     $              .eq.mmwallid)) then

                  uu(i,j,k) = zero
                  
               endif

 100        continue
 110     continue
 120  continue

      return
      end

