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
#if 1
#include <CCA/Components/MPMArches/fortran/redistribute_dragforce_cc_fort.h>
#include "../../Arches/fortran/param4.h"
#else
      subroutine redistribute_dragforce_cc(
     $     dragforcex_fcx,
     $     dragforcex_cc,
     $     ioff, joff, koff,
     $     dim_lo_fcx, dim_hi_fcx, 
     $     dim_lo_cc, dim_hi_cc,
     $     valid_lo, valid_hi)

      implicit none
      include 'param4.h'

      integer dim_lo_fcx(3)
      integer dim_hi_fcx(3)
      integer dim_lo_cc(3)
      integer dim_hi_cc(3)
      integer valid_lo(3)
      integer valid_hi(3)

      double precision dragforcex_fcx(
     $     dim_lo_fcx(1):dim_hi_fcx(1),
     $     dim_lo_fcx(2):dim_hi_fcx(2),
     $     dim_lo_fcx(3):dim_hi_fcx(3))

      double precision dragforcex_cc(
     $     dim_lo_cc(1):dim_hi_cc(1),
     $     dim_lo_cc(2):dim_hi_cc(2),
     $     dim_lo_cc(3):dim_hi_cc(3))

      integer ioff
      integer joff
      integer koff
#endif

*     local variables

      integer i
      integer j
      integer k

*     executable statements

      do k = valid_lo(3), valid_hi(3)
         do j = valid_lo(2), valid_hi(2)
            do i = valid_lo(1), valid_hi(1)

               dragforcex_fcx(i,j,k) = dragforcex_fcx(i,j,k) +
     $              pt5*(dragforcex_cc(i,j,k) + 
     $              dragforcex_cc(i-ioff,j-joff,k-koff))

            end do
         end do
      end do

      return
      end

