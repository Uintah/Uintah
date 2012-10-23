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
      subroutine interp_facetocenter(
     $     phi_cc, phi_fc,
     $     efac, wfac,
     $     ioff, joff, koff, 
     $     index,
     $     dim_lo, dim_hi,
     $     dim_lo_cc, dim_hi_cc, 
     $     dim_lo_fc, dim_hi_fc, 
     $     valid_lo, valid_hi)

c     
c     Author: Seshadri Kumar, March 6, 2001
c     This routine interpolates a quantity stored on the face
c     center (a scalar quantity or a velocity component) to the 
c     cell center
c

      implicit none

      integer dim_lo(3)
      integer dim_hi(3)
      integer dim_lo_cc(3)
      integer dim_hi_cc(3)
      integer dim_lo_fc(3)
      integer dim_hi_fc(3)

      integer valid_lo(3)
      integer valid_hi(3)
      integer index

      double precision phi_cc(
     $     dim_lo_cc(1):dim_hi_cc(1),
     $     dim_lo_cc(2):dim_hi_cc(2),
     $     dim_lo_cc(3):dim_hi_cc(3))

      double precision phi_fc(
     $     dim_lo_fc(1):dim_hi_fc(1),
     $     dim_lo_fc(2):dim_hi_fc(2),
     $     dim_lo_fc(3):dim_hi_fc(3))

      double precision efac(dim_lo(index):dim_hi(index))
      double precision wfac(dim_lo(index):dim_hi(index))

      integer ioff
      integer joff
      integer koff

*     local variables

      integer i
      integer j
      integer k
      integer idf

*     executable statements

      do 300 k = valid_lo(3), valid_hi(3)
         do 200 j = valid_lo(2), valid_hi(2)
            do 100 i = valid_lo(1), valid_hi(1)

               idf = ioff*i + joff*j + koff*k

               phi_cc(i,j,k) = efac(idf)*phi_fc(i+ioff,j+joff,k+koff) +
     $              wfac(idf)*phi_fc(i,j,k)

 100        continue
 200     continue
 300  continue

      return
      end
