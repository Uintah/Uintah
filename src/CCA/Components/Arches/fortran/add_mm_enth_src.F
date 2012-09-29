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
C 
#if 1
#include <CCA/Components/Arches/fortran/add_mm_enth_src_fort.h>
#include "../../Arches/fortran/param4.h"
#else
      SUBROUTINE add_mm_enth_src(
     $     dim_lo_su, dim_hi_su, su,
     $     dim_lo_sp, dim_hi_sp, sp,
     $     dim_lo_mm_su, dim_hi_mm_su, mm_su,
     $     dim_lo_mm_sp, dim_hi_mm_sp, mm_sp,
     $     valid_lo, valid_hi,
     &     ierr)
c---------------------------------------------------------------------
c     This routine adds the source term contribution for enthalpy
c     arising from multimaterial intrusions
c
c     Author: Seshadri Kumar, August 3, 2001
c     
c---------------------------------------------------------------------

      implicit none

c*********************************************************************
c     Arguments :
c*********************************************************************

      integer dim_lo_su(3)
      integer dim_hi_su(3)

      double precision su(
     $     dim_lo_su(1):dim_hi_su(1), 
     $     dim_lo_su(2):dim_hi_su(2), 
     &     dim_lo_su(3):dim_hi_su(3))

      integer dim_lo_sp(3)
      integer dim_hi_sp(3)

      double precision sp(
     $     dim_lo_sp(1):dim_hi_sp(1), 
     $     dim_lo_sp(2):dim_hi_sp(2), 
     &     dim_lo_sp(3):dim_hi_sp(3))

      integer dim_lo_mm_su(3)
      integer dim_hi_mm_su(3)

      double precision mm_su(
     $     dim_lo_mm_su(1):dim_hi_mm_su(1), 
     $     dim_lo_mm_su(2):dim_hi_mm_su(2), 
     &     dim_lo_mm_su(3):dim_hi_mm_su(3))

      integer dim_lo_mm_sp(3)
      integer dim_hi_mm_sp(3)

      double precision mm_sp(
     $     dim_lo_mm_sp(1):dim_hi_mm_sp(1), 
     $     dim_lo_mm_sp(2):dim_hi_mm_sp(2), 
     &     dim_lo_mm_sp(3):dim_hi_mm_sp(3))

      integer valid_lo(3)
      integer valid_hi(3)

      integer ierr

#endif

c*********************************************************************
c     Local Variables :
c*********************************************************************

      integer i
      integer j
      integer k

c*********************************************************************
c     Start :
c*********************************************************************

      do 120 k = valid_lo(3), valid_hi(3)
         do 110 j = valid_lo(2), valid_hi(2)
            do 100 i = valid_lo(1), valid_hi(1)

               su(i,j,k) = su(i,j,k) + mm_su(i,j,k)
               sp(i,j,k) = sp(i,j,k) + mm_sp(i,j,k)

 100        continue
 110     continue
 120  continue

      return
      end
