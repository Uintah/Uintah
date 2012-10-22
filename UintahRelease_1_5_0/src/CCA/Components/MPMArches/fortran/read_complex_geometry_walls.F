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
#if 1
#include <CCA/Components/MPMArches/fortran/read_complex_geometry_walls_fort.h>
#include "../../Arches/fortran/param4.h"
#else
      subroutine read_complex_geometry_walls(
     $     iccst,jccst,kccst,inext, jnext, knext,
     $     epsg, totarea, 
     $     nbar1,nbar2,nbar3,
     $     cbar1,cbar2,cbar3,
     $     gaf_x, gaf_y, gaf_z,
     $     gaf_xe, gaf_yn, gaf_zt,
     $     patchindex, tot_walls,
     $     dim_lo,dim_hi)

      integer patchindex
      integer tot_walls
      integer iccst
      integer jccst
      integer kccst

      double precision epsg(
     $     dim_lo(1):dim_hi(1),
     $     dim_lo(2):dim_hi(2),
     $     dim_lo(3):dim_hi(3))

      double precision totarea(
     $     dim_lo(1):dim_hi(1),
     $     dim_lo(2):dim_hi(2),
     $     dim_lo(3):dim_hi(3))

      double precision nbar1(
     $     dim_lo(1):dim_hi(1),
     $     dim_lo(2):dim_hi(2),
     $     dim_lo(3):dim_hi(3))

      double precision nbar2(
     $     dim_lo(1):dim_hi(1),
     $     dim_lo(2):dim_hi(2),
     $     dim_lo(3):dim_hi(3))

      double precision nbar3(
     $     dim_lo(1):dim_hi(1),
     $     dim_lo(2):dim_hi(2),
     $     dim_lo(3):dim_hi(3))

      double precision cbar1(
     $     dim_lo(1):dim_hi(1),
     $     dim_lo(2):dim_hi(2),
     $     dim_lo(3):dim_hi(3))

      double precision cbar2(
     $     dim_lo(1):dim_hi(1),
     $     dim_lo(2):dim_hi(2),
     $     dim_lo(3):dim_hi(3))

      double precision cbar3(
     $     dim_lo(1):dim_hi(1),
     $     dim_lo(2):dim_hi(2),
     $     dim_lo(3):dim_hi(3))

      double precision gaf_x(
     $     dim_lo(1):dim_hi(1),
     $     dim_lo(2):dim_hi(2),
     $     dim_lo(3):dim_hi(3))

      double precision gaf_y(
     $     dim_lo(1):dim_hi(1),
     $     dim_lo(2):dim_hi(2),
     $     dim_lo(3):dim_hi(3))

      double precision gaf_z(
     $     dim_lo(1):dim_hi(1),
     $     dim_lo(2):dim_hi(2),
     $     dim_lo(3):dim_hi(3))

      double precision gaf_xe(
     $     dim_lo(1):dim_hi(1),
     $     dim_lo(2):dim_hi(2),
     $     dim_lo(3):dim_hi(3))

      double precision gaf_yn(
     $     dim_lo(1):dim_hi(1),
     $     dim_lo(2):dim_hi(2),
     $     dim_lo(3):dim_hi(3))

      double precision gaf_zt(
     $     dim_lo(1):dim_hi(1),
     $     dim_lo(2):dim_hi(2),
     $     dim_lo(3):dim_hi(3))

#endif

c      local arrays

      character*5 ext
      character*19 filename
      integer ncells
      integer i,j,k
      integer i2,j2,k2
      integer ii,jj,kk
      integer p2
      
c     executable statements

c      print*,'The number of walls is ',tot_walls
      if (tot_walls .eq. 0) goto 222

      p2 = patchindex + 10000
      write(ext,1) p2
 1    format (i5)  

      filename = 'patch_wall' // ext // '.dat'
      open(unit=1, file = filename, status='unknown',
     $     form='formatted')

      print*, 'filename is ', filename

      read(1,*) iccst,jccst,kccst
      i = iccst
      j = jccst
      k = kccst

      do ncells = 1,tot_walls-1

         read(1,*) i2,j2,k2
         inext(i,j,k) = i2
         jnext(i,j,k) = j2
         knext(i,j,k) = k2
         i = i2
         j = j2
         k = k2

      end do
      inext(i,j,k) = 1000000
      jnext(i,j,k) = 1000000
      knext(i,j,k) = 1000000

      i = iccst
      j = jccst
      k = kccst

      do ncells = 1,tot_walls

         totarea(i,j,k) = zero
         epsg(i,j,k) = zero
         nbar1(i,j,k) = zero
         nbar2(i,j,k) = zero
         nbar3(i,j,k) = zero
         cbar1(i,j,k) = zero
         cbar2(i,j,k) = zero
         cbar3(i,j,k) = zero
         gaf_x(i,j,k) = zero
         gaf_xe(i,j,k) = zero
         gaf_y(i,j,k) = zero
         gaf_yn(i,j,k) = zero
         gaf_z(i,j,k) = zero
         gaf_zt(i,j,k) = zero

         ii = inext(i,j,k)
         jj = jnext(i,j,k) 
         kk = knext(i,j,k)
c         print*,ii,jj,kk
         i = ii
         j = jj
         k = kk
      end do

 222  continue
      
      return
      end
