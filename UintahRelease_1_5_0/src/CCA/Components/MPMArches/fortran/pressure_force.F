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
#if 1
#include <CCA/Components/MPMArches/fortran/pressure_force_fort.h>
#include "../../Arches/fortran/param4.h"
#else
      subroutine pressure_force(
     $     pfx_fcx, pfy_fcy, pfz_fcz,
     $     epsg, 
     $     epss,
     $     pres, 
     $     sew, sns, stb,
     $     dim_lo, dim_hi,
     $     dim_lo_fcx, dim_hi_fcx,
     $     dim_lo_fcy, dim_hi_fcy,
     $     dim_lo_fcz, dim_hi_fcz,
     $     dim_lo_eps, dim_hi_eps,
     $     dim_lo_epss, dim_hi_epss,
     $     dim_lo_p, dim_hi_p,
     $     valid_lo, valid_hi,
     $     pcell, wall, ffield)

C-----------------------------------------------------------------------
c     Author: Seshadri Kumar, March 21, 2001
c     Last modified: March 21, 2001
c
c     this subroutine calculates pressure forces exerted by the 
c     gas on continuous solid surfaces.
c
c     This routine calculates
c
c     pfx_fcx: gas pressure force acting on a solid wall 
c     perpendicular to the x-direction, stored at x-face centers
c
c     pfy_fcy: gas pressure force acting on a solid wall 
c     perpendicular to the y-direction, stored at y-face centers
c
c     pfz_fcz: gas pressure force acting on a solid wall
c     perpendicular to the z-direction, stored at z-face centers
c
c     The routine requires here, as inputs:
c
c     epsg: cell-centered volume fraction of gas
c
c     epss: cell-centered volume fraction of solid
c
c     pres: cell-centered pressure of gas
c     
c     sew, sns, stb: dimensions of sides of p-cell
c
c     dim_lo_fcx, dim_hi_fcx: extents of the dimension of the input 
c     array pfx_fcx
c
c     dim_lo_fcy, dim_hi_fcy: extents of the dimension of the input 
c     array pfy_fcy
c
c     dim_lo_fcz, dim_hi_fcz: extents of the dimension of the input 
c     array pfz_fcz
c
c     dim_lo_eps, dim_hi_eps: extents of the dimension of the input 
c     array epsg
c
c     dim_lo_epss, dim_hi_epss: extents of the dimension of the input 
c     array epss
c
c     dim_lo_p, dim_hi_p: extents of the dimension of the input 
c     array pres
c
c     valid_lo, valid_hi: extents of the valid region of the patch
c     (without ghost indices)
c
c     pcell: integer array that stores the information on cell typing
c
c     wall: integer array that stores the identifier for cell type wall
c
c     ffield: integer array that stores the identifier for cell type 
c     ffield
c
C-----------------------------------------------------------------------
      implicit none
      include 'param4.h'
      include 'pconst.h'

      integer dim_lo(3)
      integer dim_hi(3)

      integer dim_lo_fcx(3)
      integer dim_hi_fcx(3)

      integer dim_lo_fcy(3)
      integer dim_hi_fcy(3)

      integer dim_lo_fcz(3)
      integer dim_hi_fcz(3)

      integer dim_lo_eps(3)
      integer dim_hi_eps(3)

      integer dim_lo_epss(3)
      integer dim_hi_epss(3)

      integer dim_lo_p(3)
      integer dim_hi_p(3)
      
      double precision pfx_fcx(
     $     dim_lo_fcx(1):dim_hi_fcx(1),
     $     dim_lo_fcx(2):dim_hi_fcx(2),
     $     dim_lo_fcx(3):dim_hi_fcx(3))

      double precision pfy_fcy(
     $     dim_lo_fcy(1):dim_hi_fcy(1),
     $     dim_lo_fcy(2):dim_hi_fcy(2),
     $     dim_lo_fcy(3):dim_hi_fcy(3))

      double precision pfz_fcz(
     $     dim_lo_fcz(1):dim_hi_fcz(1),
     $     dim_lo_fcz(2):dim_hi_fcz(2),
     $     dim_lo_fcz(3):dim_hi_fcz(3))      

      double precision epsg(
     $     dim_lo_eps(1):dim_hi_eps(1),
     $     dim_lo_eps(2):dim_hi_eps(2),
     $     dim_lo_eps(3):dim_hi_eps(3))

      double precision epss(
     $     dim_lo_epss(1):dim_hi_epss(1),
     $     dim_lo_epss(2):dim_hi_epss(2),
     $     dim_lo_epss(3):dim_hi_epss(3))

      double precision pres(
     $     dim_lo_p(1):dim_hi_p(1),
     $     dim_lo_p(2):dim_hi_p(2),
     $     dim_lo_p(3):dim_hi_p(3))

      double precision sew(dim_lo(1):dim_hi(1))
      double precision sns(dim_lo(2):dim_hi(2))
      double precision stb(dim_lo(3):dim_hi(3))

      integer valid_lo(3)
      integer valid_hi(3)

      integer pcell(
     $     dim_lo(1):dim_hi(1),
     $     dim_lo(2):dim_hi(2),
     $     dim_lo(3):dim_hi(3))

      integer wall
      integer ffield
#endif

*     local variables

      integer i
      integer j
      integer k

      double precision areayz
      double precision areaxz
      double precision areaxy

*     executable statements

c--------------------------------------------------------------------
c     initialize arrays to zero
c--------------------------------------------------------------------

      call arrass(
     $     PASS3A(pfx_fcx)
     $     PASS3B(pfx_fcx),
     $     zero, valid_lo, valid_hi)

      call arrass(
     $     PASS3A(pfy_fcy)
     $     PASS3B(pfy_fcy),
     $     zero, valid_lo, valid_hi)

      call arrass(
     $     PASS3A(pfz_fcz)
     $     PASS3B(pfz_fcz),
     $     zero, valid_lo, valid_hi)

c     debug, sk, April 14, 2001
c      print*, 'printing pressure force values'

*     west, south, and bottom faces of intrusion

      do 120 k = valid_lo(3), valid_hi(3)
         do 110 j = valid_lo(2), valid_hi(2)
            do 100 i = valid_lo(1), valid_hi(1)

               if (pcell(i,j,k).eq.wall) then

*     west face

                  if (pcell(i-1,j,k).eq.ffield) then

                     areayz = sns(j)*stb(k)
                     pfx_fcx(i,j,k) = 
     $                    pfx_fcx(i,j,k) +
     $                    pres(i-1,j,k)*areayz
     $                    *epsg(i-1,j,k)*epss(i,j,k)

c     debug, sk, April 14, 2001
c                     print*, i,j,k, pres(i-1,j,k), 
c     $                    pfx_fcx(i,j,k)

                  endif

*     south face

                  if (pcell(i,j-1,k).eq.ffield) then

                     areaxz = sew(i)*stb(k)
                     pfy_fcy(i,j,k) = 
     $                    pfy_fcy(i,j,k) +
     $                    pres(i,j-1,k)*areaxz
     $                    *epsg(i,j-1,k)*epss(i,j,k)

                  endif

*     bottom face

                  if (pcell(i,j,k-1).eq.ffield) then

                     areaxy = sew(i)*sns(j)
                     pfz_fcz(i,j,k) = 
     $                    pfz_fcz(i,j,k) +
     $                    pres(i,j,k-1)*areaxy
     $                    *epsg(i,j,k-1)*epss(i,j,k)

                  endif

               endif

 100        continue
 110     continue
 120  continue

*     east face of intrusion

      do 220 k = valid_lo(3), valid_hi(3)
         do 210 j = valid_lo(2), valid_hi(2)
            do 200 i = valid_lo(1)-1, valid_hi(1)-1

               if (pcell(i,j,k).eq.wall) then

                  if (pcell(i+1,j,k).eq.ffield) then

                     areayz = sns(j)*stb(k)
                     pfx_fcx(i+1,j,k) = 
     $                    pfx_fcx(i+1,j,k) -
     $                    pres(i+1,j,k)*areayz
     $                    *epsg(i+1,j,k)*epss(i,j,k)

c     debug, sk, April 14, 2001
c                     print*, i+1,j,k, pres(i+1,j,k), 
c     $                    pfx_fcx(i+1,j,k)

                  endif

               endif

 200        continue
 210     continue
 220  continue

*     north face of intrusion

      do 320 k = valid_lo(3), valid_hi(3)
         do 310 j = valid_lo(2)-1, valid_hi(2)-1
            do 300 i = valid_lo(1), valid_hi(1)

               if (pcell(i,j,k).eq.wall) then

                  if (pcell(i,j+1,k).eq.ffield) then

                     areaxz = sew(i)*stb(k)
                     pfy_fcy(i,j+1,k) = 
     $                    pfy_fcy(i,j+1,k) -
     $                    pres(i,j+1,k)*areaxz
     $                    *epsg(i,j+1,k)*epss(i,j,k)

                  endif

               endif

 300        continue
 310     continue
 320  continue

*     top face of intrusion

      do 420 k = valid_lo(3)-1, valid_hi(3)-1
         do 410 j = valid_lo(2), valid_hi(2)
            do 400 i = valid_lo(1), valid_hi(1)

               if (pcell(i,j,k).eq.wall) then

                  if (pcell(i,j,k+1).eq.ffield) then

                     areaxy = sew(i)*sns(j)
                     pfz_fcz(i,j,k+1) = 
     $                    pfz_fcz(i,j,k+1) -
     $                    pres(i,j,k+1)*areaxy
     $                    *epsg(i,j,k+1)*epss(i,j,k)

                  endif

               endif

 400        continue
 410     continue
 420  continue

      return
      end
