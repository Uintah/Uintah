#if 1
#include <Packages/Uintah/CCA/Components/MPMArches/fortran/pressure_force_fort.h>
#include "../../Arches/fortran/param4.h"
#else
      subroutine pressure_force(
     $     pforcex_fcx, pforcey_fcy, pforcez_fcz,
     $     epsg, 
     $     epsg_solid,
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
c     pforcex_fcx: gas pressure force acting on a solid wall 
c     perpendicular to the x-direction, stored at x-face centers
c
c     pforcey_fcy: gas pressure force acting on a solid wall 
c     perpendicular to the y-direction, stored at y-face centers
c
c     pforcez_fcz: gas pressure force acting on a solid wall
c     perpendicular to the z-direction, stored at z-face centers
c
c     The routine requires here, as inputs:
c
c     epsg: cell-centered volume fraction of gas
c
c     epsg_solid: cell-centered volume fraction of solid
c
c     pres: cell-centered pressure of gas
c     
c     sew, sns, stb: dimensions of sides of p-cell
c
c     dim_lo_fcx, dim_hi_fcx: extents of the dimension of the input 
c     array pforcex_fcx
c
c     dim_lo_fcy, dim_hi_fcy: extents of the dimension of the input 
c     array pforcey_fcy
c
c     dim_lo_fcz, dim_hi_fcz: extents of the dimension of the input 
c     array pforcez_fcz
c
c     dim_lo_eps, dim_hi_eps: extents of the dimension of the input 
c     array epsg
c
c     dim_lo_epss, dim_hi_epss: extents of the dimension of the input 
c     array epsg_solid
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
      
      double precision pforcex_fcx(
     $     dim_lo_fcx(1):dim_hi_fcx(1),
     $     dim_lo_fcx(2):dim_hi_fcx(2),
     $     dim_lo_fcx(3):dim_hi_fcx(3))

      double precision pforcey_fcy(
     $     dim_lo_fcy(1):dim_hi_fcy(1),
     $     dim_lo_fcy(2):dim_hi_fcy(2),
     $     dim_lo_fcy(3):dim_hi_fcy(3))

      double precision pforcez_fcz(
     $     dim_lo_fcz(1):dim_hi_fcz(1),
     $     dim_lo_fcz(2):dim_hi_fcz(2),
     $     dim_lo_fcz(3):dim_hi_fcz(3))      

      double precision epsg(
     $     dim_lo_eps(1):dim_hi_eps(1),
     $     dim_lo_eps(2):dim_hi_eps(2),
     $     dim_lo_eps(3):dim_hi_eps(3))

      double precision epsg_solid(
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

#ifdef __GNUC__
C The variable names are too long the use the PASS3 macros,
C so we have to do this - Steve
      call arrass(pforcex_fcx_low_x, pforcex_fcx_low_y,
     $ pforcex_fcx_low_z, pforcex_fcx_high_x,
     $ pforcex_fcx_high_y, pforcex_fcx_high_z,
     $ pforcex_fcx, zero, valid_lo, valid_hi)

      call arrass(pforcey_fcy_low_x, pforcey_fcy_low_y,
     $ pforcey_fcy_low_z, pforcey_fcy_high_x,
     $ pforcey_fcy_high_y, pforcey_fcy_high_z,
     $ pforcey_fcy, zero, valid_lo, valid_hi)

      call arrass(pforcez_fcz_low_x, pforcez_fcz_low_y,
     $ pforcez_fcz_low_z, pforcez_fcz_high_x,
     $ pforcez_fcz_high_y, pforcez_fcz_high_z,
     $ pforcez_fcz, zero, valid_lo, valid_hi)

#else
      call arrass(pforcex_fcx_low, pforcex_fcx_high,
     $ pforcex_fcx, zero, valid_lo, valid_hi)

      call arrass(pforcey_fcy_low, pforcey_fcy_high,
     $ pforcey_fcy, zero, valid_lo, valid_hi)

      call arrass(pforcez_fcz_low, pforcez_fcz_high,
     $ pforcez_fcz, zero, valid_lo, valid_hi)

#endif

c     debug, sk, April 14, 2001
      print*, 'printing pressure force values'

*     west, south, and bottom faces of intrusion

      do 120 k = valid_lo(3), valid_hi(3)
         do 110 j = valid_lo(2), valid_hi(2)
            do 100 i = valid_lo(1), valid_hi(1)

               if (pcell(i,j,k).eq.wall) then

*     west face

                  if (pcell(i-1,j,k).eq.ffield) then

                     areayz = sns(j)*stb(k)
                     pforcex_fcx(i,j,k) = 
     $                    pforcex_fcx(i,j,k) +
     $                    pres(i-1,j,k)*areayz
     $                    *epsg(i-1,j,k)*epsg_solid(i,j,k)

c     debug, sk, April 14, 2001
                     print*, i,j,k, pres(i-1,j,k), 
     $                    pforcex_fcx(i,j,k)

                  endif

*     south face

                  if (pcell(i,j-1,k).eq.ffield) then

                     areaxz = sew(i)*stb(k)
                     pforcey_fcy(i,j,k) = 
     $                    pforcey_fcy(i,j,k) +
     $                    pres(i,j-1,k)*areaxz
     $                    *epsg(i,j-1,k)*epsg_solid(i,j,k)

                  endif

*     bottom face

                  if (pcell(i,j,k-1).eq.ffield) then

                     areaxy = sew(i)*sns(j)
                     pforcez_fcz(i,j,k) = 
     $                    pforcez_fcz(i,j,k) +
     $                    pres(i,j,k-1)*areaxy
     $                    *epsg(i,j,k-1)*epsg_solid(i,j,k)

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
                     pforcex_fcx(i+1,j,k) = 
     $                    pforcex_fcx(i+1,j,k) -
     $                    pres(i+1,j,k)*areayz
     $                    *epsg(i+1,j,k)*epsg_solid(i,j,k)

c     debug, sk, April 14, 2001
                     print*, i+1,j,k, pres(i+1,j,k), 
     $                    pforcex_fcx(i+1,j,k)

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
                     pforcey_fcy(i,j+1,k) = 
     $                    pforcey_fcy(i,j+1,k) -
     $                    pres(i,j+1,k)*areaxz
     $                    *epsg(i,j+1,k)*epsg_solid(i,j,k)

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
                     pforcez_fcz(i,j,k+1) = 
     $                    pforcez_fcz(i,j,k+1) -
     $                    pres(i,j,k+1)*areaxy
     $                    *epsg(i,j,k+1)*epsg_solid(i,j,k)

                  endif

               endif

 400        continue
 410     continue
 420  continue

c     debug: set pressure forces to zero

#ifdef __GNUC__
C The variable names are too long the use the PASS3 macros,
C so we have to do this - Steve
      call arrass(pforcex_fcx_low_x, pforcex_fcx_low_y,
     $ pforcex_fcx_low_z, pforcex_fcx_high_x,
     $ pforcex_fcx_high_y, pforcex_fcx_high_z,
     $ pforcex_fcx, zero, valid_lo, valid_hi)

      call arrass(pforcey_fcy_low_x, pforcey_fcy_low_y,
     $ pforcey_fcy_low_z, pforcey_fcy_high_x,
     $ pforcey_fcy_high_y, pforcey_fcy_high_z,
     $ pforcey_fcy, zero, valid_lo, valid_hi)

      call arrass(pforcez_fcz_low_x, pforcez_fcz_low_y,
     $ pforcez_fcz_low_z, pforcez_fcz_high_x,
     $ pforcez_fcz_high_y, pforcez_fcz_high_z,
     $ pforcez_fcz, zero, valid_lo, valid_hi)

#else
      call arrass(pforcex_fcx_low, pforcex_fcx_high,
     $ pforcex_fcx, zero, valid_lo, valid_hi)

      call arrass(pforcey_fcy_low, pforcey_fcy_high,
     $ pforcey_fcy, zero, valid_lo, valid_hi)

      call arrass(pforcez_fcz_low, pforcez_fcz_high,
     $ pforcez_fcz, zero, valid_lo, valid_hi)

#endif
      return
      end
