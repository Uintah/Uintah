      subroutine add_hydrostatic_term_topressure(
     $     dim_lo, dim_hi,
     $     dim_lo_ph, dim_hi_ph,
     $     dim_lo_prel, dim_hi_prel,
     $     dim_lo_den, dim_hi_den,
     $     p_plus_hydro, prel, den_micro,
     $     gx, gy, gz,
     $     xx, yy, zz,
     $     valid_lo, valid_hi,
     $     pcell, wall)

c     
c     Author: Seshadri Kumar, April 14, 2001
c     Purpose: Adds the hydrostatic component of the pressure
c     to the relative pressure calculated in the pressure solve.
c
c     Note: we will later need to change this to also include the
c     reference pressure, so that we actually calculate the absolute
c     pressure.  The point of this routine (as will be the point
c     of calculating the absolute pressure) is to correctly 
c     calculate pressurization of bounding surfaces to pass to MPM.
c
c     Note 2: This routine assumes that the location of the 
c     reference pressure is at (0.0,0.0,0.0).  Later, we need to 
c     change this routine to account for the true location of the
c     reference.
c
      implicit none

#include "param4.h"

      integer dim_lo(3)
      integer dim_hi(3)

      integer dim_lo_ph(3)
      integer dim_hi_ph(3)

      integer dim_lo_prel(3)
      integer dim_hi_prel(3)

      integer dim_lo_den(3)
      integer dim_hi_den(3)
      
      double precision p_plus_hydro(
     $     dim_lo_ph(1):dim_hi_ph(1),
     $     dim_lo_ph(2):dim_hi_ph(2),
     $     dim_lo_ph(3):dim_hi_ph(3))

      double precision prel(
     $     dim_lo_prel(1):dim_hi_prel(1),
     $     dim_lo_prel(2):dim_hi_prel(2),
     $     dim_lo_prel(3):dim_hi_prel(3))

      double precision den_micro(
     $     dim_lo_den(1):dim_hi_den(1),
     $     dim_lo_den(2):dim_hi_den(2),
     $     dim_lo_den(3):dim_hi_den(3))

      double precision gx
      double precision gy
      double precision gz

      double precision xx(dim_lo(1):dim_hi(1))
      double precision yy(dim_lo(2):dim_hi(2))
      double precision zz(dim_lo(3):dim_hi(3))

      integer valid_lo(3)
      integer valid_hi(3)

      integer pcell(
     $     dim_lo_den(1):dim_hi_den(1),
     $     dim_lo_den(2):dim_hi_den(2),
     $     dim_lo_den(3):dim_hi_den(3))
      
      integer wall

*     local variables

      integer i
      integer j
      integer k

*     executable statements

      do 300 k = valid_lo(3), valid_hi(3)
         do 200 j = valid_lo(2), valid_hi(2)
            do 100 i = valid_lo(1), valid_hi(1)

               if (pcell(i,j,k).ne.wall) then

                  p_plus_hydro(i,j,k) = prel(i,j,k) + 
     $                 den_micro(i,j,k)*
     $                 (gx*xx(i) + 
     $                 gy*yy(j) + 
     $                 gz*zz(k))

               endif

 100        continue
 200     continue
 300  continue

      return
      end

