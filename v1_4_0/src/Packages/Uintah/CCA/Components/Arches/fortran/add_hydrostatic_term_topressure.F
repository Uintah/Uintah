#include <Packages/Uintah/CCA/Components/Arches/fortran/add_hydrostatic_term_topressure_fort.h>
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

