#include <Packages/Uintah/CCA/Components/Arches/fortran/mm_modify_prescoef_fort.h>
#include "param4.h"

*     local variables

      double precision epse
      double precision epsw
      double precision epsn
      double precision epss
      double precision epst
      double precision epsb

      integer i
      integer j
      integer k

*     executable statements

      do 300 k = valid_lo(3), valid_hi(3)
         do 200 j = valid_lo(2), valid_hi(2)
            do 100 i = valid_lo(1), valid_hi(1)

               epse = pt5*(epsg(i,j,k) + epsg(i+1,j,k))
               epsw = pt5*(epsg(i,j,k) + epsg(i-1,j,k))
               epsn = pt5*(epsg(i,j,k) + epsg(i,j+1,k))
               epss = pt5*(epsg(i,j,k) + epsg(i,j-1,k))
               epst = pt5*(epsg(i,j,k) + epsg(i,j,k+1))
               epsb = pt5*(epsg(i,j,k) + epsg(i,j,k-1))

               ae(i,j,k) = ae(i,j,k)*epse
               aw(i,j,k) = aw(i,j,k)*epsw
               an(i,j,k) = an(i,j,k)*epsn
               as(i,j,k) = as(i,j,k)*epss
               at(i,j,k) = at(i,j,k)*epst
               ab(i,j,k) = ab(i,j,k)*epsb

 100        continue
 200     continue
 300  continue

      return
      end

         
