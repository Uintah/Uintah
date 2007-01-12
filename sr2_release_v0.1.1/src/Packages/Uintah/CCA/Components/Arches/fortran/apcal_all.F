#include <Packages/Uintah/CCA/Components/Arches/fortran/apcal_all_fort.h>
*-----------------------------------------------------------------------

*
*     local variables
*
      integer i
      integer j
      integer k
c-----------------------------------------------------------------------
c     executable statements
c-----------------------------------------------------------------------
      do 350 k = valid_lo(3), valid_hi(3)
         do 250 j = valid_lo(2), valid_hi(2)
            do 150 i = valid_lo(1), valid_hi(1)

               ap(i,j,k) = 0.0D0

 150        continue
 250     continue
 350  continue

      do 300 k = valid_lo(3), valid_hi(3)
         do 200 j = valid_lo(2), valid_hi(2)
            do 100 i = valid_lo(1), valid_hi(1)

               ap(i,j,k) = ae(i,j,k)+aw(i,j,k)
     $              +an(i,j,k)+as(i,j,k)
     $              +at(i,j,k)+ab(i,j,k)
     $              -sp(i,j,k)

 100        continue
 200     continue
 300  continue

      return

      end
