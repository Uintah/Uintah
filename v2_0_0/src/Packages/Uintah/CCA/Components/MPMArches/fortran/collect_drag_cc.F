#include <Packages/Uintah/CCA/Components/MPMArches/fortran/collect_drag_cc_fort.h>
#include "param4.h"

*     local variables

      integer i
      integer j
      integer k

*     executable statements

      do 300 k = valid_lo(3), valid_hi(3)
         do 200 j = valid_lo(2), valid_hi(2)
            do 100 i = valid_lo(1), valid_hi(1)

               su_dragx_cc(i,j,k) = su_dragx_cc(i,j,k) + 
     $              su_dragx_fc(i,j,k) +
     $              su_dragx_fc(i+ioff,j+joff,k+koff)

               sp_dragx_cc(i,j,k) = sp_dragx_cc(i,j,k) + 
     $              sp_dragx_fc(i,j,k) +
     $              sp_dragx_fc(i+ioff,j+joff,k+koff)

 100        continue
 200     continue
 300  continue

      return
      end

