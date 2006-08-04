#include <Packages/Uintah/CCA/Components/MPMArches/fortran/collect_scalar_fctocc_fort.h>
#include "param4.h"

*     local variables

      integer i
      integer j
      integer k

*     executable statements

      do 300 k = valid_lo(3), valid_hi(3)
         do 200 j = valid_lo(2), valid_hi(2)
            do 100 i = valid_lo(1), valid_hi(1)

               su_enth_cc(i,j,k) = su_enth_cc(i,j,k) + 
     $              su_enth_fcx(i,j,k) + su_enth_fcx(i+1,j,k) +
     $              su_enth_fcy(i,j,k) + su_enth_fcy(i,j+1,k) +
     $              su_enth_fcz(i,j,k) + su_enth_fcz(i,j,k+1)

               sp_enth_cc(i,j,k) = sp_enth_cc(i,j,k) + 
     $              sp_enth_fcx(i,j,k) + sp_enth_fcx(i+1,j,k) +
     $              sp_enth_fcy(i,j,k) + sp_enth_fcy(i,j+1,k) +
     $              sp_enth_fcz(i,j,k) + sp_enth_fcz(i,j,k+1)

 100        continue
 200     continue
 300  continue

c     debug
c      print*, 'printing debug values of su_enth and sp_enth'
c      print*, su_enth_cc(7,9,9), sp_enth_cc(7,9,9)

      return
      end

