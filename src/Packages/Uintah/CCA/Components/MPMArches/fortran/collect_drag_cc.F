#if 1
#include <Packages/Uintah/CCA/Components/MPMArches/fortran/collect_drag_cc_fort.h>
#else
      subroutine collect_drag_cc(
     $     su_dragx_cc, sp_dragx_cc,
     $     su_dragx_fc, sp_dragx_fc,
     $     ioff, joff, koff,
     $     dim_lo_su_cc, dim_hi_su_cc, 
     $     dim_lo_sp_cc, dim_hi_sp_cc, 
     $     dim_lo_su_fc, dim_hi_su_fc, 
     $     dim_lo_sp_fc, dim_hi_sp_fc, 
     $     valid_lo, valid_hi)

      implicit none
      include 'param4.h'

      integer dim_lo_su_cc(3)
      integer dim_hi_su_cc(3)

      integer dim_lo_sp_cc(3)
      integer dim_hi_sp_cc(3)

      integer dim_lo_su_fc(3)
      integer dim_hi_su_fc(3)

      integer dim_lo_sp_fc(3)
      integer dim_hi_sp_fc(3)

      integer valid_lo(3)
      integer valid_hi(3)

      double precision su_dragx_cc(
     $     dim_lo_su_cc(1):dim_hi_su_cc(1),
     $     dim_lo_su_cc(2):dim_hi_su_cc(2),
     $     dim_lo_su_cc(3):dim_hi_su_cc(3))

      double precision sp_dragx_cc(
     $     dim_lo_sp_cc(1):dim_hi_sp_cc(1),
     $     dim_lo_sp_cc(2):dim_hi_sp_cc(2),
     $     dim_lo_sp_cc(3):dim_hi_sp_cc(3))

      double precision su_dragx_fc(
     $     dim_lo_su_fc(1):dim_hi_su_fc(1),
     $     dim_lo_su_fc(2):dim_hi_su_fc(2),
     $     dim_lo_su_fc(3):dim_hi_su_fc(3))

      double precision sp_dragx_fc(
     $     dim_lo_sp_fc(1):dim_hi_sp_fc(1),
     $     dim_lo_sp_fc(2):dim_hi_sp_fc(2),
     $     dim_lo_sp_fc(3):dim_hi_sp_fc(3))

      integer ioff
      integer joff
      integer koff
#endif

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

