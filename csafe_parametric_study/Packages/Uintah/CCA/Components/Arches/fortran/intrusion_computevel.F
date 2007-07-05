#include <Packages/Uintah/CCA/Components/Arches/fortran/intrusion_computevel_fort.h>      

c---------------------------------------------------------------------
c     three-dimensional version
c     this subroutine will add pressure gradient term for momentum eqn
c     modification, Seshadri Kumar, July 30, 2001,
c     to take care of multimaterial wall intrusions
c---------------------------------------------------------------------
#include "param4.h"

c*********************************************************************
c     local variables :
c*********************************************************************
      integer i
      integer j
      integer k
      integer idf

      do 120 k =idxlo_u(3), idxhi_u(3)
         do 110 j = idxlo_u(2), idxhi_u(2)
            do 100 i = idxlo_u(1), idxhi_u(1)

               if ((pcell(i,j,k).eq.mmwallid) .or.
     $              (pcell(i-ioff, j-joff, k-koff)
     $              .eq.mmwallid)) then

                  uu(i,j,k) = zero
                  
               endif

 100        continue
 110     continue
 120  continue

      return
      end

