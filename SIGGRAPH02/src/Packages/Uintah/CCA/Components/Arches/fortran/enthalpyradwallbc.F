c
c
#include <Packages/Uintah/CCA/Components/Arches/fortran/enthalpyradwallbc_fort.h>
#include "param4.h"
      
*     local variables

      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k
      double precision radwallconst
      IST = idxLo(1)
      JST = idxLo(2)
      KST = idxLo(3)
      IEND = idxHi(1)
      JEND = idxHi(2)
      KEND = idxHi(3)
c-----------------------------------------------------------------------
      if (xminus) then
         do 120 k = KST, KEND
            do 110 j = JST, JEND
               i = ist -1
               if (pcell(i,j,k).eq.wallid) then
c     taken from pg 10-16 of fluent manual, assuming nw < .01
                  radwallconst = -SIGMA*TWO
                  qfluxw(i+1,j,k) = radwallconst*
     $                            (tg(i,j,k)**4-tg(i+1,j,k)**4)
               endif
 110        continue
 120     continue
      endif
    
      return
      end

