c
c
#include <Packages/Uintah/CCA/Components/Arches/fortran/enthalpyradthinsrc_fort.h>
#include "param4.h"
      
*     local variables

      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k
      double precision radsource, vol


      IST = idxLo(1)
      JST = idxLo(2)
      KST = idxLo(3)
      IEND = idxHi(1)
      JEND = idxHi(2)
      KEND = idxHi(3)
c-----------------------------------------------------------------------

      do 120 k = KST, KEND
         do 110 j = JST, JEND
            do 100 i = IST, IEND
               vol = sew(i)*sns(j)*stb(k)
               radsource = 4.0D0*SIGMA*Absorp(i,j,k)*
     *(tg(i,j,k)**4-tref**4)
               su(i,j,k) = su(i,j,k) - radsource*vol
 100        continue
 110     continue
 120  continue

      return
      end

