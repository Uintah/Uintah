c
c
#include <Packages/Uintah/CCA/Components/Arches/fortran/scalsrc_fort.h>
#include "param4.h"
      
*     local variables

      double precision vol, apo
      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k

      IST = idxLo(1)
      JST = idxLo(2)
      KST = idxLo(3)
      IEND = idxHi(1)
      JEND = idxHi(2)
      KEND = idxHi(3)
c-----------------------------------------------------------------------

      CALL ARRASS(PASS3A(SU)
     & PASS3B(SU),
     & ZERO, idxLo, idxHi)
      CALL ARRASS(PASS3A(SP)
     & PASS3B(SP),
     & ZERO, idxLo, idxHi)

      do 120 k = KST, KEND
         do 110 j = JST, JEND
            do 100 i = IST, IEND
               vol = sns(j)*sew(i)*stb(k)
               APO = old_den(I,J,K)*VOL/deltat
               SU(I,J,K) = SU(I,J,K) + APO*old_scal(I,J,K)
 100        continue
 110     continue
 120  continue

      return
      end
