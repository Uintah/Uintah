c
c
#include <Packages/Uintah/CCA/Components/Arches/fortran/enthalpyradsrc_fort.h>
      
*     local variables

      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k
      double precision arean, areaew, areatb


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
               arean = sew(i)*stb(k)
               areaew = sns(j)*stb(k)
               areatb = sns(j)*sew(i)
               su(i,j,k) = su(i,j,k) + 
     $                    areaew*(qfluxw(i,j,k)-qfluxe(i,j,k)) +
     $                    arean*(qfluxs(i,j,k)-qfluxn(i,j,k)) +
     $                    areatb*(qfluxb(i,j,k)-qfluxt(i,j,k)) 
 100        continue
 110     continue
 120  continue

      return
      end

