#include <Packages/Uintah/CCA/Components/Arches/Radiation/fortran/rdomvolq_fort.h>

      integer i,j,k

c     calculate incident flux at cell centers

      do 30 k=idxlo(3),idxhi(3)
         do 20 j=idxlo(2),idxhi(2)
            do 10 i=idxlo(1),idxhi(1)

            volq(i,j,k) = volq(i,j,k) + cenint(i,j,k) * wt(l)
      
 10         continue
 20      continue
 30   continue

      RETURN
      end
