#include <Packages/Uintah/CCA/Components/Arches/Radiation/fortran/rdomsrc_fort.h>

      integer i,j,k

      double precision PI


      pi = dacos(-1.0D0)
c     calculate incident flux at cell centers and radiative source

      do 30 k=idxlo(3),idxhi(3)
         do 20 j=idxlo(2),idxhi(2)
            do 10 i=idxlo(1),idxhi(1)

               src(i,j,k) = (abskg(i,j,k)) * volq(i,j,k)
     &              - 4.0D0*PI * (esrcg(i,j,k))

 10         continue
 20      continue
 30   continue

      RETURN
      end
