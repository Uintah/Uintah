#include <Packages/Uintah/CCA/Components/Arches/Radiation/fortran/rdomsrc_fort.h>

      integer i,j,k,l

      double precision PI

      DATA PI/3.141593D0/

c     calculate incident flux at cell centers and radiative source

      do 30 k=idxlo(3),idxhi(3)
         do 20 j=idxlo(2),idxhi(2)
            do 10 i=idxlo(1),idxhi(1)

               volq(i,j,k) = volq(i,j,k) + cenint(i,j,k) * wt(l)
      
               src(i,j,k) = (abskg(i,j,k)) * volq(i,j,k)
     &              - 4.0*PI * (esrcg(i,j,k))

c      if((j.eq.7).and.(k.eq.7))write(6,*)i,abs(src(i,j,k))

 10         continue
 20      continue
 30   continue

      RETURN
      end
