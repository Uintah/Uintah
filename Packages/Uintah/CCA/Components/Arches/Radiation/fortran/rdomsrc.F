#include <Packages/Uintah/CCA/Components/Arches/Radiation/fortran/rdomsrc_fort.h>

      integer i,j,k

      double precision PI


      pi = dacos(-1.0D0)
c     calculate incident flux at cell centers and radiative source

      do 30 k=idxlo(3),idxhi(3)
         do 20 j=idxlo(2),idxhi(2)
            do 10 i=idxlo(1),idxhi(1)

c               if ((i.eq.0).and.(j.eq.8).and.(k.eq.8)) then
c                  print*, 'Value of volq at 0,8,8 before cenint= ',
c     $                 volq(i,j,k)
c                  print*, 'Value of cenint at 0,8,8 = ',cenint(i,j,k)
c               endif

               volq(i,j,k) = volq(i,j,k) + cenint(i,j,k) * wt(l)
      
               src(i,j,k) = (abskg(i,j,k)) * volq(i,j,k)
     &              - 4.0D0*PI * (esrcg(i,j,k))

c               if ((i.eq.0).and.(j.eq.8).and.(k.eq.8)) then
c                  print*, 'Value of abskg at 0,8,8 = ',abskg(i,j,k)
c                  print*, 'Value of volq at 0,8,8 = ',volq(i,j,k)
c                  print*, 'Value of esrcg at 0,8,8 = ',esrcg(i,j,k)
c                  print*, 'Value of src at 0,8,8 = ',src(i,j,k)
c               endif

c      if((j.eq.7).and.(k.eq.7))write(6,*)i,abs(src(i,j,k))

 10         continue
 20      continue
 30   continue

      RETURN
      end
