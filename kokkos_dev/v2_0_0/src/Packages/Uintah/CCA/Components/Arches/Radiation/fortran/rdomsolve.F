#include <Packages/Uintah/CCA/Components/Arches/Radiation/fortran/rdomsolve_fort.h>

      integer i,j,k,ii,jj,kk,iii,jjj,kkk

      double precision PI,PI1,SIG
      double precision val, factor

      DATA SIG/5.67D-08/

      pi = dacos(-1.0D0)
      PI1 = 1.0d0 / PI

      do 30 k=idxlo(3),idxhi(3)
      do 20 j=idxlo(2),idxhi(2)
      do 10 i=idxlo(1),idxhi(1)

      if (omu(l).gt.0) then
         
         ii = i
         iii=ii-1
         plusX = .true.

      else

         ii = idxlo(1)+idxhi(1)-i
         iii=ii+1
         plusX = .false.

      end if

      if (oeta(l).gt.0) then

         jj = j 
         jjj=jj-1
         plusY = .true.

      else

         jj = idxlo(2)+idxhi(2)-j
         jjj=jj+1
         plusY = .false.

      end if

      if (oxi(l).gt.0) then
         
         kk = k
         kkk=kk-1
         plusZ = .true.

      else

         kk = idxlo(3)+idxhi(3)-k
         kkk=kk+1
         plusZ = .false.
         
      end if

      volume(ii,jj,kk) = sew(ii) * sns(jj) * stb(kk)
      su(ii,jj,kk) = esrcg(ii,jj,kk)*volume(ii,jj,kk)

      areaew = sns(jj)*stb(kk)
      arean(ii) = sew(ii)*stb(kk)                 
      areatb(ii) = sew(ii) * sns(jj)
      if (pcell(ii,jj,kk).ne.ffield) then
c  call fixval assume abskg to be one but change it in props calculation

         val = abs(fraction(bands))*SIG*PI1*tg(ii,jj,kk)**4
         factor = 1.d0
         call fixradval(val,aw(ii,jj,kk),as(ii,jj,kk),ab(ii,jj,kk),
     $                  ap(ii,jj,kk),su(ii,jj,kk),factor)
      else
         if (plusX) then          
            if (pcell(ii-1,jj,kk).eq.ffield) then
               aw(ii,jj,kk) = abs(omu(l))*areaew
            else
               aw(ii,jj,kk) = 0.0d0
               su(ii,jj,kk) = su(ii,jj,kk) + (abs(omu(l))*areaew) *
     &         PI1 * (SIG * abs(fraction(bands))* 
     &              tg(ii-1,jj,kk)**4)
            end if
         else
            if (pcell(ii+1,j,k).eq.ffield) then
               aw(ii,jj,kk) = abs(omu(l))*areaew
            else
               aw(ii,jj,kk) = 0.0d0
               su(ii,jj,kk) = su(ii,jj,kk) + (abs(omu(l)) * areaew) * 
     &         PI1 * (SIG * abs(fraction(bands))* 
     &              tg(ii+1,jj,kk)**4)
            end if
         endif
         
         if (plusY) then
            if (pcell(ii,jj-1,kk).eq.ffield) then
               as(ii,jj,kk) = abs(oeta(l))*arean(ii)
            else
               as(ii,jj,kk) = 0.0d0
               su(ii,jj,kk) = su(ii,jj,kk) + 
     &              (abs(oeta(l))*arean(ii)) *
     &         PI1 * (SIG * abs(fraction(bands)) *
     &              tg(ii,jj-1,kk)**4)
            end if
         else
            if (pcell(ii,jj+1,kk).eq.ffield) then
               as(ii,jj,kk) = abs(oeta(l))*arean(ii)
            else
               as(ii,jj,kk) = 0.0d0
               su(ii,jj,kk) = su(ii,jj,kk) + 
     &              (abs(oeta(l))*arean(ii)) *
     &         PI1 * (SIG * abs(fraction(bands)) *
     &              tg(ii,jj+1,kk)**4)
            end if
         endif

         if (plusZ) then
            if (pcell(ii,jj,kk-1).eq.ffield) then
               ab(ii,jj,kk) = abs(oxi(l))*areatb(ii)
            else
               ab(ii,jj,kk) = 0.0d0
               su(ii,jj,kk) = su(ii,jj,kk) + 
     &              (abs(oxi(l))*areatb(ii)) *
     &         PI1 * (SIG * abs(fraction(bands)) *
     &              tg(ii,jj,kk-1)**4)
            end if
         else
            if (pcell(ii,jj,kk+1).eq.ffield) then
               ab(ii,jj,kk) = abs(oxi(l))*areatb(ii)
            else
               ab(ii,jj,kk) = 0.0d0
               su(ii,jj,kk) = su(ii,jj,kk) + 
     &              (abs(oxi(l))*areatb(ii)) * 
     &         PI1 * (SIG * abs(fraction(bands)) *
     &              tg(ii,jj,kk+1)**4)
            end if
         endif
         
         ap(ii,jj,kk) =  abs(omu(l))*areaew + abs(oeta(l))*arean(ii) + 
     &        abs(oxi(l))*areatb(ii) + abskg(ii,jj,kk)*
     &        volume(ii,jj,kk)
         
      endif
10    continue
20    continue
30    continue
      RETURN
      end










