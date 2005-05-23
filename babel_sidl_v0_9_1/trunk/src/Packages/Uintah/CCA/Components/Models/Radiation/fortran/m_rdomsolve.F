#include <Packages/Uintah/CCA/Components/Models/Radiation/fortran/m_rdomsolve_fort.h>

      integer i,j,k,ii,jj,kk,iii,jjj,kkk

      double precision PI,PI1,SIG
      double precision val, factor
      double precision areaew
      double precision arean
      double precision areatb

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
      arean = sew(ii)*stb(kk)                 
      areatb = sew(ii) * sns(jj)

c      if (pcell(ii,jj,kk).ne.ffield) then
c  call fixval assume abskg to be one but change it in props calculation

c         val = abs(fraction(bands))*SIG*PI1*tg(ii,jj,kk)**4
c         factor = 1.d0
c         call fixradval(val, 
c     $        aw(ii,jj,kk),as(ii,jj,kk),ab(ii,jj,kk),
c     $        ae(ii,jj,kk),an(ii,jj,kk),at(ii,jj,kk),
c     $        ap(ii,jj,kk),su(ii,jj,kk),factor)
c      else

         if (plusX) then          
c            if (pcell(ii-1,jj,kk).eq.ffield) then

               aw(ii,jj,kk) = abs(omu(l))*areaew

c            else
c               aw(ii,jj,kk) = 0.0d0
c               su(ii,jj,kk) = su(ii,jj,kk) + (abs(omu(l))*areaew) *
c     &         PI1 * (SIG * abs(fraction(bands))*
c     &              tg(ii-1,jj,kk)**4)
c            end if
         else
c            if (pcell(ii+1,jj,kk).eq.ffield) then

               aw(ii,jj,kk) = abs(omu(l))*areaew

c            else
c               aw(ii,jj,kk) = 0.0d0
c               su(ii,jj,kk) = su(ii,jj,kk) + (abs(omu(l)) * areaew) * 
c     &         PI1 * (SIG * abs(fraction(bands))*
c     Empirical model for emissivity at boundaries
c     &              1.0d0*
c     &              tg(ii+1,jj,kk)**4)
c            end if
         endif
         
         if (plusY) then
c            if (pcell(ii,jj-1,kk).eq.ffield) then

               as(ii,jj,kk) = abs(oeta(l))*arean

c            else
c               as(ii,jj,kk) = 0.0d0
c               su(ii,jj,kk) = su(ii,jj,kk) + 
c     &              (abs(oeta(l))*arean) *
c     &         PI1 * (SIG * abs(fraction(bands)) *
c     Empirical model for emissivity at boundaries
c     &              1.0d0*
c     &              tg(ii,jj-1,kk)**4)
c            end if
         else
c            if (pcell(ii,jj+1,kk).eq.ffield) then

               as(ii,jj,kk) = abs(oeta(l))*arean

c            else
c               as(ii,jj,kk) = 0.0d0
c               su(ii,jj,kk) = su(ii,jj,kk) + 
c     &              (abs(oeta(l))*arean) *
c     &         PI1 * (SIG * abs(fraction(bands)) *
c     Empirical model for emissivity at boundaries
c     &              1.0d0*
c     &              tg(ii,jj+1,kk)**4)
c            end if
         endif

         if (plusZ) then
c            if (pcell(ii,jj,kk-1).eq.ffield) then

               ab(ii,jj,kk) = abs(oxi(l))*areatb

c            else
c               ab(ii,jj,kk) = 0.0d0
c               su(ii,jj,kk) = su(ii,jj,kk) + 
c     &              (abs(oxi(l))*areatb) *
c     &         PI1 * (SIG * abs(fraction(bands)) *
c     Empirical model for emissivity at boundaries
c     &              1.0d0*
c     &              tg(ii,jj,kk-1)**4)
c            end if
         else
c            if (pcell(ii,jj,kk+1).eq.ffield) then

               ab(ii,jj,kk) = abs(oxi(l))*areatb

c            else
c               ab(ii,jj,kk) = 0.0d0
c               su(ii,jj,kk) = su(ii,jj,kk) + 
c     &              (abs(oxi(l))*areatb) * 
c     Empirical model for emissivity at boundaries
c     &              1.0d0*
c     &         PI1 * (SIG * abs(fraction(bands)) *
c     &              tg(ii,jj,kk+1)**4)
c            end if
         endif
         
         ap(ii,jj,kk) =  abs(omu(l))*areaew + abs(oeta(l))*arean + 
     &        abs(oxi(l))*areatb + abskg(ii,jj,kk)*
     &        volume(ii,jj,kk)

c      endif
10    continue
20    continue
30    continue
         
      RETURN
      end










