#include <Packages/Uintah/CCA/Components/Arches/Radiation/fortran/rdomsolve_fort.h>

      integer i,j,k,l,ii,jj,kk

      double precision PI,PI1,SIG

      double precision volume(idxlo(1):idxhi(1),idxlo(2):idxhi(2),
     &idxlo(3):idxhi(3))
      double precision su(idxlo(1):idxhi(1),idxlo(2):idxhi(2),
     &idxlo(3):idxhi(3))
      double precision aw(idxlo(1):idxhi(1),idxlo(2):idxhi(2),
     &idxlo(3):idxhi(3))
      double precision as(idxlo(1):idxhi(1),idxlo(2):idxhi(2),
     &idxlo(3):idxhi(3))
      double precision ab(idxlo(1):idxhi(1),idxlo(2):idxhi(2),
     &idxlo(3):idxhi(3))
      double precision ap(idxlo(1):idxhi(1),idxlo(2):idxhi(2),
     &idxlo(3):idxhi(3))

      double precision areaew
      double precision arean(idxlo(1):idxhi(1))
      double precision areatb(idxlo(1):idxhi(1))

      DATA PI/3.141593D0/,SIG/5.67D-08/

      PI1 = 1.0d0 / PI

      do 30 k=idxlo(3),idxhi(3)
      do 20 j=idxlo(2),idxhi(2)
      do 10 i=idxlo(1),idxhi(1)

      if (omu(l).gt.0) then
         
         ii = i

      else

         ii = idxlo(1)+idxhi(1)-i
      end if

      if (oeta(l).gt.0) then

         jj = j 

      else

         jj = idxlo(2)+idxhi(2)-j

      end if

      if (oxi(l).gt.0) then
         
         kk = k

      else

         kk = idxlo(3)+idxhi(3)-k

      end if
      volume(ii,jj,kk) = sew(ii) * sns(jj) * stb(kk)
      su(ii,jj,kk) = esrcg(ii,jj,kk)*volume(ii,jj,kk)

      areaew = sns(jj)*stb(kk)
      arean(ii) = sew(ii)*stb(kk)                 
      areatb(ii) = sew(ii) * sns(jj)
      if (ii.eq.i) then
         if (pcell(ii-1,jj,kk).eq.ffield) then
            aw(ii,jj,kk) = abs(omu(l))*areaew
         else
            aw(ii,jj,kk) = 0.0d0
            su(ii,jj,kk) = su(ii,jj,kk) + (abs(omu(l))*areaew) *
     &           SIG * PI1 * abskg(ii-1,jj,kk) * 
     $           tg(ii-1,jj,kk)**4
         end if
      else
         if (pcell(ii+1,j,k).eq.ffield) then
            aw(ii,jj,kk) = abs(omu(l))*areaew
         else
            aw(ii,jj,kk) = 0.0d0
            su(ii,jj,kk) = su(ii,jj,kk) + (abs(omu(l)) * areaew) * 
     &           SIG * PI1 * abskg(ii+1,jj,kk) * 
     $           tg(ii+1,jj,kk)**4
         end if
      endif
      if (jj.eq.j) then
         if (pcell(ii,jj-1,kk).eq.ffield) then
            as(ii,jj,kk) = abs(oeta(l))*arean(ii)
         else
            as(ii,jj,kk) = 0.0d0
            su(ii,jj,kk) = su(ii,jj,kk) + 
     $           (abs(oeta(l))*arean(ii)) *
     &           SIG * PI1 * abskg(ii,jj-1,kk) *
     $           tg(ii,jj-1,kk)**4
         end if
      else
         if (pcell(ii,jj+1,kk).eq.ffield) then
            as(ii,jj,kk) = abs(oeta(l))*arean(ii)
         else
            as(ii,jj,kk) = 0.0d0
            su(ii,jj,kk) = su(ii,jj,kk) + 
     $           (abs(oeta(l))*arean(ii)) *
     &           SIG * PI1 * abskg(ii,jj+1,kk) * 
     $           tg(ii,jj+1,kk)**4
         end if
      endif
      if (kk.eq.k) then
         if (pcell(ii,jj,kk-1).eq.ffield) then
            ab(ii,jj,kk) = abs(oxi(l))*areatb(ii)
         else
            ab(ii,jj,kk) = 0.0d0
            su(ii,jj,kk) = su(ii,jj,kk) + 
     $           (abs(oxi(l))*areatb(ii)) *
     &           SIG * PI1 * abskg(ii,jj,kk-1) *
     $           tg(ii,jj,kk-1)**4
         end if
      else
         if (pcell(ii,jj,kk+1).eq.ffield) then
            ab(ii,jj,kk) = abs(oxi(l))*areatb(ii)
         else
            ab(ii,jj,kk) = 0.0d0
            su(ii,jj,kk) = su(ii,jj,kk) + 
     $           (abs(oxi(l))*areatb(ii)) * 
     &           SIG * PI1 * abskg(ii,jj,kk+1) * 
     $           tg(ii,jj,kk+1)**4
         end if
      endif

      ap(ii,jj,kk) =  abs(omu(l))*areaew + abs(oeta(l))*arean(ii) + 
     &             abs(oxi(l))*areatb(ii) + abskg(ii,jj,kk)*
     &volume(ii,jj,kk)

      cenint(ii,jj,kk) = (aw(ii,jj,kk)*cenint(ii,jj,kk) +
     &                    as(ii,jj,kk)*cenint(ii,jj,kk) +    
     &                    ab(ii,jj,kk)*cenint(ii,jj,kk) +
     &                    su(ii,jj,kk))/ap(ii,jj,kk)

c      if((jj.eq.7).and.(kk.eq.7))write(6,*)cenint(ii,jj,kk)

10    continue
20    continue
30    continue

      RETURN
      end






