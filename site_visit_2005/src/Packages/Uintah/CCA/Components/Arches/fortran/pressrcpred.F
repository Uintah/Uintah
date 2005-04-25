c
c
#include <Packages/Uintah/CCA/Components/Arches/fortran/pressrcpred_fort.h>
#include "param4.h"

      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k
      double precision dene, denw, denn, dens, dent, denb
      double precision ce, cw, cn, cs, ct, cb
      double precision arean, areas, areaew, areatb, vol
      double precision apo, smp

      integer imax
      integer jmax
      integer kmax
      double precision apomax
      double precision sumax
      integer isumax
      integer jsumax
      integer ksumax

      logical isnan_fort

c-----------------------------------------------------------------------
c     executable statements
c-----------------------------------------------------------------------

      IST = idxLo(1)
      JST = idxLo(2)
      KST = idxLo(3)
      IEND = idxHi(1)
      JEND = idxHi(2)
      KEND = idxHi(3)

      imax = ist
      jmax = jst
      kmax = kst
      apomax = zero

      isumax = ist
      jsumax = jst
      ksumax = kst
      sumax = zero

c-----------------------------------------------------------------------

      do 120 k = KST, KEND
         do 110 j = JST, JEND
            do 100 i = IST, IEND
               arean = sew(i)*stb(k)
               areas = sew(i)*stb(k)
               areaew = sns(j)*stb(k)
               areatb = sns(j)*sew(i)
               vol = sns(j)*sew(i)*stb(k)
               dene = pt5*(den(i+1,j,k)+den(i,j,k))
               denw = pt5*(den(i-1,j,k)+den(i,j,k))
               denn = pt5*(den(i,j+1,k)+den(i,j,k))
               dens = pt5*(den(i,j-1,k)+den(i,j,k))
               dent = pt5*(den(i,j,k+1)+den(i,j,k))
               denb = pt5*(den(i,j,k-1)+den(i,j,k))
               ce = areaew*dene
               cw = areaew*denw
               cn = arean*denn
               cs = areas*dens
               ct = areatb*dent
               cb = areatb*denb
c               apo = (4.0*den_old(i,j,k) - 3.0*den(i,j,k) - 
c     $                den_old_old(i,j,k))*vol/(2.*delta_t)
c               apo = (den_old(i,j,k) - den(i,j,k))
c     $                 *vol/delta_t
               apo = 0.0D0

#if 0
                  if (abs(apo).gt.abs(apomax)) then
                     apomax = abs(apo)
                     imax = i
                     jmax = j
                     kmax = k
                  endif

#endif
               smp = (cn*vhat(i,j+1,k)-cs*vhat(i,j,k)+
     $                ce*uhat(i+1,j,k)-cw*uhat(i,j,k)+
     $                ct*what(i,j,k+1)-cb*what(i,j,k) - apo)/delta_t
c    $              -sign*spm(i,j,k)
               su(i,j,k)= -smp
#if 0
               if (i.eq.3 .and. j.eq.9 .and. k.eq.8) then
                  print*, 'ce = ',ce
                  print*, 'cw = ',cw
                  print*, 'cn = ',cn
                  print*, 'cs = ',cs
                  print*, 'ct = ',ct
                  print*, 'cb = ',cb
                  print*, 'uhate = ',uhat(i+1,j,k)
                  print*, 'uhatw = ',uhat(i,j,k)
                  print*, 'vhatn = ',vhat(i,j+1,k)
                  print*, 'vhats = ',vhat(i,j,k)
                  print*, 'whatt = ',what(i,j,k+1)
                  print*, 'whatb = ',what(i,j,k)
               endif

               if (abs(su(i,j,k)).gt.abs(sumax)) then
                  sumax = abs(su(i,j,k))
                  isumax = i
                  jsumax = j
                  ksumax = k
               endif

               if (isnan_fort(su(i,j,k))) then
                  print*, 'found a nan in source at ',i,j,k
               endif
#endif

 100        continue
 110     continue
 120  continue

      return
      end

      logical function isnan_fort(a)
      double precision a
      
      if ((a-a).lt.100000) then
         isnan_fort = .false.
      else
         isnan_fort = .true.
      endif

      return
      end

