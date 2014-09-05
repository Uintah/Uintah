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

c-----------------------------------------------------------------------
c     executable statements
c-----------------------------------------------------------------------

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
c drho/dt is added in C code
               apo = 0.0D0

               smp = (cn*vhat(i,j+1,k)-cs*vhat(i,j,k)+
     $                ce*uhat(i+1,j,k)-cw*uhat(i,j,k)+
     $                ct*what(i,j,k+1)-cb*what(i,j,k) - apo)/delta_t
               su(i,j,k)= -smp

 100        continue
 110     continue
 120  continue

      return
      end

