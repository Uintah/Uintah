c
c
#include <Packages/Uintah/CCA/Components/Arches/fortran/bcscalar_fort.h>
#include "param4.h"
C-----------------------------------------------------------------------
C     THIS SUBROUTINE CALCULATES THE BOUNDARY CONDITONS FOR F
C-----------------------------------------------------------------------

*
*     declare local variables
*
      integer i, j, k
      

*-----------------------------------------------------------------------

*-----corrections for interior intrusions-------------------------------

      do 305 k=idxLo(3),idxHi(3)
         do 205 j=idxLo(2),idxHi(2)
            do 105 i=idxLo(1),idxHi(1)
               if (pcell(i,j,k).eq.wall) then
                call fixval(zero,ae(i,j,k),aw(i,j,k),an(i,j,k),
     &                 as(i,j,k),at(i,j,k),ab(i,j,k),su(i,j,k),
     &                 sp(i,j,k),one)
                call fixval(zero,cesav(i,j,k),cwsav(i,j,k),cnsav(i,j,k),
     &                 cssav(i,j,k),ctsav(i,j,k),cbsav(i,j,k),su(i,j,k),
     &                 sp(i,j,k),one)
                call fixval(zero,decoef(i,j,k),dwcoef(i,j,k),
     &                 dncoef(i,j,k), dscoef(i,j,k),dtcoef(i,j,k),
     &                 dbcoef(i,j,k),su(i,j,k),sp(i,j,k),one)
               end if
 105        continue
 205     continue
 305  continue

      call wallbc(idxLo, idxHi,
     & PASS3A(ae)
     & PASS3B(ae),
     & PASS3A(aw)
     & PASS3B(aw),
     & PASS3A(an)
     & PASS3B(an),
     & PASS3A(as)
     & PASS3B(as),
     & PASS3A(at)
     & PASS3B(at),
     & PASS3A(ab)
     & PASS3B(ab),
     & PASS3A(pcell)
     & PASS3B(pcell),
     & wall, xminus, xplus, yminus, yplus, zminus, zplus)
      call wallbc(idxLo, idxHi,
     & PASS3A(cesav)
     & PASS3B(cesav),
     & PASS3A(cwsav)
     & PASS3B(cwsav),
     & PASS3A(cnsav)
     & PASS3B(cnsav),
     & PASS3A(cssav)
     & PASS3B(cssav),
     & PASS3A(ctsav)
     & PASS3B(ctsav),
     & PASS3A(cbsav)
     & PASS3B(cbsav),
     & PASS3A(pcell)
     & PASS3B(pcell),
     & wall, xminus, xplus, yminus, yplus, zminus, zplus)
c set bc's for diffusion coeff's required for divergence constraint
      call wallbc(idxLo, idxHi, 
     & PASS3A(decoef)
     & PASS3B(decoef),
     & PASS3A(dwcoef)
     & PASS3B(dwcoef),
     & PASS3A(dncoef)
     & PASS3B(dncoef),
     & PASS3A(dscoef)
     & PASS3B(dscoef),
     & PASS3A(dtcoef)
     & PASS3B(dtcoef),
     & PASS3A(dbcoef)
     & PASS3B(dbcoef),
     & PASS3A(pcell)
     & PASS3B(pcell),
     & wall, xminus, xplus, yminus, yplus, zminus, zplus)

      return
      end
