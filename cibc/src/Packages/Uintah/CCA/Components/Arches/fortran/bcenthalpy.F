c
c
#include <Packages/Uintah/CCA/Components/Arches/fortran/bcenthalpy_fort.h>
#include "param4.h"
C-----------------------------------------------------------------------
C     THIS SUBROUTINE CALCULATES THE BOUNDARY CONDITONS FOR F
C-----------------------------------------------------------------------

*
*     declare local variables
*
      double precision areaew, arean, areas, areatb
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

      return
      end
