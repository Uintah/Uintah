c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/computeVel_fort.h>
C---------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     THIS SUBROUTINE WILL add pressure gradient term for momentum eqn
C---------------------------------------------------------------------

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer i, j, k, idf
      double precision dene, old_uu, old_dene
      do 120 k =idxLoU(3), idxHiU(3)
         do 110 j = idxLoU(2), idxHiU(2)
            do 100 i = idxLoU(1), idxHiU(1)
               idf = ioff*i+joff*j+koff*k
               dene = 0.5*(den(i-ioff,j-joff,k-koff)+den(i,j,k))
               old_dene = 0.5*(old_den(i-ioff,j-joff,k-koff)+
     $                         old_den(i,j,k))
               old_uu = uu(i,j,k)
               UU(I,J,K) = ((deltat*(PRESS(I-IOFF,J-JOFF,K-KOFF)-
     $                               PRESS(I,J,K))/DXPW(idf))+
     $                       old_uu*old_dene)/old_dene
 100        continue
 110     continue
 120  continue
      RETURN
      END

