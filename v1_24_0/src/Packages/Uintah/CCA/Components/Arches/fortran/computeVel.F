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
      double precision denaverage
      do 120 k =idxLoU(3), idxHiU(3)
         do 110 j = idxLoU(2), idxHiU(2)
            do 100 i = idxLoU(1), idxHiU(1)
               idf = ioff*i+joff*j+koff*k
               denaverage = 0.5D0*(den(i-ioff,j-joff,k-koff)+
     +                             den(i,j,k))
               UU(I,J,K) = deltat*(PRESS(I-IOFF,J-JOFF,K-KOFF)-
     -                      PRESS(I,J,K))/DXPW(idf)/
     /                      denaverage+uu(i,j,k)
 100        continue
 110     continue
 120  continue
      RETURN
      END

