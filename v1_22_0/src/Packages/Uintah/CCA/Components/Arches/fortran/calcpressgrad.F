c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/calcpressgrad_fort.h>
C---------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     THIS SUBROUTINE WILL calc pressure gradient term for momentum eqn
C---------------------------------------------------------------------

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer i, j, k, idf

      double precision VOL, APO
      do 120 k =idxLoU(3), idxHiU(3)
         do 110 j = idxLoU(2), idxHiU(2)
            do 100 i = idxLoU(1), idxHiU(1)
               VOL = SEWU(I)*SNS(J)*STB(K)
               idf  = i*ioff + j*joff + k*koff
               gradP(I,J,K) = (PRESS(I-IOFF,J-JOFF,K-KOFF)-
     $              PRESS(I,J,K))*VOL/DXPW(idf)
 100        continue
 110     continue
 120  continue
      RETURN
      END

