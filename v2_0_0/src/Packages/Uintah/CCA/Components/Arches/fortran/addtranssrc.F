c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/addtranssrc_fort.h>
C---------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     THIS SUBROUTINE WILL add pressure gradient term for momentum eqn
C---------------------------------------------------------------------

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer i, j, k, idf

      double precision VOL, APO
      logical*1 lexpl
      lexpl = .true.
      do 120 k =idxLoU(3), idxHiU(3)
         do 110 j = idxLoU(2), idxHiU(2)
            do 100 i = idxLoU(1), idxHiU(1)
               VOL = SEWU(I)*SNS(J)*STB(K)
               APO =  old_den(I,J,K)*VOL/deltat
               SU(I,J,K) = SU(I,J,K) + APO*UU(I,J,K)
               if (.not.lexpl) AP(I,J,K) = AP(I,J,K) + APO
 100        continue
 110     continue
 120  continue
      RETURN
      END

