c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/mmmomsrc_fort.h>
C---------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     THIS SUBROUTINE WILL compute matrix coefficients for uvelocity
C---------------------------------------------------------------------

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer ISTU, JSTU, KSTU, IENDU, JENDU, KENDU
      integer i, j, k


c*********************************************************************
c     Start :
c*********************************************************************
      ISTU = idxLoU(1)
      JSTU = idxLoU(2)
      KSTU = idxLoU(3)
      IENDU = idxHiU(1)
      JENDU = idxHiU(2)
      KENDU = idxHiU(3)

c mm momentum sources

      do 120 k = KSTU,KENDU
         do 110 j = JSTU,JENDU
            do 100 i = ISTU,IENDU

#if 0
               if (abs(mmsu(i,j,k)).gt.0.0) then
                  print*, 'printing nonzero mmsu value ',i,j,k,
     $                 mmsu(i,j,k)
               endif
               if (abs(mmsp(i,j,k)).gt.0.0) then
                  print*, 'printing nonzero mmsp value ',i,j,k,
     $                 mmsp(i,j,k)
               endif
#endif

               SU(I,J,K) = SU(I,J,K)+ mmSU(i,j,k)
               SP(i,j,k) = SP(i,j,k)+ mmSP(i,j,k)

 100        continue
 110     continue
 120  continue
      RETURN
      END

