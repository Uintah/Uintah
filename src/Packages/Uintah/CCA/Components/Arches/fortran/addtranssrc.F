c*********************************************************************
c
c
c*********************************************************************

      SUBROUTINE addtranssrc(domLoU, domHiU, domLoUng, domHiUng,
     $     idxLoU, idxHiU, UU, 
     &     SU, AP,
     &     domLo, domHi,domLong, domHing, old_den,
     &     deltat,
     &     SEWU, SNS, STB,
     &     ierr)
C---------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     THIS SUBROUTINE WILL add pressure gradient term for momentum eqn
C---------------------------------------------------------------------

      implicit none

#include "param4.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer domLoU(3), domHiU(3), domLoUng(3), domHiUng(3)
      integer idxLoU(3), idxHiU(3)
      double precision UU(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision SU(domLoUng(1):domHiUng(1), 
     $     domLoUng(2):domHiUng(2), 
     &     domLoUng(3):domHiUng(3))
      double precision AP(domLoUng(1):domHiUng(1),
     $     domLoUng(2):domHiUng(2), 
     &     domLoUng(3):domHiUng(3))
      integer domLo(3), domHi(3)
      integer domLong(3), domHing(3)
      double precision old_DEN(domLong(1):domHing(1), 
     $     domLong(2):domHing(2), 
     &     domLong(3):domHing(3))
      double precision DELTAT
      double precision SEWU(domLo(1):domHi(1))
      double precision SNS(domLo(2):domHi(2))
      double precision STB(domLo(3):domHi(3))

      integer ierr

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer i, j, k, idf

      double precision VOL, APO
      logical lexpl
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

