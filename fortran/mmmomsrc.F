c*********************************************************************
c
c
c*********************************************************************

      SUBROUTINE mmmomsrc(domLoUng, domHiUng,
     $     idxLoU, idxHiU,
     &     SU, SP,
     &     domLo, domHi,
     $     mmSU, mmSP,
     &     ierr)
C---------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     THIS SUBROUTINE WILL compute matrix coefficients for uvelocity
C---------------------------------------------------------------------

      implicit none

#include "param4.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer domLoUng(3), domHiUng(3)
      integer idxLoU(3), idxHiU(3)
      double precision SU(domLoUng(1):domHiUng(1), 
     $     domLoUng(2):domHiUng(2), 
     &     domLoUng(3):domHiUng(3))
      double precision SP(domLoUng(1):domHiUng(1), 
     $     domLoUng(2):domHiUng(2), 
     &     domLoUng(3):domHiUng(3))
      integer domLo(3), domHi(3)
      double precision mmSU(domLo(1):domHi(1), 
     $     domLo(2):domHi(2), 
     &     domLo(3):domHi(3))
      double precision mmSP(domLo(1):domHi(1), 
     $     domLo(2):domHi(2), 
     &     domLo(3):domHi(3))
      integer ierr

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
               SU(I,J,K) = SU(I,J,K)+ mmSU(i,j,k)
               SP(i,j,k) = SP(i,j,k)+ mmSP(i,j,k)
 100        continue
 110     continue
 120  continue
      RETURN
      END

