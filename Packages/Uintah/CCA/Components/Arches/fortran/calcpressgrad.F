c*********************************************************************
c
c
c*********************************************************************

      SUBROUTINE calcpressgrad(domLoU, domHiU, domLoUng, domHiUng,
     $     idxLoU, idxHiU, UU, 
     &     gradP,
     &     domLo, domHi, press,
     &     ioff, joff, koff,
     &     SEWU, SNS, STB,
     &     DXPW, 
     &     ierr)
C---------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     THIS SUBROUTINE WILL calc pressure gradient term for momentum eqn
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
      double precision gradP(domLoUng(1):domHiUng(1), 
     $     domLoUng(2):domHiUng(2), 
     &     domLoUng(3):domHiUng(3))
      integer domLo(3), domHi(3)
      double precision PRESS(domLo(1):domHi(1), domLo(2):domHi(2), 
     &     domLo(3):domHi(3))
      integer ioff, joff, koff
      double precision SEWU(domLo(1):domHi(1))
      double precision SNS(domLo(2):domHi(2))
      double precision STB(domLo(3):domHi(3))
      double precision DXPW(domLo(1*ioff+2*joff+3*koff):
     $                      domHi(1*ioff+2*joff+3*koff))

      integer ierr

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

