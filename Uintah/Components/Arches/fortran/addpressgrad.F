c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE addpressgrad(domLoU, domHiU, idxLoU, idxHiU, UU, 
     &     SU, AP,
     &     domLo, domHi, press, old_den,
     &     deltat, ioff, joff, koff,
     &     SEWU, SNS, STB,
     &     DXPW, 
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
      integer domLoU(3), domHiU(3), idxLoU(3), idxHiU(3)
      double precision UU(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision SU(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision AP(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      integer domLo(3), domHi(3)
      double precision PRESS(domLo(1):domHi(1), domLo(2):domHi(2), 
     &     domLo(3):domHi(3))
      double precision old_DEN(domLo(1):domHi(1), domLo(2):domHi(2), 
     &     domLo(3):domHi(3))
      double precision DELTAT
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
      logical ltim
      ltim = .true.
      do 120 k =idxLoU(3), idxHiU(3)
         do 110 j = idxLoU(2), idxLoU(2)
            do 100 i = idxLoU(1), idxLoU(1)
               VOL = SEWU(I)*SNS(J)*STB(K)
               idf  = i*ioff + j*joff + k*koff
               SU(I,J,K) = SU(I,J,K)+(PRESS(I-IOFF,J-JOFF,K-KOFF)-
     $              PRESS(I,J,K))*VOL/DXPW(idf)
               if (ltim) then
                  APO =  old_den(I,J,K)*VOL/deltat
                  SU(I,J,K) = SU(I,J,K) + APO*UU(I,J,K)
                  AP(I,J,K) = AP(I,J,K) + APO
               end if
 100        continue
 110     continue
 120  continue
      RETURN
      END

c*********************************************************************
c
c $Log$
c Revision 1.2  2000/07/28 02:31:04  rawat
c moved all the labels in ArchesLabel. fixed some bugs and added matrix_dw to store matrix
c coeffecients
c
c Revision 1.1  2000/07/17 22:07:02  rawat
c modified momentum source
c
c
c*********************************************************************
