c*********************************************************************
c
c
c*********************************************************************

      SUBROUTINE computeVel(domLoU, domHiU, 
     $     idxLoU, idxHiU, UU, 
     &     domLo, domHi, press,
     &     domLoDen, domHiDen, Den,
     &     domLoDenO, domHiDenO, old_Den,
     &     deltat, ioff, joff, koff,
     $     domLoGeom, domHiGeom,
     $     dxpw,
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
      integer domLoU(3), domHiU(3)
      integer idxLoU(3), idxHiU(3)
      double precision UU(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      integer domLo(3), domHi(3)
      double precision PRESS(domLo(1):domHi(1), domLo(2):domHi(2), 
     &     domLo(3):domHi(3))
      integer domLoDen(3), domHiDen(3)
      double precision Den(domLoDen(1):domHiDen(1), 
     $     domLoDen(2):domHiDen(2), 
     &     domLoDen(3):domHiDen(3))
      integer domLoDenO(3), domHiDenO(3)
      double precision old_Den(domLoDenO(1):domHiDenO(1), 
     $     domLoDenO(2):domHiDenO(2), 
     &     domLoDenO(3):domHiDenO(3))
      double precision DELTAT
      integer ioff, joff, koff
      integer domLoGeom(3), domHiGeom(3)
      integer ioff, joff, koff
      double precision DXPW(domLoGeom(1*ioff+2*joff+3*koff):
     $                      domHiGeom(1*ioff+2*joff+3*koff))

      integer ierr

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

