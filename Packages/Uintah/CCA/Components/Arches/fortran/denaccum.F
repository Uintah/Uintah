c*********************************************************************
c
c
c*********************************************************************

      SUBROUTINE denaccum(domLoDen, domHiDen, 
     $     idxLo, idxHi, DEN, 
     $     domLoD_old, domHiD_old, old_den, 
     $     domLo, domHi, 
     $     daccumulate, deltat,
     $     SEW, SNS, STB,
     &     ierr)
C-----------------------------------------------------------------------
      implicit none

#include "param4.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer domLoDen(3), domHiDen(3)
      integer idxLo(3), idxHi(3)
      double precision DEN(domLoDen(1):domHiDen(1), 
     $     domLoDen(2):domHiDen(2), 
     &     domLoDen(3):domHiDen(3))
      integer domLoD_old(3), domHiD_old(3)
      double precision old_den(domLoD_old(1):domHiD_old(1), 
     $     domLoD_old(2):domHiD_old(2), 
     &     domLoD_old(3):domHiD_old(3))
      integer domLo(3), domHi(3)
      double precision SEW(domLo(1):domHi(1))
      double precision SNS(domLo(2):domHi(2))
      double precision STB(domLo(3):domHi(3))
      double precision daccumulate, deltat
      integer ierr

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k

c*********************************************************************
c     Start :
c*********************************************************************
      IST = idxLo(1)
      JST = idxLo(2)
      KST = idxLo(3)
      IEND = idxHi(1)
      JEND = idxHi(2)
      KEND = idxHi(3)
      DO 140 K=KST,KEND
         DO 130 J=JST,JEND
            DO 120 I=IST,IEND
               daccumulate = daccumulate + 
     $                       (den(i,j,k) - old_den(i,j,k))*
     $                       (sew(i)*sns(j)*stb(k))/deltat
 120           CONTINUE
 130        CONTINUE
 140     CONTINUE

      RETURN
      END

