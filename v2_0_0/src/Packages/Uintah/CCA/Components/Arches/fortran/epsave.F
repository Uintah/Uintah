c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/epsave_fort.h>
#else
      SUBROUTINE epsave(indexLow, indexHigh, 
     &     DENP, PDEN, 
     &     I, J, K, IOFF, JOFF, KOFF,
     &     EPS,
     &     ierr)

C-----------------------------------------------------------------------
C     WRITTEN BY SESHADRI KUMAR, OCTOBER 15, 1996
C
C     epsave CALCULATES THE VOID FRACTION AT TWO POINTS, (I,J,K) AND
C     (I-IOFF,J-JOFF,K-KOFF), AND RETURNS THEIR MEAN, EPS
C-----------------------------------------------------------------------

      implicit none

#include "param4.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer indexLow(3)
      integer indexHigh(3)
      double precision DENP(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision PDEN
      integer I, J, K, IOFF, JOFF, KOFF
      double precision EPS
      integer ierr

c*********************************************************************
c     Local Variables :
c*********************************************************************
      double precision EPSP, EPSOFF

c*********************************************************************
c     Initialize Local Variables :
c*********************************************************************
      EPSP = 0.0
      EPSOFF = 0.0

C-----------------------------------------------------------------------
      CALL CALEPS(indexLow, indexHigh,DENP,PDEN,I,J,K,EPSP)
      CALL CALEPS(indexLow, indexHigh,DENP,PDEN,I-IOFF,J-JOFF,K-KOFF,
     &     EPSOFF)
      EPS = PT5*(EPSP + EPSOFF)
      RETURN
      END

c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:44  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
