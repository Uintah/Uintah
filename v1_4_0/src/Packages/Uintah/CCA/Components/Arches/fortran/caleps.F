c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/caleps_fort.h>
#else
      SUBROUTINE caleps(indexLow, indexHigh, DENP,PDEN,I,J,K,EPSG)
C-----------------------------------------------------------------------
C     WRITTEN BY SESHADRI KUMAR, OCTOBER 15, 1996
C
C     This subroutine back-calculates the value of the void fraction
C     EPSG at any location (I,J,K) of the P-cell array
C-----------------------------------------------------------------------
      implicit none

#include "param4.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer indexLow(3), indexHigh(3)
      double precision 
     $     DENP(indexLow(1):indexHigh(1),indexLow(2):indexHigh(2),
     &     indexLow(3):indexHigh(3))
      double precision pden,epsg
      integer i,j,k

c*********************************************************************
c     Locals :
c*********************************************************************
      double precision pnu

C-----------------------------------------------------------------------
      PNU = DENP(I,J,K)/PDEN
      EPSG = ONE - PNU
      RETURN
      END


c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:44  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
