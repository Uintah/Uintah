c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/arrmax_fort.h>
#else
      SUBROUTINE arrmax(indexLow, indexHigh, 
     &     A,ARMAX)
C-----------------------------------------------------------------------
C     WRITTEN BY SESHADRI KUMAR, May 29, 1996
C
C     Calculates the absolute maximum element of a 3-D array A
C
C     This subroutine assumes that A is dimensioned (NX,NY,NZ).
C     (IST,JST,KST) and (IEND,JEND,KEND) are the starting and ending 
C     locations within which the operation is to be performed.
C     
C-----------------------------------------------------------------------
      implicit none

#include "param4.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer indexLow(3)
      integer indexHigh(3)
      double precision A(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision ARMAX

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer IST,JST,KST,IEND,JEND,KEND

      integer i,j,k

c*********************************************************************
c     Local Variables :
c*********************************************************************
      IST = indexLow(1)+1
      JST = indexLow(2)+1
      KST = indexLow(3)+1
      IEND = indexHigh(1)+1
      JEND = indexHigh(2)+1
      KEND = indexHigh(3)+1

C-----------------------------------------------------------------------
C     EXECUTABLE STATEMENTS
C-----------------------------------------------------------------------
      ARMAX = TINY
      DO 300 K=KST,KEND
         DO 200 J=JST,JEND
            DO 100 I=IST,IEND
               ARMAX = MAX(ABS(A(I,J,K)),ARMAX)
 100        CONTINUE
 200     CONTINUE
 300  CONTINUE
      RETURN
      END

c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:42  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************

