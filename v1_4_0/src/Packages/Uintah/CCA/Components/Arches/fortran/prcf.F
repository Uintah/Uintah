c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/prcf_fort.h>
#else
      SUBROUTINE prcf(indexLow, indexHigh, PCELL,
     &     FF,
     &     DEN, DENP,
     &     PDEN,
     &     DENSF0, DENSF1,
     &     LMODBC, LCALP,
     &     ierr)
C~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
C
C    This subroutine calculates gas phase properties from the cardinal
C    variables, for cold flow calculations 
C
C~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      implicit none

#include "param4.h"
#include "pcell.h"

c*********************************************************************

c     Arguments :
c*********************************************************************
      integer indexLow(3), indexHigh(3)
      integer PCELL(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision FF(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision DEN(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision DENP(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision PDEN, DENSF0, DENSF1
      logical LMODBC, LCALP
      integer ierr

c*********************************************************************
c     Locals :
c*********************************************************************
      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k
      double precision EPSG
      double precision FF1, FM1

c*********************************************************************
c     Start :
c*********************************************************************
C-----------------------------------------------------------------------
c
c      FLOWT = FLOWPR + FLOWSC
c      IF (FLOWT.EQ.ZERO) THEN
c        FF = ZERO
c      ELSE
c        FF = (FLOWPR*FPR + FLOWSC*FSC)/FLOWT
c      END IF
c
      IF (LMODBC) THEN
         IST = indexLow(1)
         JST = indexLow(2)
         KST = indexLow(3)
         IEND = indexHigh(1)
         JEND = indexHigh(2)
         KEND = indexHigh(3)
      ELSE
         IST = indexLow(1)+1
         JST = indexLow(2)+1
         KST = indexLow(3)+1
         IEND = indexHigh(1)-1
         JEND = indexHigh(2)-1
         KEND = indexHigh(3)-1
      ENDIF         
      DO 140 K = KST,KEND
         DO 130 J = JST,JEND
            DO 120 I = IST,IEND
               IF (PCELL(I,J,K).EQ.WALL) THEN
C-----------------------------------------------------------------------
C             Node is in a wall.
C-----------------------------------------------------------------------
               ELSE
C-----------------------------------------------------------------------
C             Node is in computational domain.
C-----------------------------------------------------------------------
c              IF (LCALF) FF = F(I,J,K)
                  FF1 = FF(I,J,K)
                  FM1 = ONE-FF1
c
                  IF (LCALP) THEN
                     CALL CALEPS(indexLow, indexHigh,
     &                    DENP,PDEN,I,J,K,EPSG)
                  ELSE
                     EPSG = ONE
                  ENDIF
C-----------------------------------------------------------------------
C               Calculate and underrelax gas density.
C-----------------------------------------------------------------------
                  DEN(I,J,K) = EPSG/(FF1/DENSF1+FM1/DENSF0)

C                  IF (PCELL(I,J,K) .EQ. PFIELD) DEN(I,J,K) = DENPR
C                  IF (PCELL(I,J,K) .EQ. SFIELD) DEN(I,J,K) = DENSC

               END IF
 120        CONTINUE
 130     CONTINUE
 140  CONTINUE
      RETURN
      END

c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:49  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
