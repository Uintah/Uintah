c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE profscalar(domainLow, domainHigh,
     &     indexLow, indexHigh,
     &     SCALAR, PCELL,
     &     SVAL, PFIELD,
     &     ierr)
C---------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     THIS SUBROUTINE WILL ASSIGN FLAT VELOCITY PROFILES
C     AT THE APPROPRIATE BOUNDARY CONDITIONS
C     SK, 3/25/95: also sets flat profiles for density here
C     (instead of in profp.f - because if lrdpr density profiles are
C     also read in.)
C---------------------------------------------------------------------

      implicit none

#include "param4.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer domainLow(3)
      integer domainHigh(3)
      integer indexLow(3)
      integer indexHigh(3)
      integer PCELL(domainLow(1):domainHigh(1), 
     &     domainLow(2):domainHigh(2), 
     &     domainLow(3):domainHigh(3))
      double precision SCALAR(domainLow(1):domainHigh(1), 
     &     domainLow(2):domainHigh(2), domainLow(3):domainHigh(3))
      double precision SVAL
      integer PFIELD
      integer ierr

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer iBegGhost, jBegGhost, kBegGhost, NI, NJ, NK
      integer IST, JST, KST, IEND, JEND, KEND
      integer I, J, K

c*********************************************************************
c     Start :
c*********************************************************************
      iBegGhost = indexLow(1)
      jBegGhost = indexLow(2)
      kBegGhost = indexLow(3)
      NI = indexHigh(1)
      NJ = indexHigh(2)
      NK = indexHigh(3)
      IST = iBegGhost+1
      JST = jBegGhost+1
      KST = kBegGhost+1
      IEND = NI-1
      JEND = NJ-1
      KEND = NK-1

      DO 200 K = kBegGhost,NK
         DO 190 J = jBegGhost,NJ
            DO 180 I = iBegGhost, NI
               IF (PCELL(I,J,K).EQ.PFIELD) THEN
                  SCALAR(I,J,K) = SVAL
               END IF
 180        CONTINUE
 190     CONTINUE
 200  CONTINUE

      RETURN
      END

c*********************************************************************
c
c $Log$
c Revision 1.1  2000/06/20 20:42:38  rawat
c added some more boundary stuff and modified interface to IntVector. Before
c compiling the code you need to update /SCICore/Geometry/IntVector.h
c
c Revision 1.1  2000/04/12 20:31:49  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
