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
      integer iBegGhost, jBegGhost, kBegGhost
      integer iEndGhost, jEndGhost, kEndGhost
      integer IST, JST, KST, IEND, JEND, KEND
      integer I, J, K

c*********************************************************************
c     Start :
c*********************************************************************
      iBegGhost = domainLow(1)
      jBegGhost = domainLow(2)
      kBegGhost = domainLow(3)
      iEndGhost = domainHigh(1)
      jEndGhost = domainHigh(2)
      kEndGhost = domainHigh(3)
      IST = indexLow(1)
      JST = indexLow(2)
      KST = indexLow(3)
      IEND = indexHigh(1)
      JEND = indexHigh(2)
      KEND = indexHigh(3)

      DO 200 K = kBegGhost,kEndGhost
         DO 190 J = jBegGhost,jEndGhost
            DO 180 I = iBegGhost,iEndGhost
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
c Revision 1.2  2000/06/30 22:41:19  bbanerje
c Corrected behavior of profv and profscalar
c
c Revision 1.1  2000/06/20 20:42:38  rawat
c added some more boundary stuff and modified interface to IntVector. Before
c compiling the code you need to update /SCICore/Geometry/IntVector.h
c
c Revision 1.1  2000/04/12 20:31:49  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
