c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE profv(domainLow, domainHigh, indexLow, indexHigh,
     &     UU,VV,WW, PCELL,
     &     AREAPR, PFIELD, FLOWPR, DENPR,
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
      double precision UU(domainLow(1):domainHigh(1), 
     &     domainLow(2):domainHigh(2), domainLow(3):domainHigh(3))
      double precision VV(domainLow(1):domainHigh(1), 
     &     domainLow(2):domainHigh(2), domainLow(3):domainHigh(3))
      double precision WW(domainLow(1):domainHigh(1), 
     &     domainLow(2):domainHigh(2), domainLow(3):domainHigh(3))
      double precision FLOWPR, DENPR
      double precision AREAPR
      integer PFIELD
      integer ierr

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer iBegGhost, jBegGhost, kBegGhost, NI, NJ, NK
      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k
      double precision FLOW, RHO, AVGVELP, AVGVELS
      double precision YR, ZR, YVR, ZWR

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

C-------------------------------------------------------------------- 
C     CALCULATION OF U IN PRIMARY STREAMS
C---------------------------------------------------------------------
      FLOW = FLOWPR
      RHO = DENPR
      IF (AREAPR.le.TINY) THEN
        AVGVELP = ZERO
        FLOWPR = ZERO
      ELSE
        AVGVELP = FLOW/(RHO*AREAPR)
      END IF
c
c     debug to try pure diffusion case: if we don't do this, then
c     since den = 0.0, avgvelp = nan.
c     Commented for general case, 04/23/95, SK
c
c      AVGVELP = 0.1
C--------------------------------------------------------------------
C     EAST AND/OR WEST INLETS
C--------------------------------------------------------------------
      DO 200 K = kBegGhost,NK
        DO 190 J = jBegGhost,NJ
          IF (PCELL(iBegGhost,J,K).EQ.PFIELD) THEN
            UU(iBegGhost,J,K) = AVGVELP
            UU(IST,J,K) = AVGVELP
            VV(iBegGhost,J,K) = ZERO
            WW(iBegGhost,J,K) = ZERO
          END IF
          IF (PCELL(NI,J,K).EQ.PFIELD) THEN
            UU(NI,J,K) = -AVGVELP
c            UINLT(J,K) = -AVGVELP
c            UU(NI+1,J,K) = -AVGVELP
            VV(NI,J,K) = ZERO
            WW(NI,J,K) = ZERO
          END IF
  190   CONTINUE
  200 CONTINUE
C--------------------------------------------------------------------
C     NORTH AND/OR SOUTH INLETS
C--------------------------------------------------------------------
      DO 230 K = kBegGhost,NK
        DO 220 I = iBegGhost,NI
          IF (PCELL(I,jBegGhost,K).EQ.PFIELD) THEN
            VV(I,jBegGhost,K) = AVGVELP
            VV(I,JST,K) = AVGVELP
            WW(I,jBegGhost,K) = ZERO
            UU(I,jBegGhost,K) = ZERO
          END IF
          IF (PCELL(I,NJ,K).EQ.PFIELD) THEN
            VV(I,NJ,K) = -AVGVELP
c            VINLT(I,K) = -AVGVELP
c            VV(I,NJ+1,K) = -AVGVELP
            WW(I,NJ,K) = ZERO
            UU(I,NJ,K) = ZERO
          END IF
  220   CONTINUE
  230 CONTINUE
C--------------------------------------------------------------------
C     TOP AND/OR BOTTOM INLETS
C--------------------------------------------------------------------
      DO 260 J = jBegGhost,NJ
        DO 250 I = iBegGhost,NI
          IF (PCELL(I,J,kBegGhost).EQ.PFIELD) THEN
            WW(I,J,kBegGhost) = AVGVELP
            WW(I,J,KST) = AVGVELP
            VV(I,J,kBegGhost) = ZERO
            UU(I,J,kBegGhost) = ZERO
          END IF
          IF (PCELL(I,J,NK).EQ.PFIELD) THEN
            WW(I,J,NK) = -AVGVELP
c            WINLT(I,J) = -AVGVELP
c            WW(I,J,NK+1) = -AVGVELP
            VV(I,J,NK) = ZERO
            UU(I,J,NK) = ZERO
          END IF
  250   CONTINUE
  260 CONTINUE

      RETURN
      END

c*********************************************************************
c
c $Log$
c Revision 1.2  2000/06/20 20:42:38  rawat
c added some more boundary stuff and modified interface to IntVector. Before
c compiling the code you need to update /SCICore/Geometry/IntVector.h
c
c Revision 1.1  2000/04/12 20:31:49  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
