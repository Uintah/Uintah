c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE profv(domLoU, domHiU, idxLoU, idxHiU, UU,
     &     domLoV, domHiV, idxLoV, idxHiV, VV,
     &     domLoW, domHiW, idxLoW, idxHiW, WW,
     &     domLo, domHi, idxLo, idxHi, PCELL,
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
      integer domLoU(3), domHiU(3), idxLoU(3), idxHiU(3)
      double precision UU(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      integer domLoV(3), domHiV(3), idxLoV(3), idxHiV(3)
      double precision VV(domLoV(1):domHiV(1), domLoV(2):domHiV(2), 
     &     domLoV(3):domHiV(3))
      integer domLoW(3), domHiW(3), idxLoW(3), idxHiW(3)
      double precision WW(domLoW(1):domHiW(1), domLoW(2):domHiW(2), 
     &     domLoW(3):domHiW(3))
      integer domLo(3), domHi(3), idxLo(3), idxHi(3)
      integer PCELL(domLo(1):domHi(1), domLo(2):domHi(2), 
     &     domLo(3):domHi(3))
      double precision FLOWPR, DENPR
      double precision AREAPR
      integer PFIELD
      integer ierr

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer iBegGhost, jBegGhost, kBegGhost
      integer iEndGhost, jEndGhost, kEndGhost
      integer iBegGhostU, jBegGhostU, kBegGhostU
      integer iEndGhostU, jEndGhostU, kEndGhostU
      integer ISTU, JSTU, KSTU, IENDU, JENDU, KENDU
      integer iBegGhostV, jBegGhostV, kBegGhostV
      integer iEndGhostV, jEndGhostV, kEndGhostV
      integer ISTV, JSTV, KSTV, IENDV, JENDV, KENDV
      integer iBegGhostW, jBegGhostW, kBegGhostW
      integer iEndGhostW, jEndGhostW, kEndGhostW
      integer ISTW, JSTW, KSTW, IENDW, JENDW, KENDW
      integer i, j, k
      double precision FLOW, RHO, AVGVELP, AVGVELS
      double precision YR, ZR, YVR, ZWR

c*********************************************************************
c     Start :
c*********************************************************************
      iBegGhost = domLo(1)
      jBegGhost = domLo(2)
      kBegGhost = domLo(3)
      iEndGhost = domHi(1)
      jEndGhost = domHi(2)
      kEndGhost = domHi(3)
      iBegGhostU = domLoU(1)
      jBegGhostU = domLoU(2)
      kBegGhostU = domLoU(3)
      iEndGhostU = domHiU(1)
      jEndGhostU = domHiU(2)
      kEndGhostU = domHiU(3)
      ISTU = iBegGhostU + 1
      JSTU = jBegGhostU + 1
      KSTU = kBegGhostU + 1
      IENDU = iEndGhostU - 1
      JENDU = jEndGhostU - 1
      KENDU = kEndGhostU - 1
      iBegGhostV = domLoV(1)
      jBegGhostV = domLoV(2)
      kBegGhostV = domLoV(3)
      iEndGhostV = domHiV(1)
      jEndGhostV = domHiV(2)
      kEndGhostV = domHiV(3)
      ISTV = iBegGhostV + 1
      JSTV = jBegGhostV + 1
      KSTV = kBegGhostV + 1
      IENDV = iEndGhostV - 1
      JENDV = jEndGhostV - 1
      KENDV = kEndGhostV - 1
      iBegGhostW = domLoW(1)
      jBegGhostW = domLoW(2)
      kBegGhostW = domLoW(3)
      iEndGhostW = domHiW(1)
      jEndGhostW = domHiW(2)
      kEndGhostW = domHiW(3)
      ISTW = iBegGhostW + 1
      JSTW = jBegGhostW + 1
      KSTW = kBegGhostW + 1
      IENDW = iEndGhostW - 1
      JENDW = jEndGhostW - 1
      KENDW = kEndGhostW - 1

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
      DO 200 K = kBegGhost,kEndGhost
        DO 190 J = jBegGhost,jEndGhost
          IF (PCELL(iBegGhost,J,K).EQ.PFIELD) THEN
            UU(iBegGhostU,J,K) = AVGVELP
            UU(ISTU,J,K) = AVGVELP
            VV(iBegGhostU,J,K) = ZERO
            WW(iBegGhostU,J,K) = ZERO
          END IF
          IF (PCELL(iEndGhost,J,K).EQ.PFIELD) THEN
            UU(iEndGhostU,J,K) = -AVGVELP
c            UINLT(J,K) = -AVGVELP
c            UU(NI+1,J,K) = -AVGVELP
            VV(iEndGhostU,J,K) = ZERO
            WW(iEndGhostU,J,K) = ZERO
          END IF
  190   CONTINUE
  200 CONTINUE
C--------------------------------------------------------------------
C     NORTH AND/OR SOUTH INLETS
C--------------------------------------------------------------------
      DO 230 K = kBegGhost,kEndGhost
        DO 220 I = iBegGhost,iEndGhost
          IF (PCELL(I,jBegGhost,K).EQ.PFIELD) THEN
            VV(I,jBegGhostV,K) = AVGVELP
            VV(I,JSTV,K) = AVGVELP
            WW(I,jBegGhostV,K) = ZERO
            UU(I,jBegGhostV,K) = ZERO
          END IF
          IF (PCELL(I,jEndGhost,K).EQ.PFIELD) THEN
            VV(I,jEndGhostV,K) = -AVGVELP
c            VINLT(I,K) = -AVGVELP
c            VV(I,NJ+1,K) = -AVGVELP
            WW(I,jEndGhostV,K) = ZERO
            UU(I,jEndGhostV,K) = ZERO
          END IF
  220   CONTINUE
  230 CONTINUE
C--------------------------------------------------------------------
C     TOP AND/OR BOTTOM INLETS
C--------------------------------------------------------------------
      DO 260 J = jBegGhost,jEndGhost
        DO 250 I = iBegGhost,iEndGhost
          IF (PCELL(I,J,kBegGhost).EQ.PFIELD) THEN
            WW(I,J,kBegGhostW) = AVGVELP
            WW(I,J,KSTW) = AVGVELP
            VV(I,J,kBegGhostW) = ZERO
            UU(I,J,kBegGhostW) = ZERO
          END IF
          IF (PCELL(I,J,kEndGhost).EQ.PFIELD) THEN
            WW(I,J,kEndGhostW) = -AVGVELP
c            WINLT(I,J) = -AVGVELP
c            WW(I,J,NK+1) = -AVGVELP
            VV(I,J,kEndGhostW) = ZERO
            UU(I,J,kEndGhostW) = ZERO
          END IF
  250   CONTINUE
  260 CONTINUE

      RETURN
      END

c*********************************************************************
c
c $Log$
c Revision 1.5  2000/08/08 23:34:20  rawat
c fixed some bugs in profv.F and Properties.cc
c
c Revision 1.4  2000/06/30 22:41:19  bbanerje
c Corrected behavior of profv and profscalar
c
c Revision 1.3  2000/06/29 06:22:50  bbanerje
c Updated FCVariable to SFCX, SFCY, SFCZVariables and made corresponding
c changes to profv.  Code is broken until the changes are reflected
c thru all the files.
c
c Revision 1.2  2000/06/20 20:42:38  rawat
c added some more boundary stuff and modified interface to IntVector. Before
c compiling the code you need to update /SCICore/Geometry/IntVector.h
c
c Revision 1.1  2000/04/12 20:31:49  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
