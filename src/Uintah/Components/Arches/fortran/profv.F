c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE profv(indexLow, indexHigh, PCELL,
     &     UU,VV,WW,
     &     DEN,
     &     UINLT,VINLT,WINLT,
     &     YY, ZZ, YV, ZW,
     &     FLOWPR,DENPR,FLOWSC,DENSC,SWNPR,SWNSC,
     &     AREAPR, AREASC,
     &     SOR, YSCENT, ZSCENT,
     &     INCELL,
     &     LCORD,
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
#include "pcell.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer indexLow(3)
      integer indexHigh(3)
      integer PCELL(indexLow(1):indexHigh(1), indexLow(2):indexHigh(2), 
     &     indexLow(3):indexHigh(3))
      double precision UU(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision VV(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision WW(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision DEN(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision UINLT(indexLow(2):indexHigh(2), 
     &     indexLow(3):indexHigh(3))
      double precision VINLT(indexLow(1):indexHigh(1), 
     &     indexLow(3):indexHigh(3))
      double precision WINLT(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2))
      double precision YY(indexLow(2):indexHigh(2)) 
      double precision YV(indexLow(2):indexHigh(2)) 
      double precision ZZ(indexLow(3):indexHigh(3)) 
      double precision ZW(indexLow(3):indexHigh(3)) 
      double precision FLOWPR, DENPR, FLOWSC, DENSC, SWNPR, SWNSC
      double precision AREAPR, AREASC
      double precision SOR, YSCENT, ZSCENT
      integer INCELL(6, indexLow(1):indexHigh(1), 
     &     indexLow(1):indexHigh(1))
      logical LCORD
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
            INCELL(1,J,K) = 1
          END IF
          IF (PCELL(NI,J,K).EQ.PFIELD) THEN
            UU(NI,J,K) = -AVGVELP
            UINLT(J,K) = -AVGVELP
            VV(NI,J,K) = ZERO
            WW(NI,J,K) = ZERO
            INCELL(2,J,K) = NI
          END IF
          IF (PCELL(iBegGhost,J,K).EQ.SYMTRY) 
     &         UU(iBegGhost,J,K) = AVGVELP
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
            INCELL(3,I,K) = 1
          END IF
          IF (PCELL(I,NJ,K).EQ.PFIELD) THEN
            VV(I,NJ,K) = -AVGVELP
            VINLT(I,K) = -AVGVELP
            WW(I,NJ,K) = ZERO
            UU(I,NJ,K) = ZERO
            INCELL(4,I,K) = NJ
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
            INCELL(5,I,J) = 1
          END IF
          IF (PCELL(I,J,NK).EQ.PFIELD) THEN
            WW(I,J,NK) = -AVGVELP
            WINLT(I,J) = -AVGVELP
            VV(I,J,NK) = ZERO
            UU(I,J,NK) = ZERO
            INCELL(6,I,J) = NK
          END IF
  250   CONTINUE
  260 CONTINUE
C--------------------------------------------------------------------
C     SECONDARY VELOCITY 
C--------------------------------------------------------------------
      FLOW = FLOWSC
      RHO = DENSC
      IF (AREASC.le.TINY) THEN
        AVGVELS = ZERO
        FLOWSC = ZERO
      ELSE
        AVGVELS = FLOW/(RHO*AREASC)
      END IF
C--------------------------------------------------------------------
C     EAST AND/OR WEST INLETS
C--------------------------------------------------------------------
      DO 390 K = kBegGhost,NK
        DO 380 J = jBegGhost,NJ
          IF (PCELL(iBegGhost,J,K).EQ.SFIELD) THEN
            UU(iBegGhost,J,K) = AVGVELS
            UU(IST,J,K) = AVGVELS
            VV(iBegGhost,J,K) = ZERO
            WW(iBegGhost,J,K) = ZERO
            INCELL(1,J,K) = 1
          END IF
          IF (PCELL(NI,J,K).EQ.SFIELD) THEN
            UU(NI,J,K) = -AVGVELS
            UINLT(J,K) = -AVGVELS
            VV(NI,J,K) = ZERO
            WW(NI,J,K) = ZERO
            INCELL(2,J,K) = NI
          END IF
          IF (PCELL(iBegGhost,J,K).EQ.SYMTRY) 
     &         UU(iBegGhost,J,K) = AVGVELS
  380   CONTINUE
  390 CONTINUE
C--------------------------------------------------------------------
C     NORTH AND/OR SOUTH INLETS
C--------------------------------------------------------------------
      DO 420 K = kBegGhost,NK
        DO 410 I = iBegGhost,NI
          IF (PCELL(I,jBegGhost,K).EQ.SFIELD) THEN
            VV(I,jBegGhost,K) = AVGVELS
            VV(I,JST,K) = AVGVELS
            WW(I,jBegGhost,K) = ZERO
            UU(I,jBegGhost,K) = ZERO
            INCELL(3,I,K) = 1
          END IF
          IF (PCELL(I,NJ,K).EQ.SFIELD) THEN
            VV(I,NJ,K) = -AVGVELS
            VINLT(I,K) = -AVGVELS
            WW(I,NJ,K) = ZERO
            UU(I,NJ,K) = ZERO
            INCELL(4,I,K) = NJ
          END IF
  410   CONTINUE
  420 CONTINUE
C--------------------------------------------------------------------
C     TOP AND/OR BOTTOM INLETS
C--------------------------------------------------------------------
      DO 450 J = jBegGhost,NJ
        DO 440 I = iBegGhost,NI
          IF (PCELL(I,J,kBegGhost).EQ.SFIELD) THEN
            WW(I,J,kBegGhost) = AVGVELS
            WW(I,J,KST) = AVGVELS
            VV(I,J,kBegGhost) = ZERO
            UU(I,J,kBegGhost) = ZERO
            INCELL(5,I,J) = 1
          END IF
          IF (PCELL(I,J,NK).EQ.SFIELD) THEN
            WW(I,J,NK) = -AVGVELS
            WINLT(I,J) = -AVGVELS
            VV(I,J,NK) = ZERO
            UU(I,J,NK) = ZERO
            INCELL(6,I,J) = NK
          END IF
  440   CONTINUE
  450 CONTINUE
C--------------------------------------------------------------------
C     Set INCELL for domain decomposition
C--------------------------------------------------------------------
C     EAST AND/OR WEST INLETS
C--------------------------------------------------------------------
      DO 590 K = kBegGhost,NK
        DO 580 J = jBegGhost,NJ
          IF (PCELL(iBegGhost,J,K).EQ.DECOMP) INCELL(1,J,K) = 1
          IF (PCELL(NI,J,K).EQ.DECOMP) INCELL(2,J,K) = NI
  580   CONTINUE
  590 CONTINUE
C--------------------------------------------------------------------
C     NORTH AND/OR SOUTH INLETS
C--------------------------------------------------------------------
      DO 620 K = kBegGhost,NK
        DO 610 I = iBegGhost,NI
          IF (PCELL(I,jBegGhost,K).EQ.DECOMP) INCELL(3,I,K) = 1
          IF (PCELL(I,NJ,K).EQ.DECOMP) INCELL(4,I,K) = NJ
  610   CONTINUE
  620 CONTINUE
C--------------------------------------------------------------------
C     TOP AND/OR BOTTOM INLETS
C--------------------------------------------------------------------
      DO 750 J = jBegGhost,NJ
        DO 740 I = iBegGhost,NI
          IF (PCELL(I,J,kBegGhost).EQ.DECOMP) INCELL(5,I,J) = 1
          IF (PCELL(I,J,NK).EQ.DECOMP) INCELL(6,I,J) = NK
  740   CONTINUE
  750 CONTINUE
C-------------------------------------------------------------------
C     V VELOCITY PROFILE IN PRIMARY AND SECONDARY
C------------------------------------------------------------------
C
C     PROVISION IN INLETS 
C     FOR VELOCITIES THAT ARE NOT NORMAL TO THE CELL FACE :
C
C-------------------------------------------------------------------
C     TANGENTIAL VELOCITY OR SWIRL (ONLY ON WEST INLETS FOR NOW)
C     AND FOR THE CARTESIAN OPTION:
C         ONLY OUT TO A RADIUS OF SOR 
C         STARTING AT A SWIRL CENTER OF Y=YSCENT, Z=ZSCENT
C         WITH SWIRL AXIS BEING IN THE X DIRECTION
C         NOTE: these options give constant w and v velocities.
C               to input velocity profiles like in a free or
C               forced vortex you must put them in the .inl file.
C-------------------------------------------------------------------
      DO 810 K = KST,NK
        DO 800 J = JST,NJ
          IF (LCORD) THEN
            IF (PCELL(iBegGhost,J,K).EQ.PFIELD) 
     &            WW(iBegGhost,J,K) = SWNPR*PT5*
     &            (UU(iBegGhost,J,K)+UU(iBegGhost,J,K-1))
            IF (PCELL(iBegGhost,J,K).EQ.SFIELD) 
     &           WW(iBegGhost,J,K) = SWNSC*PT5*
     &           (UU(iBegGhost,J,K)+UU(iBegGhost,J,K-1))
          ELSE
            ZR = ZZ(K) - ZSCENT
            ZWR = ZW(K) - ZSCENT
            YR = YY(J) - YSCENT
            YVR = YV(J) - YSCENT
            IF (SQRT(ZR**2+YR**2) .LT. SOR) THEN
              IF (PCELL(iBegGhost,J,K).EQ.PFIELD) THEN
                VV(iBegGhost,J,K) = -ZR*SWNPR*PT5*(UU(iBegGhost,J,K)+
     &                UU(iBegGhost,J-1,K)) /SQRT(ZR**2+YVR**2)
                WW(iBegGhost,J,K) = YR*SWNPR*PT5*(UU(iBegGhost,J,K)+
     &               UU(iBegGhost,J,K-1)) /SQRT(ZWR**2+YR**2)
              END IF
              IF (PCELL(iBegGhost,J,K).EQ.SFIELD) THEN
                VV(iBegGhost,J,K) = -ZR*SWNSC*PT5*(UU(iBegGhost,J,K)+
     &                UU(iBegGhost,J-1,K)) /SQRT(ZR**2+YVR**2)
                WW(iBegGhost,J,K) = YR*SWNSC*PT5*(UU(iBegGhost,J,K)+
     &               UU(iBegGhost,J,K-1)) /SQRT(ZWR**2+YR**2)
              END IF
            END IF
          END IF
  800   CONTINUE
  810 CONTINUE
C
C     SET DENSITY PROFILES IN PRIMARY AND SECONDARY
C
C$DIR SCALAR
      DO 920 K=kBegGhost,NK
C$DIR SCALAR
         DO 910 J=jBegGhost,NJ
C$DIR SCALAR
            DO 900 I=iBegGhost,NI
               IF (PCELL(I,J,K) .EQ. PFIELD) DEN(I,J,K) = DENPR
               IF (PCELL(I,J,K) .EQ. SFIELD) DEN(I,J,K) = DENSC
 900        CONTINUE
 910     CONTINUE
 920  CONTINUE
      RETURN
      END

c*********************************************************************
c
c $Log$
c Revision 1.1  2000/04/12 20:31:49  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
