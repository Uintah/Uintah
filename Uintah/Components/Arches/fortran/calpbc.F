c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE calpbc(indexLow, indexHigh, NEL, NSP, PCELL,
     &     PP,
     &     UU, VV, WW,
     &     DEN,
     &     FF, GG, HH, ETA, GETA,
     &     XX, YY, ZZ,
     &     UINLT, VINLT, WINLT,
     &     DTIME, TTIME,
     &     FSC, ETASC, HSEC,
     &     LPRTRB, LNORM, LCALH,
     &     ierr)
C-----------------------------------------------------------------------
C
C     Routine to calculate velocity at specified pressure b.c.'s
C     also must specify all variables for pressure b.c. INLETS
C     except for turbulence variables (they will be set in PROFT)
C
C     LPBC = .TRUE. is the specified pressure boundary condition option
C              To use it you must put the pressure bc as a presssure
C              boundary (a 5 in the grd file).  Then you have the option
C              of the flow direction at the surface being specified
C              (use a domain other than the computing domain for the
C              intfc) or the direction being extrapolated from the 
C              interior computation (by using a the computing domain
C              for intfc).
C              The theory for this option makes the following three
C              assumptions:
C              1) The flow is accelerated to the boundary from a point
C                 where the pressure is the prescribed pressure in the
C                 dat file without any frictional losses (this will be
C                 ok as long as the boundary is far enough away from any
C                 action to be very low)
C              2) At the pressure reference location (IPREF,JPREF,KPREF)
C                 the velocity is zero.
C              3) The direction of the flow comes from the interior
C                 cell next to the boundary.
C              See Computational Methods for Fluid Dynamics by 
C              Joel H. Ferziger and Milovan Peric, Springer, 1996
C              page 283
C
C     FACK = is the resistance or viscous losses represented in
C              terms of kinetic energy (as in 0.5*FACK*RHO*VEL**2)
C
C     LNORM = .TRUE. is the logical to define the velocities on the
C               boundaries to be normal to the boundaries.
C               .FALSE. gets the direction from the first interior
C               node.
C
C-----------------------------------------------------------------------
      implicit none

#include "param4.h"
#include "pcell.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer indexLow(3)
      integer indexHigh(3)
      integer NEL
      integer NSP
      integer PCELL(indexLow(1):indexHigh(1), indexLow(2):indexHigh(2), 
     &     indexLow(3):indexHigh(3))
      double precision PP(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision UU(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision VV(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision WW(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision DEN(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision FF(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision GG(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision HH(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision ETA(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision GETA(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision UINLT(indexLow(2):indexHigh(2), 
     &     indexLow(3):indexHigh(3))
      double precision VINLT(indexLow(1):indexHigh(1), 
     &     indexLow(3):indexHigh(3))
      double precision WINLT(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2))
      double precision XX(indexLow(1):indexHigh(1))
      double precision YY(indexLow(2):indexHigh(2))
      double precision ZZ(indexLow(3):indexHigh(3))
      double precision DTIME, TTIME
      double precision FSC, ETASC, HSEC
      logical LPRTRB, LNORM, LCALH
      integer ierr

c*********************************************************************
c     Local Variables :
c*********************************************************************
      logical LPRTRB
      integer iBegGhost, jBegGhost, kBegGhost, NI, NJ, NK
      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k
      double precision DXPRD, DTPRD, DPMAG, PPP, P0, FACK, URFBC
      double precision EEE, FFF, GGG, HHH, GE
      double precision xpos, ypos, zpos, xyzpos
      double precision frac, dp
      double precision DENFACE, DELP
      double precision UMAG, UMAGINT, UVEL, VVEL, WVEL

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
      
      LPRTRB = .FALSE.
      DXPRD = 0.2*(XX(NI)+YY(NJ)+ZZ(NK))/3.0
      DTPRD = 25.*DTIME
      DTPRD = 2.*DTIME
      DPMAG = 10.
      DPMAG = 1.
      P0 = 0.0
      FACK = 1.0
      URFBC = 0.5
C     
C       First define what will be the variables on INLETS
C
      FFF = FSC
      GGG = 0.0
      EEE = ETASC
      GE = 0.0
      HHH = HSEC
C     
C	Calculate pbc for I=1 (West) side of domain
C
      I = iBegGhost
      DO 140 K=KST,KEND
         DO 130 J=JST,JEND
            IF (PCELL(I,J,K) .EQ. PBCFLD) THEN
                PPP = P0
                if (LPRTRB) then
                  xpos = mod(XX(I),DXPRD)/DXPRD
                  ypos = mod(YY(J),DXPRD)/DXPRD
                  zpos = mod(ZZ(K),DXPRD)/DXPRD
                  xyzpos = (xpos+ypos+zpos)*DTPRD
                  frac = mod(TTIME+xyzpos,DTPRD)/DTPRD
                  dp = DPMAG*sin(2.0*PI*FRAC)
                  PPP = PPP + dp
                end if
                PP(I,J,K) = PPP
                DENFACE = 0.5*(DEN(I,J,K)+DEN(I+1,J,K))
                DELP = PPP - PP(I-1,J,K)
                UMAG = SIGN(SQRT(2.0*ABS(DELP)/DENFACE),DELP)
                UMAG = UMAG/SQRT(FACK)
                UVEL = UU(I+2,J,K)
                VVEL = VV(I+1,J,K)
                WVEL = WW(I+1,J,K)
                UMAGINT = SQRT(UVEL*UVEL + VVEL*VVEL + WVEL*WVEL)
                IF (LNORM.OR.UMAGINT.EQ.0.0) THEN
                  UVEL = UMAG
                  VVEL = 0.0
                  WVEL = 0.0
                ELSE
                  UVEL = UMAG*ABS(UVEL)/UMAGINT
                  VVEL = VVEL*ABS(UMAG)/UMAGINT
                  WVEL = WVEL*ABS(UMAG)/UMAGINT
                END IF
                UVEL = UVEL*URFBC + UU(I,J,K)*(1.0-URFBC)
                UU(I+1,J,K) = UVEL
                UU(I,J,K) = UVEL
                IF (UVEL.GT.0.0) THEN
                  UU(I,J,K) = UVEL
                  VV(I,J,K) = VVEL
                  WW(I,J,K) = WVEL
                  PP(I,J,K) = PPP
                  FF(I,J,K) = FFF
                  GG(I,J,K) = GGG
                  ETA(I,J,K) = EEE
                  GETA(I,J,K) = GE
                  IF (LCALH) HH(I,J,K) = HHH
                END IF
	      END IF
130	    CONTINUE
140	  CONTINUE
C
C	Calculate PBC for I=NI (East) side of domain
C
          I = NI
	  DO 190 K=KST,KEND
	    DO 180 J=JST,JEND
	      IF (PCELL(I,J,K) .EQ. PBCFLD) THEN
                PPP = P0
                if (LPRTRB) then
                  xpos = mod(XX(i),DXPRD)/DXPRD
                  ypos = mod(YY(j),DXPRD)/DXPRD
                  zpos = mod(ZZ(k),DXPRD)/DXPRD
                  xyzpos = (xpos+ypos+zpos)*DTPRD
                  frac = mod(TTIME+xyzpos,DTPRD)/DTPRD
                  dp = DPMAG*SIN(2.0*PI*frac)
                  PPP = PPP + dp
                end if
                PP(I,J,K) = PPP
                DENFACE = 0.5*(DEN(I,J,K)+DEN(I-1,J,K))
                DELP = PPP - PP(I-1,J,K)
                UMAG = -SIGN(SQRT(2.0*ABS(DELP)/DENFACE),DELP)
                UMAG = UMAG/SQRT(FACK)
                UVEL = UU(I-1,J,K)
                VVEL = VV(I-1,J,K)
                WVEL = WW(I-1,J,K)
                UMAGINT = SQRT(UVEL*UVEL + VVEL*VVEL + WVEL*WVEL)
                IF (LNORM.OR.UMAGINT.EQ.0.0) THEN
                  UVEL = UMAG
                  VVEL = 0.0
                  WVEL = 0.0
                ELSE
                  UVEL = UMAG*ABS(UVEL)/UMAGINT
                  VVEL = VVEL*ABS(UMAG)/UMAGINT
                  WVEL = WVEL*ABS(UMAG)/UMAGINT
                END IF
                UVEL = UVEL*URFBC + UINLT(J,K)*(1.0-URFBC)
                UU(NI,J,K) = UVEL
                UINLT(J,K) = UVEL
                IF (UVEL.LT.0.0) THEN
                  UINLT(J,K) = UVEL
                  VV(I,J,K) = VVEL
                  WW(I,J,K) = WVEL
                  PP(I,J,K) = PPP
                  FF(I,J,K) = FFF
                  GG(I,J,K) = GGG
                  ETA(I,J,K) = EEE
                  GETA(I,J,K) = GE
                  IF (LCALH) HH(I,J,K) = HHH
                END IF
	      END IF
180	    CONTINUE
190	  CONTINUE
C
C	Compute PBC for J=1 (South) side of domain
C
	  J = jBegGhost
	  DO 240 K=KST,KEND
	    DO 230 I=IST,IEND
	      IF (PCELL(I,J,K) .EQ. PBCFLD) THEN
                PPP = P0
                if (LPRTRB) then
                  xpos = mod(XX(i),DXPRD)/DXPRD
                  ypos = mod(YY(j),DXPRD)/DXPRD
                  zpos = mod(ZZ(k),DXPRD)/DXPRD
                  xyzpos = (xpos+ypos+zpos)*DTPRD
                  frac = mod(TTIME+xyzpos,DTPRD)/DTPRD
                  dp = DPMAG*SIN(2.0*PI*frac)
                  PPP = PPP + dp
                end if
                PP(I,J,K) = PPP
                DENFACE = 0.5*(DEN(I,J,K)+DEN(I,J+1,K))
                DELP = PPP - PP(I,J+1,K)
                UMAG = SIGN(SQRT(2.0*ABS(DELP)/DENFACE),DELP)
                UMAG = UMAG/SQRT(FACK)
                UVEL = UU(I,J+1,K)
                VVEL = VV(I,J+2,K)
                WVEL = WW(I,J+1,K)
                UMAGINT = SQRT(UVEL*UVEL + VVEL*VVEL + WVEL*WVEL)
                IF (LNORM.OR.UMAGINT.EQ.0.0) THEN
                  UVEL = 0.0
                  VVEL = UMAG
                  WVEL = 0.0
                ELSE
                  UVEL = UVEL*ABS(UMAG)/UMAGINT
                  VVEL = ABS(VVEL)*UMAG/UMAGINT
                  WVEL = WVEL*ABS(UMAG)/UMAGINT
                END IF
                VVEL = URFBC*VVEL + VV(I,J,K)*(1.0-URFBC)
                VV(I,J+1,K) = VVEL
                VV(I,J,K) = VVEL
                IF (VVEL.GT.0.0) THEN
                  UU(I,J,K) = UVEL
                  VV(I,J,K) = VVEL
                  WW(I,J,K) = WVEL
                  PP(I,J,K) = PPP
                  FF(I,J,K) = FFF
                  GG(I,J,K) = GGG
                  ETA(I,J,K) = EEE
                  GETA(I,J,K) = GE
                  IF (LCALH) HH(I,J,K) = HHH
                END IF
	      END IF
230	    CONTINUE
240	  CONTINUE
C
C	Compute PBC for J=NJ (North) side of domain
C
	  J = NJ
	  DO 290 K=KST,KEND
	    DO 280 I=IST,IEND
	      IF (PCELL(I,J,K) .EQ. PBCFLD) THEN
                PPP = P0
                if (LPRTRB) then
                  xpos = mod(XX(i),DXPRD)/DXPRD
                  ypos = mod(YY(j),DXPRD)/DXPRD
                  zpos = mod(ZZ(k),DXPRD)/DXPRD
                  xyzpos = (xpos+ypos+zpos)*DTPRD
                  frac = mod(ttime+xyzpos,DTPRD)/DTPRD
                  dp = DPMAG*SIN(2.0*PI*frac)
                  PPP = PPP + dp
                end if
                PP(I,J,K) = PPP
                DENFACE = 0.5*(DEN(I,J,K)+DEN(I,J-1,K))
                DELP = PPP - PP(I,J-1,K)
                UMAG = -SIGN(SQRT(2.0*ABS(DELP)/DENFACE),DELP)
                UMAG = UMAG/SQRT(FACK)
                UVEL = UU(I,J-1,K)
                VVEL = VV(I,J-1,K)
                WVEL = WW(I,J-1,K)
                UMAGINT = SQRT(UVEL*UVEL + VVEL*VVEL + WVEL*WVEL)
                IF (LNORM.OR.UMAGINT.EQ.0.0) THEN
                  UVEL = 0.0
                  VVEL = UMAG
                  WVEL = 0.0
                ELSE
                  UVEL = UVEL*ABS(UMAG)/UMAGINT
                  VVEL = ABS(VVEL)*UMAG/UMAGINT
                  WVEL = WVEL*ABS(UMAG)/UMAGINT
                END IF
                VVEL = URFBC*VVEL + VINLT(I,K)*(1.0-URFBC)
                VV(I,NJ,K) = VVEL
                VINLT(I,K) = VVEL
                IF (VVEL.LT.0.0) THEN
                  UU(I,J,K) = UVEL
                  VINLT(I,K) = VVEL
                  WW(I,J,K) = WVEL
                  FF(I,J,K) = FFF
                  GG(I,J,K) = GGG
                  ETA(I,J,K) = EEE
                  GETA(I,J,K) = GE
                  IF (LCALH) HH(I,J,K) = HHH
                END IF
	      END IF
280	    CONTINUE
290	  CONTINUE
C
C	Calculate PBC for K=1 (Bottom) side of domain
C
	  K = kBegGhost
	  DO 340 J=JST,JEND
	    DO 330 I=IST,IEND
	      IF (PCELL(I,J,K) .EQ. PBCFLD) THEN
                PPP = P0
                if (LPRTRB) then
                  xpos = mod(XX(i),DXPRD)/DXPRD
                  ypos = mod(YY(j),DXPRD)/DXPRD
                  zpos = mod(ZZ(k),DXPRD)/DXPRD
                  xyzpos = (xpos+ypos+zpos)*DTPRD
                  frac = mod(TTIME+xyzpos,DTPRD)/DTPRD
                  dp = DPMAG*SIN(2.0*PI*frac)
                  PPP = PPP + dp
                end if
                PP(I,J,K) = PPP
                DENFACE = 0.5*(DEN(I,J,K)+DEN(I,J,K+1))
                DELP = PPP - PP(I,J,K+1)
                UMAG = SIGN(SQRT(2.0*ABS(DELP)/DENFACE),DELP)
                UMAG = UMAG/SQRT(FACK)
                UVEL = UU(I,J,K+1)
                VVEL = VV(I,J,K+1)
                WVEL = WW(I,J,K+2)
                UMAGINT = SQRT(UVEL*UVEL + VVEL*VVEL + WVEL*WVEL)
                IF (LNORM.OR.UMAGINT.EQ.0.0) THEN
                  UVEL = 0.0
                  VVEL = 0.0
                  WVEL = UMAG
                ELSE
                  UVEL = UVEL*ABS(UMAG)/UMAGINT
                  VVEL = VVEL*ABS(UMAG)/UMAGINT
                  WVEL = ABS(WVEL)*UMAG/UMAGINT
                END IF
                WVEL = URFBC*WVEL + WW(I,J,K)*(1.0-URFBC)
                WW(I,J,K+1) = WVEL
                WW(I,J,K) = WVEL
                IF (WVEL.GT.0.0) THEN
                  UU(I,J,K) = UVEL
                  VV(I,J,K) = VVEL
                  WW(I,J,K) = WVEL
                  PP(I,J,K) = PPP
                  FF(I,J,K) = FFF
                  GG(I,J,K) = GGG
                  ETA(I,J,K) = EEE
                  GETA(I,J,K) = GE
                  IF (LCALH) HH(I,J,K) = HHH
                END IF
	      END IF
330	    CONTINUE
340	  CONTINUE
C
C	Compute PBC for K=NK (Top) side of domain
C
	  K = NK
	  DO 390 J=JST,JEND
	    DO 380 I=IST,IEND
	      IF (PCELL(I,J,K) .EQ. PBCFLD) THEN
                PPP = P0
                if (LPRTRB) then
                  xpos = mod(XX(i),DXPRD)/DXPRD
                  ypos = mod(YY(j),DXPRD)/DXPRD
                  zpos = mod(ZZ(k),DXPRD)/DXPRD
                  xyzpos = (xpos+ypos+zpos)*DTPRD
                  frac = mod(TTIME+xyzpos,DTPRD)/DTPRD
                  dp = DPMAG*SIN(2.0*PI*frac)
                  PPP = PPP + dp
                end if
                PP(I,J,K) = PPP
                DENFACE = 0.5*(DEN(I,J,K)+DEN(I,J,K-1))
                DELP = PPP - PP(I,J,K-1)
                UMAG = -SIGN(SQRT(2.0*ABS(DELP)/DENFACE),DELP)
                UMAG = UMAG/SQRT(FACK)
                UVEL = UU(I,J,K-1)
                VVEL = VV(I,J,K-1)
                WVEL = WW(I,J,K-1)
                UMAGINT = SQRT(UVEL*UVEL + VVEL*VVEL + WVEL*WVEL)
                IF (LNORM.OR.UMAGINT.EQ.0.0) THEN
                  UVEL = 0.0
                  VVEL = 0.0
                  WVEL = UMAG
                ELSE
                  UVEL = UVEL*ABS(UMAG)/UMAGINT
                  VVEL = VVEL*ABS(UMAG)/UMAGINT
                  WVEL = ABS(WVEL)*UMAG/UMAGINT
                END IF
                WVEL = URFBC*WVEL + WINLT(I,J)*(1.0-URFBC)
                WW(I,J,NK) = WVEL
                WINLT(I,J) = WVEL
                IF (WVEL.LT.0.0) THEN
                  UU(I,J,K) = UVEL
                  VV(I,J,K) = VVEL
                  WINLT(I,J) = WVEL
                  PP(I,J,K) = PPP
                  FF(I,J,K) = FFF
                  GG(I,J,K) = GGG
                  ETA(I,J,K) = EEE
                  GETA(I,J,K) = GE
                  IF (LCALH) HH(I,J,K) = HHH
                END IF
	      END IF
380	    CONTINUE
390	  CONTINUE
C
	RETURN
	END

c*********************************************************************
c
c $Log$
c Revision 1.1  2000/04/12 20:31:44  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
