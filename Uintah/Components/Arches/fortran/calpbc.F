c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE calpbc(domLoU, domHiU, UU,
     &     domLoV, domHiV, VV,
     &     domLoW, domHiW, WW,
     &     domLo, domHi, idxLo, idxHi, PP, DEN, PCELL,
     &     PFIELD, BndryPress,
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

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer domLoU(3), domHiU(3)
      double precision UU(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      integer domLoV(3), domHiV(3)
      double precision VV(domLoV(1):domHiV(1), domLoV(2):domHiV(2), 
     &     domLoV(3):domHiV(3))
      integer domLoW(3), domHiW(3)
      double precision WW(domLoW(1):domHiW(1), domLoW(2):domHiW(2), 
     &     domLoW(3):domHiW(3))
      integer domLo(3), domHi(3), idxLo(3), idxHi(3)
      double precision PP(domLo(1):domHi(1), domLo(2):domHi(2), 
     &     domLo(3):domHi(3))
      double precision DEN(domLo(1):domHi(1), domLo(2):domHi(2), 
     &     domLo(3):domHi(3))
      integer PCELL(domLo(1):domHi(1), domLo(2):domHi(2), 
     &     domLo(3):domHi(3))
      double precision BndryPress
      integer PFIELD
      integer ierr

c*********************************************************************
c     Local Variables :
c*********************************************************************
      logical LNORM
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
      iBegGhost = domLoU(1)
      jBegGhost = domLoV(2)
      kBegGhost = domLoW(3)
      NI = domHiU(1) - 1
      NJ = domHiV(2) - 1
      NK = domHiW(3) - 1

      IST = idxLo(1)
      JST = idxLo(2)
      KST = idxLo(3)
      IEND = idxHi(1)
      JEND = idxHi(2)
      KEND = idxHi(3)
      
      FACK = 1.0
      URFBC = 1.0
      LNORM = .TRUE.
C     
C       First define what will be the variables on INLETS
C

C     
C	Calculate pbc for I=1 (West) side of domain
C
      I = iBegGhost
      DO 140 K=KST,KEND
         DO 130 J=JST,JEND
            IF (PCELL(I,J,K) .EQ. PFIELD) THEN
                PPP = BndryPress
                PP(I,J,K) = PPP
                DENFACE = 0.5*(DEN(I,J,K)+DEN(I+1,J,K))
                DELP = PPP - PP(I+1,J,K)
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
                  VV(I,J,K) = VVEL
                  WW(I,J,K) = WVEL
                  PP(I,J,K) = PPP
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
	      IF (PCELL(I,J,K) .EQ. PFIELD) THEN
                PPP = BndryPress
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
                UVEL = UVEL*URFBC + UU(I+1,J,K)*(1.0-URFBC)
                UU(NI,J,K) = UVEL
                UU(NI+1,J,K) = UVEL
                IF (UVEL.LT.0.0) THEN
                  VV(I,J,K) = VVEL
                  WW(I,J,K) = WVEL
                  PP(I,J,K) = PPP
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
	      IF (PCELL(I,J,K) .EQ. PFIELD) THEN
                PPP = BndryPress
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
                  WW(I,J,K) = WVEL
                  PP(I,J,K) = PPP
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
	      IF (PCELL(I,J,K) .EQ. PFIELD) THEN
                PPP = BndryPress
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
                VVEL = URFBC*VVEL + VV(I,J+1,K)*(1.0-URFBC)
                VV(I,NJ,K) = VVEL
                VV(I,J+1,K) = VVEL
                IF (VVEL.LT.0.0) THEN
                  UU(I,J,K) = UVEL
                  WW(I,J,K) = WVEL
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
	      IF (PCELL(I,J,K) .EQ. PFIELD) THEN
                PPP = BndryPress
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
                  PP(I,J,K) = PPP
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
	      IF (PCELL(I,J,K) .EQ. PFIELD) THEN
                PPP = BndryPress
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
                WVEL = URFBC*WVEL + WW(I,J,K+1)*(1.0-URFBC)
                WW(I,J,NK) = WVEL
                WW(I,J,K+1) = WVEL
                IF (WVEL.LT.0.0) THEN
                  UU(I,J,K) = UVEL
                  VV(I,J,K) = VVEL
                  PP(I,J,K) = PPP
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
c Revision 1.2  2000/07/07 23:07:47  rawat
c added inlet bc's
c
c Revision 1.1  2000/04/12 20:31:44  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
