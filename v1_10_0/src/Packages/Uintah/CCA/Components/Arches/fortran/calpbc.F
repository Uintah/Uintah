c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/calpbc_fort.h>
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

c*********************************************************************
c     Local Variables :
c*********************************************************************
      logical*1 LNORM, LTRACTION
      integer IST, JST, KST, IEND, JEND, KEND, jst_mod, kst_mod
      integer i, j, k
      double precision DXPRD, DTPRD, DPMAG, PPP, P0, FACK, URFBC
      double precision EEE, FFF, GGG, HHH, GE
      double precision xpos, ypos, zpos, xyzpos
      double precision frac, dp
      double precision DENFACE, DELP
      double precision UMAG, UMAGINT, UVEL, VVEL, WVEL
      double precision uuvel, vvvel, wwvel

c*********************************************************************
c     Start :
c*********************************************************************
      IST = idxLo(1)
      JST = idxLo(2)
      KST = idxLo(3)
      IEND = idxHi(1)
      JEND = idxHi(2)
      KEND = idxHi(3)
#ifdef ARCHES_BC_DEBUG
      print*,'ist',ist
      print*,'jst',jst
      print*,'kst',kst
      print*,'iend',iend
      print*,'jend',jend
      print*,'kend',kend
#endif
      FACK = 1.0D0
      URFBC = 0.1D0
      LNORM = .TRUE.
      LTRACTION = .TRUE.
C     
C       First define what will be the variables on INLETS
C

C     
C	Calculate pbc for I=1 (West) side of domain
C
      if (xminus) then
         I = ist -1
         DO 140 K=KST,KEND
            DO 130 J=JST,JEND
               IF (PCELL(I,J,K) .EQ. PFIELD) THEN
                  PPP = BndryPress
                  PP(I,J,K) = PPP
                  DENFACE = 0.5D0*(DEN(I,J,K)+DEN(I+1,J,K))
                  DELP = PPP - PP(I+1,J,K)
                  UMAG = SIGN(SQRT(2.0D0*ABS(DELP)/DENFACE),DELP)
                  UMAG = UMAG/SQRT(FACK)
                  UVEL = UU(I+2,J,K)
                  VVEL = VV(I+1,J,K)
                  WVEL = WW(I+1,J,K)
                  if (.not.ltraction) then
                     UMAGINT = SQRT(UVEL*UVEL + VVEL*VVEL + WVEL*WVEL)
                     IF (LNORM.OR.UMAGINT.EQ.0.0D0) THEN
                        UVEL = UMAG
c                        VVEL = 0.0
c                        WVEL = 0.0
                     ELSE
                        UVEL = UMAG*ABS(UVEL)/UMAGINT
                        VVEL = VVEL*ABS(UMAG)/UMAGINT
                        WVEL = WVEL*ABS(UMAG)/UMAGINT
                     END IF
                     UVEL = UVEL*URFBC + UU(I,J,K)*(1.0D0-URFBC)
                  endif
                  UU(I+1,J,K) = UVEL
                  UU(I,J,K) = UVEL
c                  IF (UVEL.GT.0.0) THEN
                     VV(I,J,K) = VVEL
                     WW(I,J,K) = WVEL
                     PP(I,J,K) = PPP
c                  END IF
               END IF
 130        CONTINUE
 140     CONTINUE
      endif
C
C	Calculate PBC for I=NI (East) side of domain
C
      if (xplus) then
         I = iend+1
         DO 190 K=KST,KEND
	    DO 180 J=JST,JEND
               IF (PCELL(I,J,K) .EQ. PFIELD) THEN
                  PPP = BndryPress
                  PP(I,J,K) = PPP
                  DENFACE = 0.5D0*(DEN(I,J,K)+DEN(I-1,J,K))
                  DELP = PPP - PP(I-1,J,K)
                  UMAG = -SIGN(SQRT(ABS(DELP)/DENFACE),DELP)
                  UMAG = UMAG/SQRT(FACK)
                  UVEL = UU(I-1,J,K)
                  VVEL = VV(I-1,J,K)
                  WVEL = WW(I-1,J,K)
                  if (.not.ltraction) then
                     UMAGINT = SQRT(UVEL*UVEL + VVEL*VVEL + WVEL*WVEL)
                     IF (LNORM.OR.UMAGINT.EQ.0.0D0) THEN
                        UVEL = UMAG
c                        VVEL = 0.0
c                        WVEL = 0.0
                     ELSE
                        UVEL = UMAG*ABS(UVEL)/UMAGINT
                     END IF
                     UVEL = UVEL*URFBC + UU(I+1,J,K)*(1.0D0-URFBC)
                  end if
                  UU(I,J,K) = UVEL
                  UU(I+1,J,K) = UVEL
c                  if (uvel.lt.0.0) then
                     PP(I,J,K) = PPP
c                  endif
c                     uu(i,j,k) =  sqrt(umag*umag+uu(I-1,j,k)*
c     $                                 uu(i-1,j,k))
c                     uu(i+1,j,k) = uu(i,j,k)
               END IF
 180        CONTINUE
 190     CONTINUE
         jst_mod=jst
         if (yminus) jst_mod=jst_mod+1
         DO K=KST,KEND
	    DO J=JST_mod,JEND
               IF (PCELL(I,J,K) .EQ. PFIELD) THEN
                  PPP = BndryPress
                  DENFACE = 0.5D0*(DEN(I,J,K)+DEN(I-1,J,K))
                  DELP = PPP - PP(I-1,J,K)
                  UMAG = -SIGN(SQRT(ABS(DELP)/DENFACE),DELP)
                  UMAG = UMAG/SQRT(FACK)
                  UVEL = UU(I-1,J,K)
                  VVEL = VV(I-1,J,K)
                  WVEL = WW(I-1,J,K)
                  if (.not.ltraction) then
                     UMAGINT = SQRT(UVEL*UVEL + VVEL*VVEL + WVEL*WVEL)
                     IF (LNORM.OR.UMAGINT.EQ.0.0D0) THEN
                        UVEL = UMAG
c                        VVEL = 0.0
c                        WVEL = 0.0
                     ELSE
                        VVEL = VVEL*ABS(UMAG)/UMAGINT
                     END IF
                  end if
c                  if (uvel.lt.0.0) then
                     VV(I,J,K) = VVEL
c                  endif
               END IF
            end do
         end do
         kst_mod=kst
         if (zminus) kst_mod=kst_mod+1
         DO K=KST_mod,KEND
	    DO J=JST,JEND
               IF (PCELL(I,J,K) .EQ. PFIELD) THEN
                  PPP = BndryPress
                  DENFACE = 0.5D0*(DEN(I,J,K)+DEN(I-1,J,K))
                  DELP = PPP - PP(I-1,J,K)
                  UMAG = -SIGN(SQRT(ABS(DELP)/DENFACE),DELP)
                  UMAG = UMAG/SQRT(FACK)
                  UVEL = UU(I-1,J,K)
                  VVEL = VV(I-1,J,K)
                  WVEL = WW(I-1,J,K)
                  if (.not.ltraction) then
                     UMAGINT = SQRT(UVEL*UVEL + VVEL*VVEL + WVEL*WVEL)
                     IF (LNORM.OR.UMAGINT.EQ.0.0D0) THEN
                        UVEL = UMAG
c                        VVEL = 0.0
c                        WVEL = 0.0
                     ELSE
                        WVEL = WVEL*ABS(UMAG)/UMAGINT
                     END IF
                  end if
c                  if (uvel.lt.0.0) then
                     WW(I,J,K) = WVEL
c                  endif
               END IF
            end do
         end do
      endif
C
C	Compute PBC for J=1 (South) side of domain
C
      if (yminus) then
         J = jst-1
         DO 240 K=KST,KEND
            DO 230 I=IST,IEND
               IF (PCELL(I,J,K) .EQ. PFIELD) THEN
                  PPP = BndryPress
                  PP(I,J,K) = PPP
                  DENFACE = 0.5D0*(DEN(I,J,K)+DEN(I,J+1,K))
                  DELP = PPP - PP(I,J+1,K)
c                  UMAG = SIGN(SQRT(ABS(DELP)/DENFACE),DELP)
c                  UMAG = UMAG/SQRT(FACK)
                  UMAG = SQRT(2.0d0*ABS(DELP)/DENFACE)
                  if (vv(i,j+2,k).gt.0.0d0) then
                     vvel = umag
                  else
                     vvel = vv(i,j+2,k)
                  endif
                  UVEL = UU(I,J+1,K)
                  VVEL = VV(I,J+2,K)
                  WVEL = WW(I,J+1,K)
                  if (.not.ltraction) then
                     UMAGINT = SQRT(UVEL*UVEL + VVEL*VVEL + WVEL*WVEL)
                     IF (LNORM.OR.UMAGINT.EQ.0.0D0) THEN
c                        UVEL = 0.0
                        VVEL = UMAG
c                        WVEL = 0.0
                     ELSE
                        UVEL = UVEL*ABS(UMAG)/UMAGINT
                        VVEL = ABS(VVEL)*UMAG/UMAGINT
                        WVEL = WVEL*ABS(UMAG)/UMAGINT
                     END IF
                     VVEL = URFBC*VVEL + VV(I,J,K)*(1.0D0-URFBC)
                  endif
                  VVvel = urfbc*(VVEL 
     $                 - pp(i,j+1,k)*dynpv(j+1)/vis(i,j+1,k)) +
     $                   vv(i,j+1,k)*(1.0D0-urfbc)
                  vv(i,j+1,k) = vvel
                  VV(I,J,K) = VV(i,j+1,k)
                  UU(I,J,K) = UVEL
c     $                        + (vv(i,j+2,k)-vv(i-1,j+2,k))
                  if (.not.(zminus.and.k.eq.kst))  WW(I,J,K) = WVEL
c     $                        + (vv(i,j+2,k)-vv(i,j+2,k-1))
                  PP(I,J,K) = PPP
               END IF
 230        CONTINUE
 240     CONTINUE
      endif
C
C	Compute PBC for J=NJ (North) side of domain
C
      if (yplus) then
         J = jend+1
         DO 290 K=KST,KEND
	    DO 280 I=IST,IEND
               IF (PCELL(I,J,K) .EQ. PFIELD) THEN
                  PPP = BndryPress
                  PP(I,J,K) = PPP
                  DENFACE = 0.5D0*(DEN(I,J,K)+DEN(I,J-1,K))
                  DELP = PPP - PP(I,J-1,K)
c                  UMAG = -SIGN(SQRT(ABS(DELP)/DENFACE),DELP)
c                  UMAG = UMAG/SQRT(FACK)
                  UMAG = -SQRT(2.0d0*ABS(DELP)/DENFACE)
                  if (vv(i,j-1,k).lt.0.0d0) then
                     vvel = umag
                  else
                     vvel = vv(i,j-1,k)
                  endif
                  UVEL = UU(I,J-1,K)
                  VVEL = VV(I,J-1,K)
                  WVEL = WW(I,J-1,K)
                  if (.not.ltraction) then
                     UMAGINT = SQRT(UVEL*UVEL + VVEL*VVEL + WVEL*WVEL)
                     IF (LNORM.OR.UMAGINT.EQ.0.0D0) THEN
c                        UVEL = 0.0
                        VVEL = UMAG
c                        WVEL = 0.0
                     ELSE
                        UVEL = UVEL*ABS(UMAG)/UMAGINT
                        VVEL = ABS(VVEL)*UMAG/UMAGINT
                        WVEL = WVEL*ABS(UMAG)/UMAGINT
                     END IF
                     VVEL = URFBC*VVEL + VV(I,J+1,K)*(1.0D0-URFBC)
                  endif
                  VVvel = urfbc*(VVEL
     $                 + pp(i,j-1,k)*dynpv(j-1)/vis(i,j-1,k)) +
     $                 (1.0D0 - urfbc)*vv(i,j,k)
                  vv(i,j,k) = vvel
                  VV(I,J+1,K) = VV(i,j,k)
                  UU(I,J,K) = UVEL
c     $                        - (vv(i,j-1,k)-vv(i-1,j-1,k))
                  if (.not.(zminus.and.k.eq.kst))  WW(I,J,K) = WVEL
c                  WW(I,J,K) = WVEL
c     $                        - (vv(i,j-1,k)-vv(i,j-1,k-1))
                  PP(I,J,K) = PPP
               END IF
 280        CONTINUE
 290     CONTINUE
      endif

C
C	Calculate PBC for K=1 (Bottom) side of domain
C
      if (zminus) then
         K = kst-1
         DO 340 J=JST,JEND
	    DO 330 I=IST,IEND
               IF (PCELL(I,J,K) .EQ. PFIELD) THEN
                  PPP = BndryPress
                  PP(I,J,K) = PPP
                  DENFACE = 0.5D0*(DEN(I,J,K)+DEN(I,J,K+1))
                  DELP = PPP - PP(I,J,K+1)
c                  UMAG = SIGN(SQRT(ABS(DELP)/DENFACE),DELP)
c                  UMAG = UMAG/SQRT(FACK)
                  UMAG = SQRT(2.0d0*ABS(DELP)/DENFACE)
                  if (ww(i,j,k+2).gt.0.d0) then
                     wvel = umag
                  else
                     WVEL = WW(I,J,K+2)
                  endif
                  UVEL = UU(I,J,K+1)
                  VVEL = VV(I,J,K+1)
                  WVEL = WW(I,J,K+2)
                  if (.not.ltraction) then
                     UMAGINT = SQRT(UVEL*UVEL + VVEL*VVEL + WVEL*WVEL)
                     IF (LNORM.OR.UMAGINT.EQ.0.0D0) THEN
c                        UVEL = 0.0
c                        VVEL = 0.0
                        WVEL = UMAG
                     ELSE
                        UVEL = UVEL*ABS(UMAG)/UMAGINT
                        VVEL = VVEL*ABS(UMAG)/UMAGINT
                        WVEL = ABS(WVEL)*UMAG/UMAGINT
                     END IF
                     WVEL = URFBC*WVEL + WW(I,J,K)*(1.0-URFBC)
                  endif
                  WWvel = urfbc*(WVEL
     $                 - pp(i,j,k+1)*dztpw(k+1)/vis(i,j,k+1)) +
     $                 (1.0D0-urfbc)*ww(i,j,k+1)
                  ww(i,j,k+1) = wvel
                  WW(I,J,K) = WW(i,j,k+1)
                  UU(I,J,K) = UVEL
c     $                        + (WW(i,j,k+2) - ww(i-1,j,k+2))
                  if (.not.(yminus.and.j.eq.jst))  vv(I,J,K) = vVEL
c     $                        + (ww(i,j,k+2) - ww(i,j-1,k+2))
                  PP(I,J,K) = PPP
               END IF
 330        CONTINUE
 340     CONTINUE
      endif
C
C	Compute PBC for K=NK (Top) side of domain
C
      if (zplus) then
         K = kend+1
         DO 390 J=JST,JEND
	    DO 380 I=IST,IEND
               IF (PCELL(I,J,K) .EQ. PFIELD) THEN
                  PPP = BndryPress
                  PP(I,J,K) = PPP
                  DENFACE = 0.5D0*(DEN(I,J,K)+DEN(I,J,K-1))
                  DELP = PPP - PP(I,J,K-1)
c                  UMAG = -SIGN(SQRT(ABS(DELP)/DENFACE),DELP)
c                  UMAG = UMAG/SQRT(FACK)
                  UMAG = -SQRT(2.0d0*ABS(DELP)/DENFACE)
                  UVEL = UU(I,J,K-1)
                  VVEL = VV(I,J,K-1)
                  if (ww(i,j,k-1).lt.0.d0) then
                     wvel = umag
                  else
                     WVEL = WW(I,J,K-1)
                  endif
                  WVEL = WW(I,J,K-1)
                  if (.not.ltraction) then
                     UMAGINT = SQRT(UVEL*UVEL + VVEL*VVEL + WVEL*WVEL)
                     IF (LNORM.OR.UMAGINT.EQ.0.0D0) THEN
c                        UVEL = 0.0
c                        VVEL = 0.0
                        WVEL = UMAG
                     ELSE
                        UVEL = UVEL*ABS(UMAG)/UMAGINT
                        VVEL = VVEL*ABS(UMAG)/UMAGINT
                        WVEL = ABS(WVEL)*UMAG/UMAGINT
                     END IF
                     WVEL = URFBC*WVEL + WW(I,J,K+1)*(1.0D0-URFBC)
                  endif
                  WWvel = urfbc*(WVEL
     $                 + pp(i,j,k-1)*dztpw(k-1)/vis(i,j,k-1)) +
     $                 (1.0D0-urfbc)*ww(i,j,k)
                  ww(i,j,k) = wvel
                  WW(I,J,K+1) = WW(i,j,k)
                  UU(I,J,K) = UVEL
c     $                 - (WW(i,j,k-1) - ww(i-1,j,k-1))
                  if (.not.(yminus.and.j.eq.jst))  vv(I,J,K) = vVEL
c                  VV(I,J,K) = VVEL
c     $                 - (WW(i,j,k-1) - ww(i,j-1,k-1))
                  PP(I,J,K) = PPP
               END IF
 380        CONTINUE
 390     CONTINUE
      endif
C
      RETURN
      END

c*********************************************************************
c
c Revision 1.6  2000/10/12 20:07:41  sparker
c Commented out debug print statements
c Fixed bug in profscalar
c
c Revision 1.5  2000/10/11 16:37:31  rawat
c modified calpbc for ghost cells
c
c Revision 1.4  2000/10/10 19:31:03  rawat
c added scalarsolver
c
c Revision 1.3  2000/08/10 00:56:38  rawat
c added pressure bc for scalar and changed discretization option for velocity
c
c Revision 1.2  2000/07/07 23:07:47  rawat
c added inlet bc's
c
c Revision 1.1  2000/04/12 20:31:44  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
