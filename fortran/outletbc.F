c*********************************************************************
c
c
c*********************************************************************

      SUBROUTINE outletbc(domLoU, domHiU, UU,
     &     domLoV, domHiV, VV,
     &     domLoW, domHiW, WW,
     $     domLoS, domHiS, FF,
     $     domLoU_old, domHiU_old, old_UU,
     &     domLoV_old, domHiV_old, old_VV,
     &     domLoW_old, domHiW_old, old_WW,
     $     domLoS_old, domHiS_old, old_FF,    
     &     domLoCT, domHiCT,
     $     idxLo, idxHi, PCELL,
     &     PFIELD, uvwout,
     $     xminus, xplus, yminus, yplus, zminus, zplus,
     $     delta_t,
     $     domLoGeom, domHiGeom,
     $     dxpwu, dxpw,
     &     ierr)
C-----------------------------------------------------------------------
C
C     Routine to calculate outlet velocity at specified pressure b.c.'s
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
      integer domLoS(3), domHiS(3)
      double precision FF(domLoS(1):domHiS(1), domLoS(2):domHiS(2), 
     &     domLoS(3):domHiS(3))
      integer domLoU_old(3), domHiU_old(3)
      double precision old_UU(domLoU_old(1):domHiU_old(1), 
     $     domLoU_old(2):domHiU_old(2), 
     &     domLoU_old(3):domHiU_old(3))
      integer domLoV_old(3), domHiV_old(3)
      double precision old_VV(domLoV_old(1):domHiV_old(1), 
     $     domLoV_old(2):domHiV_old(2), 
     &     domLoV_old(3):domHiV_old(3))
      integer domLoW_old(3), domHiW_old(3)
      double precision old_WW(domLoW_old(1):domHiW_old(1),
     $     domLoW_old(2):domHiW_old(2), 
     &     domLoW_old(3):domHiW_old(3))
      integer domLoS_old(3), domHiS_old(3)
      double precision old_FF(domLoS_old(1):domHiS_old(1),
     $     domLoS_old(2):domHiS_old(2), 
     &     domLoS_old(3):domHiS_old(3))
      integer domLoCT(3), domHiCT(3)
      integer idxLo(3), idxHi(3)
      integer PCELL(domLoCT(1):domHiCT(1), domLoCT(2):domHiCT(2), 
     &     domLoCT(3):domHiCT(3))
      double precision uvwout
      integer PFIELD
      integer xminus, xplus, yminus, yplus, zminus, zplus
      double precision delta_t
      integer domLoGeom(3), domHiGeom(3)
      double precision dxpwu(domLoGeom(1):domHiGeom(1))
      double precision dxpw(domLoGeom(1):domHiGeom(1))
      integer ierr

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k

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
      if (xplus) then
         I = iend+1
         DO 190 K=KST,KEND
	    DO 180 J=JST,JEND
               IF (PCELL(I,J,K) .EQ. PFIELD) THEN
                  UU(I,J,K) = -delta_t*uvwout*(old_uu(i,j,k)-
     $                         old_uu(i-1,j,k))/dxpwu(i) + 
     $                         old_uu(i,j,k)
                  UU(I+1,J,K) = uu(i,j,k)
                  VV(I,J,K) = -delta_t*uvwout*(old_vv(i,j,k)-
     $                         old_vv(i-1,j,k))/dxpw(i) + 
     $                         old_vv(i,j,k)
                  WW(I,J,K) = -delta_t*uvwout*(old_ww(i,j,k)-
     $                         old_ww(i-1,j,k))/dxpw(i) + 
     $                         old_ww(i,j,k)
                  ff(i,j,k) = -delta_t*uvwout*(old_ff(i,j,k)-
     $                         old_ff(i-1,j,k))/dxpw(i) + 
     $                         old_ff(i,j,k)
               endif
 180        CONTINUE
 190     CONTINUE
      endif
      
      RETURN
      END
