c*********************************************************************
c
c
c*********************************************************************

      SUBROUTINE outletbcenth(domLoS, domHiS, H,
     $     domLoS_old, domHiS_old, old_H,    
     &     domLoCT, domHiCT,
     $     idxLo, idxHi, PCELL,
     &     PFIELD, uvwout,
     $     xminus, xplus, yminus, yplus, zminus, zplus,
     $     delta_t,
     $     domLoGeom, domHiGeom,
     $     dxpw,
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
      integer domLoS(3), domHiS(3)
      double precision H(domLoS(1):domHiS(1), domLoS(2):domHiS(2), 
     &     domLoS(3):domHiS(3))
      integer domLoS_old(3), domHiS_old(3)
      double precision old_H(domLoS_old(1):domHiS_old(1),
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
                  H(i,j,k) = -delta_t*uvwout*(old_H(i,j,k)-
     $                         old_H(i-1,j,k))/dxpw(i) + 
     $                         old_H(i,j,k)
               endif
 180        CONTINUE
 190     CONTINUE
      endif
      
      RETURN
      END
