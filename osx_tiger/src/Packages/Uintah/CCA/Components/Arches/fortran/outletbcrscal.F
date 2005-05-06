c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/outletbcrscal_fort.h>
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
                 H(i,j,k) = (-delta_t*uvwout*(den(i,j,k)*old_H(i,j,k)-
     $                         den(i-1,j,k)*old_H(i-1,j,k))/dxpw(i) + 
     $                         den(i,j,k)*old_H(i,j,k))/den(i,j,k)
                  if (H(i,j,k) .lt. 0.0D0) then 
                     H(i,j,k) = 0.0D0
                  else if (H(i,j,k) .gt. 1.0D0) then
                     H(i,j,k) = 1.0D0
                  endif
               endif
 180        CONTINUE
 190     CONTINUE
      endif
      
      RETURN
      END
