c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/outletbc_fort.h>
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
      integer IST, JST, KST, IEND, JEND, KEND, jst_mod, kst_mod
      integer i, j, k
      double precision avden, avdenlow

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
c      if (yminus) jst = jst + 1
c      if (zminus) kst = kst + 1
      if (xplus) then
         I = iend+1
         DO 190 K=KST,KEND
	    DO 180 J=JST,JEND
               IF (PCELL(I,J,K) .EQ. PFIELD) THEN
                  avden = 0.5D0*(den(i,j,k)+den(i-1,j,k))
                  avdenlow = 0.5D0*(den(i-1,j,k)+den(i-2,j,k))
c                  avden = 1.0
c                  avdenlow = 1.0
                  UU(I,J,K) = (-delta_t*old_uu(i-1,j,k)*
c                  UU(I,J,K) = (-delta_t*uvwout*
     $                        (avden*old_uu(i,j,k)-
     $                         avdenlow*old_uu(i-1,j,k))/(dxpwu(i)) + 
     $                         avden*old_uu(i,j,k))/avden
                  flowout = flowout+ avden*
     $                 uu(i,j,k)*sns(j)*stb(k)
                  UU(I+1,J,K) = uu(i,j,k)
                 ff(i,j,k)=(-delta_t*old_uu(i-1,j,k)*
c                 ff(i,j,k)=(-delta_t*uvwout*
     $                       (den(i,j,k)*old_ff(i,j,k)-
     $                        den(i-1,j,k)*old_ff(i-1,j,k))/(dxpw(i))+ 
     $                         den(i,j,k)*old_ff(i,j,k))/den(i,j,k)
                  if (ff(i,j,k) .lt. 0.0D0) then 
                     ff(i,j,k) = 0.0D0
                  else if (ff(i,j,k) .gt. 1.0D0) then
                     ff(i,j,k) = 1.0D0
                  endif
               endif
 180        CONTINUE
 190     CONTINUE
         jst_mod=jst
         if (yminus) jst_mod=jst_mod+1
         DO K=KST,KEND
	    DO J=JST_mod,JEND
               IF (PCELL(I,J,K) .EQ. PFIELD) THEN
                  avden = 0.5D0*(den(i,j,k)+den(i,j-1,k))
                  avdenlow = 0.5D0*(den(i-1,j,k)+den(i-1,j-1,k))
#if 0
                  VV(I,J,K) = (-delta_t*old_uu(i-1,j,k)*
c                  VV(I,J,K) = (-delta_t*uvwout*
     $                        (avden*old_vv(i,j,k)-
     $                         avdenlow*old_vv(i-1,j,k))/(dxpw(i)) + 
     $                         avden*old_vv(i,j,k))/avden
#else
                  vv(i,j,k) = vv(i-1,j,k)
#endif
               endif
            end do
         end do
         kst_mod=kst
         if (zminus) kst_mod=kst_mod+1
         DO K=KST_mod,KEND
	    DO J=JST,JEND
               IF (PCELL(I,J,K) .EQ. PFIELD) THEN
                  avden = 0.5D0*(den(i,j,k)+den(i,j,k-1))
                  avdenlow = 0.5D0*(den(i-1,j,k)+den(i-1,j,k-1))
#if 0
                  WW(I,J,K) = (-delta_t*old_uu(i-1,j,k)*
c                  WW(I,J,K) = (-delta_t*uvwout*
     $                        (avden*old_ww(i,j,k)-
     $                         avdenlow*old_ww(i-1,j,k))/(dxpw(i)) + 
     $                         avden*old_ww(i,j,k))/avden
#else
                  ww(i,j,k) = ww(i-1,j,k)
#endif
               endif
            end do
         end do
      endif
      
      RETURN
      END
