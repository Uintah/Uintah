#if 1
#include <Packages/Uintah/CCA/Components/MPMArches/fortran/momentum_exchange_term_continuous_cc_fort.h>
#include "../../Arches/fortran/param4.h"
#else
      subroutine momentum_exchange_term_continuous_cc(
     $     su_dragx_fcy, 
     $     sp_dragx_fcy,
     $     su_dragx_fcz, 
     $     sp_dragx_fcz,
     $     su_dragx_cc, 
     $     sp_dragx_cc,
     $     dragforcex_fcy, 
     $     dragforcex_fcz, 
     $     dragforcex_cc, 
     $     ug_cc, 
     $     up_cc, 
     $     up_fcy, 
     $     up_fcz, 
     $     eps_Gas, 
     $     eps_Solid,
     $     viscos, csmag,
     $     sew, sns, stb, yy, zz, yv, zw,
     $     dim_lo, dim_hi, 
     $     dim_lo_su_fcy, dim_hi_su_fcy, 
     $     dim_lo_sp_fcy, dim_hi_sp_fcy, 
     $     dim_lo_su_fcz, dim_hi_su_fcz, 
     $     dim_lo_sp_fcz, dim_hi_sp_fcz, 
     $     dim_lo_su_cc, dim_hi_su_cc,
     $     dim_lo_sp_cc, dim_hi_sp_cc,
     $     dim_lo_dx_fcy, dim_hi_dx_fcy,
     $     dim_lo_dx_fcz, dim_hi_dx_fcz,
     $     dim_lo_dx_cc, dim_hi_dx_cc,
     $     dim_lo_ugc, dim_hi_ugc,
     $     dim_lo_upc, dim_hi_upc,
     $     dim_lo_upy, dim_hi_upy,
     $     dim_lo_upz, dim_hi_upz,
     $     dim_lo_eps, dim_hi_eps,
     $     dim_lo_epss, dim_hi_epss,
     $     valid_lo, valid_hi,
     $     ioff, joff, koff, indexflo, indext1, indext2,
     $     pcell, wall, ffield)

C-----------------------------------------------------------------------
c     Author: Seshadri Kumar, March 1, 2001
c     Last modified: March 21, 2001
c
c     this subroutine calculates momentum exchange terms for the
c     continuous solid-gas system
c
c     This is a cell-centered version, which means that the 
c     momentum exchange terms for the gas are calculated at the 
c     cell center.  The corresponding terms for the solid material
c     are calculated at the face-center.
c
c     The forces calculated by this routine calculate forces 
c     on the faces of a p-cell for the solid phase and the relevant
c     source terms for the gas phase.  Because the routine is 
c     direction-invariant, the routine must be called three times
c     to get all the nine components of the forces on the faces.
c     The routine is written as though the x-directed u-velocity
c     is under consideration and therefore interactions with walls 
c     perpendicular to the y- and z-directions are relevant.
c     By calling the variables in cyclic order (xyz, yzx, zxy), 
c     all the needed forces are obtained.
c
c     This routine calculates, for the gas phase,
c
c     su_dragx_cc: nonlinear source term for the drag, calculated here
c     at cell centers (used to interpolate to su_dragx_fcx)
c
c     sp_dragx_cc: linear source term for the drag, calculated here 
c     at cell centers (used to interpolate to sp_dragx_fcx)
c
c     and, for the particle phase,
c
c     dragforcex_fcy: drag force in the x-direction for interaction 
c     with solid walls perpendicular to the y-direction, stored at 
c     y-face centers
c
c     dragforcex_fcz: drag force in the x-direction for interaction 
c     with solid walls perpendicular to the z-direction, stored at 
c     z-face centers
c
c     dragforcex_cc: drag force in the x-direction for cells that 
c     contain both gas and solid 
c
c     The routine requires here, as inputs:
c
c     ug_cc: cell-centered u-velocities for the gas phase
c     
c     up_cc: cell-centered u-velocities for the solid phase
c
c     up_cc is further used to calculate (by interpolation)
c
c     up_fcy: solid velocity in the x-direction, stored at y-face 
c     centers
c     
c     up_fcz: solid velocity in the x-direction, stored at z-face 
c     centers
c
c     eps_Gas: cell-centered volume fraction of gas
c
c     eps_Solid: cell-centered volume fraction of solid of current 
c     material index
c
c     discrete_fraction: cell-centered discrete fraction of solid
c
c     viscos: molecular (dynamic) viscosity of gas
c
c     csmag: Smagorinsky constant for LES turbulence model
c
c     sew, sns, stb: dimensions of sides of p-cell
c
c     yy, yv: y co-ordinates of p-cell and v-cell
c
c     zz, zw: z co-ordinates of p-cell and w-cell
c
c     dim_lo, dim_hi: extents of the dimension of the input arrays
c
c     valid_lo_ext, valid_hi_ext: extents of the valid region of 
c     the patch (+ ghost indices)
c
c     valid_lo, valid_hi: extents of the valid region of the patch
c     (without ghost indices)
c
c     ioff, joff, koff: indices that represent which direction (and
c     therefore which momentum component) is being addressed; ioff = 1
c     represents the u-velocity
c
c     indexflo: index for the flow direction (for dimensioning);
c     for the u-velocity component, this is 1.
c
c     indext1: index for the first transverse direction 
c     (for dimensioning); for the u-velocity component, this is 2.
c
c     indext2: index for the second transverse direction
c     (for dimensioning); for the u-velocity component, this is 3.
c
c     pcell: character array that stores the information on cell typing
c
C-----------------------------------------------------------------------
      implicit none
      include 'param4.h'
      include 'pconst.h'

      integer dim_lo(3)
      integer dim_hi(3)

      integer dim_lo_su_fcy(3)
      integer dim_hi_su_fcy(3)

      integer dim_lo_sp_fcy(3)
      integer dim_hi_sp_fcy(3)

      integer dim_lo_su_fcz(3)
      integer dim_hi_su_fcz(3)

      integer dim_lo_sp_fcz(3)
      integer dim_hi_sp_fcz(3)

      integer dim_lo_su_cc(3)
      integer dim_hi_su_cc(3)
      
      integer dim_lo_sp_cc(3)
      integer dim_hi_sp_cc(3)

      integer dim_lo_dx_cc(3)
      integer dim_hi_dx_cc(3)

      integer dim_lo_dx_fcy(3)
      integer dim_hi_dx_fcy(3)
      
      integer dim_lo_dx_fcz(3)
      integer dim_hi_dx_fcz(3)

      integer dim_lo_ugc(3)
      integer dim_hi_ugc(3)

      integer dim_lo_upc(3)
      integer dim_hi_upc(3)

      integer dim_lo_upy(3)
      integer dim_hi_upy(3)

      integer dim_lo_upz(3)
      integer dim_hi_upz(3)

      integer dim_lo_eps(3)
      integer dim_hi_eps(3)

      integer dim_lo_epss(3)
      integer dim_hi_epss(3)

      integer indexflo
      integer indext1
      integer indext2

      double precision su_dragx_fcy(
     $     dim_lo_su_fcy(1):dim_hi_su_fcy(1),
     $     dim_lo_su_fcy(2):dim_hi_su_fcy(2),
     $     dim_lo_su_fcy(3):dim_hi_su_fcy(3))

      double precision sp_dragx_fcy(
     $     dim_lo_sp_fcy(1):dim_hi_sp_fcy(1),
     $     dim_lo_sp_fcy(2):dim_hi_sp_fcy(2),
     $     dim_lo_sp_fcy(3):dim_hi_sp_fcy(3))

      double precision su_dragx_fcz(
     $     dim_lo_su_fcz(1):dim_hi_su_fcz(1),
     $     dim_lo_su_fcz(2):dim_hi_su_fcz(2),
     $     dim_lo_su_fcz(3):dim_hi_su_fcz(3))

      double precision sp_dragx_fcz(
     $     dim_lo_sp_fcz(1):dim_hi_sp_fcz(1),
     $     dim_lo_sp_fcz(2):dim_hi_sp_fcz(2),
     $     dim_lo_sp_fcz(3):dim_hi_sp_fcz(3))

      double precision su_dragx_cc(
     $     dim_lo_su_cc(1):dim_hi_su_cc(1),
     $     dim_lo_su_cc(2):dim_hi_su_cc(2),
     $     dim_lo_su_cc(3):dim_hi_su_cc(3))

      double precision sp_dragx_cc(
     $     dim_lo_sp_cc(1):dim_hi_sp_cc(1),
     $     dim_lo_sp_cc(2):dim_hi_sp_cc(2),
     $     dim_lo_sp_cc(3):dim_hi_sp_cc(3))

      double precision dragforcex_fcy(
     $     dim_lo_dx_fcy(1):dim_hi_dx_fcy(1),
     $     dim_lo_dx_fcy(2):dim_hi_dx_fcy(2),
     $     dim_lo_dx_fcy(3):dim_hi_dx_fcy(3))

      double precision dragforcex_fcz(
     $     dim_lo_dx_fcz(1):dim_hi_dx_fcz(1),
     $     dim_lo_dx_fcz(2):dim_hi_dx_fcz(2),
     $     dim_lo_dx_fcz(3):dim_hi_dx_fcz(3))

      double precision dragforcex_cc(
     $     dim_lo_dx_cc(1):dim_hi_dx_cc(1),
     $     dim_lo_dx_cc(2):dim_hi_dx_cc(2),
     $     dim_lo_dx_cc(3):dim_hi_dx_cc(3))

      double precision ug_cc(
     $     dim_lo_ugc(1):dim_hi_ugc(1),
     $     dim_lo_ugc(2):dim_hi_ugc(2),
     $     dim_lo_ugc(3):dim_hi_ugc(3))

      double precision up_cc(
     $     dim_lo_upc(1):dim_hi_upc(1),
     $     dim_lo_upc(2):dim_hi_upc(2),
     $     dim_lo_upc(3):dim_hi_upc(3))

      double precision up_fcy(
     $     dim_lo_upy(1):dim_hi_upy(1),
     $     dim_lo_upy(2):dim_hi_upy(2),
     $     dim_lo_upy(3):dim_hi_upy(3))

      double precision up_fcz(
     $     dim_lo_upz(1):dim_hi_upz(1),
     $     dim_lo_upz(2):dim_hi_upz(2),
     $     dim_lo_upz(3):dim_hi_upz(3))

      double precision eps_Gas(
     $     dim_lo_eps(1):dim_hi_eps(1),
     $     dim_lo_eps(2):dim_hi_eps(2),
     $     dim_lo_eps(3):dim_hi_eps(3))

      double precision eps_Solid(
     $     dim_lo_epss(1):dim_hi_epss(1),
     $     dim_lo_epss(2):dim_hi_epss(2),
     $     dim_lo_epss(3):dim_hi_epss(3))

      double precision viscos
      double precision csmag

      double precision sew(dim_lo(indexflo):dim_hi(indexflo))
      double precision sns(dim_lo(indext1):dim_hi(indext1))
      double precision stb(dim_lo(indext2):dim_hi(indext2))

      double precision yy(dim_lo(indext1):dim_hi(indext1))
      double precision yv(dim_lo(indext1):dim_hi(indext1))

      double precision zz(dim_lo(indext2):dim_hi(indext2))
      double precision zw(dim_lo(indext2):dim_hi(indext2))

      integer valid_lo(3)
      integer valid_hi(3)

      integer ioff
      integer joff
      integer koff

      integer pcell(
     $     dim_lo(1):dim_hi(1),
     $     dim_lo(2):dim_hi(2),
     $     dim_lo(3):dim_hi(3))

      integer wall
      
      integer ffield
#endif

*     local variables

      integer valid_lo_ext(3)
      integer valid_hi_ext(3)

      double precision eps_Gas_cont
      
      integer idf
      integer idt1
      integer idt2

      integer i
      integer j
      integer k

      integer idir

      double precision drep
      double precision darea
      double precision dist
      double precision area
      double precision rmean
      double precision tmult
      double precision factor

      logical lmltm

*     executable statements

      lmltm = .false.

c--------------------------------------------------------------------
c     define arrays for one ghost around patch
c--------------------------------------------------------------------

      do idir = 1, 3

         valid_lo_ext(idir) = valid_lo(idir) - 1
         valid_hi_ext(idir) = valid_hi(idir) + 1

      end do

c--------------------------------------------------------------------
c     initialize arrays to zero
c--------------------------------------------------------------------

c     Some initializations are performed here.
c     The initializations for the following arrays must be done
c     at a level above this one, because they are cumulative
c     over all materials:
c
c     su_dragx_fcy
c     sp_dragx_fcy
c     su_dragx_fcz
c     sp_dragx_fcz
c     su_dragx_cc
c     sp_dragx_cc
c

#ifdef __GNUC__
C The variable names are too long the use the PASS3 macros,
C so we have to do this - Steve
      call arrass(dragforcex_fcy_low_x, dragforcex_fcy_low_y,
     $ dragforcex_fcy_low_z, dragforcex_fcy_high_x,
     $ dragforcex_fcy_high_y, dragforcex_fcy_high_z,
     $ dragforcex_fcy, zero, valid_lo, valid_hi)

      call arrass(dragforcex_fcz_low_x, dragforcex_fcz_low_y,
     $ dragforcex_fcz_low_z, dragforcex_fcz_high_x,
     $ dragforcex_fcz_high_y, dragforcex_fcz_high_z,
     $ dragforcex_fcz, zero, valid_lo, valid_hi)

      call arrass(dragforcex_cc_low_x, dragforcex_cc_low_y,
     $ dragforcex_cc_low_z, dragforcex_cc_high_x,
     $ dragforcex_cc_high_y, dragforcex_cc_high_z,
     $ dragforcex_cc, zero, valid_lo, valid_hi)

#else
      call arrass(dragforcex_fcy_low, dragforcex_fcy_high,
     $ dragforcex_fcy, zero, valid_lo, valid_hi)

      call arrass(dragforcex_fcz_low, dragforcex_fcz_high,
     $ dragforcex_fcz, zero, valid_lo, valid_hi)

      call arrass(dragforcex_cc_low, dragforcex_cc_high,
     $ dragforcex_cc, zero, valid_lo, valid_hi)

#endif

*     start computation
c
c     part 1:
c     calculation of momentum exchange terms for cells that have both
c     solid and fluid in them.  This calculation is done for both
c     solid and fluid at the cell center; and is later interpolated for
c     the solid to the face centers.
c     

c     debug, sk, April 14, 2001

      if (ioff.eq.1) then
         print*, 'printing drag forces'
      endif

c     end debug

      do 120 k = valid_lo(3), valid_hi(3)
         do 110 j = valid_lo(2), valid_hi(2)
            do 100 i = valid_lo(1), valid_hi(1)

c               eps_Gas_cont = (one - discrete_fraction(i,j,k))*
c     $              eps_Gas(i,j,k)
c     later we will get discrete fraction from mpm to input here.

               eps_Gas_cont = eps_Gas(i,j,k)
               if (eps_Gas_cont.lt.(one - small).and. 
     $              pcell(i,j,k).ne.wall) then

                  idf  = ioff*i + joff*j + koff*k
                  idt1 = koff*i + ioff*j + joff*k
                  idt2 = joff*i + koff*j + ioff*k

                  drep = (eps_Gas_cont*
     $                 sew(idf)*sns(idt1)*stb(idt2))**
     $                 (1.0/3.0)
                  darea = ((one - eps_Gas_cont)*
     $                 sew(idf)*sns(idt1)*stb(idt2))**
     $                 (1.0/3.0)
                  dist = drep/two
                  area = darea*darea

                  factor = eps_Solid(i,j,k)/(one - eps_Gas(i,j,k))
                  
                  tmult = factor*viscos/dist
                     
                  su_dragx_cc(i,j,k) = su_dragx_cc(i,j,k) + 
     $                 tmult*area*up_cc(i,j,k) 

                  sp_dragx_cc(i,j,k) = sp_dragx_cc(i,j,k) -
     $                 tmult*area
                     
                  dragforcex_cc(i,j,k) = dragforcex_cc(i,j,k) +
     $                 tmult*area*
     $                 (ug_cc(i,j,k) - up_cc(i,j,k))

c     debug, sk, April 14, 2001

                  if (ioff.eq.1) then
c                     print*, i,j,k, dragforcex_cc(i,j,k)
                  endif

c     end debug

               end if

 100        continue
 110     continue
 120  continue

c
c     part 2:
c     calculation of momentum exchange terms for cells 
c     adjacent to wall cells
c

      do 220 k = valid_lo_ext(3), valid_hi(3)
         do 210 j = valid_lo_ext(2), valid_hi(2)
            do 200 i = valid_lo_ext(1), valid_hi(1)

               if (pcell(i,j,k).eq.wall) then

                  idf  = ioff*i + joff*j + koff*k
                  idt1 = koff*i + ioff*j + joff*k
                  idt2 = joff*i + koff*j + ioff*k

c     lower and upper faces of intrusion for transverse 
c     direction 1 (i.e., south and north faces)

#if 1
      write(6,*)'walmom_cc broke - Steve'
      stop
#else
                  call walmom_cc(
     $                 su_dragx_fcy, sp_dragx_fcy, 
     $                 dragforcex_fcy, 
     $                 ug_cc, up_fcy, eps_Gas, 
     $                 eps_Solid,
     $                 sew, stb, yv, yy,
     $                 viscos, csmag,
     $                 idf, idt1, idt2, 
     $                 i, j, k, 
     $                 ioff, joff, koff,
     $                 koff, ioff, joff,
     $                 dim_lo, dim_hi, 
     $                 dim_lo_dx_fcy, dim_hi_dx_fcy,
     $                 dim_lo_su_fcy, dim_hi_su_fcy,
     $                 dim_lo_sp_fcy, dim_hi_sp_fcy,
     $                 dim_lo_ugc, dim_hi_ugc, 
     $                 dim_lo_upy, dim_hi_upy,
     $                 dim_lo_eps, dim_hi_eps, 
     $                 dim_lo_epss, dim_hi_epss, 
     $                 indexflo, indext1, indext2,
     $                 valid_lo_ext, valid_hi_ext, 
     $                 valid_lo, valid_hi,
     $                 lmltm,
     $                 pcell, ffield)

c     lower and upper faces of intrusion for transverse 
c     direction 2 (i.e., bottom and top faces)

                  call walmom_cc(
     $                 su_dragx_fcz, sp_dragx_fcz,
     $                 dragforcex_fcz, 
     $                 ug_cc, up_fcz, eps_Gas, 
     $                 eps_Solid,
     $                 sew, sns, zw, zz,
     $                 viscos, csmag,
     $                 idf, idt2, idt1, 
     $                 i, j, k, 
     $                 ioff, joff, koff,
     $                 joff, koff, ioff,
     $                 dim_lo, dim_hi, 
     $                 dim_lo_dx_fcz, dim_hi_dx_fcz,
     $                 dim_lo_su_fcz, dim_hi_su_fcz,
     $                 dim_lo_sp_fcz, dim_hi_sp_fcz,
     $                 dim_lo_ugc, dim_hi_ugc, 
     $                 dim_lo_upz, dim_hi_upz,
     $                 dim_lo_eps, dim_hi_eps, 
     $                 dim_lo_epss, dim_hi_epss, 
     $                 indexflo, indext2, indext1,
     $                 valid_lo_ext, valid_hi_ext,
     $                 valid_lo, valid_hi,
     $                 lmltm,
     $                 pcell, ffield)
#endif

               endif

 200        continue
 210     continue
 220  continue

      return
      end
