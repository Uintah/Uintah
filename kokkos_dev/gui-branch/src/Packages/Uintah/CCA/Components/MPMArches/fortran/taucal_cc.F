      subroutine taucal_cc(
     $     su_drag_fcy, sp_drag_fcy, 
     $     dragforcex_fcy, 
     $     ug_cc, up_fcy, eps_Gas, 
     $     eps_Solid,
     $     sew, stb, yv, yy,
     $     viscos, csmag, sign,
     $     idist, 
     $     idf, idt1, idt2,
     $     i, j, k, 
     $     iorig, jorig, korig, 
     $     ioff, joff, koff, 
     $     dim_lo, dim_hi, 
     $     dim_lo_dx_fcy, dim_hi_dx_fcy,
     $     dim_lo_su_fcy, dim_hi_su_fcy,
     $     dim_lo_sp_fcy, dim_hi_sp_fcy,
     $     dim_lo_ugc, dim_hi_ugc, 
     $     dim_lo_upy, dim_hi_upy,
     $     dim_lo_eps, dim_hi_eps, 
     $     dim_lo_epss, dim_hi_epss, 
     $     indexflo, indext1, indext2,
     $     lmltm, 
     $     pcell, ffield)
C--------------------------------------------------------------------
c     calculates the wall shear stress for the gas phase
c     written by Seshadri Kumar, August 8, 2000
c     
c     modified, January 8, 2000, Seshadri Kumar
c     Now taucal also calculates the drag force 
c     on the solid intrusions due to the fluid
c
c     modified, March 5, 2001
c     This is a cell-centered version.  Although dragforcex_fcy is
c     stored as (iorig,jorig,korig), this must be understood
c     to mean their face-centered representations.
c     Further, this version does not have anything for k-epsilon
c     However, it is augmented for Smagorinsky LES.
C--------------------------------------------------------------------

      implicit none
      INCLUDE 'param4.h'
      
      integer dim_lo(3)
      integer dim_hi(3)

      integer dim_lo_dx_fcy(3)
      integer dim_hi_dx_fcy(3)

      integer dim_lo_su_fcy(3)
      integer dim_hi_su_fcy(3)

      integer dim_lo_sp_fcy(3)
      integer dim_hi_sp_fcy(3)

      integer dim_lo_ugc(3)
      integer dim_hi_ugc(3)

      integer dim_lo_upy(3)
      integer dim_hi_upy(3)

      integer dim_lo_eps(3)
      integer dim_hi_eps(3)

      integer dim_lo_epss(3)
      integer dim_hi_epss(3)

      integer indexflo
      integer indext1
      integer indext2

      double precision su_drag_fcy(
     $     dim_lo_su_fcy(1):dim_hi_su_fcy(1),
     $     dim_lo_su_fcy(2):dim_hi_su_fcy(2),
     $     dim_lo_su_fcy(3):dim_hi_su_fcy(3))

      double precision sp_drag_fcy(
     $     dim_lo_sp_fcy(1):dim_hi_sp_fcy(1),
     $     dim_lo_sp_fcy(2):dim_hi_sp_fcy(2),
     $     dim_lo_sp_fcy(3):dim_hi_sp_fcy(3))

      double precision dragforcex_fcy(
     $     dim_lo_dx_fcy(1):dim_hi_dx_fcy(1),
     $     dim_lo_dx_fcy(2):dim_hi_dx_fcy(2),
     $     dim_lo_dx_fcy(3):dim_hi_dx_fcy(3))

      double precision ug_cc(
     $     dim_lo_ugc(1):dim_hi_ugc(1),
     $     dim_lo_ugc(2):dim_hi_ugc(2),
     $     dim_lo_ugc(3):dim_hi_ugc(3))

      double precision up_fcy(
     $     dim_lo_upy(1):dim_hi_upy(1),
     $     dim_lo_upy(2):dim_hi_upy(2),
     $     dim_lo_upy(3):dim_hi_upy(3))

      double precision eps_Gas(
     $     dim_lo_eps(1):dim_hi_eps(1),
     $     dim_lo_eps(2):dim_hi_eps(2),
     $     dim_lo_eps(3):dim_hi_eps(3))

      double precision eps_Solid(
     $     dim_lo_epss(1):dim_hi_epss(1),
     $     dim_lo_epss(2):dim_hi_epss(2),
     $     dim_lo_epss(3):dim_hi_epss(3))

      double precision sew(dim_lo(indexflo):dim_hi(indexflo))
      double precision stb(dim_lo(indext2):dim_hi(indext2))

      double precision yv(dim_lo(indext1):dim_hi(indext1))
      double precision yy(dim_lo(indext1):dim_hi(indext1))

      double precision viscos
      double precision csmag
      double precision sign

      integer idist
      integer idf
      integer idt1
      integer idt2
      integer i
      integer j
      integer k
      integer iorig
      integer jorig
      integer korig
      integer ioff
      integer joff
      integer koff

      logical lmltm

      integer pcell(
     $     dim_lo(1):dim_hi(1),
     $     dim_lo(2):dim_hi(2),
     $     dim_lo(3):dim_hi(3))

      integer ffield

*     local variables
      
      double precision yp     
      double precision rmean
      double precision eps
      double precision tmult

      integer ii
      integer jj
      integer kk

*     externals

      external rmean

C--------------------------------------------------------------------

      if (pcell(i,j,k).eq.ffield) then 

         yp = sign*(yv(idt1)-yy(idt1-idist))
         eps = eps_Gas(i,j,k)

         ii = idist*iorig + (1-idist)*i
         jj = idist*jorig + (1-idist)*j
         kk = idist*korig + (1-idist)*k

         if (lmltm) then

c     smagorinsky model: with delta near wall treated as
c     distance from wall, in analogy with classical mixing length
c     model

            tmult = eps*(csmag*yp)**2*abs((up_fcy(ii,jj,kk) - 
     $           ug_cc(i,j,k))/yp)/yp
            tmult = tmult*eps_Solid(iorig,jorig,korig)

         else

c     laminar flow

            tmult = eps*viscos/yp
            tmult = tmult*eps_Solid(iorig,jorig,korig)

         end if

         su_drag_fcy(ii,jj,kk) = 
     $        su_drag_fcy(ii,jj,kk) + 
     $        tmult*sew(idf)*stb(idt2)*up_fcy(ii,jj,kk)

         sp_drag_fcy(ii,jj,kk) = 
     $        sp_drag_fcy(ii,jj,kk) - 
     $        tmult*sew(idf)*stb(idt2)

         dragforcex_fcy(ii,jj,kk) = 
     $        dragforcex_fcy(ii,jj,kk) +
     $        tmult*sew(idf)*stb(idt2)*
     $        (ug_cc(i,j,k) - up_fcy(ii,jj,kk))

      end if
      return
      end
