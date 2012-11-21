C
C The MIT License
C
C Copyright (c) 1997-2012 The University of Utah
C
C Permission is hereby granted, free of charge, to any person obtaining a copy
C of this software and associated documentation files (the "Software"), to
C deal in the Software without restriction, including without limitation the
C rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
C sell copies of the Software, and to permit persons to whom the Software is
C furnished to do so, subject to the following conditions:
C
C The above copyright notice and this permission notice shall be included in
C all copies or substantial portions of the Software.
C
C THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
C IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
C FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
C AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
C LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
C FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
C IN THE SOFTWARE.
C
C 
C 
#if 1
#include <CCA/Components/MPMArches/fortran/taucal_cc_fort.h>
#include "../../Arches/fortran/param4.h"
#else
      subroutine taucal_cc(
     $     su_fcy, 
     $     sp_fcy, 
     $     dfx_fcy, 
     $     ug_cc, 
     $     up_fcy, 
     $     epsg, 
     $     epss,
     $     sew, 
     $     stb, 
     $     yv, 
     $     yy,
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
c
c     Calculates the wall shear stress for the gas phase
c     written by Seshadri Kumar, August 8, 2000
c     
c     modified, January 8, 2001, Seshadri Kumar
c     Now taucal also calculates the drag force 
c     on the solid intrusions due to the fluid
c
c     modified, March 5, 2001
c     This is a cell-centered version.  Although dfx_fcy is
c     stored as (iorig,jorig,korig), this must be understood
c     to mean their face-centered representations.
c     Further, this version does not have anything for 
c     k-epsilon. However, it is augmented for Smagorinsky 
c     LES.
c     
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

      double precision su_fcy(
     $     dim_lo_su_fcy(1):dim_hi_su_fcy(1),
     $     dim_lo_su_fcy(2):dim_hi_su_fcy(2),
     $     dim_lo_su_fcy(3):dim_hi_su_fcy(3))

      double precision sp_fcy(
     $     dim_lo_sp_fcy(1):dim_hi_sp_fcy(1),
     $     dim_lo_sp_fcy(2):dim_hi_sp_fcy(2),
     $     dim_lo_sp_fcy(3):dim_hi_sp_fcy(3))

      double precision dfx_fcy(
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

      double precision epsg(
     $     dim_lo_eps(1):dim_hi_eps(1),
     $     dim_lo_eps(2):dim_hi_eps(2),
     $     dim_lo_eps(3):dim_hi_eps(3))

      double precision epss(
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

#endif

*     local variables
      
      double precision yp     
      double precision rmean
      double precision eps
      double precision tmult

      integer ii
      integer jj
      integer kk

      double precision rho
      double precision visturb

*     externals

C      external rmean

C--------------------------------------------------------------------

      if (pcell(i,j,k).eq.ffield) then 

         yp = sign*(yv(idt1)-yy(idt1-idist))
         eps = epsg(i,j,k)

         ii = idist*iorig + (1-idist)*i
         jj = idist*jorig + (1-idist)*j
         kk = idist*korig + (1-idist)*k

         if (lmltm) then

c     smagorinsky model: with delta near wall treated as
c     distance from wall, in analogy with classical mixing 
c     length model

            rho = den(i,j,k)
            
            visturb = (csmag*yp)**2*rho*
     $           abs((up_fcy(ii,jj,kk) - ug_cc(i,j,k))/
     $           yp)

            tmult = (visturb + eps*viscos)/yp

            tmult = tmult*epss(iorig,jorig,korig)

c     This kstabu causes a parallelization bug; we need to 
c     (perhaps) have a separate function for the stability
c     conditions for all four (momentum + energy) mm equations.
c     Seshadri Kumar, July 25, 2003
c
c            kstabu(i,j,k) = kstabu(i,j,k) + 
c     $           ((csmag*yp)**2*
c     $           abs((up_fcy(ii,jj,kk) - ug_cc(i,j,k))/yp)+
c     $           viscos/dmicr(i,j,k))*
c     $           epss(iorig,jorig,korig)*
c     $           sew(idf)*stb(idt2)/yp

         else

c     laminar flow

            tmult = eps*viscos/yp
            tmult = tmult*epss(iorig,jorig,korig)

c            kstabu(i,j,k) = kstabu(i,j,k) + 
c     $           (viscos/dmicr(i,j,k))*
c     $           epss(iorig,jorig,korig)*
c     $           sew(idf)*stb(idt2)/yp

         endif

         su_fcy(ii,jj,kk) = 
     $        su_fcy(ii,jj,kk) + 
     $        tmult*sew(idf)*stb(idt2)*up_fcy(ii,jj,kk)

         sp_fcy(ii,jj,kk) = 
     $        sp_fcy(ii,jj,kk) - 
     $        tmult*sew(idf)*stb(idt2)

         dfx_fcy(ii,jj,kk) = 
     $        dfx_fcy(ii,jj,kk) +
     $        tmult*sew(idf)*stb(idt2)*
     $        (ug_cc(i,j,k) - up_fcy(ii,jj,kk))

      end if

      return
      end
