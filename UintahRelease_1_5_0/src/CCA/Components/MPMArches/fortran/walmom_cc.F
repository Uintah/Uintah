C
C Copyright (c) 1997-2012 The University of Utah
C
C Permission is hereby granted, free of charge, to any person obtaining a copy
C of this software and associated documentation files (the \"Software\"), to
C deal in the Software without restriction, including without limitation the
C rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
C sell copies of the Software, and to permit persons to whom the Software is
C furnished to do so, subject to the following conditions:
C
C The above copyright notice and this permission notice shall be included in
C all copies or substantial portions of the Software.
C
C THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
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
#include <CCA/Components/MPMArches/fortran/walmom_cc_fort.h>
#include "../../Arches/fortran/param4.h"
#else
      subroutine walmom_cc(
     $     sux_fcy, 
     $     spx_fcy, 
     $     dfx_fcy, 
     $     ug_cc, 
     $     up_fcy, 
     $     epsg, 
     $     epss,
     $     sew, stb, yv, yy,
     $     viscos, csmag, 
     $     idf, idt1, idt2, 
     $     i, j, k, 
     $     ioff, joff, koff,
     $     ioff2, joff2, koff2, 
     $     dim_lo, dim_hi, 
     $     dim_lo_dx_fcy, dim_hi_dx_fcy,
     $     dim_lo_su_fcy, dim_hi_su_fcy,
     $     dim_lo_sp_fcy, dim_hi_sp_fcy,
     $     dim_lo_ugc, dim_hi_ugc, 
     $     dim_lo_upy, dim_hi_upy,
     $     dim_lo_eps, dim_hi_eps, 
     $     dim_lo_epss, dim_hi_epss,
     $     indexflo, indext1, indext2,
     $     valid_lo_ext, valid_hi_ext, 
     $     valid_lo, valid_hi,
     $     lmltm,
     $     pcell, ffield)
c--------------------------------------------------------------------
      implicit none
      include 'param4.h'
c--------------------------------------------------------------------

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

      integer valid_lo_ext(3)
      integer valid_hi_ext(3)
      integer valid_lo(3)
      integer valid_hi(3)

      double precision sux_fcy(
     $     dim_lo_su_fcy(1):dim_hi_su_fcy(1),
     $     dim_lo_su_fcy(2):dim_hi_su_fcy(2),
     $     dim_lo_su_fcy(3):dim_hi_su_fcy(3))

      double precision spx_fcy(
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
      
      integer idf
      integer idt1
      integer idt2

      integer i
      integer j
      integer k

      integer ioff
      integer joff
      integer koff

      integer ioff2
      integer joff2
      integer koff2

      logical lmltm

      integer pcell(
     $     dim_lo(1):dim_hi(1),
     $     dim_lo(2):dim_hi(2),
     $     dim_lo(3):dim_hi(3))

      integer ffield

#endif

*     executable statements

c--------------------------------------------------------------------
c             lower face of intrusion\wall
c--------------------------------------------------------------------

      if (idt1.ne.valid_lo_ext(indext1) .and.
     $     idf.ne.valid_lo_ext(indexflo) .and.
     $     idt2.ne.valid_lo_ext(indext2)) then

         call taucal_cc(
     $        PASS3A(sux_fcy)
     $        PASS3B(sux_fcy), 
     $        PASS3A(spx_fcy)
     $        PASS3B(spx_fcy), 
     $        PASS3A(dfx_fcy)
     $        PASS3B(dfx_fcy), 
     $        PASS3A(kstabu)
     $        PASS3B(kstabu), 
     $        PASS3A(ug_cc)
     $        PASS3B(ug_cc), 
     $        PASS3A(up_fcy)
     $        PASS3B(up_fcy), 
     $        PASS3A(epsg)
     $        PASS3B(epsg), 
     $        PASS3A(den)
     $        PASS3B(den), 
     $        PASS3A(dmicr)
     $        PASS3B(dmicr), 
     $        PASS3A(epss)
     $        PASS3B(epss),
     $        PASS1(sew), 
     $        PASS1(stb), 
     $        PASS1(yv), 
     $        PASS1(yy),
     $        viscos, csmag, one,
     $        1,
     $        idf, idt1, idt2,
     $        i-ioff2, j-joff2, k-koff2, 
     $        i, j, k, 
     $        ioff, joff, koff,
     $        indexflo, indext1, indext2,
     $        lmltm,
     $        PASS3A(pcell)
     $        PASS3B(pcell), 
     $        ffield)

      end if

c--------------------------------------------------------------------
c             upper face of intrusion\wall
c--------------------------------------------------------------------

      if (idt1.ne.valid_hi(indext1) .and.
     $     idf.ne.valid_lo_ext(indexflo) .and.
     $     idt2.ne.valid_lo_ext(indext2)) then

         call taucal_cc(
     $        PASS3A(sux_fcy)
     $        PASS3B(sux_fcy), 
     $        PASS3A(spx_fcy)
     $        PASS3B(spx_fcy), 
     $        PASS3A(dfx_fcy)
     $        PASS3B(dfx_fcy), 
     $        PASS3A(kstabu)
     $        PASS3B(kstabu), 
     $        PASS3A(ug_cc)
     $        PASS3B(ug_cc), 
     $        PASS3A(up_fcy)
     $        PASS3B(up_fcy), 
     $        PASS3A(epsg)
     $        PASS3B(epsg), 
     $        PASS3A(den)
     $        PASS3B(den), 
     $        PASS3A(dmicr)
     $        PASS3B(dmicr), 
     $        PASS3A(epss)
     $        PASS3B(epss), 
     $        PASS1(sew), 
     $        PASS1(stb), 
     $        PASS1(yv), 
     $        PASS1(yy),
     $        viscos, csmag, -one,
     $        0, 
     $        idf, idt1+1, idt2,
     $        i+ioff2, j+joff2, k+koff2, 
     $        i, j, k, 
     $        ioff, joff, koff, 
     $        indexflo, indext1, indext2,
     $        lmltm,
     $        PASS3A(pcell)
     $        PASS3B(pcell), 
     $        ffield)

      end if

      return
      end
