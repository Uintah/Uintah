#if 1
#include <Packages/Uintah/CCA/Components/MPMArches/fortran/energy_exchange_term_fort.h>
#include "../../Arches/fortran/param4.h"
#else
      subroutine energy_exchange_term(
     $     dim_lo_patch,   dim_hi_patch,
     $     dim_lo_pcell,   dim_hi_pcell,
     $     dim_lo_hts_fcx, dim_hi_hts_fcx,
     $     dim_lo_hts_fcy, dim_hi_hts_fcy,
     $     dim_lo_hts_fcz, dim_hi_hts_fcz,
     $     dim_lo_hts_cc,  dim_hi_hts_cc,
     $     dim_lo_sug_cc,  dim_hi_sug_cc,
     $     dim_lo_spg_cc,  dim_hi_spg_cc,
     $     dim_lo_sug_fcx, dim_hi_sug_fcx,
     $     dim_lo_spg_fcx, dim_hi_spg_fcx,
     $     dim_lo_sug_fcy, dim_hi_sug_fcy,
     $     dim_lo_spg_fcy, dim_hi_spg_fcy,
     $     dim_lo_sug_fcz, dim_hi_sug_fcz,
     $     dim_lo_spg_fcz, dim_hi_spg_fcz,
     $     dim_lo_tg,      dim_hi_tg,
     $     dim_lo_ts_cc,   dim_hi_ts_cc,
     $     dim_lo_ts_fcx,  dim_hi_ts_fcx,
     $     dim_lo_ts_fcy,  dim_hi_ts_fcy,
     $     dim_lo_ts_fcz,  dim_hi_ts_fcz,
     $     dim_lo_eps,     dim_hi_eps,
     $     dim_lo_epss,    dim_hi_epss,
     $     hts_fcx, 
     $     hts_fcy, 
     $     hts_fcz,
     $     hts_cc,
     $     sug_cc,
     $     spg_cc,
     $     sug_fcx, 
     $     spg_fcx,
     $     sug_fcy,
     $     spg_fcy,
     $     sug_fcz,
     $     spg_fcz,
     $     tg,
     $     ts_cc,
     $     ts_fcx,
     $     ts_fcy,
     $     ts_fcz,
     $     rfluxE,
     $     rfluxW,
     $     rfluxN,
     $     rfluxS,
     $     rfluxT,
     $     rfluxB,
     $     epsg, 
     $     epss,
     $     sew, sns, stb,
     $     heat_trans_coeff, 
     $     valid_lo, valid_hi,
     $     pcell, wall, ffield)

C-----------------------------------------------------------------------
c     Author: Seshadri Kumar, July 2, 2001
c     Last modified: July 2, 2001
c
C-----------------------------------------------------------------------

      implicit none
      include 'param4.h'
      include 'pconst.h'

      integer dim_lo_patch(3)
      integer dim_hi_patch(3)

      integer dim_lo_pcell(3)
      integer dim_hi_pcell(3)

      integer dim_lo_hts_fcx(3)
      integer dim_hi_hts_fcx(3)

      integer dim_lo_hts_fcy(3)
      integer dim_hi_hts_fcy(3)

      integer dim_lo_hts_fcz(3)
      integer dim_hi_hts_fcz(3)

      integer dim_lo_hts_cc(3)
      integer dim_hi_hts_cc(3)

      integer dim_lo_sug_cc(3)
      integer dim_hi_sug_cc(3)

      integer dim_lo_spg_cc(3)
      integer dim_hi_spg_cc(3)

      integer dim_lo_sug_fcx(3)
      integer dim_hi_sug_fcx(3)
      
      integer dim_lo_spg_fcx(3)
      integer dim_hi_spg_fcx(3)

      integer dim_lo_sug_fcy(3)
      integer dim_hi_sug_fcy(3)
      
      integer dim_lo_spg_fcy(3)
      integer dim_hi_spg_fcy(3)

      integer dim_lo_sug_fcz(3)
      integer dim_hi_sug_fcz(3)
      
      integer dim_lo_spg_fcz(3)
      integer dim_hi_spg_fcz(3)

      integer dim_lo_tg(3)
      integer dim_hi_tg(3)
      
      integer dim_lo_ts_cc(3)
      integer dim_hi_ts_cc(3)
      
      integer dim_lo_ts_fcx(3)
      integer dim_hi_ts_fcx(3)
      
      integer dim_lo_ts_fcy(3)
      integer dim_hi_ts_fcy(3)

      integer dim_lo_ts_fcz(3)
      integer dim_hi_ts_fcz(3)

      integer dim_lo_eps(3)
      integer dim_hi_eps(3)

      integer dim_lo_epss(3)
      integer dim_hi_epss(3)

      double precision hts_fcx(
     $     dim_lo_hts_fcx(1):dim_hi_hts_fcx(1),
     $     dim_lo_hts_fcx(2):dim_hi_hts_fcx(2),
     $     dim_lo_hts_fcx(3):dim_hi_hts_fcx(3))

      double precision hts_fcy(
     $     dim_lo_hts_fcy(1):dim_hi_hts_fcy(1),
     $     dim_lo_hts_fcy(2):dim_hi_hts_fcy(2),
     $     dim_lo_hts_fcy(3):dim_hi_hts_fcy(3))

      double precision hts_fcz(
     $     dim_lo_hts_fcz(1):dim_hi_hts_fcz(1),
     $     dim_lo_hts_fcz(2):dim_hi_hts_fcz(2),
     $     dim_lo_hts_fcz(3):dim_hi_hts_fcz(3))      

      double precision hts_fcx_rad(
     $     dim_lo_hts_fcx(1):dim_hi_hts_fcx(1),
     $     dim_lo_hts_fcx(2):dim_hi_hts_fcx(2),
     $     dim_lo_hts_fcx(3):dim_hi_hts_fcx(3))

      double precision hts_fcy_rad(
     $     dim_lo_hts_fcy(1):dim_hi_hts_fcy(1),
     $     dim_lo_hts_fcy(2):dim_hi_hts_fcy(2),
     $     dim_lo_hts_fcy(3):dim_hi_hts_fcy(3))

      double precision hts_fcz_rad(
     $     dim_lo_hts_fcz(1):dim_hi_hts_fcz(1),
     $     dim_lo_hts_fcz(2):dim_hi_hts_fcz(2),
     $     dim_lo_hts_fcz(3):dim_hi_hts_fcz(3))      

      double precision hts_cc(
     $     dim_lo_hts_cc(1):dim_hi_hts_cc(1),
     $     dim_lo_hts_cc(2):dim_hi_hts_cc(2),
     $     dim_lo_hts_cc(3):dim_hi_hts_cc(3))      

      double precision sug_cc(
     $     dim_lo_sug_cc(1):dim_hi_sug_cc(1),
     $     dim_lo_sug_cc(2):dim_hi_sug_cc(2),
     $     dim_lo_sug_cc(3):dim_hi_sug_cc(3))      

      double precision spg_cc(
     $     dim_lo_spg_cc(1):dim_hi_spg_cc(1),
     $     dim_lo_spg_cc(2):dim_hi_spg_cc(2),
     $     dim_lo_spg_cc(3):dim_hi_spg_cc(3))      

      double precision sug_fcx(
     $     dim_lo_sug_fcx(1):dim_hi_sug_fcx(1),
     $     dim_lo_sug_fcx(2):dim_hi_sug_fcx(2),
     $     dim_lo_sug_fcx(3):dim_hi_sug_fcx(3))      

      double precision spg_fcx(
     $     dim_lo_spg_fcx(1):dim_hi_spg_fcx(1),
     $     dim_lo_spg_fcx(2):dim_hi_spg_fcx(2),
     $     dim_lo_spg_fcx(3):dim_hi_spg_fcx(3))      

      double precision sug_fcy(
     $     dim_lo_sug_fcy(1):dim_hi_sug_fcy(1),
     $     dim_lo_sug_fcy(2):dim_hi_sug_fcy(2),
     $     dim_lo_sug_fcy(3):dim_hi_sug_fcy(3))      

      double precision spg_fcy(
     $     dim_lo_spg_fcy(1):dim_hi_spg_fcy(1),
     $     dim_lo_spg_fcy(2):dim_hi_spg_fcy(2),
     $     dim_lo_spg_fcy(3):dim_hi_spg_fcy(3))      

      double precision sug_fcz(
     $     dim_lo_sug_fcz(1):dim_hi_sug_fcz(1),
     $     dim_lo_sug_fcz(2):dim_hi_sug_fcz(2),
     $     dim_lo_sug_fcz(3):dim_hi_sug_fcz(3))      

      double precision spg_fcz(
     $     dim_lo_spg_fcz(1):dim_hi_spg_fcz(1),
     $     dim_lo_spg_fcz(2):dim_hi_spg_fcz(2),
     $     dim_lo_spg_fcz(3):dim_hi_spg_fcz(3))      

      double precision tg(
     $     dim_lo_tg(1):dim_hi_tg(1),
     $     dim_lo_tg(2):dim_hi_tg(2),
     $     dim_lo_tg(3):dim_hi_tg(3))      

      double precision ts_cc(
     $     dim_lo_ts_cc(1):dim_hi_ts_cc(1),
     $     dim_lo_ts_cc(2):dim_hi_ts_cc(2),
     $     dim_lo_ts_cc(3):dim_hi_ts_cc(3))      

      double precision ts_fcx(
     $     dim_lo_ts_fcx(1):dim_hi_ts_fcx(1),
     $     dim_lo_ts_fcx(2):dim_hi_ts_fcx(2),
     $     dim_lo_ts_fcx(3):dim_hi_ts_fcx(3))      

      double precision ts_fcy(
     $     dim_lo_ts_fcy(1):dim_hi_ts_fcy(1),
     $     dim_lo_ts_fcy(2):dim_hi_ts_fcy(2),
     $     dim_lo_ts_fcy(3):dim_hi_ts_fcy(3))      

      double precision ts_fcz(
     $     dim_lo_ts_fcz(1):dim_hi_ts_fcz(1),
     $     dim_lo_ts_fcz(2):dim_hi_ts_fcz(2),
     $     dim_lo_ts_fcz(3):dim_hi_ts_fcz(3))      

      double precision rfluxE(
     $     dim_lo_epss(1):dim_hi_epss(1),
     $     dim_lo_epss(2):dim_hi_epss(2),
     $     dim_lo_epss(3):dim_hi_epss(3))

      double precision rfluxW(
     $     dim_lo_epss(1):dim_hi_epss(1),
     $     dim_lo_epss(2):dim_hi_epss(2),
     $     dim_lo_epss(3):dim_hi_epss(3))

      double precision rfluxN(
     $     dim_lo_epss(1):dim_hi_epss(1),
     $     dim_lo_epss(2):dim_hi_epss(2),
     $     dim_lo_epss(3):dim_hi_epss(3))

      double precision rfluxS(
     $     dim_lo_epss(1):dim_hi_epss(1),
     $     dim_lo_epss(2):dim_hi_epss(2),
     $     dim_lo_epss(3):dim_hi_epss(3))

      double precision rfluxT(
     $     dim_lo_epss(1):dim_hi_epss(1),
     $     dim_lo_epss(2):dim_hi_epss(2),
     $     dim_lo_epss(3):dim_hi_epss(3))

      double precision rfluxB(
     $     dim_lo_epss(1):dim_hi_epss(1),
     $     dim_lo_epss(2):dim_hi_epss(2),
     $     dim_lo_epss(3):dim_hi_epss(3))

      double precision epsg(
     $     dim_lo_eps(1):dim_hi_eps(1),
     $     dim_lo_eps(2):dim_hi_eps(2),
     $     dim_lo_eps(3):dim_hi_eps(3))

      double precision epss(
     $     dim_lo_epss(1):dim_hi_epss(1),
     $     dim_lo_epss(2):dim_hi_epss(2),
     $     dim_lo_epss(3):dim_hi_epss(3))

      double precision sew(dim_lo_patch(1):dim_hi_patch(1))
      double precision sns(dim_lo_patch(2):dim_hi_patch(2))
      double precision stb(dim_lo_patch(3):dim_hi_patch(3))

      double precision heat_trans_coeff

      integer valid_lo(3)
      integer valid_hi(3)

      integer pcell(
     $     dim_lo_pcell(1):dim_hi_pcell(1),
     $     dim_lo_pcell(2):dim_hi_pcell(2),
     $     dim_lo_pcell(3):dim_hi_pcell(3))

      integer wall
      integer ffield

#endif

*     local variables

      integer i
      integer j
      integer k

      double precision areayz
      double precision areaxz
      double precision areaxy

      double precision qmult

      double precision eps_gas_cont
      double precision darea
      double precision area
      double precision factor

c--------------------------------------------------------------------
*     executable statements
c--------------------------------------------------------------------

c--------------------------------------------------------------------
c     part 1: intrusion cells and the cells next to them
c--------------------------------------------------------------------

*     west face of intrusion

      do 120 k = valid_lo(3), valid_hi(3)
         do 110 j = valid_lo(2), valid_hi(2)
            do 100 i = valid_lo(1), valid_hi(1)

               if (pcell(i,j,k) .eq. wall) then

                  if (pcell(i-1,j,k) .eq. ffield) then

                     areayz = sns(j)*stb(k)
                     qmult = heat_trans_coeff*areayz*
     $                    epss(i,j,k)*epsg(i-1,j,k)

c     We multiply by epsg because we want to only include the
c     area that is actually seen by the gas.  The assumption made
c     here is that the area fraction is well represented by the 
c     volume fraction.  This of course implies that we need to account
c     for the heat transfer in partially filled cells separately.
c     We further multiply by epss because this is 
c     the fraction of the total heat transfer that is passed on
c     to the individual MPM material that has this epss passed
c     into this code.  Note that since in a wall cell, there is no
c     gas, epss is the absolute solid fraction of the current
c     MPM material (relative to all solid and gas materials) as 
c     well as its relative fraction (relative to all MPM materials
c     alone,) so we don't need to divide by (1-epsg).

                     hts_fcx(i,j,k) = 
     $                    hts_fcx(i,j,k) + qmult*
     $                    (tg(i-1,j,k) - 
     $                    ts_fcx(i,j,k))

                     sug_fcx(i,j,k) = sug_fcx(i,j,k) -
     $                    qmult*
     $                    (tg(i-1,j,k) - 
     $                    ts_fcx(i,j,k))

c#if 0
c     radiative heat transfer to intrusions
c     note that at this point (November 9, 2002) we
c     have no model for partial cells. Qn: should there
c     be a multiplication by epsg, since in a partial
c     cell not all the intrusion is seen by the gas?
c     That would require a model for partial cells
c     as well.

                     hts_fcx(i,j,k) = 
     $                    hts_fcx(i,j,k) + 
     $                    rfluxE(i-1,j,k)*areayz*
     $                    epss(i,j,k)

                     hts_fcx_rad(i,j,k) = 
     $                    hts_fcx_rad(i,j,k) + 
     $                    rfluxE(i-1,j,k)*areayz*
     $                    epss(i,j,k)

c#endif

#ifdef debugEnergyExchange
                     print*, i,j,k, hts_fcx(i,j,k), 
     $                    tg(i-1,j,k), ts_fcx(i,j,k),
     $                    epsg(i-1,j,k)
#endif

                  endif

               endif
               
 100        continue
 110     continue
 120  continue

*     east face of intrusion

      do 220 k = valid_lo(3), valid_hi(3)
         do 210 j = valid_lo(2), valid_hi(2)
            do 200 i = valid_lo(1)-1, valid_hi(1)-1

               if (pcell(i,j,k).eq.wall) then

                  if (pcell(i+1,j,k).eq.ffield) then

                     areayz = sns(j)*stb(k)
                     qmult = heat_trans_coeff*areayz*
     $                    epss(i,j,k)*epsg(i+1,j,k)

                     hts_fcx(i+1,j,k) = 
     $                    hts_fcx(i+1,j,k) + qmult*
     $                    (tg(i+1,j,k) - 
     $                    ts_fcx(i,j,k))

                     sug_fcx(i+1,j,k) = sug_fcx(i+1,j,k) -
     $                    qmult*
     $                    (tg(i+1,j,k) - 
     $                    ts_fcx(i,j,k))

c#if 0
c     radiative heat transfer to intrusions
c     note that at this point (November 9, 2002) we
c     have no model for partial cells. Qn: should there
c     be a multiplication by epsg, since in a partial
c     cell not all the intrusion is seen by the gas?
c     That would require a model for partial cells
c     as well.

                     hts_fcx(i+1,j,k) = 
     $                    hts_fcx(i+1,j,k) + 
     $                    rfluxW(i+1,j,k)*areayz*
     $                    epss(i,j,k)

                     hts_fcx_rad(i+1,j,k) = 
     $                    hts_fcx_rad(i+1,j,k) + 
     $                    rfluxW(i+1,j,k)*areayz*
     $                    epss(i,j,k)

c#endif

#ifdef debugEnergyExchange
                     print*, i+1,j,k, hts_fcx(i+1,j,k),
     $                    tg(i+1,j,k), ts_fcx(i,j,k),
     $                    epsg(i+1,j,k)
#endif

                  endif

               endif

 200        continue
 210     continue
 220  continue

c     south face of intrusion

      do 320 k = valid_lo(3), valid_hi(3)
         do 310 j = valid_lo(2), valid_hi(2)
            do 300 i = valid_lo(1), valid_hi(1)

               if (pcell(i,j,k).eq.wall) then

                  if (pcell(i,j-1,k) .eq. ffield) then

                     areaxz = sew(i)*stb(k)
                     qmult = heat_trans_coeff*areaxz*
     $                    epsg(i,j-1,k)*epss(i,j,k)

                     hts_fcy(i,j,k) = 
     $                    hts_fcy(i,j,k) + qmult*
     $                    (tg(i,j-1,k) -
     $                    ts_fcy(i,j,k))

                     sug_fcy(i,j,k) = sug_fcy(i,j,k) -
     $                    qmult*
     $                    (tg(i,j-1,k) -
     $                    ts_fcy(i,j,k))

c#if 0
c     radiative heat transfer to intrusions
c     note that at this point (November 9, 2002) we
c     have no model for partial cells. Qn: should there
c     be a multiplication by epsg, since in a partial
c     cell not all the intrusion is seen by the gas?
c     That would require a model for partial cells
c     as well.

                     hts_fcy(i,j,k) = 
     $                    hts_fcy(i,j,k) +
     $                    rfluxN(i,j-1,k)*areaxz*
     $                    epss(i,j,k)

                     hts_fcy_rad(i,j,k) = 
     $                    hts_fcy_rad(i,j,k) +
     $                    rfluxN(i,j-1,k)*areaxz*
     $                    epss(i,j,k)

c#endif

#ifdef debugEnergyExchange
                     print*, i,j,k, hts_fcy(i,j,k),
     $                    tg(i,j-1,k), ts_fcy(i,j,k),
     $                    epsg(i,j-1,k)
#endif

                  endif

               endif

 300        continue
 310     continue
 320  continue

*     north face of intrusion

      do 420 k = valid_lo(3), valid_hi(3)
         do 410 j = valid_lo(2)-1, valid_hi(2)-1
            do 400 i = valid_lo(1), valid_hi(1)

               if (pcell(i,j,k) .eq. wall) then

                  if (pcell(i,j+1,k).eq.ffield) then

                     areaxz = sew(i)*stb(k)
                     qmult = heat_trans_coeff*areaxz*
     $                    epsg(i,j+1,k)*epss(i,j,k)

                     hts_fcy(i,j+1,k) = 
     $                    hts_fcy(i,j+1,k) + qmult*
     $                    (tg(i,j+1,k) - 
     $                    ts_fcx(i,j,k))

                     sug_fcy(i,j+1,k) = sug_fcy(i,j+1,k) -
     $                    qmult*
     $                    (tg(i,j+1,k) - 
     $                    ts_fcx(i,j,k))

c#if 0
c     radiative heat transfer to intrusions
c     note that at this point (November 9, 2002) we
c     have no model for partial cells. Qn: should there
c     be a multiplication by epsg, since in a partial
c     cell not all the intrusion is seen by the gas?
c     That would require a model for partial cells
c     as well.

                     hts_fcy(i,j+1,k) = 
     $                    hts_fcy(i,j+1,k) +
     $                    rfluxS(i,j+1,k)*areaxz*
     $                    epss(i,j,k)

                     hts_fcy_rad(i,j+1,k) = 
     $                    hts_fcy_rad(i,j+1,k) +
     $                    rfluxS(i,j+1,k)*areaxz*
     $                    epss(i,j,k)

c#endif

#ifdef debugEnergyExchange
                     print*, i,j+1,k, hts_fcy(i,j+1,k),
     $                    tg(i,j+1,k), ts_fcx(i,j,k),
     $                    epsg(i,j+1,k)
#endif

                  endif

               endif

 400        continue
 410     continue
 420  continue

*     bottom face

      do 520 k = valid_lo(3), valid_hi(3)
         do 510 j = valid_lo(2), valid_hi(2)
            do 500 i = valid_lo(1), valid_hi(1)

               if (pcell(i,j,k) .eq. wall) then

                  if (pcell(i,j,k-1).eq.ffield) then

                     areaxy = sew(i)*sns(j)
                     qmult = heat_trans_coeff*areaxy*
     $                    epsg(i,j,k-1)*epss(i,j,k)

                     hts_fcz(i,j,k) = 
     $                    hts_fcz(i,j,k) + qmult*
     $                    (tg(i,j,k-1) - 
     $                    ts_fcz(i,j,k))

                     sug_fcz(i,j,k) = sug_fcz(i,j,k) -
     $                    qmult*
     $                    (tg(i,j,k-1) - 
     $                    ts_fcz(i,j,k))

c#if 0
c     radiative heat transfer to intrusions
c     note that at this point (November 9, 2002) we
c     have no model for partial cells. Qn: should there
c     be a multiplication by epsg, since in a partial
c     cell not all the intrusion is seen by the gas?
c     That would require a model for partial cells
c     as well.

                      hts_fcz(i,j,k) = 
     $                    hts_fcz(i,j,k) +
     $                    rfluxT(i,j,k-1)*areaxy*
     $                    epss(i,j,k)

                      hts_fcz_rad(i,j,k) = 
     $                    hts_fcz_rad(i,j,k) +
     $                    rfluxT(i,j,k-1)*areaxy*
     $                    epss(i,j,k)

c#endif

#ifdef debugEnergyExchange
                     print*, i,j,k, hts_fcz(i,j,k),
     $                    tg(i,j,k-1), ts_fcz(i,j,k),
     $                    epsg(i,j,k-1)
#endif

                  endif

               endif

 500        continue
 510     continue
 520  continue

*     top face of intrusion

      do 620 k = valid_lo(3)-1, valid_hi(3)-1
         do 610 j = valid_lo(2), valid_hi(2)
            do 600 i = valid_lo(1), valid_hi(1)

               if (pcell(i,j,k).eq.wall) then

                  if (pcell(i,j,k+1).eq.ffield) then

                     areaxy = sew(i)*sns(j)
                     qmult = heat_trans_coeff*areaxy*
     $                    epsg(i,j,k+1)*epss(i,j,k)

                     hts_fcz(i,j,k+1) = 
     $                    hts_fcz(i,j,k+1) + qmult*
     $                    (tg(i,j,k+1) - 
     $                    ts_fcz(i,j,k))

                     sug_fcz(i,j,k+1) = sug_fcz(i,j,k+1) -
     $                    qmult*
     $                    (tg(i,j,k+1) - 
     $                    ts_fcz(i,j,k))

c#if 0
c     radiative heat transfer to intrusions
c     note that at this point (November 9, 2002) we
c     have no model for partial cells. Qn: should there
c     be a multiplication by epsg, since in a partial
c     cell not all the intrusion is seen by the gas?
c     That would require a model for partial cells
c     as well.

                     hts_fcz(i,j,k+1) =
     $                    hts_fcz(i,j,k+1) + 
     $                    rfluxB(i,j,k+1)*areaxy*
     $                    epss(i,j,k)

                     hts_fcz_rad(i,j,k+1) =
     $                    hts_fcz_rad(i,j,k+1) + 
     $                    rfluxB(i,j,k+1)*areaxy*
     $                    epss(i,j,k)

c#endif

#ifdef debugEnergyExchange
                     print*, i,j,k+1, hts_fcz(i,j,k+1),
     $                    tg(i,j,k+1), ts_fcz(i,j,k),
     $                    epsg(i,j,k+1)
#endif

                  endif

               endif

 600        continue
 610     continue
 620  continue

#ifdef debugEnergyExchange
      print*, 'now for mixed cells'
#endif

c     part 2: cells that have both fluid and solid
c     For now, this has only convective heat transfer.  When we
c     do complex geometry treatment, then we will consider heat
c     transfer to partial cells as well.

      do 720 i = valid_lo(1), valid_hi(1)
         do 710 j = valid_lo(2), valid_hi(2)
            do 700 k = valid_lo(3), valid_hi(3)

c     modif. for discrete fraction a la mom exch?

               eps_gas_cont = epsg(i,j,k)
               if (eps_gas_cont.lt.(one - small).and. 
     $              pcell(i,j,k).ne.wall) then

                  darea = ((one - eps_gas_cont)*
     $                 sew(i)*sns(j)*stb(k))**(1.0/3.0)
                  area = darea*darea

c     this is the area of solid actually exposed to the gas

                  factor = epss(i,j,k)/(one - epsg(i,j,k))

c     This factor is the fraction of the total heat transfer
c     attributable to the current solid material whose epsg is
c     being passed here.  If there is only one solid material,
c     then factor = one.

                  qmult = heat_trans_coeff*area*factor
                  hts_cc(i,j,k) = 
     $                 hts_cc(i,j,k) + qmult*
     $                 (tg(i,j,k) - ts_cc(i,j,k))

                  sug_cc(i,j,k) = sug_cc(i,j,k) +
     $                 qmult*(ts_cc(i,j,k) -
     $                 tg(i,j,k))

#ifdef debugEnergyExchange
                  print*, i,j,k, hts_cc(i,j,k),
     $                 tg(i,j,k), ts_cc(i,j,k),
     $                 epsg(i,j,k)
#endif
                     
               end if

 700        continue
 710     continue
 720  continue

      return
      end
