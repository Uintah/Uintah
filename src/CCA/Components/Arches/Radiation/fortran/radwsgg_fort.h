
#ifndef fspec_radwsgg
#define fspec_radwsgg

#ifdef __cplusplus

#include <CCA/Components/Arches/Radiation/fortran/FortranNameMangle.h>

extern "C" void F_radwsgg(int* idxlo,
                        int* idxhi,
                        int* abskg_low_x, int* abskg_low_y, int* abskg_low_z, int* abskg_high_x, int* abskg_high_y, int* abskg_high_z, double* abskg_ptr,
                        int* esrcg_low_x, int* esrcg_low_y, int* esrcg_low_z, int* esrcg_high_x, int* esrcg_high_y, int* esrcg_high_z, double* esrcg_ptr,
                        int* shgamma_low_x, int* shgamma_low_y, int* shgamma_low_z, int* shgamma_high_x, int* shgamma_high_y, int* shgamma_high_z, double* shgamma_ptr,
                        int* bands,
                        int* pcell_low_x, int* pcell_low_y, int* pcell_low_z, int* pcell_high_x, int* pcell_high_y, int* pcell_high_z, int* pcell_ptr,
                        int* ffield,
                        int* co2_low_x, int* co2_low_y, int* co2_low_z, int* co2_high_x, int* co2_high_y, int* co2_high_z, double* co2_ptr,
                        int* h2o_low_x, int* h2o_low_y, int* h2o_low_z, int* h2o_high_x, int* h2o_high_y, int* h2o_high_z, double* h2o_ptr,
                        int* sfv_low_x, int* sfv_low_y, int* sfv_low_z, int* sfv_high_x, int* sfv_high_y, int* sfv_high_z, double* sfv_ptr,
                        int* tg_low_x, int* tg_low_y, int* tg_low_z, int* tg_high_x, int* tg_high_y, int* tg_high_z, double* tg_ptr,
                        int* lambda,
                        int* fraction_low, int* fraction_high, double* fraction_ptr,
                        int* fractiontwo_low, int* fractiontwo_high, double* fractiontwo_ptr);

static void fort_radwsgg( Uintah::IntVector & idxlo,
                          Uintah::IntVector & idxhi,
                          Uintah::CCVariable<double> & abskg,
                          Uintah::CCVariable<double> & esrcg,
                          Uintah::CCVariable<double> & shgamma,
                          int & bands,
                          Uintah::constCCVariable<int> & pcell,
                          int & ffield,
                          Uintah::constCCVariable<double> & co2,
                          Uintah::constCCVariable<double> & h2o,
                          Uintah::constCCVariable<double> & sfv,
                          Uintah::CCVariable<double> & tg,
                          int & lambda,
                          Uintah::OffsetArray1<double> & fraction,
                          Uintah::OffsetArray1<double> & fractiontwo )
{
  Uintah::IntVector abskg_low = abskg.getWindow()->getOffset();
  Uintah::IntVector abskg_high = abskg.getWindow()->getData()->size() + abskg_low - Uintah::IntVector(1, 1, 1);
  int abskg_low_x = abskg_low.x();
  int abskg_high_x = abskg_high.x();
  int abskg_low_y = abskg_low.y();
  int abskg_high_y = abskg_high.y();
  int abskg_low_z = abskg_low.z();
  int abskg_high_z = abskg_high.z();
  Uintah::IntVector esrcg_low = esrcg.getWindow()->getOffset();
  Uintah::IntVector esrcg_high = esrcg.getWindow()->getData()->size() + esrcg_low - Uintah::IntVector(1, 1, 1);
  int esrcg_low_x = esrcg_low.x();
  int esrcg_high_x = esrcg_high.x();
  int esrcg_low_y = esrcg_low.y();
  int esrcg_high_y = esrcg_high.y();
  int esrcg_low_z = esrcg_low.z();
  int esrcg_high_z = esrcg_high.z();
  Uintah::IntVector shgamma_low = shgamma.getWindow()->getOffset();
  Uintah::IntVector shgamma_high = shgamma.getWindow()->getData()->size() + shgamma_low - Uintah::IntVector(1, 1, 1);
  int shgamma_low_x = shgamma_low.x();
  int shgamma_high_x = shgamma_high.x();
  int shgamma_low_y = shgamma_low.y();
  int shgamma_high_y = shgamma_high.y();
  int shgamma_low_z = shgamma_low.z();
  int shgamma_high_z = shgamma_high.z();
  Uintah::IntVector pcell_low = pcell.getWindow()->getOffset();
  Uintah::IntVector pcell_high = pcell.getWindow()->getData()->size() + pcell_low - Uintah::IntVector(1, 1, 1);
  int pcell_low_x = pcell_low.x();
  int pcell_high_x = pcell_high.x();
  int pcell_low_y = pcell_low.y();
  int pcell_high_y = pcell_high.y();
  int pcell_low_z = pcell_low.z();
  int pcell_high_z = pcell_high.z();
  Uintah::IntVector co2_low = co2.getWindow()->getOffset();
  Uintah::IntVector co2_high = co2.getWindow()->getData()->size() + co2_low - Uintah::IntVector(1, 1, 1);
  int co2_low_x = co2_low.x();
  int co2_high_x = co2_high.x();
  int co2_low_y = co2_low.y();
  int co2_high_y = co2_high.y();
  int co2_low_z = co2_low.z();
  int co2_high_z = co2_high.z();
  Uintah::IntVector h2o_low = h2o.getWindow()->getOffset();
  Uintah::IntVector h2o_high = h2o.getWindow()->getData()->size() + h2o_low - Uintah::IntVector(1, 1, 1);
  int h2o_low_x = h2o_low.x();
  int h2o_high_x = h2o_high.x();
  int h2o_low_y = h2o_low.y();
  int h2o_high_y = h2o_high.y();
  int h2o_low_z = h2o_low.z();
  int h2o_high_z = h2o_high.z();
  Uintah::IntVector sfv_low = sfv.getWindow()->getOffset();
  Uintah::IntVector sfv_high = sfv.getWindow()->getData()->size() + sfv_low - Uintah::IntVector(1, 1, 1);
  int sfv_low_x = sfv_low.x();
  int sfv_high_x = sfv_high.x();
  int sfv_low_y = sfv_low.y();
  int sfv_high_y = sfv_high.y();
  int sfv_low_z = sfv_low.z();
  int sfv_high_z = sfv_high.z();
  Uintah::IntVector tg_low = tg.getWindow()->getOffset();
  Uintah::IntVector tg_high = tg.getWindow()->getData()->size() + tg_low - Uintah::IntVector(1, 1, 1);
  int tg_low_x = tg_low.x();
  int tg_high_x = tg_high.x();
  int tg_low_y = tg_low.y();
  int tg_high_y = tg_high.y();
  int tg_low_z = tg_low.z();
  int tg_high_z = tg_high.z();
  int fraction_low = fraction.low();
  int fraction_high = fraction.high();
  int fractiontwo_low = fractiontwo.low();
  int fractiontwo_high = fractiontwo.high();
  F_radwsgg( idxlo.get_pointer(),
           idxhi.get_pointer(),
           &abskg_low_x, &abskg_low_y, &abskg_low_z, &abskg_high_x, &abskg_high_y, &abskg_high_z, abskg.getPointer(),
           &esrcg_low_x, &esrcg_low_y, &esrcg_low_z, &esrcg_high_x, &esrcg_high_y, &esrcg_high_z, esrcg.getPointer(),
           &shgamma_low_x, &shgamma_low_y, &shgamma_low_z, &shgamma_high_x, &shgamma_high_y, &shgamma_high_z, shgamma.getPointer(),
           &bands,
           &pcell_low_x, &pcell_low_y, &pcell_low_z, &pcell_high_x, &pcell_high_y, &pcell_high_z, const_cast<int*>(pcell.getPointer()),
           &ffield,
           &co2_low_x, &co2_low_y, &co2_low_z, &co2_high_x, &co2_high_y, &co2_high_z, const_cast<double*>(co2.getPointer()),
           &h2o_low_x, &h2o_low_y, &h2o_low_z, &h2o_high_x, &h2o_high_y, &h2o_high_z, const_cast<double*>(h2o.getPointer()),
           &sfv_low_x, &sfv_low_y, &sfv_low_z, &sfv_high_x, &sfv_high_y, &sfv_high_z, const_cast<double*>(sfv.getPointer()),
           &tg_low_x, &tg_low_y, &tg_low_z, &tg_high_x, &tg_high_y, &tg_high_z, tg.getPointer(),
           &lambda,
           &fraction_low, &fraction_high, fraction.get_objs(),
           &fractiontwo_low, &fractiontwo_high, fractiontwo.get_objs() );
}

#else /* !__cplusplus */

C This is the FORTRAN code portion of the file:

      subroutine RADWSGG(idxlo, idxhi, abskg_low_x, abskg_low_y,
     & abskg_low_z, abskg_high_x, abskg_high_y, abskg_high_z, abskg, 
     & esrcg_low_x, esrcg_low_y, esrcg_low_z, esrcg_high_x, 
     & esrcg_high_y, esrcg_high_z, esrcg, shgamma_low_x, shgamma_low_y,
     &  shgamma_low_z, shgamma_high_x, shgamma_high_y, shgamma_high_z, 
     & shgamma, bands, pcell_low_x, pcell_low_y, pcell_low_z, 
     & pcell_high_x, pcell_high_y, pcell_high_z, pcell, ffield, 
     & co2_low_x, co2_low_y, co2_low_z, co2_high_x, co2_high_y, 
     & co2_high_z, co2, h2o_low_x, h2o_low_y, h2o_low_z, h2o_high_x, 
     & h2o_high_y, h2o_high_z, h2o, sfv_low_x, sfv_low_y, sfv_low_z, 
     & sfv_high_x, sfv_high_y, sfv_high_z, sfv, tg_low_x, tg_low_y, 
     & tg_low_z, tg_high_x, tg_high_y, tg_high_z, tg, lambda, 
     & fraction_low, fraction_high, fraction, fractiontwo_low, 
     & fractiontwo_high, fractiontwo)

      implicit none
      integer idxlo(3)
      integer idxhi(3)
      integer abskg_low_x, abskg_low_y, abskg_low_z, abskg_high_x, 
     & abskg_high_y, abskg_high_z
      double precision abskg(abskg_low_x:abskg_high_x, abskg_low_y:
     & abskg_high_y, abskg_low_z:abskg_high_z)
      integer esrcg_low_x, esrcg_low_y, esrcg_low_z, esrcg_high_x, 
     & esrcg_high_y, esrcg_high_z
      double precision esrcg(esrcg_low_x:esrcg_high_x, esrcg_low_y:
     & esrcg_high_y, esrcg_low_z:esrcg_high_z)
      integer shgamma_low_x, shgamma_low_y, shgamma_low_z, 
     & shgamma_high_x, shgamma_high_y, shgamma_high_z
      double precision shgamma(shgamma_low_x:shgamma_high_x, 
     & shgamma_low_y:shgamma_high_y, shgamma_low_z:shgamma_high_z)
      integer bands
      integer pcell_low_x, pcell_low_y, pcell_low_z, pcell_high_x, 
     & pcell_high_y, pcell_high_z
      integer pcell(pcell_low_x:pcell_high_x, pcell_low_y:pcell_high_y,
     &  pcell_low_z:pcell_high_z)
      integer ffield
      integer co2_low_x, co2_low_y, co2_low_z, co2_high_x, co2_high_y, 
     & co2_high_z
      double precision co2(co2_low_x:co2_high_x, co2_low_y:co2_high_y, 
     & co2_low_z:co2_high_z)
      integer h2o_low_x, h2o_low_y, h2o_low_z, h2o_high_x, h2o_high_y, 
     & h2o_high_z
      double precision h2o(h2o_low_x:h2o_high_x, h2o_low_y:h2o_high_y, 
     & h2o_low_z:h2o_high_z)
      integer sfv_low_x, sfv_low_y, sfv_low_z, sfv_high_x, sfv_high_y, 
     & sfv_high_z
      double precision sfv(sfv_low_x:sfv_high_x, sfv_low_y:sfv_high_y, 
     & sfv_low_z:sfv_high_z)
      integer tg_low_x, tg_low_y, tg_low_z, tg_high_x, tg_high_y, 
     & tg_high_z
      double precision tg(tg_low_x:tg_high_x, tg_low_y:tg_high_y, 
     & tg_low_z:tg_high_z)
      integer lambda
      integer fraction_low
      integer fraction_high
      double precision fraction(fraction_low:fraction_high)
      integer fractiontwo_low
      integer fractiontwo_high
      double precision fractiontwo(fractiontwo_low:fractiontwo_high)
#endif /* __cplusplus */

#endif /* fspec_radwsgg */
