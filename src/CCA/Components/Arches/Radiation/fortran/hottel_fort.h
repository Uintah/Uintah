
#ifndef fspec_hottel
#define fspec_hottel

#ifdef __cplusplus

#include <CCA/Components/Arches/Radiation/fortran/FortranNameMangle.h>

extern "C" void F_hottel(int* idxlo,
                       int* idxhi,
                       int* tg_low_x, int* tg_low_y, int* tg_low_z, int* tg_high_x, int* tg_high_y, int* tg_high_z, double* tg_ptr,
                       int* co2_low_x, int* co2_low_y, int* co2_low_z, int* co2_high_x, int* co2_high_y, int* co2_high_z, double* co2_ptr,
                       int* h2o_low_x, int* h2o_low_y, int* h2o_low_z, int* h2o_high_x, int* h2o_high_y, int* h2o_high_z, double* h2o_ptr,
                       int* volfrac_low_x, int* volfrac_low_y, int* volfrac_low_z, int* volfrac_high_x, int* volfrac_high_y, int* volfrac_high_z, double* volfrac_ptr,
                       double* opl,
                       int* sfv_low_x, int* sfv_low_y, int* sfv_low_z, int* sfv_high_x, int* sfv_high_y, int* sfv_high_z, double* sfv_ptr,
                       int* abskg_low_x, int* abskg_low_y, int* abskg_low_z, int* abskg_high_x, int* abskg_high_y, int* abskg_high_z, double* abskg_ptr,
                       double* pressure);

static void fort_hottel( Uintah::IntVector & idxlo,
                         Uintah::IntVector & idxhi,
                         Uintah::constCCVariable<double> & tg,
                         Uintah::constCCVariable<double> & co2,
                         Uintah::constCCVariable<double> & h2o,
                         Uintah::constCCVariable<double> & volfrac,
                         double & opl,
                         Uintah::constCCVariable<double> & sfv,
                         Uintah::CCVariable<double> & abskg,
                         double & pressure )
{
  Uintah::IntVector tg_low = tg.getWindow()->getOffset();
  Uintah::IntVector tg_high = tg.getWindow()->getData()->size() + tg_low - Uintah::IntVector(1, 1, 1);
  int tg_low_x = tg_low.x();
  int tg_high_x = tg_high.x();
  int tg_low_y = tg_low.y();
  int tg_high_y = tg_high.y();
  int tg_low_z = tg_low.z();
  int tg_high_z = tg_high.z();
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
  Uintah::IntVector volfrac_low = volfrac.getWindow()->getOffset();
  Uintah::IntVector volfrac_high = volfrac.getWindow()->getData()->size() + volfrac_low - Uintah::IntVector(1, 1, 1);
  int volfrac_low_x = volfrac_low.x();
  int volfrac_high_x = volfrac_high.x();
  int volfrac_low_y = volfrac_low.y();
  int volfrac_high_y = volfrac_high.y();
  int volfrac_low_z = volfrac_low.z();
  int volfrac_high_z = volfrac_high.z();
  Uintah::IntVector sfv_low = sfv.getWindow()->getOffset();
  Uintah::IntVector sfv_high = sfv.getWindow()->getData()->size() + sfv_low - Uintah::IntVector(1, 1, 1);
  int sfv_low_x = sfv_low.x();
  int sfv_high_x = sfv_high.x();
  int sfv_low_y = sfv_low.y();
  int sfv_high_y = sfv_high.y();
  int sfv_low_z = sfv_low.z();
  int sfv_high_z = sfv_high.z();
  Uintah::IntVector abskg_low = abskg.getWindow()->getOffset();
  Uintah::IntVector abskg_high = abskg.getWindow()->getData()->size() + abskg_low - Uintah::IntVector(1, 1, 1);
  int abskg_low_x = abskg_low.x();
  int abskg_high_x = abskg_high.x();
  int abskg_low_y = abskg_low.y();
  int abskg_high_y = abskg_high.y();
  int abskg_low_z = abskg_low.z();
  int abskg_high_z = abskg_high.z();
  F_hottel( idxlo.get_pointer(),
          idxhi.get_pointer(),
          &tg_low_x, &tg_low_y, &tg_low_z, &tg_high_x, &tg_high_y, &tg_high_z, const_cast<double*>(tg.getPointer()),
          &co2_low_x, &co2_low_y, &co2_low_z, &co2_high_x, &co2_high_y, &co2_high_z, const_cast<double*>(co2.getPointer()),
          &h2o_low_x, &h2o_low_y, &h2o_low_z, &h2o_high_x, &h2o_high_y, &h2o_high_z, const_cast<double*>(h2o.getPointer()),
          &volfrac_low_x, &volfrac_low_y, &volfrac_low_z, &volfrac_high_x, &volfrac_high_y, &volfrac_high_z, const_cast<double*>(volfrac.getPointer()),
          &opl,
          &sfv_low_x, &sfv_low_y, &sfv_low_z, &sfv_high_x, &sfv_high_y, &sfv_high_z, const_cast<double*>(sfv.getPointer()),
          &abskg_low_x, &abskg_low_y, &abskg_low_z, &abskg_high_x, &abskg_high_y, &abskg_high_z, abskg.getPointer(),
          &pressure );
}

#else /* !__cplusplus */

C This is the FORTRAN code portion of the file:

      subroutine HOTTEL(idxlo, idxhi, tg_low_x, tg_low_y, tg_low_z,
     & tg_high_x, tg_high_y, tg_high_z, tg, co2_low_x, co2_low_y, 
     & co2_low_z, co2_high_x, co2_high_y, co2_high_z, co2, h2o_low_x, 
     & h2o_low_y, h2o_low_z, h2o_high_x, h2o_high_y, h2o_high_z, h2o, 
     & volfrac_low_x, volfrac_low_y, volfrac_low_z, volfrac_high_x, 
     & volfrac_high_y, volfrac_high_z, volfrac, opl, sfv_low_x, 
     & sfv_low_y, sfv_low_z, sfv_high_x, sfv_high_y, sfv_high_z, sfv, 
     & abskg_low_x, abskg_low_y, abskg_low_z, abskg_high_x, 
     & abskg_high_y, abskg_high_z, abskg, pressure)

      implicit none
      integer idxlo(3)
      integer idxhi(3)
      integer tg_low_x, tg_low_y, tg_low_z, tg_high_x, tg_high_y, 
     & tg_high_z
      double precision tg(tg_low_x:tg_high_x, tg_low_y:tg_high_y, 
     & tg_low_z:tg_high_z)
      integer co2_low_x, co2_low_y, co2_low_z, co2_high_x, co2_high_y, 
     & co2_high_z
      double precision co2(co2_low_x:co2_high_x, co2_low_y:co2_high_y, 
     & co2_low_z:co2_high_z)
      integer h2o_low_x, h2o_low_y, h2o_low_z, h2o_high_x, h2o_high_y, 
     & h2o_high_z
      double precision h2o(h2o_low_x:h2o_high_x, h2o_low_y:h2o_high_y, 
     & h2o_low_z:h2o_high_z)
      integer volfrac_low_x, volfrac_low_y, volfrac_low_z, 
     & volfrac_high_x, volfrac_high_y, volfrac_high_z
      double precision volfrac(volfrac_low_x:volfrac_high_x, 
     & volfrac_low_y:volfrac_high_y, volfrac_low_z:volfrac_high_z)
      double precision opl
      integer sfv_low_x, sfv_low_y, sfv_low_z, sfv_high_x, sfv_high_y, 
     & sfv_high_z
      double precision sfv(sfv_low_x:sfv_high_x, sfv_low_y:sfv_high_y, 
     & sfv_low_z:sfv_high_z)
      integer abskg_low_x, abskg_low_y, abskg_low_z, abskg_high_x, 
     & abskg_high_y, abskg_high_z
      double precision abskg(abskg_low_x:abskg_high_x, abskg_low_y:
     & abskg_high_y, abskg_low_z:abskg_high_z)
      double precision pressure
#endif /* __cplusplus */

#endif /* fspec_hottel */
