
#ifndef fspec_mascalscalar
#define fspec_mascalscalar

#ifdef __cplusplus

#include <CCA/Components/Arches/fortran/FortranNameMangle.h>

extern "C" void MASCALSCALAR(int* valid_lo,
                             int* valid_hi,
                             int* phi_low_x, int* phi_low_y, int* phi_low_z, int* phi_high_x, int* phi_high_y, int* phi_high_z, double* phi_ptr,
                             int* ae_low_x, int* ae_low_y, int* ae_low_z, int* ae_high_x, int* ae_high_y, int* ae_high_z, double* ae_ptr,
                             int* aw_low_x, int* aw_low_y, int* aw_low_z, int* aw_high_x, int* aw_high_y, int* aw_high_z, double* aw_ptr,
                             int* an_low_x, int* an_low_y, int* an_low_z, int* an_high_x, int* an_high_y, int* an_high_z, double* an_ptr,
                             int* as_low_x, int* as_low_y, int* as_low_z, int* as_high_x, int* as_high_y, int* as_high_z, double* as_ptr,
                             int* at_low_x, int* at_low_y, int* at_low_z, int* at_high_x, int* at_high_y, int* at_high_z, double* at_ptr,
                             int* ab_low_x, int* ab_low_y, int* ab_low_z, int* ab_high_x, int* ab_high_y, int* ab_high_z, double* ab_ptr,
                             int* su_low_x, int* su_low_y, int* su_low_z, int* su_high_x, int* su_high_y, int* su_high_z, double* su_ptr,
                             int* cesav_low_x, int* cesav_low_y, int* cesav_low_z, int* cesav_high_x, int* cesav_high_y, int* cesav_high_z, double* cesav_ptr,
                             int* cwsav_low_x, int* cwsav_low_y, int* cwsav_low_z, int* cwsav_high_x, int* cwsav_high_y, int* cwsav_high_z, double* cwsav_ptr,
                             int* cnsav_low_x, int* cnsav_low_y, int* cnsav_low_z, int* cnsav_high_x, int* cnsav_high_y, int* cnsav_high_z, double* cnsav_ptr,
                             int* cssav_low_x, int* cssav_low_y, int* cssav_low_z, int* cssav_high_x, int* cssav_high_y, int* cssav_high_z, double* cssav_ptr,
                             int* ctsav_low_x, int* ctsav_low_y, int* ctsav_low_z, int* ctsav_high_x, int* ctsav_high_y, int* ctsav_high_z, double* ctsav_ptr,
                             int* cbsav_low_x, int* cbsav_low_y, int* cbsav_low_z, int* cbsav_high_x, int* cbsav_high_y, int* cbsav_high_z, double* cbsav_ptr,
                             int* conv_scheme);

static void fort_mascalscalar( Uintah::IntVector & valid_lo,
                               Uintah::IntVector & valid_hi,
                               const Uintah::Array3<double> & phi,
                               Uintah::Array3<double> & ae,
                               Uintah::Array3<double> & aw,
                               Uintah::Array3<double> & an,
                               Uintah::Array3<double> & as,
                               Uintah::Array3<double> & at,
                               Uintah::Array3<double> & ab,
                               Uintah::Array3<double> & su,
                               Uintah::Array3<double> & cesav,
                               Uintah::Array3<double> & cwsav,
                               Uintah::Array3<double> & cnsav,
                               Uintah::Array3<double> & cssav,
                               Uintah::Array3<double> & ctsav,
                               Uintah::Array3<double> & cbsav,
                               int & conv_scheme )
{
  Uintah::IntVector phi_low = phi.getWindow()->getOffset();
  Uintah::IntVector phi_high = phi.getWindow()->getData()->size() + phi_low - Uintah::IntVector(1, 1, 1);
  int phi_low_x = phi_low.x();
  int phi_high_x = phi_high.x();
  int phi_low_y = phi_low.y();
  int phi_high_y = phi_high.y();
  int phi_low_z = phi_low.z();
  int phi_high_z = phi_high.z();
  Uintah::IntVector ae_low = ae.getWindow()->getOffset();
  Uintah::IntVector ae_high = ae.getWindow()->getData()->size() + ae_low - Uintah::IntVector(1, 1, 1);
  int ae_low_x = ae_low.x();
  int ae_high_x = ae_high.x();
  int ae_low_y = ae_low.y();
  int ae_high_y = ae_high.y();
  int ae_low_z = ae_low.z();
  int ae_high_z = ae_high.z();
  Uintah::IntVector aw_low = aw.getWindow()->getOffset();
  Uintah::IntVector aw_high = aw.getWindow()->getData()->size() + aw_low - Uintah::IntVector(1, 1, 1);
  int aw_low_x = aw_low.x();
  int aw_high_x = aw_high.x();
  int aw_low_y = aw_low.y();
  int aw_high_y = aw_high.y();
  int aw_low_z = aw_low.z();
  int aw_high_z = aw_high.z();
  Uintah::IntVector an_low = an.getWindow()->getOffset();
  Uintah::IntVector an_high = an.getWindow()->getData()->size() + an_low - Uintah::IntVector(1, 1, 1);
  int an_low_x = an_low.x();
  int an_high_x = an_high.x();
  int an_low_y = an_low.y();
  int an_high_y = an_high.y();
  int an_low_z = an_low.z();
  int an_high_z = an_high.z();
  Uintah::IntVector as_low = as.getWindow()->getOffset();
  Uintah::IntVector as_high = as.getWindow()->getData()->size() + as_low - Uintah::IntVector(1, 1, 1);
  int as_low_x = as_low.x();
  int as_high_x = as_high.x();
  int as_low_y = as_low.y();
  int as_high_y = as_high.y();
  int as_low_z = as_low.z();
  int as_high_z = as_high.z();
  Uintah::IntVector at_low = at.getWindow()->getOffset();
  Uintah::IntVector at_high = at.getWindow()->getData()->size() + at_low - Uintah::IntVector(1, 1, 1);
  int at_low_x = at_low.x();
  int at_high_x = at_high.x();
  int at_low_y = at_low.y();
  int at_high_y = at_high.y();
  int at_low_z = at_low.z();
  int at_high_z = at_high.z();
  Uintah::IntVector ab_low = ab.getWindow()->getOffset();
  Uintah::IntVector ab_high = ab.getWindow()->getData()->size() + ab_low - Uintah::IntVector(1, 1, 1);
  int ab_low_x = ab_low.x();
  int ab_high_x = ab_high.x();
  int ab_low_y = ab_low.y();
  int ab_high_y = ab_high.y();
  int ab_low_z = ab_low.z();
  int ab_high_z = ab_high.z();
  Uintah::IntVector su_low = su.getWindow()->getOffset();
  Uintah::IntVector su_high = su.getWindow()->getData()->size() + su_low - Uintah::IntVector(1, 1, 1);
  int su_low_x = su_low.x();
  int su_high_x = su_high.x();
  int su_low_y = su_low.y();
  int su_high_y = su_high.y();
  int su_low_z = su_low.z();
  int su_high_z = su_high.z();
  Uintah::IntVector cesav_low = cesav.getWindow()->getOffset();
  Uintah::IntVector cesav_high = cesav.getWindow()->getData()->size() + cesav_low - Uintah::IntVector(1, 1, 1);
  int cesav_low_x = cesav_low.x();
  int cesav_high_x = cesav_high.x();
  int cesav_low_y = cesav_low.y();
  int cesav_high_y = cesav_high.y();
  int cesav_low_z = cesav_low.z();
  int cesav_high_z = cesav_high.z();
  Uintah::IntVector cwsav_low = cwsav.getWindow()->getOffset();
  Uintah::IntVector cwsav_high = cwsav.getWindow()->getData()->size() + cwsav_low - Uintah::IntVector(1, 1, 1);
  int cwsav_low_x = cwsav_low.x();
  int cwsav_high_x = cwsav_high.x();
  int cwsav_low_y = cwsav_low.y();
  int cwsav_high_y = cwsav_high.y();
  int cwsav_low_z = cwsav_low.z();
  int cwsav_high_z = cwsav_high.z();
  Uintah::IntVector cnsav_low = cnsav.getWindow()->getOffset();
  Uintah::IntVector cnsav_high = cnsav.getWindow()->getData()->size() + cnsav_low - Uintah::IntVector(1, 1, 1);
  int cnsav_low_x = cnsav_low.x();
  int cnsav_high_x = cnsav_high.x();
  int cnsav_low_y = cnsav_low.y();
  int cnsav_high_y = cnsav_high.y();
  int cnsav_low_z = cnsav_low.z();
  int cnsav_high_z = cnsav_high.z();
  Uintah::IntVector cssav_low = cssav.getWindow()->getOffset();
  Uintah::IntVector cssav_high = cssav.getWindow()->getData()->size() + cssav_low - Uintah::IntVector(1, 1, 1);
  int cssav_low_x = cssav_low.x();
  int cssav_high_x = cssav_high.x();
  int cssav_low_y = cssav_low.y();
  int cssav_high_y = cssav_high.y();
  int cssav_low_z = cssav_low.z();
  int cssav_high_z = cssav_high.z();
  Uintah::IntVector ctsav_low = ctsav.getWindow()->getOffset();
  Uintah::IntVector ctsav_high = ctsav.getWindow()->getData()->size() + ctsav_low - Uintah::IntVector(1, 1, 1);
  int ctsav_low_x = ctsav_low.x();
  int ctsav_high_x = ctsav_high.x();
  int ctsav_low_y = ctsav_low.y();
  int ctsav_high_y = ctsav_high.y();
  int ctsav_low_z = ctsav_low.z();
  int ctsav_high_z = ctsav_high.z();
  Uintah::IntVector cbsav_low = cbsav.getWindow()->getOffset();
  Uintah::IntVector cbsav_high = cbsav.getWindow()->getData()->size() + cbsav_low - Uintah::IntVector(1, 1, 1);
  int cbsav_low_x = cbsav_low.x();
  int cbsav_high_x = cbsav_high.x();
  int cbsav_low_y = cbsav_low.y();
  int cbsav_high_y = cbsav_high.y();
  int cbsav_low_z = cbsav_low.z();
  int cbsav_high_z = cbsav_high.z();
  MASCALSCALAR( valid_lo.get_pointer(),
                valid_hi.get_pointer(),
                &phi_low_x, &phi_low_y, &phi_low_z, &phi_high_x, &phi_high_y, &phi_high_z, const_cast<double*>(phi.getPointer()),
                &ae_low_x, &ae_low_y, &ae_low_z, &ae_high_x, &ae_high_y, &ae_high_z, ae.getPointer(),
                &aw_low_x, &aw_low_y, &aw_low_z, &aw_high_x, &aw_high_y, &aw_high_z, aw.getPointer(),
                &an_low_x, &an_low_y, &an_low_z, &an_high_x, &an_high_y, &an_high_z, an.getPointer(),
                &as_low_x, &as_low_y, &as_low_z, &as_high_x, &as_high_y, &as_high_z, as.getPointer(),
                &at_low_x, &at_low_y, &at_low_z, &at_high_x, &at_high_y, &at_high_z, at.getPointer(),
                &ab_low_x, &ab_low_y, &ab_low_z, &ab_high_x, &ab_high_y, &ab_high_z, ab.getPointer(),
                &su_low_x, &su_low_y, &su_low_z, &su_high_x, &su_high_y, &su_high_z, su.getPointer(),
                &cesav_low_x, &cesav_low_y, &cesav_low_z, &cesav_high_x, &cesav_high_y, &cesav_high_z, cesav.getPointer(),
                &cwsav_low_x, &cwsav_low_y, &cwsav_low_z, &cwsav_high_x, &cwsav_high_y, &cwsav_high_z, cwsav.getPointer(),
                &cnsav_low_x, &cnsav_low_y, &cnsav_low_z, &cnsav_high_x, &cnsav_high_y, &cnsav_high_z, cnsav.getPointer(),
                &cssav_low_x, &cssav_low_y, &cssav_low_z, &cssav_high_x, &cssav_high_y, &cssav_high_z, cssav.getPointer(),
                &ctsav_low_x, &ctsav_low_y, &ctsav_low_z, &ctsav_high_x, &ctsav_high_y, &ctsav_high_z, ctsav.getPointer(),
                &cbsav_low_x, &cbsav_low_y, &cbsav_low_z, &cbsav_high_x, &cbsav_high_y, &cbsav_high_z, cbsav.getPointer(),
                &conv_scheme );
}

#else /* !__cplusplus */

C This is the FORTRAN code portion of the file:

      subroutine MASCALSCALAR(valid_lo, valid_hi, phi_low_x, phi_low_y,
     &  phi_low_z, phi_high_x, phi_high_y, phi_high_z, phi, ae_low_x, 
     & ae_low_y, ae_low_z, ae_high_x, ae_high_y, ae_high_z, ae, 
     & aw_low_x, aw_low_y, aw_low_z, aw_high_x, aw_high_y, aw_high_z, 
     & aw, an_low_x, an_low_y, an_low_z, an_high_x, an_high_y, 
     & an_high_z, an, as_low_x, as_low_y, as_low_z, as_high_x, 
     & as_high_y, as_high_z, as, at_low_x, at_low_y, at_low_z, 
     & at_high_x, at_high_y, at_high_z, at, ab_low_x, ab_low_y, 
     & ab_low_z, ab_high_x, ab_high_y, ab_high_z, ab, su_low_x, 
     & su_low_y, su_low_z, su_high_x, su_high_y, su_high_z, su, 
     & cesav_low_x, cesav_low_y, cesav_low_z, cesav_high_x, 
     & cesav_high_y, cesav_high_z, cesav, cwsav_low_x, cwsav_low_y, 
     & cwsav_low_z, cwsav_high_x, cwsav_high_y, cwsav_high_z, cwsav, 
     & cnsav_low_x, cnsav_low_y, cnsav_low_z, cnsav_high_x, 
     & cnsav_high_y, cnsav_high_z, cnsav, cssav_low_x, cssav_low_y, 
     & cssav_low_z, cssav_high_x, cssav_high_y, cssav_high_z, cssav, 
     & ctsav_low_x, ctsav_low_y, ctsav_low_z, ctsav_high_x, 
     & ctsav_high_y, ctsav_high_z, ctsav, cbsav_low_x, cbsav_low_y, 
     & cbsav_low_z, cbsav_high_x, cbsav_high_y, cbsav_high_z, cbsav, 
     & conv_scheme)

      implicit none
      integer valid_lo(3)
      integer valid_hi(3)
      integer phi_low_x, phi_low_y, phi_low_z, phi_high_x, phi_high_y, 
     & phi_high_z
      double precision phi(phi_low_x:phi_high_x, phi_low_y:phi_high_y, 
     & phi_low_z:phi_high_z)
      integer ae_low_x, ae_low_y, ae_low_z, ae_high_x, ae_high_y, 
     & ae_high_z
      double precision ae(ae_low_x:ae_high_x, ae_low_y:ae_high_y, 
     & ae_low_z:ae_high_z)
      integer aw_low_x, aw_low_y, aw_low_z, aw_high_x, aw_high_y, 
     & aw_high_z
      double precision aw(aw_low_x:aw_high_x, aw_low_y:aw_high_y, 
     & aw_low_z:aw_high_z)
      integer an_low_x, an_low_y, an_low_z, an_high_x, an_high_y, 
     & an_high_z
      double precision an(an_low_x:an_high_x, an_low_y:an_high_y, 
     & an_low_z:an_high_z)
      integer as_low_x, as_low_y, as_low_z, as_high_x, as_high_y, 
     & as_high_z
      double precision as(as_low_x:as_high_x, as_low_y:as_high_y, 
     & as_low_z:as_high_z)
      integer at_low_x, at_low_y, at_low_z, at_high_x, at_high_y, 
     & at_high_z
      double precision at(at_low_x:at_high_x, at_low_y:at_high_y, 
     & at_low_z:at_high_z)
      integer ab_low_x, ab_low_y, ab_low_z, ab_high_x, ab_high_y, 
     & ab_high_z
      double precision ab(ab_low_x:ab_high_x, ab_low_y:ab_high_y, 
     & ab_low_z:ab_high_z)
      integer su_low_x, su_low_y, su_low_z, su_high_x, su_high_y, 
     & su_high_z
      double precision su(su_low_x:su_high_x, su_low_y:su_high_y, 
     & su_low_z:su_high_z)
      integer cesav_low_x, cesav_low_y, cesav_low_z, cesav_high_x, 
     & cesav_high_y, cesav_high_z
      double precision cesav(cesav_low_x:cesav_high_x, cesav_low_y:
     & cesav_high_y, cesav_low_z:cesav_high_z)
      integer cwsav_low_x, cwsav_low_y, cwsav_low_z, cwsav_high_x, 
     & cwsav_high_y, cwsav_high_z
      double precision cwsav(cwsav_low_x:cwsav_high_x, cwsav_low_y:
     & cwsav_high_y, cwsav_low_z:cwsav_high_z)
      integer cnsav_low_x, cnsav_low_y, cnsav_low_z, cnsav_high_x, 
     & cnsav_high_y, cnsav_high_z
      double precision cnsav(cnsav_low_x:cnsav_high_x, cnsav_low_y:
     & cnsav_high_y, cnsav_low_z:cnsav_high_z)
      integer cssav_low_x, cssav_low_y, cssav_low_z, cssav_high_x, 
     & cssav_high_y, cssav_high_z
      double precision cssav(cssav_low_x:cssav_high_x, cssav_low_y:
     & cssav_high_y, cssav_low_z:cssav_high_z)
      integer ctsav_low_x, ctsav_low_y, ctsav_low_z, ctsav_high_x, 
     & ctsav_high_y, ctsav_high_z
      double precision ctsav(ctsav_low_x:ctsav_high_x, ctsav_low_y:
     & ctsav_high_y, ctsav_low_z:ctsav_high_z)
      integer cbsav_low_x, cbsav_low_y, cbsav_low_z, cbsav_high_x, 
     & cbsav_high_y, cbsav_high_z
      double precision cbsav(cbsav_low_x:cbsav_high_x, cbsav_low_y:
     & cbsav_high_y, cbsav_low_z:cbsav_high_z)
      integer conv_scheme
#endif /* __cplusplus */

#endif /* fspec_mascalscalar */
