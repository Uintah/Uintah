
#ifndef fspec_wvelcoef_central
#define fspec_wvelcoef_central

#ifdef __cplusplus

#include <CCA/Components/Arches/fortran/FortranNameMangle.h>

extern "C" void F_wvelcoef_central(int* ww_low_x, int* ww_low_y, int* ww_low_z, int* ww_high_x, int* ww_high_y, int* ww_high_z, double* ww_ptr,
                                 int* cesav_low_x, int* cesav_low_y, int* cesav_low_z, int* cesav_high_x, int* cesav_high_y, int* cesav_high_z, double* cesav_ptr,
                                 int* cwsav_low_x, int* cwsav_low_y, int* cwsav_low_z, int* cwsav_high_x, int* cwsav_high_y, int* cwsav_high_z, double* cwsav_ptr,
                                 int* cnsav_low_x, int* cnsav_low_y, int* cnsav_low_z, int* cnsav_high_x, int* cnsav_high_y, int* cnsav_high_z, double* cnsav_ptr,
                                 int* cssav_low_x, int* cssav_low_y, int* cssav_low_z, int* cssav_high_x, int* cssav_high_y, int* cssav_high_z, double* cssav_ptr,
                                 int* ctsav_low_x, int* ctsav_low_y, int* ctsav_low_z, int* ctsav_high_x, int* ctsav_high_y, int* ctsav_high_z, double* ctsav_ptr,
                                 int* cbsav_low_x, int* cbsav_low_y, int* cbsav_low_z, int* cbsav_high_x, int* cbsav_high_y, int* cbsav_high_z, double* cbsav_ptr,
                                 int* ap_low_x, int* ap_low_y, int* ap_low_z, int* ap_high_x, int* ap_high_y, int* ap_high_z, double* ap_ptr,
                                 int* ae_low_x, int* ae_low_y, int* ae_low_z, int* ae_high_x, int* ae_high_y, int* ae_high_z, double* ae_ptr,
                                 int* aw_low_x, int* aw_low_y, int* aw_low_z, int* aw_high_x, int* aw_high_y, int* aw_high_z, double* aw_ptr,
                                 int* an_low_x, int* an_low_y, int* an_low_z, int* an_high_x, int* an_high_y, int* an_high_z, double* an_ptr,
                                 int* as_low_x, int* as_low_y, int* as_low_z, int* as_high_x, int* as_high_y, int* as_high_z, double* as_ptr,
                                 int* at_low_x, int* at_low_y, int* at_low_z, int* at_high_x, int* at_high_y, int* at_high_z, double* at_ptr,
                                 int* ab_low_x, int* ab_low_y, int* ab_low_z, int* ab_high_x, int* ab_high_y, int* ab_high_z, double* ab_ptr,
                                 int* uu_low_x, int* uu_low_y, int* uu_low_z, int* uu_high_x, int* uu_high_y, int* uu_high_z, double* uu_ptr,
                                 int* vv_low_x, int* vv_low_y, int* vv_low_z, int* vv_high_x, int* vv_high_y, int* vv_high_z, double* vv_ptr,
                                 int* den_low_x, int* den_low_y, int* den_low_z, int* den_high_x, int* den_high_y, int* den_high_z, double* den_ptr,
                                 int* vis_low_x, int* vis_low_y, int* vis_low_z, int* vis_high_x, int* vis_high_y, int* vis_high_z, double* vis_ptr,
                                 int* den_ref_low_x, int* den_ref_low_y, int* den_ref_low_z, int* den_ref_high_x, int* den_ref_high_y, int* den_ref_high_z, double* den_ref_ptr,
                                 int* SU_low_x, int* SU_low_y, int* SU_low_z, int* SU_high_x, int* SU_high_y, int* SU_high_z, double* SU_ptr,
                                 int* eps_low_x, int* eps_low_y, int* eps_low_z, int* eps_high_x, int* eps_high_y, int* eps_high_z, double* eps_ptr,
                                 double* deltat,
                                 double* grav,
                                 double* dx,
                                 double* dy,
                                 double* dz,
                                 int* idxLoW,
                                 int* idxHiW);

static void fort_wvelcoef_central( Uintah::constSFCZVariable<double> & ww,
                                   Uintah::SFCZVariable<double> & cesav,
                                   Uintah::SFCZVariable<double> & cwsav,
                                   Uintah::SFCZVariable<double> & cnsav,
                                   Uintah::SFCZVariable<double> & cssav,
                                   Uintah::SFCZVariable<double> & ctsav,
                                   Uintah::SFCZVariable<double> & cbsav,
                                   Uintah::SFCZVariable<double> & ap,
                                   Uintah::SFCZVariable<double> & ae,
                                   Uintah::SFCZVariable<double> & aw,
                                   Uintah::SFCZVariable<double> & an,
                                   Uintah::SFCZVariable<double> & as,
                                   Uintah::SFCZVariable<double> & at,
                                   Uintah::SFCZVariable<double> & ab,
                                   Uintah::constSFCXVariable<double> & uu,
                                   Uintah::constSFCYVariable<double> & vv,
                                   Uintah::constCCVariable<double> & den,
                                   Uintah::constCCVariable<double> & vis,
                                   Uintah::constCCVariable<double> & den_ref,
                                   Uintah::SFCZVariable<double> & SU,
                                   Uintah::constCCVariable<double> & eps,
                                   double & deltat,
                                   double & grav,
                                   double & dx,
                                   double & dy,
                                   double & dz,
                                   Uintah::IntVector & idxLoW,
                                   Uintah::IntVector & idxHiW )
{
  Uintah::IntVector ww_low = ww.getWindow()->getOffset();
  Uintah::IntVector ww_high = ww.getWindow()->getData()->size() + ww_low - Uintah::IntVector(1, 1, 1);
  int ww_low_x = ww_low.x();
  int ww_high_x = ww_high.x();
  int ww_low_y = ww_low.y();
  int ww_high_y = ww_high.y();
  int ww_low_z = ww_low.z();
  int ww_high_z = ww_high.z();
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
  Uintah::IntVector ap_low = ap.getWindow()->getOffset();
  Uintah::IntVector ap_high = ap.getWindow()->getData()->size() + ap_low - Uintah::IntVector(1, 1, 1);
  int ap_low_x = ap_low.x();
  int ap_high_x = ap_high.x();
  int ap_low_y = ap_low.y();
  int ap_high_y = ap_high.y();
  int ap_low_z = ap_low.z();
  int ap_high_z = ap_high.z();
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
  Uintah::IntVector uu_low = uu.getWindow()->getOffset();
  Uintah::IntVector uu_high = uu.getWindow()->getData()->size() + uu_low - Uintah::IntVector(1, 1, 1);
  int uu_low_x = uu_low.x();
  int uu_high_x = uu_high.x();
  int uu_low_y = uu_low.y();
  int uu_high_y = uu_high.y();
  int uu_low_z = uu_low.z();
  int uu_high_z = uu_high.z();
  Uintah::IntVector vv_low = vv.getWindow()->getOffset();
  Uintah::IntVector vv_high = vv.getWindow()->getData()->size() + vv_low - Uintah::IntVector(1, 1, 1);
  int vv_low_x = vv_low.x();
  int vv_high_x = vv_high.x();
  int vv_low_y = vv_low.y();
  int vv_high_y = vv_high.y();
  int vv_low_z = vv_low.z();
  int vv_high_z = vv_high.z();
  Uintah::IntVector den_low = den.getWindow()->getOffset();
  Uintah::IntVector den_high = den.getWindow()->getData()->size() + den_low - Uintah::IntVector(1, 1, 1);
  int den_low_x = den_low.x();
  int den_high_x = den_high.x();
  int den_low_y = den_low.y();
  int den_high_y = den_high.y();
  int den_low_z = den_low.z();
  int den_high_z = den_high.z();
  Uintah::IntVector vis_low = vis.getWindow()->getOffset();
  Uintah::IntVector vis_high = vis.getWindow()->getData()->size() + vis_low - Uintah::IntVector(1, 1, 1);
  int vis_low_x = vis_low.x();
  int vis_high_x = vis_high.x();
  int vis_low_y = vis_low.y();
  int vis_high_y = vis_high.y();
  int vis_low_z = vis_low.z();
  int vis_high_z = vis_high.z();
  Uintah::IntVector den_ref_low = den_ref.getWindow()->getOffset();
  Uintah::IntVector den_ref_high = den_ref.getWindow()->getData()->size() + den_ref_low - Uintah::IntVector(1, 1, 1);
  int den_ref_low_x = den_ref_low.x();
  int den_ref_high_x = den_ref_high.x();
  int den_ref_low_y = den_ref_low.y();
  int den_ref_high_y = den_ref_high.y();
  int den_ref_low_z = den_ref_low.z();
  int den_ref_high_z = den_ref_high.z();
  Uintah::IntVector SU_low = SU.getWindow()->getOffset();
  Uintah::IntVector SU_high = SU.getWindow()->getData()->size() + SU_low - Uintah::IntVector(1, 1, 1);
  int SU_low_x = SU_low.x();
  int SU_high_x = SU_high.x();
  int SU_low_y = SU_low.y();
  int SU_high_y = SU_high.y();
  int SU_low_z = SU_low.z();
  int SU_high_z = SU_high.z();
  Uintah::IntVector eps_low = eps.getWindow()->getOffset();
  Uintah::IntVector eps_high = eps.getWindow()->getData()->size() + eps_low - Uintah::IntVector(1, 1, 1);
  int eps_low_x = eps_low.x();
  int eps_high_x = eps_high.x();
  int eps_low_y = eps_low.y();
  int eps_high_y = eps_high.y();
  int eps_low_z = eps_low.z();
  int eps_high_z = eps_high.z();
  F_wvelcoef_central( &ww_low_x, &ww_low_y, &ww_low_z, &ww_high_x, &ww_high_y, &ww_high_z, const_cast<double*>(ww.getPointer()),
                    &cesav_low_x, &cesav_low_y, &cesav_low_z, &cesav_high_x, &cesav_high_y, &cesav_high_z, cesav.getPointer(),
                    &cwsav_low_x, &cwsav_low_y, &cwsav_low_z, &cwsav_high_x, &cwsav_high_y, &cwsav_high_z, cwsav.getPointer(),
                    &cnsav_low_x, &cnsav_low_y, &cnsav_low_z, &cnsav_high_x, &cnsav_high_y, &cnsav_high_z, cnsav.getPointer(),
                    &cssav_low_x, &cssav_low_y, &cssav_low_z, &cssav_high_x, &cssav_high_y, &cssav_high_z, cssav.getPointer(),
                    &ctsav_low_x, &ctsav_low_y, &ctsav_low_z, &ctsav_high_x, &ctsav_high_y, &ctsav_high_z, ctsav.getPointer(),
                    &cbsav_low_x, &cbsav_low_y, &cbsav_low_z, &cbsav_high_x, &cbsav_high_y, &cbsav_high_z, cbsav.getPointer(),
                    &ap_low_x, &ap_low_y, &ap_low_z, &ap_high_x, &ap_high_y, &ap_high_z, ap.getPointer(),
                    &ae_low_x, &ae_low_y, &ae_low_z, &ae_high_x, &ae_high_y, &ae_high_z, ae.getPointer(),
                    &aw_low_x, &aw_low_y, &aw_low_z, &aw_high_x, &aw_high_y, &aw_high_z, aw.getPointer(),
                    &an_low_x, &an_low_y, &an_low_z, &an_high_x, &an_high_y, &an_high_z, an.getPointer(),
                    &as_low_x, &as_low_y, &as_low_z, &as_high_x, &as_high_y, &as_high_z, as.getPointer(),
                    &at_low_x, &at_low_y, &at_low_z, &at_high_x, &at_high_y, &at_high_z, at.getPointer(),
                    &ab_low_x, &ab_low_y, &ab_low_z, &ab_high_x, &ab_high_y, &ab_high_z, ab.getPointer(),
                    &uu_low_x, &uu_low_y, &uu_low_z, &uu_high_x, &uu_high_y, &uu_high_z, const_cast<double*>(uu.getPointer()),
                    &vv_low_x, &vv_low_y, &vv_low_z, &vv_high_x, &vv_high_y, &vv_high_z, const_cast<double*>(vv.getPointer()),
                    &den_low_x, &den_low_y, &den_low_z, &den_high_x, &den_high_y, &den_high_z, const_cast<double*>(den.getPointer()),
                    &vis_low_x, &vis_low_y, &vis_low_z, &vis_high_x, &vis_high_y, &vis_high_z, const_cast<double*>(vis.getPointer()),
                    &den_ref_low_x, &den_ref_low_y, &den_ref_low_z, &den_ref_high_x, &den_ref_high_y, &den_ref_high_z, const_cast<double*>(den_ref.getPointer()),
                    &SU_low_x, &SU_low_y, &SU_low_z, &SU_high_x, &SU_high_y, &SU_high_z, SU.getPointer(),
                    &eps_low_x, &eps_low_y, &eps_low_z, &eps_high_x, &eps_high_y, &eps_high_z, const_cast<double*>(eps.getPointer()),
                    &deltat,
                    &grav,
                    &dx,
                    &dy,
                    &dz,
                    idxLoW.get_pointer(),
                    idxHiW.get_pointer() );
}

#else /* !__cplusplus */

C This is the FORTRAN code portion of the file:

      subroutine WVELCOEF_CENTRAL(ww_low_x, ww_low_y, ww_low_z,
     & ww_high_x, ww_high_y, ww_high_z, ww, cesav_low_x, cesav_low_y, 
     & cesav_low_z, cesav_high_x, cesav_high_y, cesav_high_z, cesav, 
     & cwsav_low_x, cwsav_low_y, cwsav_low_z, cwsav_high_x, 
     & cwsav_high_y, cwsav_high_z, cwsav, cnsav_low_x, cnsav_low_y, 
     & cnsav_low_z, cnsav_high_x, cnsav_high_y, cnsav_high_z, cnsav, 
     & cssav_low_x, cssav_low_y, cssav_low_z, cssav_high_x, 
     & cssav_high_y, cssav_high_z, cssav, ctsav_low_x, ctsav_low_y, 
     & ctsav_low_z, ctsav_high_x, ctsav_high_y, ctsav_high_z, ctsav, 
     & cbsav_low_x, cbsav_low_y, cbsav_low_z, cbsav_high_x, 
     & cbsav_high_y, cbsav_high_z, cbsav, ap_low_x, ap_low_y, ap_low_z,
     &  ap_high_x, ap_high_y, ap_high_z, ap, ae_low_x, ae_low_y, 
     & ae_low_z, ae_high_x, ae_high_y, ae_high_z, ae, aw_low_x, 
     & aw_low_y, aw_low_z, aw_high_x, aw_high_y, aw_high_z, aw, 
     & an_low_x, an_low_y, an_low_z, an_high_x, an_high_y, an_high_z, 
     & an, as_low_x, as_low_y, as_low_z, as_high_x, as_high_y, 
     & as_high_z, as, at_low_x, at_low_y, at_low_z, at_high_x, 
     & at_high_y, at_high_z, at, ab_low_x, ab_low_y, ab_low_z, 
     & ab_high_x, ab_high_y, ab_high_z, ab, uu_low_x, uu_low_y, 
     & uu_low_z, uu_high_x, uu_high_y, uu_high_z, uu, vv_low_x, 
     & vv_low_y, vv_low_z, vv_high_x, vv_high_y, vv_high_z, vv, 
     & den_low_x, den_low_y, den_low_z, den_high_x, den_high_y, 
     & den_high_z, den, vis_low_x, vis_low_y, vis_low_z, vis_high_x, 
     & vis_high_y, vis_high_z, vis, den_ref_low_x, den_ref_low_y, 
     & den_ref_low_z, den_ref_high_x, den_ref_high_y, den_ref_high_z, 
     & den_ref, SU_low_x, SU_low_y, SU_low_z, SU_high_x, SU_high_y, 
     & SU_high_z, SU, eps_low_x, eps_low_y, eps_low_z, eps_high_x, 
     & eps_high_y, eps_high_z, eps, deltat, grav, dx, dy, dz, idxLoW, 
     & idxHiW)

      implicit none
      integer ww_low_x, ww_low_y, ww_low_z, ww_high_x, ww_high_y, 
     & ww_high_z
      double precision ww(ww_low_x:ww_high_x, ww_low_y:ww_high_y, 
     & ww_low_z:ww_high_z)
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
      integer ap_low_x, ap_low_y, ap_low_z, ap_high_x, ap_high_y, 
     & ap_high_z
      double precision ap(ap_low_x:ap_high_x, ap_low_y:ap_high_y, 
     & ap_low_z:ap_high_z)
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
      integer uu_low_x, uu_low_y, uu_low_z, uu_high_x, uu_high_y, 
     & uu_high_z
      double precision uu(uu_low_x:uu_high_x, uu_low_y:uu_high_y, 
     & uu_low_z:uu_high_z)
      integer vv_low_x, vv_low_y, vv_low_z, vv_high_x, vv_high_y, 
     & vv_high_z
      double precision vv(vv_low_x:vv_high_x, vv_low_y:vv_high_y, 
     & vv_low_z:vv_high_z)
      integer den_low_x, den_low_y, den_low_z, den_high_x, den_high_y, 
     & den_high_z
      double precision den(den_low_x:den_high_x, den_low_y:den_high_y, 
     & den_low_z:den_high_z)
      integer vis_low_x, vis_low_y, vis_low_z, vis_high_x, vis_high_y, 
     & vis_high_z
      double precision vis(vis_low_x:vis_high_x, vis_low_y:vis_high_y, 
     & vis_low_z:vis_high_z)
      integer den_ref_low_x, den_ref_low_y, den_ref_low_z, 
     & den_ref_high_x, den_ref_high_y, den_ref_high_z
      double precision den_ref(den_ref_low_x:den_ref_high_x, 
     & den_ref_low_y:den_ref_high_y, den_ref_low_z:den_ref_high_z)
      integer SU_low_x, SU_low_y, SU_low_z, SU_high_x, SU_high_y, 
     & SU_high_z
      double precision SU(SU_low_x:SU_high_x, SU_low_y:SU_high_y, 
     & SU_low_z:SU_high_z)
      integer eps_low_x, eps_low_y, eps_low_z, eps_high_x, eps_high_y, 
     & eps_high_z
      double precision eps(eps_low_x:eps_high_x, eps_low_y:eps_high_y, 
     & eps_low_z:eps_high_z)
      double precision deltat
      double precision grav
      double precision dx
      double precision dy
      double precision dz
      integer idxLoW(3)
      integer idxHiW(3)
#endif /* __cplusplus */

#endif /* fspec_wvelcoef_central */
