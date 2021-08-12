
#ifndef fspec_vvelcoef
#define fspec_vvelcoef

#ifdef __cplusplus

#include <CCA/Components/Arches/fortran/FortranNameMangle.h>

extern "C" void F_vvelcoef(int* vv_low_x, int* vv_low_y, int* vv_low_z, int* vv_high_x, int* vv_high_y, int* vv_high_z, double* vv_ptr,
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
                         int* ww_low_x, int* ww_low_y, int* ww_low_z, int* ww_high_x, int* ww_high_y, int* ww_high_z, double* ww_ptr,
                         int* den_low_x, int* den_low_y, int* den_low_z, int* den_high_x, int* den_high_y, int* den_high_z, double* den_ptr,
                         int* vis_low_x, int* vis_low_y, int* vis_low_z, int* vis_high_x, int* vis_high_y, int* vis_high_z, double* vis_ptr,
                         int* den_ref_low_x, int* den_ref_low_y, int* den_ref_low_z, int* den_ref_high_x, int* den_ref_high_y, int* den_ref_high_z, double* den_ref_ptr,
                         int* SU_low_x, int* SU_low_y, int* SU_low_z, int* SU_high_x, int* SU_high_y, int* SU_high_z, double* SU_ptr,
                         int* eps_low_x, int* eps_low_y, int* eps_low_z, int* eps_high_x, int* eps_high_y, int* eps_high_z, double* eps_ptr,
                         double* deltat,
                         double* grav,
                         bool* lcend,
                         int* cee_low, int* cee_high, double* cee_ptr,
                         int* cwe_low, int* cwe_high, double* cwe_ptr,
                         int* cww_low, int* cww_high, double* cww_ptr,
                         int* cnnv_low, int* cnnv_high, double* cnnv_ptr,
                         int* csnv_low, int* csnv_high, double* csnv_ptr,
                         int* cssv_low, int* cssv_high, double* cssv_ptr,
                         int* ctt_low, int* ctt_high, double* ctt_ptr,
                         int* cbt_low, int* cbt_high, double* cbt_ptr,
                         int* cbb_low, int* cbb_high, double* cbb_ptr,
                         int* sew_low, int* sew_high, double* sew_ptr,
                         int* snsv_low, int* snsv_high, double* snsv_ptr,
                         int* sns_low, int* sns_high, double* sns_ptr,
                         int* stb_low, int* stb_high, double* stb_ptr,
                         int* dxep_low, int* dxep_high, double* dxep_ptr,
                         int* dxpw_low, int* dxpw_high, double* dxpw_ptr,
                         int* dynpv_low, int* dynpv_high, double* dynpv_ptr,
                         int* dypsv_low, int* dypsv_high, double* dypsv_ptr,
                         int* dyps_low, int* dyps_high, double* dyps_ptr,
                         int* dztp_low, int* dztp_high, double* dztp_ptr,
                         int* dzpb_low, int* dzpb_high, double* dzpb_ptr,
                         int* fac1v_low, int* fac1v_high, double* fac1v_ptr,
                         int* fac2v_low, int* fac2v_high, double* fac2v_ptr,
                         int* fac3v_low, int* fac3v_high, double* fac3v_ptr,
                         int* fac4v_low, int* fac4v_high, double* fac4v_ptr,
                         int* jnsdv_low, int* jnsdv_high, int* jnsdv_ptr,
                         int* jssdv_low, int* jssdv_high, int* jssdv_ptr,
                         int* efac_low, int* efac_high, double* efac_ptr,
                         int* wfac_low, int* wfac_high, double* wfac_ptr,
                         int* tfac_low, int* tfac_high, double* tfac_ptr,
                         int* bfac_low, int* bfac_high, double* bfac_ptr,
                         int* fac1ew_low, int* fac1ew_high, double* fac1ew_ptr,
                         int* fac2ew_low, int* fac2ew_high, double* fac2ew_ptr,
                         int* fac3ew_low, int* fac3ew_high, double* fac3ew_ptr,
                         int* fac4ew_low, int* fac4ew_high, double* fac4ew_ptr,
                         int* e_shift_low, int* e_shift_high, int* e_shift_ptr,
                         int* w_shift_low, int* w_shift_high, int* w_shift_ptr,
                         int* fac1tb_low, int* fac1tb_high, double* fac1tb_ptr,
                         int* fac2tb_low, int* fac2tb_high, double* fac2tb_ptr,
                         int* fac3tb_low, int* fac3tb_high, double* fac3tb_ptr,
                         int* fac4tb_low, int* fac4tb_high, double* fac4tb_ptr,
                         int* t_shift_low, int* t_shift_high, int* t_shift_ptr,
                         int* b_shift_low, int* b_shift_high, int* b_shift_ptr,
                         int* idxLoV,
                         int* idxHiV);

static void fort_vvelcoef( Uintah::constSFCYVariable<double> & vv,
                           Uintah::SFCYVariable<double> & cesav,
                           Uintah::SFCYVariable<double> & cwsav,
                           Uintah::SFCYVariable<double> & cnsav,
                           Uintah::SFCYVariable<double> & cssav,
                           Uintah::SFCYVariable<double> & ctsav,
                           Uintah::SFCYVariable<double> & cbsav,
                           Uintah::SFCYVariable<double> & ap,
                           Uintah::SFCYVariable<double> & ae,
                           Uintah::SFCYVariable<double> & aw,
                           Uintah::SFCYVariable<double> & an,
                           Uintah::SFCYVariable<double> & as,
                           Uintah::SFCYVariable<double> & at,
                           Uintah::SFCYVariable<double> & ab,
                           Uintah::constSFCXVariable<double> & uu,
                           Uintah::constSFCZVariable<double> & ww,
                           Uintah::constCCVariable<double> & den,
                           Uintah::constCCVariable<double> & vis,
                           Uintah::constCCVariable<double> & den_ref,
                           Uintah::SFCYVariable<double> & SU,
                           Uintah::constCCVariable<double> & eps,
                           double & deltat,
                           double & grav,
                           bool & lcend,
                           Uintah::OffsetArray1<double> & cee,
                           Uintah::OffsetArray1<double> & cwe,
                           Uintah::OffsetArray1<double> & cww,
                           Uintah::OffsetArray1<double> & cnnv,
                           Uintah::OffsetArray1<double> & csnv,
                           Uintah::OffsetArray1<double> & cssv,
                           Uintah::OffsetArray1<double> & ctt,
                           Uintah::OffsetArray1<double> & cbt,
                           Uintah::OffsetArray1<double> & cbb,
                           Uintah::OffsetArray1<double> & sew,
                           Uintah::OffsetArray1<double> & snsv,
                           Uintah::OffsetArray1<double> & sns,
                           Uintah::OffsetArray1<double> & stb,
                           Uintah::OffsetArray1<double> & dxep,
                           Uintah::OffsetArray1<double> & dxpw,
                           Uintah::OffsetArray1<double> & dynpv,
                           Uintah::OffsetArray1<double> & dypsv,
                           Uintah::OffsetArray1<double> & dyps,
                           Uintah::OffsetArray1<double> & dztp,
                           Uintah::OffsetArray1<double> & dzpb,
                           Uintah::OffsetArray1<double> & fac1v,
                           Uintah::OffsetArray1<double> & fac2v,
                           Uintah::OffsetArray1<double> & fac3v,
                           Uintah::OffsetArray1<double> & fac4v,
                           Uintah::OffsetArray1<int> & jnsdv,
                           Uintah::OffsetArray1<int> & jssdv,
                           Uintah::OffsetArray1<double> & efac,
                           Uintah::OffsetArray1<double> & wfac,
                           Uintah::OffsetArray1<double> & tfac,
                           Uintah::OffsetArray1<double> & bfac,
                           Uintah::OffsetArray1<double> & fac1ew,
                           Uintah::OffsetArray1<double> & fac2ew,
                           Uintah::OffsetArray1<double> & fac3ew,
                           Uintah::OffsetArray1<double> & fac4ew,
                           Uintah::OffsetArray1<int> & e_shift,
                           Uintah::OffsetArray1<int> & w_shift,
                           Uintah::OffsetArray1<double> & fac1tb,
                           Uintah::OffsetArray1<double> & fac2tb,
                           Uintah::OffsetArray1<double> & fac3tb,
                           Uintah::OffsetArray1<double> & fac4tb,
                           Uintah::OffsetArray1<int> & t_shift,
                           Uintah::OffsetArray1<int> & b_shift,
                           Uintah::IntVector & idxLoV,
                           Uintah::IntVector & idxHiV )
{
  Uintah::IntVector vv_low = vv.getWindow()->getOffset();
  Uintah::IntVector vv_high = vv.getWindow()->getData()->size() + vv_low - Uintah::IntVector(1, 1, 1);
  int vv_low_x = vv_low.x();
  int vv_high_x = vv_high.x();
  int vv_low_y = vv_low.y();
  int vv_high_y = vv_high.y();
  int vv_low_z = vv_low.z();
  int vv_high_z = vv_high.z();
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
  Uintah::IntVector ww_low = ww.getWindow()->getOffset();
  Uintah::IntVector ww_high = ww.getWindow()->getData()->size() + ww_low - Uintah::IntVector(1, 1, 1);
  int ww_low_x = ww_low.x();
  int ww_high_x = ww_high.x();
  int ww_low_y = ww_low.y();
  int ww_high_y = ww_high.y();
  int ww_low_z = ww_low.z();
  int ww_high_z = ww_high.z();
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
  int cee_low = cee.low();
  int cee_high = cee.high();
  int cwe_low = cwe.low();
  int cwe_high = cwe.high();
  int cww_low = cww.low();
  int cww_high = cww.high();
  int cnnv_low = cnnv.low();
  int cnnv_high = cnnv.high();
  int csnv_low = csnv.low();
  int csnv_high = csnv.high();
  int cssv_low = cssv.low();
  int cssv_high = cssv.high();
  int ctt_low = ctt.low();
  int ctt_high = ctt.high();
  int cbt_low = cbt.low();
  int cbt_high = cbt.high();
  int cbb_low = cbb.low();
  int cbb_high = cbb.high();
  int sew_low = sew.low();
  int sew_high = sew.high();
  int snsv_low = snsv.low();
  int snsv_high = snsv.high();
  int sns_low = sns.low();
  int sns_high = sns.high();
  int stb_low = stb.low();
  int stb_high = stb.high();
  int dxep_low = dxep.low();
  int dxep_high = dxep.high();
  int dxpw_low = dxpw.low();
  int dxpw_high = dxpw.high();
  int dynpv_low = dynpv.low();
  int dynpv_high = dynpv.high();
  int dypsv_low = dypsv.low();
  int dypsv_high = dypsv.high();
  int dyps_low = dyps.low();
  int dyps_high = dyps.high();
  int dztp_low = dztp.low();
  int dztp_high = dztp.high();
  int dzpb_low = dzpb.low();
  int dzpb_high = dzpb.high();
  int fac1v_low = fac1v.low();
  int fac1v_high = fac1v.high();
  int fac2v_low = fac2v.low();
  int fac2v_high = fac2v.high();
  int fac3v_low = fac3v.low();
  int fac3v_high = fac3v.high();
  int fac4v_low = fac4v.low();
  int fac4v_high = fac4v.high();
  int jnsdv_low = jnsdv.low();
  int jnsdv_high = jnsdv.high();
  int jssdv_low = jssdv.low();
  int jssdv_high = jssdv.high();
  int efac_low = efac.low();
  int efac_high = efac.high();
  int wfac_low = wfac.low();
  int wfac_high = wfac.high();
  int tfac_low = tfac.low();
  int tfac_high = tfac.high();
  int bfac_low = bfac.low();
  int bfac_high = bfac.high();
  int fac1ew_low = fac1ew.low();
  int fac1ew_high = fac1ew.high();
  int fac2ew_low = fac2ew.low();
  int fac2ew_high = fac2ew.high();
  int fac3ew_low = fac3ew.low();
  int fac3ew_high = fac3ew.high();
  int fac4ew_low = fac4ew.low();
  int fac4ew_high = fac4ew.high();
  int e_shift_low = e_shift.low();
  int e_shift_high = e_shift.high();
  int w_shift_low = w_shift.low();
  int w_shift_high = w_shift.high();
  int fac1tb_low = fac1tb.low();
  int fac1tb_high = fac1tb.high();
  int fac2tb_low = fac2tb.low();
  int fac2tb_high = fac2tb.high();
  int fac3tb_low = fac3tb.low();
  int fac3tb_high = fac3tb.high();
  int fac4tb_low = fac4tb.low();
  int fac4tb_high = fac4tb.high();
  int t_shift_low = t_shift.low();
  int t_shift_high = t_shift.high();
  int b_shift_low = b_shift.low();
  int b_shift_high = b_shift.high();
  F_vvelcoef( &vv_low_x, &vv_low_y, &vv_low_z, &vv_high_x, &vv_high_y, &vv_high_z, const_cast<double*>(vv.getPointer()),
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
            &ww_low_x, &ww_low_y, &ww_low_z, &ww_high_x, &ww_high_y, &ww_high_z, const_cast<double*>(ww.getPointer()),
            &den_low_x, &den_low_y, &den_low_z, &den_high_x, &den_high_y, &den_high_z, const_cast<double*>(den.getPointer()),
            &vis_low_x, &vis_low_y, &vis_low_z, &vis_high_x, &vis_high_y, &vis_high_z, const_cast<double*>(vis.getPointer()),
            &den_ref_low_x, &den_ref_low_y, &den_ref_low_z, &den_ref_high_x, &den_ref_high_y, &den_ref_high_z, const_cast<double*>(den_ref.getPointer()),
            &SU_low_x, &SU_low_y, &SU_low_z, &SU_high_x, &SU_high_y, &SU_high_z, SU.getPointer(),
            &eps_low_x, &eps_low_y, &eps_low_z, &eps_high_x, &eps_high_y, &eps_high_z, const_cast<double*>(eps.getPointer()),
            &deltat,
            &grav,
            &lcend,
            &cee_low, &cee_high, cee.get_objs(),
            &cwe_low, &cwe_high, cwe.get_objs(),
            &cww_low, &cww_high, cww.get_objs(),
            &cnnv_low, &cnnv_high, cnnv.get_objs(),
            &csnv_low, &csnv_high, csnv.get_objs(),
            &cssv_low, &cssv_high, cssv.get_objs(),
            &ctt_low, &ctt_high, ctt.get_objs(),
            &cbt_low, &cbt_high, cbt.get_objs(),
            &cbb_low, &cbb_high, cbb.get_objs(),
            &sew_low, &sew_high, sew.get_objs(),
            &snsv_low, &snsv_high, snsv.get_objs(),
            &sns_low, &sns_high, sns.get_objs(),
            &stb_low, &stb_high, stb.get_objs(),
            &dxep_low, &dxep_high, dxep.get_objs(),
            &dxpw_low, &dxpw_high, dxpw.get_objs(),
            &dynpv_low, &dynpv_high, dynpv.get_objs(),
            &dypsv_low, &dypsv_high, dypsv.get_objs(),
            &dyps_low, &dyps_high, dyps.get_objs(),
            &dztp_low, &dztp_high, dztp.get_objs(),
            &dzpb_low, &dzpb_high, dzpb.get_objs(),
            &fac1v_low, &fac1v_high, fac1v.get_objs(),
            &fac2v_low, &fac2v_high, fac2v.get_objs(),
            &fac3v_low, &fac3v_high, fac3v.get_objs(),
            &fac4v_low, &fac4v_high, fac4v.get_objs(),
            &jnsdv_low, &jnsdv_high, jnsdv.get_objs(),
            &jssdv_low, &jssdv_high, jssdv.get_objs(),
            &efac_low, &efac_high, efac.get_objs(),
            &wfac_low, &wfac_high, wfac.get_objs(),
            &tfac_low, &tfac_high, tfac.get_objs(),
            &bfac_low, &bfac_high, bfac.get_objs(),
            &fac1ew_low, &fac1ew_high, fac1ew.get_objs(),
            &fac2ew_low, &fac2ew_high, fac2ew.get_objs(),
            &fac3ew_low, &fac3ew_high, fac3ew.get_objs(),
            &fac4ew_low, &fac4ew_high, fac4ew.get_objs(),
            &e_shift_low, &e_shift_high, e_shift.get_objs(),
            &w_shift_low, &w_shift_high, w_shift.get_objs(),
            &fac1tb_low, &fac1tb_high, fac1tb.get_objs(),
            &fac2tb_low, &fac2tb_high, fac2tb.get_objs(),
            &fac3tb_low, &fac3tb_high, fac3tb.get_objs(),
            &fac4tb_low, &fac4tb_high, fac4tb.get_objs(),
            &t_shift_low, &t_shift_high, t_shift.get_objs(),
            &b_shift_low, &b_shift_high, b_shift.get_objs(),
            idxLoV.get_pointer(),
            idxHiV.get_pointer() );
}

#else /* !__cplusplus */

C This is the FORTRAN code portion of the file:

      subroutine VVELCOEF(vv_low_x, vv_low_y, vv_low_z, vv_high_x,
     & vv_high_y, vv_high_z, vv, cesav_low_x, cesav_low_y, cesav_low_z,
     &  cesav_high_x, cesav_high_y, cesav_high_z, cesav, cwsav_low_x, 
     & cwsav_low_y, cwsav_low_z, cwsav_high_x, cwsav_high_y, 
     & cwsav_high_z, cwsav, cnsav_low_x, cnsav_low_y, cnsav_low_z, 
     & cnsav_high_x, cnsav_high_y, cnsav_high_z, cnsav, cssav_low_x, 
     & cssav_low_y, cssav_low_z, cssav_high_x, cssav_high_y, 
     & cssav_high_z, cssav, ctsav_low_x, ctsav_low_y, ctsav_low_z, 
     & ctsav_high_x, ctsav_high_y, ctsav_high_z, ctsav, cbsav_low_x, 
     & cbsav_low_y, cbsav_low_z, cbsav_high_x, cbsav_high_y, 
     & cbsav_high_z, cbsav, ap_low_x, ap_low_y, ap_low_z, ap_high_x, 
     & ap_high_y, ap_high_z, ap, ae_low_x, ae_low_y, ae_low_z, 
     & ae_high_x, ae_high_y, ae_high_z, ae, aw_low_x, aw_low_y, 
     & aw_low_z, aw_high_x, aw_high_y, aw_high_z, aw, an_low_x, 
     & an_low_y, an_low_z, an_high_x, an_high_y, an_high_z, an, 
     & as_low_x, as_low_y, as_low_z, as_high_x, as_high_y, as_high_z, 
     & as, at_low_x, at_low_y, at_low_z, at_high_x, at_high_y, 
     & at_high_z, at, ab_low_x, ab_low_y, ab_low_z, ab_high_x, 
     & ab_high_y, ab_high_z, ab, uu_low_x, uu_low_y, uu_low_z, 
     & uu_high_x, uu_high_y, uu_high_z, uu, ww_low_x, ww_low_y, 
     & ww_low_z, ww_high_x, ww_high_y, ww_high_z, ww, den_low_x, 
     & den_low_y, den_low_z, den_high_x, den_high_y, den_high_z, den, 
     & vis_low_x, vis_low_y, vis_low_z, vis_high_x, vis_high_y, 
     & vis_high_z, vis, den_ref_low_x, den_ref_low_y, den_ref_low_z, 
     & den_ref_high_x, den_ref_high_y, den_ref_high_z, den_ref, 
     & SU_low_x, SU_low_y, SU_low_z, SU_high_x, SU_high_y, SU_high_z, 
     & SU, eps_low_x, eps_low_y, eps_low_z, eps_high_x, eps_high_y, 
     & eps_high_z, eps, deltat, grav, lcend, cee_low, cee_high, cee, 
     & cwe_low, cwe_high, cwe, cww_low, cww_high, cww, cnnv_low, 
     & cnnv_high, cnnv, csnv_low, csnv_high, csnv, cssv_low, cssv_high,
     &  cssv, ctt_low, ctt_high, ctt, cbt_low, cbt_high, cbt, cbb_low, 
     & cbb_high, cbb, sew_low, sew_high, sew, snsv_low, snsv_high, snsv
     & , sns_low, sns_high, sns, stb_low, stb_high, stb, dxep_low, 
     & dxep_high, dxep, dxpw_low, dxpw_high, dxpw, dynpv_low, 
     & dynpv_high, dynpv, dypsv_low, dypsv_high, dypsv, dyps_low, 
     & dyps_high, dyps, dztp_low, dztp_high, dztp, dzpb_low, dzpb_high,
     &  dzpb, fac1v_low, fac1v_high, fac1v, fac2v_low, fac2v_high, 
     & fac2v, fac3v_low, fac3v_high, fac3v, fac4v_low, fac4v_high, 
     & fac4v, jnsdv_low, jnsdv_high, jnsdv, jssdv_low, jssdv_high, 
     & jssdv, efac_low, efac_high, efac, wfac_low, wfac_high, wfac, 
     & tfac_low, tfac_high, tfac, bfac_low, bfac_high, bfac, fac1ew_low
     & , fac1ew_high, fac1ew, fac2ew_low, fac2ew_high, fac2ew, 
     & fac3ew_low, fac3ew_high, fac3ew, fac4ew_low, fac4ew_high, fac4ew
     & , e_shift_low, e_shift_high, e_shift, w_shift_low, w_shift_high,
     &  w_shift, fac1tb_low, fac1tb_high, fac1tb, fac2tb_low, 
     & fac2tb_high, fac2tb, fac3tb_low, fac3tb_high, fac3tb, fac4tb_low
     & , fac4tb_high, fac4tb, t_shift_low, t_shift_high, t_shift, 
     & b_shift_low, b_shift_high, b_shift, idxLoV, idxHiV)

      implicit none
      integer vv_low_x, vv_low_y, vv_low_z, vv_high_x, vv_high_y, 
     & vv_high_z
      double precision vv(vv_low_x:vv_high_x, vv_low_y:vv_high_y, 
     & vv_low_z:vv_high_z)
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
      integer ww_low_x, ww_low_y, ww_low_z, ww_high_x, ww_high_y, 
     & ww_high_z
      double precision ww(ww_low_x:ww_high_x, ww_low_y:ww_high_y, 
     & ww_low_z:ww_high_z)
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
      logical*1 lcend
      integer cee_low
      integer cee_high
      double precision cee(cee_low:cee_high)
      integer cwe_low
      integer cwe_high
      double precision cwe(cwe_low:cwe_high)
      integer cww_low
      integer cww_high
      double precision cww(cww_low:cww_high)
      integer cnnv_low
      integer cnnv_high
      double precision cnnv(cnnv_low:cnnv_high)
      integer csnv_low
      integer csnv_high
      double precision csnv(csnv_low:csnv_high)
      integer cssv_low
      integer cssv_high
      double precision cssv(cssv_low:cssv_high)
      integer ctt_low
      integer ctt_high
      double precision ctt(ctt_low:ctt_high)
      integer cbt_low
      integer cbt_high
      double precision cbt(cbt_low:cbt_high)
      integer cbb_low
      integer cbb_high
      double precision cbb(cbb_low:cbb_high)
      integer sew_low
      integer sew_high
      double precision sew(sew_low:sew_high)
      integer snsv_low
      integer snsv_high
      double precision snsv(snsv_low:snsv_high)
      integer sns_low
      integer sns_high
      double precision sns(sns_low:sns_high)
      integer stb_low
      integer stb_high
      double precision stb(stb_low:stb_high)
      integer dxep_low
      integer dxep_high
      double precision dxep(dxep_low:dxep_high)
      integer dxpw_low
      integer dxpw_high
      double precision dxpw(dxpw_low:dxpw_high)
      integer dynpv_low
      integer dynpv_high
      double precision dynpv(dynpv_low:dynpv_high)
      integer dypsv_low
      integer dypsv_high
      double precision dypsv(dypsv_low:dypsv_high)
      integer dyps_low
      integer dyps_high
      double precision dyps(dyps_low:dyps_high)
      integer dztp_low
      integer dztp_high
      double precision dztp(dztp_low:dztp_high)
      integer dzpb_low
      integer dzpb_high
      double precision dzpb(dzpb_low:dzpb_high)
      integer fac1v_low
      integer fac1v_high
      double precision fac1v(fac1v_low:fac1v_high)
      integer fac2v_low
      integer fac2v_high
      double precision fac2v(fac2v_low:fac2v_high)
      integer fac3v_low
      integer fac3v_high
      double precision fac3v(fac3v_low:fac3v_high)
      integer fac4v_low
      integer fac4v_high
      double precision fac4v(fac4v_low:fac4v_high)
      integer jnsdv_low
      integer jnsdv_high
      integer jnsdv(jnsdv_low:jnsdv_high)
      integer jssdv_low
      integer jssdv_high
      integer jssdv(jssdv_low:jssdv_high)
      integer efac_low
      integer efac_high
      double precision efac(efac_low:efac_high)
      integer wfac_low
      integer wfac_high
      double precision wfac(wfac_low:wfac_high)
      integer tfac_low
      integer tfac_high
      double precision tfac(tfac_low:tfac_high)
      integer bfac_low
      integer bfac_high
      double precision bfac(bfac_low:bfac_high)
      integer fac1ew_low
      integer fac1ew_high
      double precision fac1ew(fac1ew_low:fac1ew_high)
      integer fac2ew_low
      integer fac2ew_high
      double precision fac2ew(fac2ew_low:fac2ew_high)
      integer fac3ew_low
      integer fac3ew_high
      double precision fac3ew(fac3ew_low:fac3ew_high)
      integer fac4ew_low
      integer fac4ew_high
      double precision fac4ew(fac4ew_low:fac4ew_high)
      integer e_shift_low
      integer e_shift_high
      integer e_shift(e_shift_low:e_shift_high)
      integer w_shift_low
      integer w_shift_high
      integer w_shift(w_shift_low:w_shift_high)
      integer fac1tb_low
      integer fac1tb_high
      double precision fac1tb(fac1tb_low:fac1tb_high)
      integer fac2tb_low
      integer fac2tb_high
      double precision fac2tb(fac2tb_low:fac2tb_high)
      integer fac3tb_low
      integer fac3tb_high
      double precision fac3tb(fac3tb_low:fac3tb_high)
      integer fac4tb_low
      integer fac4tb_high
      double precision fac4tb(fac4tb_low:fac4tb_high)
      integer t_shift_low
      integer t_shift_high
      integer t_shift(t_shift_low:t_shift_high)
      integer b_shift_low
      integer b_shift_high
      integer b_shift(b_shift_low:b_shift_high)
      integer idxLoV(3)
      integer idxHiV(3)
#endif /* __cplusplus */

#endif /* fspec_vvelcoef */
