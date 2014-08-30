
#ifndef fspec_energy_exchange_term
#define fspec_energy_exchange_term

#ifdef __cplusplus

extern "C" void energy_exchange_term_(int* hts_fcx_low_x, int* hts_fcx_low_y, int* hts_fcx_low_z, int* hts_fcx_high_x, int* hts_fcx_high_y, int* hts_fcx_high_z, double* hts_fcx_ptr,
                                      int* hts_fcy_low_x, int* hts_fcy_low_y, int* hts_fcy_low_z, int* hts_fcy_high_x, int* hts_fcy_high_y, int* hts_fcy_high_z, double* hts_fcy_ptr,
                                      int* hts_fcz_low_x, int* hts_fcz_low_y, int* hts_fcz_low_z, int* hts_fcz_high_x, int* hts_fcz_high_y, int* hts_fcz_high_z, double* hts_fcz_ptr,
                                      int* hts_fcx_rad_low_x, int* hts_fcx_rad_low_y, int* hts_fcx_rad_low_z, int* hts_fcx_rad_high_x, int* hts_fcx_rad_high_y, int* hts_fcx_rad_high_z, double* hts_fcx_rad_ptr,
                                      int* hts_fcy_rad_low_x, int* hts_fcy_rad_low_y, int* hts_fcy_rad_low_z, int* hts_fcy_rad_high_x, int* hts_fcy_rad_high_y, int* hts_fcy_rad_high_z, double* hts_fcy_rad_ptr,
                                      int* hts_fcz_rad_low_x, int* hts_fcz_rad_low_y, int* hts_fcz_rad_low_z, int* hts_fcz_rad_high_x, int* hts_fcz_rad_high_y, int* hts_fcz_rad_high_z, double* hts_fcz_rad_ptr,
                                      int* hts_cc_low_x, int* hts_cc_low_y, int* hts_cc_low_z, int* hts_cc_high_x, int* hts_cc_high_y, int* hts_cc_high_z, double* hts_cc_ptr,
                                      int* htflux_convX_low_x, int* htflux_convX_low_y, int* htflux_convX_low_z, int* htflux_convX_high_x, int* htflux_convX_high_y, int* htflux_convX_high_z, double* htflux_convX_ptr,
                                      int* htflux_radX_low_x, int* htflux_radX_low_y, int* htflux_radX_low_z, int* htflux_radX_high_x, int* htflux_radX_high_y, int* htflux_radX_high_z, double* htflux_radX_ptr,
                                      int* htfluxX_low_x, int* htfluxX_low_y, int* htfluxX_low_z, int* htfluxX_high_x, int* htfluxX_high_y, int* htfluxX_high_z, double* htfluxX_ptr,
                                      int* htflux_convY_low_x, int* htflux_convY_low_y, int* htflux_convY_low_z, int* htflux_convY_high_x, int* htflux_convY_high_y, int* htflux_convY_high_z, double* htflux_convY_ptr,
                                      int* htflux_radY_low_x, int* htflux_radY_low_y, int* htflux_radY_low_z, int* htflux_radY_high_x, int* htflux_radY_high_y, int* htflux_radY_high_z, double* htflux_radY_ptr,
                                      int* htfluxY_low_x, int* htfluxY_low_y, int* htfluxY_low_z, int* htfluxY_high_x, int* htfluxY_high_y, int* htfluxY_high_z, double* htfluxY_ptr,
                                      int* htflux_convZ_low_x, int* htflux_convZ_low_y, int* htflux_convZ_low_z, int* htflux_convZ_high_x, int* htflux_convZ_high_y, int* htflux_convZ_high_z, double* htflux_convZ_ptr,
                                      int* htflux_radZ_low_x, int* htflux_radZ_low_y, int* htflux_radZ_low_z, int* htflux_radZ_high_x, int* htflux_radZ_high_y, int* htflux_radZ_high_z, double* htflux_radZ_ptr,
                                      int* htfluxZ_low_x, int* htfluxZ_low_y, int* htfluxZ_low_z, int* htfluxZ_high_x, int* htfluxZ_high_y, int* htfluxZ_high_z, double* htfluxZ_ptr,
                                      int* htflux_convCC_low_x, int* htflux_convCC_low_y, int* htflux_convCC_low_z, int* htflux_convCC_high_x, int* htflux_convCC_high_y, int* htflux_convCC_high_z, double* htflux_convCC_ptr,
                                      int* sug_cc_low_x, int* sug_cc_low_y, int* sug_cc_low_z, int* sug_cc_high_x, int* sug_cc_high_y, int* sug_cc_high_z, double* sug_cc_ptr,
                                      int* spg_cc_low_x, int* spg_cc_low_y, int* spg_cc_low_z, int* spg_cc_high_x, int* spg_cc_high_y, int* spg_cc_high_z, double* spg_cc_ptr,
                                      int* sug_fcx_low_x, int* sug_fcx_low_y, int* sug_fcx_low_z, int* sug_fcx_high_x, int* sug_fcx_high_y, int* sug_fcx_high_z, double* sug_fcx_ptr,
                                      int* spg_fcx_low_x, int* spg_fcx_low_y, int* spg_fcx_low_z, int* spg_fcx_high_x, int* spg_fcx_high_y, int* spg_fcx_high_z, double* spg_fcx_ptr,
                                      int* sug_fcy_low_x, int* sug_fcy_low_y, int* sug_fcy_low_z, int* sug_fcy_high_x, int* sug_fcy_high_y, int* sug_fcy_high_z, double* sug_fcy_ptr,
                                      int* spg_fcy_low_x, int* spg_fcy_low_y, int* spg_fcy_low_z, int* spg_fcy_high_x, int* spg_fcy_high_y, int* spg_fcy_high_z, double* spg_fcy_ptr,
                                      int* sug_fcz_low_x, int* sug_fcz_low_y, int* sug_fcz_low_z, int* sug_fcz_high_x, int* sug_fcz_high_y, int* sug_fcz_high_z, double* sug_fcz_ptr,
                                      int* spg_fcz_low_x, int* spg_fcz_low_y, int* spg_fcz_low_z, int* spg_fcz_high_x, int* spg_fcz_high_y, int* spg_fcz_high_z, double* spg_fcz_ptr,
                                      int* kstabh_low_x, int* kstabh_low_y, int* kstabh_low_z, int* kstabh_high_x, int* kstabh_high_y, int* kstabh_high_z, double* kstabh_ptr,
                                      int* tg_low_x, int* tg_low_y, int* tg_low_z, int* tg_high_x, int* tg_high_y, int* tg_high_z, double* tg_ptr,
                                      int* ts_cc_low_x, int* ts_cc_low_y, int* ts_cc_low_z, int* ts_cc_high_x, int* ts_cc_high_y, int* ts_cc_high_z, double* ts_cc_ptr,
                                      int* ts_fcx_low_x, int* ts_fcx_low_y, int* ts_fcx_low_z, int* ts_fcx_high_x, int* ts_fcx_high_y, int* ts_fcx_high_z, double* ts_fcx_ptr,
                                      int* ts_fcy_low_x, int* ts_fcy_low_y, int* ts_fcy_low_z, int* ts_fcy_high_x, int* ts_fcy_high_y, int* ts_fcy_high_z, double* ts_fcy_ptr,
                                      int* ts_fcz_low_x, int* ts_fcz_low_y, int* ts_fcz_low_z, int* ts_fcz_high_x, int* ts_fcz_high_y, int* ts_fcz_high_z, double* ts_fcz_ptr,
                                      int* ug_cc_low_x, int* ug_cc_low_y, int* ug_cc_low_z, int* ug_cc_high_x, int* ug_cc_high_y, int* ug_cc_high_z, double* ug_cc_ptr,
                                      int* vg_cc_low_x, int* vg_cc_low_y, int* vg_cc_low_z, int* vg_cc_high_x, int* vg_cc_high_y, int* vg_cc_high_z, double* vg_cc_ptr,
                                      int* wg_cc_low_x, int* wg_cc_low_y, int* wg_cc_low_z, int* wg_cc_high_x, int* wg_cc_high_y, int* wg_cc_high_z, double* wg_cc_ptr,
                                      int* up_cc_low_x, int* up_cc_low_y, int* up_cc_low_z, int* up_cc_high_x, int* up_cc_high_y, int* up_cc_high_z, double* up_cc_ptr,
                                      int* vp_cc_low_x, int* vp_cc_low_y, int* vp_cc_low_z, int* vp_cc_high_x, int* vp_cc_high_y, int* vp_cc_high_z, double* vp_cc_ptr,
                                      int* wp_cc_low_x, int* wp_cc_low_y, int* wp_cc_low_z, int* wp_cc_high_x, int* wp_cc_high_y, int* wp_cc_high_z, double* wp_cc_ptr,
                                      int* vp_fcx_low_x, int* vp_fcx_low_y, int* vp_fcx_low_z, int* vp_fcx_high_x, int* vp_fcx_high_y, int* vp_fcx_high_z, double* vp_fcx_ptr,
                                      int* wp_fcx_low_x, int* wp_fcx_low_y, int* wp_fcx_low_z, int* wp_fcx_high_x, int* wp_fcx_high_y, int* wp_fcx_high_z, double* wp_fcx_ptr,
                                      int* up_fcy_low_x, int* up_fcy_low_y, int* up_fcy_low_z, int* up_fcy_high_x, int* up_fcy_high_y, int* up_fcy_high_z, double* up_fcy_ptr,
                                      int* wp_fcy_low_x, int* wp_fcy_low_y, int* wp_fcy_low_z, int* wp_fcy_high_x, int* wp_fcy_high_y, int* wp_fcy_high_z, double* wp_fcy_ptr,
                                      int* up_fcz_low_x, int* up_fcz_low_y, int* up_fcz_low_z, int* up_fcz_high_x, int* up_fcz_high_y, int* up_fcz_high_z, double* up_fcz_ptr,
                                      int* vp_fcz_low_x, int* vp_fcz_low_y, int* vp_fcz_low_z, int* vp_fcz_high_x, int* vp_fcz_high_y, int* vp_fcz_high_z, double* vp_fcz_ptr,
                                      int* denMicro_low_x, int* denMicro_low_y, int* denMicro_low_z, int* denMicro_high_x, int* denMicro_high_y, int* denMicro_high_z, double* denMicro_ptr,
                                      int* enth_low_x, int* enth_low_y, int* enth_low_z, int* enth_high_x, int* enth_high_y, int* enth_high_z, double* enth_ptr,
                                      int* rfluxE_low_x, int* rfluxE_low_y, int* rfluxE_low_z, int* rfluxE_high_x, int* rfluxE_high_y, int* rfluxE_high_z, double* rfluxE_ptr,
                                      int* rfluxW_low_x, int* rfluxW_low_y, int* rfluxW_low_z, int* rfluxW_high_x, int* rfluxW_high_y, int* rfluxW_high_z, double* rfluxW_ptr,
                                      int* rfluxN_low_x, int* rfluxN_low_y, int* rfluxN_low_z, int* rfluxN_high_x, int* rfluxN_high_y, int* rfluxN_high_z, double* rfluxN_ptr,
                                      int* rfluxS_low_x, int* rfluxS_low_y, int* rfluxS_low_z, int* rfluxS_high_x, int* rfluxS_high_y, int* rfluxS_high_z, double* rfluxS_ptr,
                                      int* rfluxT_low_x, int* rfluxT_low_y, int* rfluxT_low_z, int* rfluxT_high_x, int* rfluxT_high_y, int* rfluxT_high_z, double* rfluxT_ptr,
                                      int* rfluxB_low_x, int* rfluxB_low_y, int* rfluxB_low_z, int* rfluxB_high_x, int* rfluxB_high_y, int* rfluxB_high_z, double* rfluxB_ptr,
                                      int* epsg_low_x, int* epsg_low_y, int* epsg_low_z, int* epsg_high_x, int* epsg_high_y, int* epsg_high_z, double* epsg_ptr,
                                      int* epss_low_x, int* epss_low_y, int* epss_low_z, int* epss_high_x, int* epss_high_y, int* epss_high_z, double* epss_ptr,
                                      double* dx,
                                      double* dy,
                                      double* dz,
                                      double* tcond,
                                      double* csmag,
                                      double* prturb,
                                      double* cpfluid,
                                      int* valid_lo,
                                      int* valid_hi,
                                      int* pcell_low_x, int* pcell_low_y, int* pcell_low_z, int* pcell_high_x, int* pcell_high_y, int* pcell_high_z, int* pcell_ptr,
                                      int* wall,
                                      int* ffield);

static void fort_energy_exchange_term( Uintah::SFCXVariable<double> & hts_fcx,
                                       Uintah::SFCYVariable<double> & hts_fcy,
                                       Uintah::SFCZVariable<double> & hts_fcz,
                                       Uintah::SFCXVariable<double> & hts_fcx_rad,
                                       Uintah::SFCYVariable<double> & hts_fcy_rad,
                                       Uintah::SFCZVariable<double> & hts_fcz_rad,
                                       Uintah::CCVariable<double> & hts_cc,
                                       Uintah::SFCXVariable<double> & htflux_convX,
                                       Uintah::SFCXVariable<double> & htflux_radX,
                                       Uintah::SFCXVariable<double> & htfluxX,
                                       Uintah::SFCYVariable<double> & htflux_convY,
                                       Uintah::SFCYVariable<double> & htflux_radY,
                                       Uintah::SFCYVariable<double> & htfluxY,
                                       Uintah::SFCZVariable<double> & htflux_convZ,
                                       Uintah::SFCZVariable<double> & htflux_radZ,
                                       Uintah::SFCZVariable<double> & htfluxZ,
                                       Uintah::CCVariable<double> & htflux_convCC,
                                       Uintah::CCVariable<double> & sug_cc,
                                       Uintah::CCVariable<double> & spg_cc,
                                       Uintah::SFCXVariable<double> & sug_fcx,
                                       Uintah::SFCXVariable<double> & spg_fcx,
                                       Uintah::SFCYVariable<double> & sug_fcy,
                                       Uintah::SFCYVariable<double> & spg_fcy,
                                       Uintah::SFCZVariable<double> & sug_fcz,
                                       Uintah::SFCZVariable<double> & spg_fcz,
                                       Uintah::CCVariable<double> & kstabh,
                                       Uintah::constCCVariable<double> & tg,
                                       Uintah::constCCVariable<double> & ts_cc,
                                       Uintah::constSFCXVariable<double> & ts_fcx,
                                       Uintah::constSFCYVariable<double> & ts_fcy,
                                       Uintah::constSFCZVariable<double> & ts_fcz,
                                       Uintah::CCVariable<double> & ug_cc,
                                       Uintah::CCVariable<double> & vg_cc,
                                       Uintah::CCVariable<double> & wg_cc,
                                       Uintah::constCCVariable<double> & up_cc,
                                       Uintah::constCCVariable<double> & vp_cc,
                                       Uintah::constCCVariable<double> & wp_cc,
                                       Uintah::constSFCXVariable<double> & vp_fcx,
                                       Uintah::constSFCXVariable<double> & wp_fcx,
                                       Uintah::constSFCYVariable<double> & up_fcy,
                                       Uintah::constSFCYVariable<double> & wp_fcy,
                                       Uintah::constSFCZVariable<double> & up_fcz,
                                       Uintah::constSFCZVariable<double> & vp_fcz,
                                       Uintah::constCCVariable<double> & denMicro,
                                       Uintah::constCCVariable<double> & enth,
                                       Uintah::CCVariable<double> & rfluxE,
                                       Uintah::CCVariable<double> & rfluxW,
                                       Uintah::CCVariable<double> & rfluxN,
                                       Uintah::CCVariable<double> & rfluxS,
                                       Uintah::CCVariable<double> & rfluxT,
                                       Uintah::CCVariable<double> & rfluxB,
                                       Uintah::constCCVariable<double> & epsg,
                                       Uintah::constCCVariable<double> & epss,
                                       double & dx,
                                       double & dy,
                                       double & dz,
                                       double & tcond,
                                       double & csmag,
                                       double & prturb,
                                       double & cpfluid,
                                       Uintah::IntVector & valid_lo,
                                       Uintah::IntVector & valid_hi,
                                       Uintah::constCCVariable<int> & pcell,
                                       int & wall,
                                       int & ffield )
{
  Uintah::IntVector hts_fcx_low = hts_fcx.getWindow()->getOffset();
  Uintah::IntVector hts_fcx_high = hts_fcx.getWindow()->getData()->size() + hts_fcx_low - Uintah::IntVector(1, 1, 1);
  int hts_fcx_low_x = hts_fcx_low.x();
  int hts_fcx_high_x = hts_fcx_high.x();
  int hts_fcx_low_y = hts_fcx_low.y();
  int hts_fcx_high_y = hts_fcx_high.y();
  int hts_fcx_low_z = hts_fcx_low.z();
  int hts_fcx_high_z = hts_fcx_high.z();
  Uintah::IntVector hts_fcy_low = hts_fcy.getWindow()->getOffset();
  Uintah::IntVector hts_fcy_high = hts_fcy.getWindow()->getData()->size() + hts_fcy_low - Uintah::IntVector(1, 1, 1);
  int hts_fcy_low_x = hts_fcy_low.x();
  int hts_fcy_high_x = hts_fcy_high.x();
  int hts_fcy_low_y = hts_fcy_low.y();
  int hts_fcy_high_y = hts_fcy_high.y();
  int hts_fcy_low_z = hts_fcy_low.z();
  int hts_fcy_high_z = hts_fcy_high.z();
  Uintah::IntVector hts_fcz_low = hts_fcz.getWindow()->getOffset();
  Uintah::IntVector hts_fcz_high = hts_fcz.getWindow()->getData()->size() + hts_fcz_low - Uintah::IntVector(1, 1, 1);
  int hts_fcz_low_x = hts_fcz_low.x();
  int hts_fcz_high_x = hts_fcz_high.x();
  int hts_fcz_low_y = hts_fcz_low.y();
  int hts_fcz_high_y = hts_fcz_high.y();
  int hts_fcz_low_z = hts_fcz_low.z();
  int hts_fcz_high_z = hts_fcz_high.z();
  Uintah::IntVector hts_fcx_rad_low = hts_fcx_rad.getWindow()->getOffset();
  Uintah::IntVector hts_fcx_rad_high = hts_fcx_rad.getWindow()->getData()->size() + hts_fcx_rad_low - Uintah::IntVector(1, 1, 1);
  int hts_fcx_rad_low_x = hts_fcx_rad_low.x();
  int hts_fcx_rad_high_x = hts_fcx_rad_high.x();
  int hts_fcx_rad_low_y = hts_fcx_rad_low.y();
  int hts_fcx_rad_high_y = hts_fcx_rad_high.y();
  int hts_fcx_rad_low_z = hts_fcx_rad_low.z();
  int hts_fcx_rad_high_z = hts_fcx_rad_high.z();
  Uintah::IntVector hts_fcy_rad_low = hts_fcy_rad.getWindow()->getOffset();
  Uintah::IntVector hts_fcy_rad_high = hts_fcy_rad.getWindow()->getData()->size() + hts_fcy_rad_low - Uintah::IntVector(1, 1, 1);
  int hts_fcy_rad_low_x = hts_fcy_rad_low.x();
  int hts_fcy_rad_high_x = hts_fcy_rad_high.x();
  int hts_fcy_rad_low_y = hts_fcy_rad_low.y();
  int hts_fcy_rad_high_y = hts_fcy_rad_high.y();
  int hts_fcy_rad_low_z = hts_fcy_rad_low.z();
  int hts_fcy_rad_high_z = hts_fcy_rad_high.z();
  Uintah::IntVector hts_fcz_rad_low = hts_fcz_rad.getWindow()->getOffset();
  Uintah::IntVector hts_fcz_rad_high = hts_fcz_rad.getWindow()->getData()->size() + hts_fcz_rad_low - Uintah::IntVector(1, 1, 1);
  int hts_fcz_rad_low_x = hts_fcz_rad_low.x();
  int hts_fcz_rad_high_x = hts_fcz_rad_high.x();
  int hts_fcz_rad_low_y = hts_fcz_rad_low.y();
  int hts_fcz_rad_high_y = hts_fcz_rad_high.y();
  int hts_fcz_rad_low_z = hts_fcz_rad_low.z();
  int hts_fcz_rad_high_z = hts_fcz_rad_high.z();
  Uintah::IntVector hts_cc_low = hts_cc.getWindow()->getOffset();
  Uintah::IntVector hts_cc_high = hts_cc.getWindow()->getData()->size() + hts_cc_low - Uintah::IntVector(1, 1, 1);
  int hts_cc_low_x = hts_cc_low.x();
  int hts_cc_high_x = hts_cc_high.x();
  int hts_cc_low_y = hts_cc_low.y();
  int hts_cc_high_y = hts_cc_high.y();
  int hts_cc_low_z = hts_cc_low.z();
  int hts_cc_high_z = hts_cc_high.z();
  Uintah::IntVector htflux_convX_low = htflux_convX.getWindow()->getOffset();
  Uintah::IntVector htflux_convX_high = htflux_convX.getWindow()->getData()->size() + htflux_convX_low - Uintah::IntVector(1, 1, 1);
  int htflux_convX_low_x = htflux_convX_low.x();
  int htflux_convX_high_x = htflux_convX_high.x();
  int htflux_convX_low_y = htflux_convX_low.y();
  int htflux_convX_high_y = htflux_convX_high.y();
  int htflux_convX_low_z = htflux_convX_low.z();
  int htflux_convX_high_z = htflux_convX_high.z();
  Uintah::IntVector htflux_radX_low = htflux_radX.getWindow()->getOffset();
  Uintah::IntVector htflux_radX_high = htflux_radX.getWindow()->getData()->size() + htflux_radX_low - Uintah::IntVector(1, 1, 1);
  int htflux_radX_low_x = htflux_radX_low.x();
  int htflux_radX_high_x = htflux_radX_high.x();
  int htflux_radX_low_y = htflux_radX_low.y();
  int htflux_radX_high_y = htflux_radX_high.y();
  int htflux_radX_low_z = htflux_radX_low.z();
  int htflux_radX_high_z = htflux_radX_high.z();
  Uintah::IntVector htfluxX_low = htfluxX.getWindow()->getOffset();
  Uintah::IntVector htfluxX_high = htfluxX.getWindow()->getData()->size() + htfluxX_low - Uintah::IntVector(1, 1, 1);
  int htfluxX_low_x = htfluxX_low.x();
  int htfluxX_high_x = htfluxX_high.x();
  int htfluxX_low_y = htfluxX_low.y();
  int htfluxX_high_y = htfluxX_high.y();
  int htfluxX_low_z = htfluxX_low.z();
  int htfluxX_high_z = htfluxX_high.z();
  Uintah::IntVector htflux_convY_low = htflux_convY.getWindow()->getOffset();
  Uintah::IntVector htflux_convY_high = htflux_convY.getWindow()->getData()->size() + htflux_convY_low - Uintah::IntVector(1, 1, 1);
  int htflux_convY_low_x = htflux_convY_low.x();
  int htflux_convY_high_x = htflux_convY_high.x();
  int htflux_convY_low_y = htflux_convY_low.y();
  int htflux_convY_high_y = htflux_convY_high.y();
  int htflux_convY_low_z = htflux_convY_low.z();
  int htflux_convY_high_z = htflux_convY_high.z();
  Uintah::IntVector htflux_radY_low = htflux_radY.getWindow()->getOffset();
  Uintah::IntVector htflux_radY_high = htflux_radY.getWindow()->getData()->size() + htflux_radY_low - Uintah::IntVector(1, 1, 1);
  int htflux_radY_low_x = htflux_radY_low.x();
  int htflux_radY_high_x = htflux_radY_high.x();
  int htflux_radY_low_y = htflux_radY_low.y();
  int htflux_radY_high_y = htflux_radY_high.y();
  int htflux_radY_low_z = htflux_radY_low.z();
  int htflux_radY_high_z = htflux_radY_high.z();
  Uintah::IntVector htfluxY_low = htfluxY.getWindow()->getOffset();
  Uintah::IntVector htfluxY_high = htfluxY.getWindow()->getData()->size() + htfluxY_low - Uintah::IntVector(1, 1, 1);
  int htfluxY_low_x = htfluxY_low.x();
  int htfluxY_high_x = htfluxY_high.x();
  int htfluxY_low_y = htfluxY_low.y();
  int htfluxY_high_y = htfluxY_high.y();
  int htfluxY_low_z = htfluxY_low.z();
  int htfluxY_high_z = htfluxY_high.z();
  Uintah::IntVector htflux_convZ_low = htflux_convZ.getWindow()->getOffset();
  Uintah::IntVector htflux_convZ_high = htflux_convZ.getWindow()->getData()->size() + htflux_convZ_low - Uintah::IntVector(1, 1, 1);
  int htflux_convZ_low_x = htflux_convZ_low.x();
  int htflux_convZ_high_x = htflux_convZ_high.x();
  int htflux_convZ_low_y = htflux_convZ_low.y();
  int htflux_convZ_high_y = htflux_convZ_high.y();
  int htflux_convZ_low_z = htflux_convZ_low.z();
  int htflux_convZ_high_z = htflux_convZ_high.z();
  Uintah::IntVector htflux_radZ_low = htflux_radZ.getWindow()->getOffset();
  Uintah::IntVector htflux_radZ_high = htflux_radZ.getWindow()->getData()->size() + htflux_radZ_low - Uintah::IntVector(1, 1, 1);
  int htflux_radZ_low_x = htflux_radZ_low.x();
  int htflux_radZ_high_x = htflux_radZ_high.x();
  int htflux_radZ_low_y = htflux_radZ_low.y();
  int htflux_radZ_high_y = htflux_radZ_high.y();
  int htflux_radZ_low_z = htflux_radZ_low.z();
  int htflux_radZ_high_z = htflux_radZ_high.z();
  Uintah::IntVector htfluxZ_low = htfluxZ.getWindow()->getOffset();
  Uintah::IntVector htfluxZ_high = htfluxZ.getWindow()->getData()->size() + htfluxZ_low - Uintah::IntVector(1, 1, 1);
  int htfluxZ_low_x = htfluxZ_low.x();
  int htfluxZ_high_x = htfluxZ_high.x();
  int htfluxZ_low_y = htfluxZ_low.y();
  int htfluxZ_high_y = htfluxZ_high.y();
  int htfluxZ_low_z = htfluxZ_low.z();
  int htfluxZ_high_z = htfluxZ_high.z();
  Uintah::IntVector htflux_convCC_low = htflux_convCC.getWindow()->getOffset();
  Uintah::IntVector htflux_convCC_high = htflux_convCC.getWindow()->getData()->size() + htflux_convCC_low - Uintah::IntVector(1, 1, 1);
  int htflux_convCC_low_x = htflux_convCC_low.x();
  int htflux_convCC_high_x = htflux_convCC_high.x();
  int htflux_convCC_low_y = htflux_convCC_low.y();
  int htflux_convCC_high_y = htflux_convCC_high.y();
  int htflux_convCC_low_z = htflux_convCC_low.z();
  int htflux_convCC_high_z = htflux_convCC_high.z();
  Uintah::IntVector sug_cc_low = sug_cc.getWindow()->getOffset();
  Uintah::IntVector sug_cc_high = sug_cc.getWindow()->getData()->size() + sug_cc_low - Uintah::IntVector(1, 1, 1);
  int sug_cc_low_x = sug_cc_low.x();
  int sug_cc_high_x = sug_cc_high.x();
  int sug_cc_low_y = sug_cc_low.y();
  int sug_cc_high_y = sug_cc_high.y();
  int sug_cc_low_z = sug_cc_low.z();
  int sug_cc_high_z = sug_cc_high.z();
  Uintah::IntVector spg_cc_low = spg_cc.getWindow()->getOffset();
  Uintah::IntVector spg_cc_high = spg_cc.getWindow()->getData()->size() + spg_cc_low - Uintah::IntVector(1, 1, 1);
  int spg_cc_low_x = spg_cc_low.x();
  int spg_cc_high_x = spg_cc_high.x();
  int spg_cc_low_y = spg_cc_low.y();
  int spg_cc_high_y = spg_cc_high.y();
  int spg_cc_low_z = spg_cc_low.z();
  int spg_cc_high_z = spg_cc_high.z();
  Uintah::IntVector sug_fcx_low = sug_fcx.getWindow()->getOffset();
  Uintah::IntVector sug_fcx_high = sug_fcx.getWindow()->getData()->size() + sug_fcx_low - Uintah::IntVector(1, 1, 1);
  int sug_fcx_low_x = sug_fcx_low.x();
  int sug_fcx_high_x = sug_fcx_high.x();
  int sug_fcx_low_y = sug_fcx_low.y();
  int sug_fcx_high_y = sug_fcx_high.y();
  int sug_fcx_low_z = sug_fcx_low.z();
  int sug_fcx_high_z = sug_fcx_high.z();
  Uintah::IntVector spg_fcx_low = spg_fcx.getWindow()->getOffset();
  Uintah::IntVector spg_fcx_high = spg_fcx.getWindow()->getData()->size() + spg_fcx_low - Uintah::IntVector(1, 1, 1);
  int spg_fcx_low_x = spg_fcx_low.x();
  int spg_fcx_high_x = spg_fcx_high.x();
  int spg_fcx_low_y = spg_fcx_low.y();
  int spg_fcx_high_y = spg_fcx_high.y();
  int spg_fcx_low_z = spg_fcx_low.z();
  int spg_fcx_high_z = spg_fcx_high.z();
  Uintah::IntVector sug_fcy_low = sug_fcy.getWindow()->getOffset();
  Uintah::IntVector sug_fcy_high = sug_fcy.getWindow()->getData()->size() + sug_fcy_low - Uintah::IntVector(1, 1, 1);
  int sug_fcy_low_x = sug_fcy_low.x();
  int sug_fcy_high_x = sug_fcy_high.x();
  int sug_fcy_low_y = sug_fcy_low.y();
  int sug_fcy_high_y = sug_fcy_high.y();
  int sug_fcy_low_z = sug_fcy_low.z();
  int sug_fcy_high_z = sug_fcy_high.z();
  Uintah::IntVector spg_fcy_low = spg_fcy.getWindow()->getOffset();
  Uintah::IntVector spg_fcy_high = spg_fcy.getWindow()->getData()->size() + spg_fcy_low - Uintah::IntVector(1, 1, 1);
  int spg_fcy_low_x = spg_fcy_low.x();
  int spg_fcy_high_x = spg_fcy_high.x();
  int spg_fcy_low_y = spg_fcy_low.y();
  int spg_fcy_high_y = spg_fcy_high.y();
  int spg_fcy_low_z = spg_fcy_low.z();
  int spg_fcy_high_z = spg_fcy_high.z();
  Uintah::IntVector sug_fcz_low = sug_fcz.getWindow()->getOffset();
  Uintah::IntVector sug_fcz_high = sug_fcz.getWindow()->getData()->size() + sug_fcz_low - Uintah::IntVector(1, 1, 1);
  int sug_fcz_low_x = sug_fcz_low.x();
  int sug_fcz_high_x = sug_fcz_high.x();
  int sug_fcz_low_y = sug_fcz_low.y();
  int sug_fcz_high_y = sug_fcz_high.y();
  int sug_fcz_low_z = sug_fcz_low.z();
  int sug_fcz_high_z = sug_fcz_high.z();
  Uintah::IntVector spg_fcz_low = spg_fcz.getWindow()->getOffset();
  Uintah::IntVector spg_fcz_high = spg_fcz.getWindow()->getData()->size() + spg_fcz_low - Uintah::IntVector(1, 1, 1);
  int spg_fcz_low_x = spg_fcz_low.x();
  int spg_fcz_high_x = spg_fcz_high.x();
  int spg_fcz_low_y = spg_fcz_low.y();
  int spg_fcz_high_y = spg_fcz_high.y();
  int spg_fcz_low_z = spg_fcz_low.z();
  int spg_fcz_high_z = spg_fcz_high.z();
  Uintah::IntVector kstabh_low = kstabh.getWindow()->getOffset();
  Uintah::IntVector kstabh_high = kstabh.getWindow()->getData()->size() + kstabh_low - Uintah::IntVector(1, 1, 1);
  int kstabh_low_x = kstabh_low.x();
  int kstabh_high_x = kstabh_high.x();
  int kstabh_low_y = kstabh_low.y();
  int kstabh_high_y = kstabh_high.y();
  int kstabh_low_z = kstabh_low.z();
  int kstabh_high_z = kstabh_high.z();
  Uintah::IntVector tg_low = tg.getWindow()->getOffset();
  Uintah::IntVector tg_high = tg.getWindow()->getData()->size() + tg_low - Uintah::IntVector(1, 1, 1);
  int tg_low_x = tg_low.x();
  int tg_high_x = tg_high.x();
  int tg_low_y = tg_low.y();
  int tg_high_y = tg_high.y();
  int tg_low_z = tg_low.z();
  int tg_high_z = tg_high.z();
  Uintah::IntVector ts_cc_low = ts_cc.getWindow()->getOffset();
  Uintah::IntVector ts_cc_high = ts_cc.getWindow()->getData()->size() + ts_cc_low - Uintah::IntVector(1, 1, 1);
  int ts_cc_low_x = ts_cc_low.x();
  int ts_cc_high_x = ts_cc_high.x();
  int ts_cc_low_y = ts_cc_low.y();
  int ts_cc_high_y = ts_cc_high.y();
  int ts_cc_low_z = ts_cc_low.z();
  int ts_cc_high_z = ts_cc_high.z();
  Uintah::IntVector ts_fcx_low = ts_fcx.getWindow()->getOffset();
  Uintah::IntVector ts_fcx_high = ts_fcx.getWindow()->getData()->size() + ts_fcx_low - Uintah::IntVector(1, 1, 1);
  int ts_fcx_low_x = ts_fcx_low.x();
  int ts_fcx_high_x = ts_fcx_high.x();
  int ts_fcx_low_y = ts_fcx_low.y();
  int ts_fcx_high_y = ts_fcx_high.y();
  int ts_fcx_low_z = ts_fcx_low.z();
  int ts_fcx_high_z = ts_fcx_high.z();
  Uintah::IntVector ts_fcy_low = ts_fcy.getWindow()->getOffset();
  Uintah::IntVector ts_fcy_high = ts_fcy.getWindow()->getData()->size() + ts_fcy_low - Uintah::IntVector(1, 1, 1);
  int ts_fcy_low_x = ts_fcy_low.x();
  int ts_fcy_high_x = ts_fcy_high.x();
  int ts_fcy_low_y = ts_fcy_low.y();
  int ts_fcy_high_y = ts_fcy_high.y();
  int ts_fcy_low_z = ts_fcy_low.z();
  int ts_fcy_high_z = ts_fcy_high.z();
  Uintah::IntVector ts_fcz_low = ts_fcz.getWindow()->getOffset();
  Uintah::IntVector ts_fcz_high = ts_fcz.getWindow()->getData()->size() + ts_fcz_low - Uintah::IntVector(1, 1, 1);
  int ts_fcz_low_x = ts_fcz_low.x();
  int ts_fcz_high_x = ts_fcz_high.x();
  int ts_fcz_low_y = ts_fcz_low.y();
  int ts_fcz_high_y = ts_fcz_high.y();
  int ts_fcz_low_z = ts_fcz_low.z();
  int ts_fcz_high_z = ts_fcz_high.z();
  Uintah::IntVector ug_cc_low = ug_cc.getWindow()->getOffset();
  Uintah::IntVector ug_cc_high = ug_cc.getWindow()->getData()->size() + ug_cc_low - Uintah::IntVector(1, 1, 1);
  int ug_cc_low_x = ug_cc_low.x();
  int ug_cc_high_x = ug_cc_high.x();
  int ug_cc_low_y = ug_cc_low.y();
  int ug_cc_high_y = ug_cc_high.y();
  int ug_cc_low_z = ug_cc_low.z();
  int ug_cc_high_z = ug_cc_high.z();
  Uintah::IntVector vg_cc_low = vg_cc.getWindow()->getOffset();
  Uintah::IntVector vg_cc_high = vg_cc.getWindow()->getData()->size() + vg_cc_low - Uintah::IntVector(1, 1, 1);
  int vg_cc_low_x = vg_cc_low.x();
  int vg_cc_high_x = vg_cc_high.x();
  int vg_cc_low_y = vg_cc_low.y();
  int vg_cc_high_y = vg_cc_high.y();
  int vg_cc_low_z = vg_cc_low.z();
  int vg_cc_high_z = vg_cc_high.z();
  Uintah::IntVector wg_cc_low = wg_cc.getWindow()->getOffset();
  Uintah::IntVector wg_cc_high = wg_cc.getWindow()->getData()->size() + wg_cc_low - Uintah::IntVector(1, 1, 1);
  int wg_cc_low_x = wg_cc_low.x();
  int wg_cc_high_x = wg_cc_high.x();
  int wg_cc_low_y = wg_cc_low.y();
  int wg_cc_high_y = wg_cc_high.y();
  int wg_cc_low_z = wg_cc_low.z();
  int wg_cc_high_z = wg_cc_high.z();
  Uintah::IntVector up_cc_low = up_cc.getWindow()->getOffset();
  Uintah::IntVector up_cc_high = up_cc.getWindow()->getData()->size() + up_cc_low - Uintah::IntVector(1, 1, 1);
  int up_cc_low_x = up_cc_low.x();
  int up_cc_high_x = up_cc_high.x();
  int up_cc_low_y = up_cc_low.y();
  int up_cc_high_y = up_cc_high.y();
  int up_cc_low_z = up_cc_low.z();
  int up_cc_high_z = up_cc_high.z();
  Uintah::IntVector vp_cc_low = vp_cc.getWindow()->getOffset();
  Uintah::IntVector vp_cc_high = vp_cc.getWindow()->getData()->size() + vp_cc_low - Uintah::IntVector(1, 1, 1);
  int vp_cc_low_x = vp_cc_low.x();
  int vp_cc_high_x = vp_cc_high.x();
  int vp_cc_low_y = vp_cc_low.y();
  int vp_cc_high_y = vp_cc_high.y();
  int vp_cc_low_z = vp_cc_low.z();
  int vp_cc_high_z = vp_cc_high.z();
  Uintah::IntVector wp_cc_low = wp_cc.getWindow()->getOffset();
  Uintah::IntVector wp_cc_high = wp_cc.getWindow()->getData()->size() + wp_cc_low - Uintah::IntVector(1, 1, 1);
  int wp_cc_low_x = wp_cc_low.x();
  int wp_cc_high_x = wp_cc_high.x();
  int wp_cc_low_y = wp_cc_low.y();
  int wp_cc_high_y = wp_cc_high.y();
  int wp_cc_low_z = wp_cc_low.z();
  int wp_cc_high_z = wp_cc_high.z();
  Uintah::IntVector vp_fcx_low = vp_fcx.getWindow()->getOffset();
  Uintah::IntVector vp_fcx_high = vp_fcx.getWindow()->getData()->size() + vp_fcx_low - Uintah::IntVector(1, 1, 1);
  int vp_fcx_low_x = vp_fcx_low.x();
  int vp_fcx_high_x = vp_fcx_high.x();
  int vp_fcx_low_y = vp_fcx_low.y();
  int vp_fcx_high_y = vp_fcx_high.y();
  int vp_fcx_low_z = vp_fcx_low.z();
  int vp_fcx_high_z = vp_fcx_high.z();
  Uintah::IntVector wp_fcx_low = wp_fcx.getWindow()->getOffset();
  Uintah::IntVector wp_fcx_high = wp_fcx.getWindow()->getData()->size() + wp_fcx_low - Uintah::IntVector(1, 1, 1);
  int wp_fcx_low_x = wp_fcx_low.x();
  int wp_fcx_high_x = wp_fcx_high.x();
  int wp_fcx_low_y = wp_fcx_low.y();
  int wp_fcx_high_y = wp_fcx_high.y();
  int wp_fcx_low_z = wp_fcx_low.z();
  int wp_fcx_high_z = wp_fcx_high.z();
  Uintah::IntVector up_fcy_low = up_fcy.getWindow()->getOffset();
  Uintah::IntVector up_fcy_high = up_fcy.getWindow()->getData()->size() + up_fcy_low - Uintah::IntVector(1, 1, 1);
  int up_fcy_low_x = up_fcy_low.x();
  int up_fcy_high_x = up_fcy_high.x();
  int up_fcy_low_y = up_fcy_low.y();
  int up_fcy_high_y = up_fcy_high.y();
  int up_fcy_low_z = up_fcy_low.z();
  int up_fcy_high_z = up_fcy_high.z();
  Uintah::IntVector wp_fcy_low = wp_fcy.getWindow()->getOffset();
  Uintah::IntVector wp_fcy_high = wp_fcy.getWindow()->getData()->size() + wp_fcy_low - Uintah::IntVector(1, 1, 1);
  int wp_fcy_low_x = wp_fcy_low.x();
  int wp_fcy_high_x = wp_fcy_high.x();
  int wp_fcy_low_y = wp_fcy_low.y();
  int wp_fcy_high_y = wp_fcy_high.y();
  int wp_fcy_low_z = wp_fcy_low.z();
  int wp_fcy_high_z = wp_fcy_high.z();
  Uintah::IntVector up_fcz_low = up_fcz.getWindow()->getOffset();
  Uintah::IntVector up_fcz_high = up_fcz.getWindow()->getData()->size() + up_fcz_low - Uintah::IntVector(1, 1, 1);
  int up_fcz_low_x = up_fcz_low.x();
  int up_fcz_high_x = up_fcz_high.x();
  int up_fcz_low_y = up_fcz_low.y();
  int up_fcz_high_y = up_fcz_high.y();
  int up_fcz_low_z = up_fcz_low.z();
  int up_fcz_high_z = up_fcz_high.z();
  Uintah::IntVector vp_fcz_low = vp_fcz.getWindow()->getOffset();
  Uintah::IntVector vp_fcz_high = vp_fcz.getWindow()->getData()->size() + vp_fcz_low - Uintah::IntVector(1, 1, 1);
  int vp_fcz_low_x = vp_fcz_low.x();
  int vp_fcz_high_x = vp_fcz_high.x();
  int vp_fcz_low_y = vp_fcz_low.y();
  int vp_fcz_high_y = vp_fcz_high.y();
  int vp_fcz_low_z = vp_fcz_low.z();
  int vp_fcz_high_z = vp_fcz_high.z();
  Uintah::IntVector denMicro_low = denMicro.getWindow()->getOffset();
  Uintah::IntVector denMicro_high = denMicro.getWindow()->getData()->size() + denMicro_low - Uintah::IntVector(1, 1, 1);
  int denMicro_low_x = denMicro_low.x();
  int denMicro_high_x = denMicro_high.x();
  int denMicro_low_y = denMicro_low.y();
  int denMicro_high_y = denMicro_high.y();
  int denMicro_low_z = denMicro_low.z();
  int denMicro_high_z = denMicro_high.z();
  Uintah::IntVector enth_low = enth.getWindow()->getOffset();
  Uintah::IntVector enth_high = enth.getWindow()->getData()->size() + enth_low - Uintah::IntVector(1, 1, 1);
  int enth_low_x = enth_low.x();
  int enth_high_x = enth_high.x();
  int enth_low_y = enth_low.y();
  int enth_high_y = enth_high.y();
  int enth_low_z = enth_low.z();
  int enth_high_z = enth_high.z();
  Uintah::IntVector rfluxE_low = rfluxE.getWindow()->getOffset();
  Uintah::IntVector rfluxE_high = rfluxE.getWindow()->getData()->size() + rfluxE_low - Uintah::IntVector(1, 1, 1);
  int rfluxE_low_x = rfluxE_low.x();
  int rfluxE_high_x = rfluxE_high.x();
  int rfluxE_low_y = rfluxE_low.y();
  int rfluxE_high_y = rfluxE_high.y();
  int rfluxE_low_z = rfluxE_low.z();
  int rfluxE_high_z = rfluxE_high.z();
  Uintah::IntVector rfluxW_low = rfluxW.getWindow()->getOffset();
  Uintah::IntVector rfluxW_high = rfluxW.getWindow()->getData()->size() + rfluxW_low - Uintah::IntVector(1, 1, 1);
  int rfluxW_low_x = rfluxW_low.x();
  int rfluxW_high_x = rfluxW_high.x();
  int rfluxW_low_y = rfluxW_low.y();
  int rfluxW_high_y = rfluxW_high.y();
  int rfluxW_low_z = rfluxW_low.z();
  int rfluxW_high_z = rfluxW_high.z();
  Uintah::IntVector rfluxN_low = rfluxN.getWindow()->getOffset();
  Uintah::IntVector rfluxN_high = rfluxN.getWindow()->getData()->size() + rfluxN_low - Uintah::IntVector(1, 1, 1);
  int rfluxN_low_x = rfluxN_low.x();
  int rfluxN_high_x = rfluxN_high.x();
  int rfluxN_low_y = rfluxN_low.y();
  int rfluxN_high_y = rfluxN_high.y();
  int rfluxN_low_z = rfluxN_low.z();
  int rfluxN_high_z = rfluxN_high.z();
  Uintah::IntVector rfluxS_low = rfluxS.getWindow()->getOffset();
  Uintah::IntVector rfluxS_high = rfluxS.getWindow()->getData()->size() + rfluxS_low - Uintah::IntVector(1, 1, 1);
  int rfluxS_low_x = rfluxS_low.x();
  int rfluxS_high_x = rfluxS_high.x();
  int rfluxS_low_y = rfluxS_low.y();
  int rfluxS_high_y = rfluxS_high.y();
  int rfluxS_low_z = rfluxS_low.z();
  int rfluxS_high_z = rfluxS_high.z();
  Uintah::IntVector rfluxT_low = rfluxT.getWindow()->getOffset();
  Uintah::IntVector rfluxT_high = rfluxT.getWindow()->getData()->size() + rfluxT_low - Uintah::IntVector(1, 1, 1);
  int rfluxT_low_x = rfluxT_low.x();
  int rfluxT_high_x = rfluxT_high.x();
  int rfluxT_low_y = rfluxT_low.y();
  int rfluxT_high_y = rfluxT_high.y();
  int rfluxT_low_z = rfluxT_low.z();
  int rfluxT_high_z = rfluxT_high.z();
  Uintah::IntVector rfluxB_low = rfluxB.getWindow()->getOffset();
  Uintah::IntVector rfluxB_high = rfluxB.getWindow()->getData()->size() + rfluxB_low - Uintah::IntVector(1, 1, 1);
  int rfluxB_low_x = rfluxB_low.x();
  int rfluxB_high_x = rfluxB_high.x();
  int rfluxB_low_y = rfluxB_low.y();
  int rfluxB_high_y = rfluxB_high.y();
  int rfluxB_low_z = rfluxB_low.z();
  int rfluxB_high_z = rfluxB_high.z();
  Uintah::IntVector epsg_low = epsg.getWindow()->getOffset();
  Uintah::IntVector epsg_high = epsg.getWindow()->getData()->size() + epsg_low - Uintah::IntVector(1, 1, 1);
  int epsg_low_x = epsg_low.x();
  int epsg_high_x = epsg_high.x();
  int epsg_low_y = epsg_low.y();
  int epsg_high_y = epsg_high.y();
  int epsg_low_z = epsg_low.z();
  int epsg_high_z = epsg_high.z();
  Uintah::IntVector epss_low = epss.getWindow()->getOffset();
  Uintah::IntVector epss_high = epss.getWindow()->getData()->size() + epss_low - Uintah::IntVector(1, 1, 1);
  int epss_low_x = epss_low.x();
  int epss_high_x = epss_high.x();
  int epss_low_y = epss_low.y();
  int epss_high_y = epss_high.y();
  int epss_low_z = epss_low.z();
  int epss_high_z = epss_high.z();
  Uintah::IntVector pcell_low = pcell.getWindow()->getOffset();
  Uintah::IntVector pcell_high = pcell.getWindow()->getData()->size() + pcell_low - Uintah::IntVector(1, 1, 1);
  int pcell_low_x = pcell_low.x();
  int pcell_high_x = pcell_high.x();
  int pcell_low_y = pcell_low.y();
  int pcell_high_y = pcell_high.y();
  int pcell_low_z = pcell_low.z();
  int pcell_high_z = pcell_high.z();
  energy_exchange_term_( &hts_fcx_low_x, &hts_fcx_low_y, &hts_fcx_low_z, &hts_fcx_high_x, &hts_fcx_high_y, &hts_fcx_high_z, hts_fcx.getPointer(),
                         &hts_fcy_low_x, &hts_fcy_low_y, &hts_fcy_low_z, &hts_fcy_high_x, &hts_fcy_high_y, &hts_fcy_high_z, hts_fcy.getPointer(),
                         &hts_fcz_low_x, &hts_fcz_low_y, &hts_fcz_low_z, &hts_fcz_high_x, &hts_fcz_high_y, &hts_fcz_high_z, hts_fcz.getPointer(),
                         &hts_fcx_rad_low_x, &hts_fcx_rad_low_y, &hts_fcx_rad_low_z, &hts_fcx_rad_high_x, &hts_fcx_rad_high_y, &hts_fcx_rad_high_z, hts_fcx_rad.getPointer(),
                         &hts_fcy_rad_low_x, &hts_fcy_rad_low_y, &hts_fcy_rad_low_z, &hts_fcy_rad_high_x, &hts_fcy_rad_high_y, &hts_fcy_rad_high_z, hts_fcy_rad.getPointer(),
                         &hts_fcz_rad_low_x, &hts_fcz_rad_low_y, &hts_fcz_rad_low_z, &hts_fcz_rad_high_x, &hts_fcz_rad_high_y, &hts_fcz_rad_high_z, hts_fcz_rad.getPointer(),
                         &hts_cc_low_x, &hts_cc_low_y, &hts_cc_low_z, &hts_cc_high_x, &hts_cc_high_y, &hts_cc_high_z, hts_cc.getPointer(),
                         &htflux_convX_low_x, &htflux_convX_low_y, &htflux_convX_low_z, &htflux_convX_high_x, &htflux_convX_high_y, &htflux_convX_high_z, htflux_convX.getPointer(),
                         &htflux_radX_low_x, &htflux_radX_low_y, &htflux_radX_low_z, &htflux_radX_high_x, &htflux_radX_high_y, &htflux_radX_high_z, htflux_radX.getPointer(),
                         &htfluxX_low_x, &htfluxX_low_y, &htfluxX_low_z, &htfluxX_high_x, &htfluxX_high_y, &htfluxX_high_z, htfluxX.getPointer(),
                         &htflux_convY_low_x, &htflux_convY_low_y, &htflux_convY_low_z, &htflux_convY_high_x, &htflux_convY_high_y, &htflux_convY_high_z, htflux_convY.getPointer(),
                         &htflux_radY_low_x, &htflux_radY_low_y, &htflux_radY_low_z, &htflux_radY_high_x, &htflux_radY_high_y, &htflux_radY_high_z, htflux_radY.getPointer(),
                         &htfluxY_low_x, &htfluxY_low_y, &htfluxY_low_z, &htfluxY_high_x, &htfluxY_high_y, &htfluxY_high_z, htfluxY.getPointer(),
                         &htflux_convZ_low_x, &htflux_convZ_low_y, &htflux_convZ_low_z, &htflux_convZ_high_x, &htflux_convZ_high_y, &htflux_convZ_high_z, htflux_convZ.getPointer(),
                         &htflux_radZ_low_x, &htflux_radZ_low_y, &htflux_radZ_low_z, &htflux_radZ_high_x, &htflux_radZ_high_y, &htflux_radZ_high_z, htflux_radZ.getPointer(),
                         &htfluxZ_low_x, &htfluxZ_low_y, &htfluxZ_low_z, &htfluxZ_high_x, &htfluxZ_high_y, &htfluxZ_high_z, htfluxZ.getPointer(),
                         &htflux_convCC_low_x, &htflux_convCC_low_y, &htflux_convCC_low_z, &htflux_convCC_high_x, &htflux_convCC_high_y, &htflux_convCC_high_z, htflux_convCC.getPointer(),
                         &sug_cc_low_x, &sug_cc_low_y, &sug_cc_low_z, &sug_cc_high_x, &sug_cc_high_y, &sug_cc_high_z, sug_cc.getPointer(),
                         &spg_cc_low_x, &spg_cc_low_y, &spg_cc_low_z, &spg_cc_high_x, &spg_cc_high_y, &spg_cc_high_z, spg_cc.getPointer(),
                         &sug_fcx_low_x, &sug_fcx_low_y, &sug_fcx_low_z, &sug_fcx_high_x, &sug_fcx_high_y, &sug_fcx_high_z, sug_fcx.getPointer(),
                         &spg_fcx_low_x, &spg_fcx_low_y, &spg_fcx_low_z, &spg_fcx_high_x, &spg_fcx_high_y, &spg_fcx_high_z, spg_fcx.getPointer(),
                         &sug_fcy_low_x, &sug_fcy_low_y, &sug_fcy_low_z, &sug_fcy_high_x, &sug_fcy_high_y, &sug_fcy_high_z, sug_fcy.getPointer(),
                         &spg_fcy_low_x, &spg_fcy_low_y, &spg_fcy_low_z, &spg_fcy_high_x, &spg_fcy_high_y, &spg_fcy_high_z, spg_fcy.getPointer(),
                         &sug_fcz_low_x, &sug_fcz_low_y, &sug_fcz_low_z, &sug_fcz_high_x, &sug_fcz_high_y, &sug_fcz_high_z, sug_fcz.getPointer(),
                         &spg_fcz_low_x, &spg_fcz_low_y, &spg_fcz_low_z, &spg_fcz_high_x, &spg_fcz_high_y, &spg_fcz_high_z, spg_fcz.getPointer(),
                         &kstabh_low_x, &kstabh_low_y, &kstabh_low_z, &kstabh_high_x, &kstabh_high_y, &kstabh_high_z, kstabh.getPointer(),
                         &tg_low_x, &tg_low_y, &tg_low_z, &tg_high_x, &tg_high_y, &tg_high_z, const_cast<double*>(tg.getPointer()),
                         &ts_cc_low_x, &ts_cc_low_y, &ts_cc_low_z, &ts_cc_high_x, &ts_cc_high_y, &ts_cc_high_z, const_cast<double*>(ts_cc.getPointer()),
                         &ts_fcx_low_x, &ts_fcx_low_y, &ts_fcx_low_z, &ts_fcx_high_x, &ts_fcx_high_y, &ts_fcx_high_z, const_cast<double*>(ts_fcx.getPointer()),
                         &ts_fcy_low_x, &ts_fcy_low_y, &ts_fcy_low_z, &ts_fcy_high_x, &ts_fcy_high_y, &ts_fcy_high_z, const_cast<double*>(ts_fcy.getPointer()),
                         &ts_fcz_low_x, &ts_fcz_low_y, &ts_fcz_low_z, &ts_fcz_high_x, &ts_fcz_high_y, &ts_fcz_high_z, const_cast<double*>(ts_fcz.getPointer()),
                         &ug_cc_low_x, &ug_cc_low_y, &ug_cc_low_z, &ug_cc_high_x, &ug_cc_high_y, &ug_cc_high_z, ug_cc.getPointer(),
                         &vg_cc_low_x, &vg_cc_low_y, &vg_cc_low_z, &vg_cc_high_x, &vg_cc_high_y, &vg_cc_high_z, vg_cc.getPointer(),
                         &wg_cc_low_x, &wg_cc_low_y, &wg_cc_low_z, &wg_cc_high_x, &wg_cc_high_y, &wg_cc_high_z, wg_cc.getPointer(),
                         &up_cc_low_x, &up_cc_low_y, &up_cc_low_z, &up_cc_high_x, &up_cc_high_y, &up_cc_high_z, const_cast<double*>(up_cc.getPointer()),
                         &vp_cc_low_x, &vp_cc_low_y, &vp_cc_low_z, &vp_cc_high_x, &vp_cc_high_y, &vp_cc_high_z, const_cast<double*>(vp_cc.getPointer()),
                         &wp_cc_low_x, &wp_cc_low_y, &wp_cc_low_z, &wp_cc_high_x, &wp_cc_high_y, &wp_cc_high_z, const_cast<double*>(wp_cc.getPointer()),
                         &vp_fcx_low_x, &vp_fcx_low_y, &vp_fcx_low_z, &vp_fcx_high_x, &vp_fcx_high_y, &vp_fcx_high_z, const_cast<double*>(vp_fcx.getPointer()),
                         &wp_fcx_low_x, &wp_fcx_low_y, &wp_fcx_low_z, &wp_fcx_high_x, &wp_fcx_high_y, &wp_fcx_high_z, const_cast<double*>(wp_fcx.getPointer()),
                         &up_fcy_low_x, &up_fcy_low_y, &up_fcy_low_z, &up_fcy_high_x, &up_fcy_high_y, &up_fcy_high_z, const_cast<double*>(up_fcy.getPointer()),
                         &wp_fcy_low_x, &wp_fcy_low_y, &wp_fcy_low_z, &wp_fcy_high_x, &wp_fcy_high_y, &wp_fcy_high_z, const_cast<double*>(wp_fcy.getPointer()),
                         &up_fcz_low_x, &up_fcz_low_y, &up_fcz_low_z, &up_fcz_high_x, &up_fcz_high_y, &up_fcz_high_z, const_cast<double*>(up_fcz.getPointer()),
                         &vp_fcz_low_x, &vp_fcz_low_y, &vp_fcz_low_z, &vp_fcz_high_x, &vp_fcz_high_y, &vp_fcz_high_z, const_cast<double*>(vp_fcz.getPointer()),
                         &denMicro_low_x, &denMicro_low_y, &denMicro_low_z, &denMicro_high_x, &denMicro_high_y, &denMicro_high_z, const_cast<double*>(denMicro.getPointer()),
                         &enth_low_x, &enth_low_y, &enth_low_z, &enth_high_x, &enth_high_y, &enth_high_z, const_cast<double*>(enth.getPointer()),
                         &rfluxE_low_x, &rfluxE_low_y, &rfluxE_low_z, &rfluxE_high_x, &rfluxE_high_y, &rfluxE_high_z, rfluxE.getPointer(),
                         &rfluxW_low_x, &rfluxW_low_y, &rfluxW_low_z, &rfluxW_high_x, &rfluxW_high_y, &rfluxW_high_z, rfluxW.getPointer(),
                         &rfluxN_low_x, &rfluxN_low_y, &rfluxN_low_z, &rfluxN_high_x, &rfluxN_high_y, &rfluxN_high_z, rfluxN.getPointer(),
                         &rfluxS_low_x, &rfluxS_low_y, &rfluxS_low_z, &rfluxS_high_x, &rfluxS_high_y, &rfluxS_high_z, rfluxS.getPointer(),
                         &rfluxT_low_x, &rfluxT_low_y, &rfluxT_low_z, &rfluxT_high_x, &rfluxT_high_y, &rfluxT_high_z, rfluxT.getPointer(),
                         &rfluxB_low_x, &rfluxB_low_y, &rfluxB_low_z, &rfluxB_high_x, &rfluxB_high_y, &rfluxB_high_z, rfluxB.getPointer(),
                         &epsg_low_x, &epsg_low_y, &epsg_low_z, &epsg_high_x, &epsg_high_y, &epsg_high_z, const_cast<double*>(epsg.getPointer()),
                         &epss_low_x, &epss_low_y, &epss_low_z, &epss_high_x, &epss_high_y, &epss_high_z, const_cast<double*>(epss.getPointer()),
                         &dx,
                         &dy,
                         &dz,
                         &tcond,
                         &csmag,
                         &prturb,
                         &cpfluid,
                         valid_lo.get_pointer(),
                         valid_hi.get_pointer(),
                         &pcell_low_x, &pcell_low_y, &pcell_low_z, &pcell_high_x, &pcell_high_y, &pcell_high_z, const_cast<int*>(pcell.getPointer()),
                         &wall,
                         &ffield );
}

#else /* !__cplusplus */

C This is the FORTRAN code portion of the file:

      subroutine energy_exchange_term(hts_fcx_low_x, hts_fcx_low_y, 
     & hts_fcx_low_z, hts_fcx_high_x, hts_fcx_high_y, hts_fcx_high_z, 
     & hts_fcx, hts_fcy_low_x, hts_fcy_low_y, hts_fcy_low_z, 
     & hts_fcy_high_x, hts_fcy_high_y, hts_fcy_high_z, hts_fcy, 
     & hts_fcz_low_x, hts_fcz_low_y, hts_fcz_low_z, hts_fcz_high_x, 
     & hts_fcz_high_y, hts_fcz_high_z, hts_fcz, hts_fcx_rad_low_x, 
     & hts_fcx_rad_low_y, hts_fcx_rad_low_z, hts_fcx_rad_high_x, 
     & hts_fcx_rad_high_y, hts_fcx_rad_high_z, hts_fcx_rad, 
     & hts_fcy_rad_low_x, hts_fcy_rad_low_y, hts_fcy_rad_low_z, 
     & hts_fcy_rad_high_x, hts_fcy_rad_high_y, hts_fcy_rad_high_z, 
     & hts_fcy_rad, hts_fcz_rad_low_x, hts_fcz_rad_low_y, 
     & hts_fcz_rad_low_z, hts_fcz_rad_high_x, hts_fcz_rad_high_y, 
     & hts_fcz_rad_high_z, hts_fcz_rad, hts_cc_low_x, hts_cc_low_y, 
     & hts_cc_low_z, hts_cc_high_x, hts_cc_high_y, hts_cc_high_z, 
     & hts_cc, htflux_convX_low_x, htflux_convX_low_y, 
     & htflux_convX_low_z, htflux_convX_high_x, htflux_convX_high_y, 
     & htflux_convX_high_z, htflux_convX, htflux_radX_low_x, 
     & htflux_radX_low_y, htflux_radX_low_z, htflux_radX_high_x, 
     & htflux_radX_high_y, htflux_radX_high_z, htflux_radX, 
     & htfluxX_low_x, htfluxX_low_y, htfluxX_low_z, htfluxX_high_x, 
     & htfluxX_high_y, htfluxX_high_z, htfluxX, htflux_convY_low_x, 
     & htflux_convY_low_y, htflux_convY_low_z, htflux_convY_high_x, 
     & htflux_convY_high_y, htflux_convY_high_z, htflux_convY, 
     & htflux_radY_low_x, htflux_radY_low_y, htflux_radY_low_z, 
     & htflux_radY_high_x, htflux_radY_high_y, htflux_radY_high_z, 
     & htflux_radY, htfluxY_low_x, htfluxY_low_y, htfluxY_low_z, 
     & htfluxY_high_x, htfluxY_high_y, htfluxY_high_z, htfluxY, 
     & htflux_convZ_low_x, htflux_convZ_low_y, htflux_convZ_low_z, 
     & htflux_convZ_high_x, htflux_convZ_high_y, htflux_convZ_high_z, 
     & htflux_convZ, htflux_radZ_low_x, htflux_radZ_low_y, 
     & htflux_radZ_low_z, htflux_radZ_high_x, htflux_radZ_high_y, 
     & htflux_radZ_high_z, htflux_radZ, htfluxZ_low_x, htfluxZ_low_y, 
     & htfluxZ_low_z, htfluxZ_high_x, htfluxZ_high_y, htfluxZ_high_z, 
     & htfluxZ, htflux_convCC_low_x, htflux_convCC_low_y, 
     & htflux_convCC_low_z, htflux_convCC_high_x, htflux_convCC_high_y,
     &  htflux_convCC_high_z, htflux_convCC, sug_cc_low_x, sug_cc_low_y
     & , sug_cc_low_z, sug_cc_high_x, sug_cc_high_y, sug_cc_high_z, 
     & sug_cc, spg_cc_low_x, spg_cc_low_y, spg_cc_low_z, spg_cc_high_x,
     &  spg_cc_high_y, spg_cc_high_z, spg_cc, sug_fcx_low_x, 
     & sug_fcx_low_y, sug_fcx_low_z, sug_fcx_high_x, sug_fcx_high_y, 
     & sug_fcx_high_z, sug_fcx, spg_fcx_low_x, spg_fcx_low_y, 
     & spg_fcx_low_z, spg_fcx_high_x, spg_fcx_high_y, spg_fcx_high_z, 
     & spg_fcx, sug_fcy_low_x, sug_fcy_low_y, sug_fcy_low_z, 
     & sug_fcy_high_x, sug_fcy_high_y, sug_fcy_high_z, sug_fcy, 
     & spg_fcy_low_x, spg_fcy_low_y, spg_fcy_low_z, spg_fcy_high_x, 
     & spg_fcy_high_y, spg_fcy_high_z, spg_fcy, sug_fcz_low_x, 
     & sug_fcz_low_y, sug_fcz_low_z, sug_fcz_high_x, sug_fcz_high_y, 
     & sug_fcz_high_z, sug_fcz, spg_fcz_low_x, spg_fcz_low_y, 
     & spg_fcz_low_z, spg_fcz_high_x, spg_fcz_high_y, spg_fcz_high_z, 
     & spg_fcz, kstabh_low_x, kstabh_low_y, kstabh_low_z, kstabh_high_x
     & , kstabh_high_y, kstabh_high_z, kstabh, tg_low_x, tg_low_y, 
     & tg_low_z, tg_high_x, tg_high_y, tg_high_z, tg, ts_cc_low_x, 
     & ts_cc_low_y, ts_cc_low_z, ts_cc_high_x, ts_cc_high_y, 
     & ts_cc_high_z, ts_cc, ts_fcx_low_x, ts_fcx_low_y, ts_fcx_low_z, 
     & ts_fcx_high_x, ts_fcx_high_y, ts_fcx_high_z, ts_fcx, 
     & ts_fcy_low_x, ts_fcy_low_y, ts_fcy_low_z, ts_fcy_high_x, 
     & ts_fcy_high_y, ts_fcy_high_z, ts_fcy, ts_fcz_low_x, ts_fcz_low_y
     & , ts_fcz_low_z, ts_fcz_high_x, ts_fcz_high_y, ts_fcz_high_z, 
     & ts_fcz, ug_cc_low_x, ug_cc_low_y, ug_cc_low_z, ug_cc_high_x, 
     & ug_cc_high_y, ug_cc_high_z, ug_cc, vg_cc_low_x, vg_cc_low_y, 
     & vg_cc_low_z, vg_cc_high_x, vg_cc_high_y, vg_cc_high_z, vg_cc, 
     & wg_cc_low_x, wg_cc_low_y, wg_cc_low_z, wg_cc_high_x, 
     & wg_cc_high_y, wg_cc_high_z, wg_cc, up_cc_low_x, up_cc_low_y, 
     & up_cc_low_z, up_cc_high_x, up_cc_high_y, up_cc_high_z, up_cc, 
     & vp_cc_low_x, vp_cc_low_y, vp_cc_low_z, vp_cc_high_x, 
     & vp_cc_high_y, vp_cc_high_z, vp_cc, wp_cc_low_x, wp_cc_low_y, 
     & wp_cc_low_z, wp_cc_high_x, wp_cc_high_y, wp_cc_high_z, wp_cc, 
     & vp_fcx_low_x, vp_fcx_low_y, vp_fcx_low_z, vp_fcx_high_x, 
     & vp_fcx_high_y, vp_fcx_high_z, vp_fcx, wp_fcx_low_x, wp_fcx_low_y
     & , wp_fcx_low_z, wp_fcx_high_x, wp_fcx_high_y, wp_fcx_high_z, 
     & wp_fcx, up_fcy_low_x, up_fcy_low_y, up_fcy_low_z, up_fcy_high_x,
     &  up_fcy_high_y, up_fcy_high_z, up_fcy, wp_fcy_low_x, 
     & wp_fcy_low_y, wp_fcy_low_z, wp_fcy_high_x, wp_fcy_high_y, 
     & wp_fcy_high_z, wp_fcy, up_fcz_low_x, up_fcz_low_y, up_fcz_low_z,
     &  up_fcz_high_x, up_fcz_high_y, up_fcz_high_z, up_fcz, 
     & vp_fcz_low_x, vp_fcz_low_y, vp_fcz_low_z, vp_fcz_high_x, 
     & vp_fcz_high_y, vp_fcz_high_z, vp_fcz, denMicro_low_x, 
     & denMicro_low_y, denMicro_low_z, denMicro_high_x, denMicro_high_y
     & , denMicro_high_z, denMicro, enth_low_x, enth_low_y, enth_low_z,
     &  enth_high_x, enth_high_y, enth_high_z, enth, rfluxE_low_x, 
     & rfluxE_low_y, rfluxE_low_z, rfluxE_high_x, rfluxE_high_y, 
     & rfluxE_high_z, rfluxE, rfluxW_low_x, rfluxW_low_y, rfluxW_low_z,
     &  rfluxW_high_x, rfluxW_high_y, rfluxW_high_z, rfluxW, 
     & rfluxN_low_x, rfluxN_low_y, rfluxN_low_z, rfluxN_high_x, 
     & rfluxN_high_y, rfluxN_high_z, rfluxN, rfluxS_low_x, rfluxS_low_y
     & , rfluxS_low_z, rfluxS_high_x, rfluxS_high_y, rfluxS_high_z, 
     & rfluxS, rfluxT_low_x, rfluxT_low_y, rfluxT_low_z, rfluxT_high_x,
     &  rfluxT_high_y, rfluxT_high_z, rfluxT, rfluxB_low_x, 
     & rfluxB_low_y, rfluxB_low_z, rfluxB_high_x, rfluxB_high_y, 
     & rfluxB_high_z, rfluxB, epsg_low_x, epsg_low_y, epsg_low_z, 
     & epsg_high_x, epsg_high_y, epsg_high_z, epsg, epss_low_x, 
     & epss_low_y, epss_low_z, epss_high_x, epss_high_y, epss_high_z, 
     & epss, dx, dy, dz, tcond, csmag, prturb, cpfluid, valid_lo, 
     & valid_hi, pcell_low_x, pcell_low_y, pcell_low_z, pcell_high_x, 
     & pcell_high_y, pcell_high_z, pcell, wall, ffield)

      implicit none
      integer hts_fcx_low_x, hts_fcx_low_y, hts_fcx_low_z, 
     & hts_fcx_high_x, hts_fcx_high_y, hts_fcx_high_z
      double precision hts_fcx(hts_fcx_low_x:hts_fcx_high_x, 
     & hts_fcx_low_y:hts_fcx_high_y, hts_fcx_low_z:hts_fcx_high_z)
      integer hts_fcy_low_x, hts_fcy_low_y, hts_fcy_low_z, 
     & hts_fcy_high_x, hts_fcy_high_y, hts_fcy_high_z
      double precision hts_fcy(hts_fcy_low_x:hts_fcy_high_x, 
     & hts_fcy_low_y:hts_fcy_high_y, hts_fcy_low_z:hts_fcy_high_z)
      integer hts_fcz_low_x, hts_fcz_low_y, hts_fcz_low_z, 
     & hts_fcz_high_x, hts_fcz_high_y, hts_fcz_high_z
      double precision hts_fcz(hts_fcz_low_x:hts_fcz_high_x, 
     & hts_fcz_low_y:hts_fcz_high_y, hts_fcz_low_z:hts_fcz_high_z)
      integer hts_fcx_rad_low_x, hts_fcx_rad_low_y, hts_fcx_rad_low_z, 
     & hts_fcx_rad_high_x, hts_fcx_rad_high_y, hts_fcx_rad_high_z
      double precision hts_fcx_rad(hts_fcx_rad_low_x:hts_fcx_rad_high_x
     & , hts_fcx_rad_low_y:hts_fcx_rad_high_y, hts_fcx_rad_low_z:
     & hts_fcx_rad_high_z)
      integer hts_fcy_rad_low_x, hts_fcy_rad_low_y, hts_fcy_rad_low_z, 
     & hts_fcy_rad_high_x, hts_fcy_rad_high_y, hts_fcy_rad_high_z
      double precision hts_fcy_rad(hts_fcy_rad_low_x:hts_fcy_rad_high_x
     & , hts_fcy_rad_low_y:hts_fcy_rad_high_y, hts_fcy_rad_low_z:
     & hts_fcy_rad_high_z)
      integer hts_fcz_rad_low_x, hts_fcz_rad_low_y, hts_fcz_rad_low_z, 
     & hts_fcz_rad_high_x, hts_fcz_rad_high_y, hts_fcz_rad_high_z
      double precision hts_fcz_rad(hts_fcz_rad_low_x:hts_fcz_rad_high_x
     & , hts_fcz_rad_low_y:hts_fcz_rad_high_y, hts_fcz_rad_low_z:
     & hts_fcz_rad_high_z)
      integer hts_cc_low_x, hts_cc_low_y, hts_cc_low_z, hts_cc_high_x, 
     & hts_cc_high_y, hts_cc_high_z
      double precision hts_cc(hts_cc_low_x:hts_cc_high_x, hts_cc_low_y:
     & hts_cc_high_y, hts_cc_low_z:hts_cc_high_z)
      integer htflux_convX_low_x, htflux_convX_low_y, 
     & htflux_convX_low_z, htflux_convX_high_x, htflux_convX_high_y, 
     & htflux_convX_high_z
      double precision htflux_convX(htflux_convX_low_x:
     & htflux_convX_high_x, htflux_convX_low_y:htflux_convX_high_y, 
     & htflux_convX_low_z:htflux_convX_high_z)
      integer htflux_radX_low_x, htflux_radX_low_y, htflux_radX_low_z, 
     & htflux_radX_high_x, htflux_radX_high_y, htflux_radX_high_z
      double precision htflux_radX(htflux_radX_low_x:htflux_radX_high_x
     & , htflux_radX_low_y:htflux_radX_high_y, htflux_radX_low_z:
     & htflux_radX_high_z)
      integer htfluxX_low_x, htfluxX_low_y, htfluxX_low_z, 
     & htfluxX_high_x, htfluxX_high_y, htfluxX_high_z
      double precision htfluxX(htfluxX_low_x:htfluxX_high_x, 
     & htfluxX_low_y:htfluxX_high_y, htfluxX_low_z:htfluxX_high_z)
      integer htflux_convY_low_x, htflux_convY_low_y, 
     & htflux_convY_low_z, htflux_convY_high_x, htflux_convY_high_y, 
     & htflux_convY_high_z
      double precision htflux_convY(htflux_convY_low_x:
     & htflux_convY_high_x, htflux_convY_low_y:htflux_convY_high_y, 
     & htflux_convY_low_z:htflux_convY_high_z)
      integer htflux_radY_low_x, htflux_radY_low_y, htflux_radY_low_z, 
     & htflux_radY_high_x, htflux_radY_high_y, htflux_radY_high_z
      double precision htflux_radY(htflux_radY_low_x:htflux_radY_high_x
     & , htflux_radY_low_y:htflux_radY_high_y, htflux_radY_low_z:
     & htflux_radY_high_z)
      integer htfluxY_low_x, htfluxY_low_y, htfluxY_low_z, 
     & htfluxY_high_x, htfluxY_high_y, htfluxY_high_z
      double precision htfluxY(htfluxY_low_x:htfluxY_high_x, 
     & htfluxY_low_y:htfluxY_high_y, htfluxY_low_z:htfluxY_high_z)
      integer htflux_convZ_low_x, htflux_convZ_low_y, 
     & htflux_convZ_low_z, htflux_convZ_high_x, htflux_convZ_high_y, 
     & htflux_convZ_high_z
      double precision htflux_convZ(htflux_convZ_low_x:
     & htflux_convZ_high_x, htflux_convZ_low_y:htflux_convZ_high_y, 
     & htflux_convZ_low_z:htflux_convZ_high_z)
      integer htflux_radZ_low_x, htflux_radZ_low_y, htflux_radZ_low_z, 
     & htflux_radZ_high_x, htflux_radZ_high_y, htflux_radZ_high_z
      double precision htflux_radZ(htflux_radZ_low_x:htflux_radZ_high_x
     & , htflux_radZ_low_y:htflux_radZ_high_y, htflux_radZ_low_z:
     & htflux_radZ_high_z)
      integer htfluxZ_low_x, htfluxZ_low_y, htfluxZ_low_z, 
     & htfluxZ_high_x, htfluxZ_high_y, htfluxZ_high_z
      double precision htfluxZ(htfluxZ_low_x:htfluxZ_high_x, 
     & htfluxZ_low_y:htfluxZ_high_y, htfluxZ_low_z:htfluxZ_high_z)
      integer htflux_convCC_low_x, htflux_convCC_low_y, 
     & htflux_convCC_low_z, htflux_convCC_high_x, htflux_convCC_high_y,
     &  htflux_convCC_high_z
      double precision htflux_convCC(htflux_convCC_low_x:
     & htflux_convCC_high_x, htflux_convCC_low_y:htflux_convCC_high_y, 
     & htflux_convCC_low_z:htflux_convCC_high_z)
      integer sug_cc_low_x, sug_cc_low_y, sug_cc_low_z, sug_cc_high_x, 
     & sug_cc_high_y, sug_cc_high_z
      double precision sug_cc(sug_cc_low_x:sug_cc_high_x, sug_cc_low_y:
     & sug_cc_high_y, sug_cc_low_z:sug_cc_high_z)
      integer spg_cc_low_x, spg_cc_low_y, spg_cc_low_z, spg_cc_high_x, 
     & spg_cc_high_y, spg_cc_high_z
      double precision spg_cc(spg_cc_low_x:spg_cc_high_x, spg_cc_low_y:
     & spg_cc_high_y, spg_cc_low_z:spg_cc_high_z)
      integer sug_fcx_low_x, sug_fcx_low_y, sug_fcx_low_z, 
     & sug_fcx_high_x, sug_fcx_high_y, sug_fcx_high_z
      double precision sug_fcx(sug_fcx_low_x:sug_fcx_high_x, 
     & sug_fcx_low_y:sug_fcx_high_y, sug_fcx_low_z:sug_fcx_high_z)
      integer spg_fcx_low_x, spg_fcx_low_y, spg_fcx_low_z, 
     & spg_fcx_high_x, spg_fcx_high_y, spg_fcx_high_z
      double precision spg_fcx(spg_fcx_low_x:spg_fcx_high_x, 
     & spg_fcx_low_y:spg_fcx_high_y, spg_fcx_low_z:spg_fcx_high_z)
      integer sug_fcy_low_x, sug_fcy_low_y, sug_fcy_low_z, 
     & sug_fcy_high_x, sug_fcy_high_y, sug_fcy_high_z
      double precision sug_fcy(sug_fcy_low_x:sug_fcy_high_x, 
     & sug_fcy_low_y:sug_fcy_high_y, sug_fcy_low_z:sug_fcy_high_z)
      integer spg_fcy_low_x, spg_fcy_low_y, spg_fcy_low_z, 
     & spg_fcy_high_x, spg_fcy_high_y, spg_fcy_high_z
      double precision spg_fcy(spg_fcy_low_x:spg_fcy_high_x, 
     & spg_fcy_low_y:spg_fcy_high_y, spg_fcy_low_z:spg_fcy_high_z)
      integer sug_fcz_low_x, sug_fcz_low_y, sug_fcz_low_z, 
     & sug_fcz_high_x, sug_fcz_high_y, sug_fcz_high_z
      double precision sug_fcz(sug_fcz_low_x:sug_fcz_high_x, 
     & sug_fcz_low_y:sug_fcz_high_y, sug_fcz_low_z:sug_fcz_high_z)
      integer spg_fcz_low_x, spg_fcz_low_y, spg_fcz_low_z, 
     & spg_fcz_high_x, spg_fcz_high_y, spg_fcz_high_z
      double precision spg_fcz(spg_fcz_low_x:spg_fcz_high_x, 
     & spg_fcz_low_y:spg_fcz_high_y, spg_fcz_low_z:spg_fcz_high_z)
      integer kstabh_low_x, kstabh_low_y, kstabh_low_z, kstabh_high_x, 
     & kstabh_high_y, kstabh_high_z
      double precision kstabh(kstabh_low_x:kstabh_high_x, kstabh_low_y:
     & kstabh_high_y, kstabh_low_z:kstabh_high_z)
      integer tg_low_x, tg_low_y, tg_low_z, tg_high_x, tg_high_y, 
     & tg_high_z
      double precision tg(tg_low_x:tg_high_x, tg_low_y:tg_high_y, 
     & tg_low_z:tg_high_z)
      integer ts_cc_low_x, ts_cc_low_y, ts_cc_low_z, ts_cc_high_x, 
     & ts_cc_high_y, ts_cc_high_z
      double precision ts_cc(ts_cc_low_x:ts_cc_high_x, ts_cc_low_y:
     & ts_cc_high_y, ts_cc_low_z:ts_cc_high_z)
      integer ts_fcx_low_x, ts_fcx_low_y, ts_fcx_low_z, ts_fcx_high_x, 
     & ts_fcx_high_y, ts_fcx_high_z
      double precision ts_fcx(ts_fcx_low_x:ts_fcx_high_x, ts_fcx_low_y:
     & ts_fcx_high_y, ts_fcx_low_z:ts_fcx_high_z)
      integer ts_fcy_low_x, ts_fcy_low_y, ts_fcy_low_z, ts_fcy_high_x, 
     & ts_fcy_high_y, ts_fcy_high_z
      double precision ts_fcy(ts_fcy_low_x:ts_fcy_high_x, ts_fcy_low_y:
     & ts_fcy_high_y, ts_fcy_low_z:ts_fcy_high_z)
      integer ts_fcz_low_x, ts_fcz_low_y, ts_fcz_low_z, ts_fcz_high_x, 
     & ts_fcz_high_y, ts_fcz_high_z
      double precision ts_fcz(ts_fcz_low_x:ts_fcz_high_x, ts_fcz_low_y:
     & ts_fcz_high_y, ts_fcz_low_z:ts_fcz_high_z)
      integer ug_cc_low_x, ug_cc_low_y, ug_cc_low_z, ug_cc_high_x, 
     & ug_cc_high_y, ug_cc_high_z
      double precision ug_cc(ug_cc_low_x:ug_cc_high_x, ug_cc_low_y:
     & ug_cc_high_y, ug_cc_low_z:ug_cc_high_z)
      integer vg_cc_low_x, vg_cc_low_y, vg_cc_low_z, vg_cc_high_x, 
     & vg_cc_high_y, vg_cc_high_z
      double precision vg_cc(vg_cc_low_x:vg_cc_high_x, vg_cc_low_y:
     & vg_cc_high_y, vg_cc_low_z:vg_cc_high_z)
      integer wg_cc_low_x, wg_cc_low_y, wg_cc_low_z, wg_cc_high_x, 
     & wg_cc_high_y, wg_cc_high_z
      double precision wg_cc(wg_cc_low_x:wg_cc_high_x, wg_cc_low_y:
     & wg_cc_high_y, wg_cc_low_z:wg_cc_high_z)
      integer up_cc_low_x, up_cc_low_y, up_cc_low_z, up_cc_high_x, 
     & up_cc_high_y, up_cc_high_z
      double precision up_cc(up_cc_low_x:up_cc_high_x, up_cc_low_y:
     & up_cc_high_y, up_cc_low_z:up_cc_high_z)
      integer vp_cc_low_x, vp_cc_low_y, vp_cc_low_z, vp_cc_high_x, 
     & vp_cc_high_y, vp_cc_high_z
      double precision vp_cc(vp_cc_low_x:vp_cc_high_x, vp_cc_low_y:
     & vp_cc_high_y, vp_cc_low_z:vp_cc_high_z)
      integer wp_cc_low_x, wp_cc_low_y, wp_cc_low_z, wp_cc_high_x, 
     & wp_cc_high_y, wp_cc_high_z
      double precision wp_cc(wp_cc_low_x:wp_cc_high_x, wp_cc_low_y:
     & wp_cc_high_y, wp_cc_low_z:wp_cc_high_z)
      integer vp_fcx_low_x, vp_fcx_low_y, vp_fcx_low_z, vp_fcx_high_x, 
     & vp_fcx_high_y, vp_fcx_high_z
      double precision vp_fcx(vp_fcx_low_x:vp_fcx_high_x, vp_fcx_low_y:
     & vp_fcx_high_y, vp_fcx_low_z:vp_fcx_high_z)
      integer wp_fcx_low_x, wp_fcx_low_y, wp_fcx_low_z, wp_fcx_high_x, 
     & wp_fcx_high_y, wp_fcx_high_z
      double precision wp_fcx(wp_fcx_low_x:wp_fcx_high_x, wp_fcx_low_y:
     & wp_fcx_high_y, wp_fcx_low_z:wp_fcx_high_z)
      integer up_fcy_low_x, up_fcy_low_y, up_fcy_low_z, up_fcy_high_x, 
     & up_fcy_high_y, up_fcy_high_z
      double precision up_fcy(up_fcy_low_x:up_fcy_high_x, up_fcy_low_y:
     & up_fcy_high_y, up_fcy_low_z:up_fcy_high_z)
      integer wp_fcy_low_x, wp_fcy_low_y, wp_fcy_low_z, wp_fcy_high_x, 
     & wp_fcy_high_y, wp_fcy_high_z
      double precision wp_fcy(wp_fcy_low_x:wp_fcy_high_x, wp_fcy_low_y:
     & wp_fcy_high_y, wp_fcy_low_z:wp_fcy_high_z)
      integer up_fcz_low_x, up_fcz_low_y, up_fcz_low_z, up_fcz_high_x, 
     & up_fcz_high_y, up_fcz_high_z
      double precision up_fcz(up_fcz_low_x:up_fcz_high_x, up_fcz_low_y:
     & up_fcz_high_y, up_fcz_low_z:up_fcz_high_z)
      integer vp_fcz_low_x, vp_fcz_low_y, vp_fcz_low_z, vp_fcz_high_x, 
     & vp_fcz_high_y, vp_fcz_high_z
      double precision vp_fcz(vp_fcz_low_x:vp_fcz_high_x, vp_fcz_low_y:
     & vp_fcz_high_y, vp_fcz_low_z:vp_fcz_high_z)
      integer denMicro_low_x, denMicro_low_y, denMicro_low_z, 
     & denMicro_high_x, denMicro_high_y, denMicro_high_z
      double precision denMicro(denMicro_low_x:denMicro_high_x, 
     & denMicro_low_y:denMicro_high_y, denMicro_low_z:denMicro_high_z)
      integer enth_low_x, enth_low_y, enth_low_z, enth_high_x, 
     & enth_high_y, enth_high_z
      double precision enth(enth_low_x:enth_high_x, enth_low_y:
     & enth_high_y, enth_low_z:enth_high_z)
      integer rfluxE_low_x, rfluxE_low_y, rfluxE_low_z, rfluxE_high_x, 
     & rfluxE_high_y, rfluxE_high_z
      double precision rfluxE(rfluxE_low_x:rfluxE_high_x, rfluxE_low_y:
     & rfluxE_high_y, rfluxE_low_z:rfluxE_high_z)
      integer rfluxW_low_x, rfluxW_low_y, rfluxW_low_z, rfluxW_high_x, 
     & rfluxW_high_y, rfluxW_high_z
      double precision rfluxW(rfluxW_low_x:rfluxW_high_x, rfluxW_low_y:
     & rfluxW_high_y, rfluxW_low_z:rfluxW_high_z)
      integer rfluxN_low_x, rfluxN_low_y, rfluxN_low_z, rfluxN_high_x, 
     & rfluxN_high_y, rfluxN_high_z
      double precision rfluxN(rfluxN_low_x:rfluxN_high_x, rfluxN_low_y:
     & rfluxN_high_y, rfluxN_low_z:rfluxN_high_z)
      integer rfluxS_low_x, rfluxS_low_y, rfluxS_low_z, rfluxS_high_x, 
     & rfluxS_high_y, rfluxS_high_z
      double precision rfluxS(rfluxS_low_x:rfluxS_high_x, rfluxS_low_y:
     & rfluxS_high_y, rfluxS_low_z:rfluxS_high_z)
      integer rfluxT_low_x, rfluxT_low_y, rfluxT_low_z, rfluxT_high_x, 
     & rfluxT_high_y, rfluxT_high_z
      double precision rfluxT(rfluxT_low_x:rfluxT_high_x, rfluxT_low_y:
     & rfluxT_high_y, rfluxT_low_z:rfluxT_high_z)
      integer rfluxB_low_x, rfluxB_low_y, rfluxB_low_z, rfluxB_high_x, 
     & rfluxB_high_y, rfluxB_high_z
      double precision rfluxB(rfluxB_low_x:rfluxB_high_x, rfluxB_low_y:
     & rfluxB_high_y, rfluxB_low_z:rfluxB_high_z)
      integer epsg_low_x, epsg_low_y, epsg_low_z, epsg_high_x, 
     & epsg_high_y, epsg_high_z
      double precision epsg(epsg_low_x:epsg_high_x, epsg_low_y:
     & epsg_high_y, epsg_low_z:epsg_high_z)
      integer epss_low_x, epss_low_y, epss_low_z, epss_high_x, 
     & epss_high_y, epss_high_z
      double precision epss(epss_low_x:epss_high_x, epss_low_y:
     & epss_high_y, epss_low_z:epss_high_z)
      double precision dx
      double precision dy
      double precision dz
      double precision tcond
      double precision csmag
      double precision prturb
      double precision cpfluid
      integer valid_lo(3)
      integer valid_hi(3)
      integer pcell_low_x, pcell_low_y, pcell_low_z, pcell_high_x, 
     & pcell_high_y, pcell_high_z
      integer pcell(pcell_low_x:pcell_high_x, pcell_low_y:pcell_high_y,
     &  pcell_low_z:pcell_high_z)
      integer wall
      integer ffield
#endif /* __cplusplus */

#endif /* fspec_energy_exchange_term */

#ifndef PASS1
#  define PASS1(x) x/**/_low, x/**/_high, x
#endif

#ifndef PASS3
#  define PASS3A(x) x/**/_low_x, x/**/_low_y, x/**/_low_z, 
#  define PASS3B(x) x/**/_high_x, x/**/_high_y, x/**/_high_z, x
#endif
