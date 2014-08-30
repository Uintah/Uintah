
#ifndef fspec_momentum_exchange_cont_cc
#define fspec_momentum_exchange_cont_cc

#ifdef __cplusplus

extern "C" void momentum_exchange_cont_cc_(int* sux_fcy_low_x, int* sux_fcy_low_y, int* sux_fcy_low_z, int* sux_fcy_high_x, int* sux_fcy_high_y, int* sux_fcy_high_z, double* sux_fcy_ptr,
                                           int* spx_fcy_low_x, int* spx_fcy_low_y, int* spx_fcy_low_z, int* spx_fcy_high_x, int* spx_fcy_high_y, int* spx_fcy_high_z, double* spx_fcy_ptr,
                                           int* sux_fcz_low_x, int* sux_fcz_low_y, int* sux_fcz_low_z, int* sux_fcz_high_x, int* sux_fcz_high_y, int* sux_fcz_high_z, double* sux_fcz_ptr,
                                           int* spx_fcz_low_x, int* spx_fcz_low_y, int* spx_fcz_low_z, int* spx_fcz_high_x, int* spx_fcz_high_y, int* spx_fcz_high_z, double* spx_fcz_ptr,
                                           int* sux_cc_low_x, int* sux_cc_low_y, int* sux_cc_low_z, int* sux_cc_high_x, int* sux_cc_high_y, int* sux_cc_high_z, double* sux_cc_ptr,
                                           int* spx_cc_low_x, int* spx_cc_low_y, int* spx_cc_low_z, int* spx_cc_high_x, int* spx_cc_high_y, int* spx_cc_high_z, double* spx_cc_ptr,
                                           int* kstabu_low_x, int* kstabu_low_y, int* kstabu_low_z, int* kstabu_high_x, int* kstabu_high_y, int* kstabu_high_z, double* kstabu_ptr,
                                           int* dfx_fcy_low_x, int* dfx_fcy_low_y, int* dfx_fcy_low_z, int* dfx_fcy_high_x, int* dfx_fcy_high_y, int* dfx_fcy_high_z, double* dfx_fcy_ptr,
                                           int* dfx_fcz_low_x, int* dfx_fcz_low_y, int* dfx_fcz_low_z, int* dfx_fcz_high_x, int* dfx_fcz_high_y, int* dfx_fcz_high_z, double* dfx_fcz_ptr,
                                           int* dfx_cc_low_x, int* dfx_cc_low_y, int* dfx_cc_low_z, int* dfx_cc_high_x, int* dfx_cc_high_y, int* dfx_cc_high_z, double* dfx_cc_ptr,
                                           int* ug_cc_low_x, int* ug_cc_low_y, int* ug_cc_low_z, int* ug_cc_high_x, int* ug_cc_high_y, int* ug_cc_high_z, double* ug_cc_ptr,
                                           int* up_cc_low_x, int* up_cc_low_y, int* up_cc_low_z, int* up_cc_high_x, int* up_cc_high_y, int* up_cc_high_z, double* up_cc_ptr,
                                           int* up_fcy_low_x, int* up_fcy_low_y, int* up_fcy_low_z, int* up_fcy_high_x, int* up_fcy_high_y, int* up_fcy_high_z, double* up_fcy_ptr,
                                           int* up_fcz_low_x, int* up_fcz_low_y, int* up_fcz_low_z, int* up_fcz_high_x, int* up_fcz_high_y, int* up_fcz_high_z, double* up_fcz_ptr,
                                           int* epsg_low_x, int* epsg_low_y, int* epsg_low_z, int* epsg_high_x, int* epsg_high_y, int* epsg_high_z, double* epsg_ptr,
                                           int* den_low_x, int* den_low_y, int* den_low_z, int* den_high_x, int* den_high_y, int* den_high_z, double* den_ptr,
                                           int* denmicro_low_x, int* denmicro_low_y, int* denmicro_low_z, int* denmicro_high_x, int* denmicro_high_y, int* denmicro_high_z, double* denmicro_ptr,
                                           int* epss_low_x, int* epss_low_y, int* epss_low_z, int* epss_high_x, int* epss_high_y, int* epss_high_z, double* epss_ptr,
                                           double* viscos,
                                           double* csmag,
                                           int* sew_low, int* sew_high, double* sew_ptr,
                                           int* sns_low, int* sns_high, double* sns_ptr,
                                           int* stb_low, int* stb_high, double* stb_ptr,
                                           int* yy_low, int* yy_high, double* yy_ptr,
                                           int* zz_low, int* zz_high, double* zz_ptr,
                                           int* yv_low, int* yv_high, double* yv_ptr,
                                           int* zw_low, int* zw_high, double* zw_ptr,
                                           int* valid_lo,
                                           int* valid_hi,
                                           int* ioff,
                                           int* joff,
                                           int* koff,
                                           int* indexflo,
                                           int* indext1,
                                           int* indext2,
                                           int* pcell_low_x, int* pcell_low_y, int* pcell_low_z, int* pcell_high_x, int* pcell_high_y, int* pcell_high_z, int* pcell_ptr,
                                           int* wall,
                                           int* ffield);

static void fort_momentum_exchange_cont_cc( Uintah::Array3<double> & sux_fcy,
                                            Uintah::Array3<double> & spx_fcy,
                                            Uintah::Array3<double> & sux_fcz,
                                            Uintah::Array3<double> & spx_fcz,
                                            Uintah::CCVariable<double> & sux_cc,
                                            Uintah::CCVariable<double> & spx_cc,
                                            Uintah::Array3<double> & kstabu,
                                            Uintah::Array3<double> & dfx_fcy,
                                            Uintah::Array3<double> & dfx_fcz,
                                            Uintah::CCVariable<double> & dfx_cc,
                                            Uintah::CCVariable<double> & ug_cc,
                                            Uintah::constCCVariable<double> & up_cc,
                                            const Uintah::Array3<double> & up_fcy,
                                            const Uintah::Array3<double> & up_fcz,
                                            Uintah::constCCVariable<double> & epsg,
                                            Uintah::constCCVariable<double> & den,
                                            Uintah::constCCVariable<double> & denmicro,
                                            Uintah::constCCVariable<double> & epss,
                                            double & viscos,
                                            double & csmag,
                                            Uintah::OffsetArray1<double> & sew,
                                            Uintah::OffsetArray1<double> & sns,
                                            Uintah::OffsetArray1<double> & stb,
                                            Uintah::OffsetArray1<double> & yy,
                                            Uintah::OffsetArray1<double> & zz,
                                            Uintah::OffsetArray1<double> & yv,
                                            Uintah::OffsetArray1<double> & zw,
                                            Uintah::IntVector & valid_lo,
                                            Uintah::IntVector & valid_hi,
                                            int & ioff,
                                            int & joff,
                                            int & koff,
                                            int & indexflo,
                                            int & indext1,
                                            int & indext2,
                                            Uintah::constCCVariable<int> & pcell,
                                            int & wall,
                                            int & ffield )
{
  Uintah::IntVector sux_fcy_low = sux_fcy.getWindow()->getOffset();
  Uintah::IntVector sux_fcy_high = sux_fcy.getWindow()->getData()->size() + sux_fcy_low - Uintah::IntVector(1, 1, 1);
  int sux_fcy_low_x = sux_fcy_low.x();
  int sux_fcy_high_x = sux_fcy_high.x();
  int sux_fcy_low_y = sux_fcy_low.y();
  int sux_fcy_high_y = sux_fcy_high.y();
  int sux_fcy_low_z = sux_fcy_low.z();
  int sux_fcy_high_z = sux_fcy_high.z();
  Uintah::IntVector spx_fcy_low = spx_fcy.getWindow()->getOffset();
  Uintah::IntVector spx_fcy_high = spx_fcy.getWindow()->getData()->size() + spx_fcy_low - Uintah::IntVector(1, 1, 1);
  int spx_fcy_low_x = spx_fcy_low.x();
  int spx_fcy_high_x = spx_fcy_high.x();
  int spx_fcy_low_y = spx_fcy_low.y();
  int spx_fcy_high_y = spx_fcy_high.y();
  int spx_fcy_low_z = spx_fcy_low.z();
  int spx_fcy_high_z = spx_fcy_high.z();
  Uintah::IntVector sux_fcz_low = sux_fcz.getWindow()->getOffset();
  Uintah::IntVector sux_fcz_high = sux_fcz.getWindow()->getData()->size() + sux_fcz_low - Uintah::IntVector(1, 1, 1);
  int sux_fcz_low_x = sux_fcz_low.x();
  int sux_fcz_high_x = sux_fcz_high.x();
  int sux_fcz_low_y = sux_fcz_low.y();
  int sux_fcz_high_y = sux_fcz_high.y();
  int sux_fcz_low_z = sux_fcz_low.z();
  int sux_fcz_high_z = sux_fcz_high.z();
  Uintah::IntVector spx_fcz_low = spx_fcz.getWindow()->getOffset();
  Uintah::IntVector spx_fcz_high = spx_fcz.getWindow()->getData()->size() + spx_fcz_low - Uintah::IntVector(1, 1, 1);
  int spx_fcz_low_x = spx_fcz_low.x();
  int spx_fcz_high_x = spx_fcz_high.x();
  int spx_fcz_low_y = spx_fcz_low.y();
  int spx_fcz_high_y = spx_fcz_high.y();
  int spx_fcz_low_z = spx_fcz_low.z();
  int spx_fcz_high_z = spx_fcz_high.z();
  Uintah::IntVector sux_cc_low = sux_cc.getWindow()->getOffset();
  Uintah::IntVector sux_cc_high = sux_cc.getWindow()->getData()->size() + sux_cc_low - Uintah::IntVector(1, 1, 1);
  int sux_cc_low_x = sux_cc_low.x();
  int sux_cc_high_x = sux_cc_high.x();
  int sux_cc_low_y = sux_cc_low.y();
  int sux_cc_high_y = sux_cc_high.y();
  int sux_cc_low_z = sux_cc_low.z();
  int sux_cc_high_z = sux_cc_high.z();
  Uintah::IntVector spx_cc_low = spx_cc.getWindow()->getOffset();
  Uintah::IntVector spx_cc_high = spx_cc.getWindow()->getData()->size() + spx_cc_low - Uintah::IntVector(1, 1, 1);
  int spx_cc_low_x = spx_cc_low.x();
  int spx_cc_high_x = spx_cc_high.x();
  int spx_cc_low_y = spx_cc_low.y();
  int spx_cc_high_y = spx_cc_high.y();
  int spx_cc_low_z = spx_cc_low.z();
  int spx_cc_high_z = spx_cc_high.z();
  Uintah::IntVector kstabu_low = kstabu.getWindow()->getOffset();
  Uintah::IntVector kstabu_high = kstabu.getWindow()->getData()->size() + kstabu_low - Uintah::IntVector(1, 1, 1);
  int kstabu_low_x = kstabu_low.x();
  int kstabu_high_x = kstabu_high.x();
  int kstabu_low_y = kstabu_low.y();
  int kstabu_high_y = kstabu_high.y();
  int kstabu_low_z = kstabu_low.z();
  int kstabu_high_z = kstabu_high.z();
  Uintah::IntVector dfx_fcy_low = dfx_fcy.getWindow()->getOffset();
  Uintah::IntVector dfx_fcy_high = dfx_fcy.getWindow()->getData()->size() + dfx_fcy_low - Uintah::IntVector(1, 1, 1);
  int dfx_fcy_low_x = dfx_fcy_low.x();
  int dfx_fcy_high_x = dfx_fcy_high.x();
  int dfx_fcy_low_y = dfx_fcy_low.y();
  int dfx_fcy_high_y = dfx_fcy_high.y();
  int dfx_fcy_low_z = dfx_fcy_low.z();
  int dfx_fcy_high_z = dfx_fcy_high.z();
  Uintah::IntVector dfx_fcz_low = dfx_fcz.getWindow()->getOffset();
  Uintah::IntVector dfx_fcz_high = dfx_fcz.getWindow()->getData()->size() + dfx_fcz_low - Uintah::IntVector(1, 1, 1);
  int dfx_fcz_low_x = dfx_fcz_low.x();
  int dfx_fcz_high_x = dfx_fcz_high.x();
  int dfx_fcz_low_y = dfx_fcz_low.y();
  int dfx_fcz_high_y = dfx_fcz_high.y();
  int dfx_fcz_low_z = dfx_fcz_low.z();
  int dfx_fcz_high_z = dfx_fcz_high.z();
  Uintah::IntVector dfx_cc_low = dfx_cc.getWindow()->getOffset();
  Uintah::IntVector dfx_cc_high = dfx_cc.getWindow()->getData()->size() + dfx_cc_low - Uintah::IntVector(1, 1, 1);
  int dfx_cc_low_x = dfx_cc_low.x();
  int dfx_cc_high_x = dfx_cc_high.x();
  int dfx_cc_low_y = dfx_cc_low.y();
  int dfx_cc_high_y = dfx_cc_high.y();
  int dfx_cc_low_z = dfx_cc_low.z();
  int dfx_cc_high_z = dfx_cc_high.z();
  Uintah::IntVector ug_cc_low = ug_cc.getWindow()->getOffset();
  Uintah::IntVector ug_cc_high = ug_cc.getWindow()->getData()->size() + ug_cc_low - Uintah::IntVector(1, 1, 1);
  int ug_cc_low_x = ug_cc_low.x();
  int ug_cc_high_x = ug_cc_high.x();
  int ug_cc_low_y = ug_cc_low.y();
  int ug_cc_high_y = ug_cc_high.y();
  int ug_cc_low_z = ug_cc_low.z();
  int ug_cc_high_z = ug_cc_high.z();
  Uintah::IntVector up_cc_low = up_cc.getWindow()->getOffset();
  Uintah::IntVector up_cc_high = up_cc.getWindow()->getData()->size() + up_cc_low - Uintah::IntVector(1, 1, 1);
  int up_cc_low_x = up_cc_low.x();
  int up_cc_high_x = up_cc_high.x();
  int up_cc_low_y = up_cc_low.y();
  int up_cc_high_y = up_cc_high.y();
  int up_cc_low_z = up_cc_low.z();
  int up_cc_high_z = up_cc_high.z();
  Uintah::IntVector up_fcy_low = up_fcy.getWindow()->getOffset();
  Uintah::IntVector up_fcy_high = up_fcy.getWindow()->getData()->size() + up_fcy_low - Uintah::IntVector(1, 1, 1);
  int up_fcy_low_x = up_fcy_low.x();
  int up_fcy_high_x = up_fcy_high.x();
  int up_fcy_low_y = up_fcy_low.y();
  int up_fcy_high_y = up_fcy_high.y();
  int up_fcy_low_z = up_fcy_low.z();
  int up_fcy_high_z = up_fcy_high.z();
  Uintah::IntVector up_fcz_low = up_fcz.getWindow()->getOffset();
  Uintah::IntVector up_fcz_high = up_fcz.getWindow()->getData()->size() + up_fcz_low - Uintah::IntVector(1, 1, 1);
  int up_fcz_low_x = up_fcz_low.x();
  int up_fcz_high_x = up_fcz_high.x();
  int up_fcz_low_y = up_fcz_low.y();
  int up_fcz_high_y = up_fcz_high.y();
  int up_fcz_low_z = up_fcz_low.z();
  int up_fcz_high_z = up_fcz_high.z();
  Uintah::IntVector epsg_low = epsg.getWindow()->getOffset();
  Uintah::IntVector epsg_high = epsg.getWindow()->getData()->size() + epsg_low - Uintah::IntVector(1, 1, 1);
  int epsg_low_x = epsg_low.x();
  int epsg_high_x = epsg_high.x();
  int epsg_low_y = epsg_low.y();
  int epsg_high_y = epsg_high.y();
  int epsg_low_z = epsg_low.z();
  int epsg_high_z = epsg_high.z();
  Uintah::IntVector den_low = den.getWindow()->getOffset();
  Uintah::IntVector den_high = den.getWindow()->getData()->size() + den_low - Uintah::IntVector(1, 1, 1);
  int den_low_x = den_low.x();
  int den_high_x = den_high.x();
  int den_low_y = den_low.y();
  int den_high_y = den_high.y();
  int den_low_z = den_low.z();
  int den_high_z = den_high.z();
  Uintah::IntVector denmicro_low = denmicro.getWindow()->getOffset();
  Uintah::IntVector denmicro_high = denmicro.getWindow()->getData()->size() + denmicro_low - Uintah::IntVector(1, 1, 1);
  int denmicro_low_x = denmicro_low.x();
  int denmicro_high_x = denmicro_high.x();
  int denmicro_low_y = denmicro_low.y();
  int denmicro_high_y = denmicro_high.y();
  int denmicro_low_z = denmicro_low.z();
  int denmicro_high_z = denmicro_high.z();
  Uintah::IntVector epss_low = epss.getWindow()->getOffset();
  Uintah::IntVector epss_high = epss.getWindow()->getData()->size() + epss_low - Uintah::IntVector(1, 1, 1);
  int epss_low_x = epss_low.x();
  int epss_high_x = epss_high.x();
  int epss_low_y = epss_low.y();
  int epss_high_y = epss_high.y();
  int epss_low_z = epss_low.z();
  int epss_high_z = epss_high.z();
  int sew_low = sew.low();
  int sew_high = sew.high();
  int sns_low = sns.low();
  int sns_high = sns.high();
  int stb_low = stb.low();
  int stb_high = stb.high();
  int yy_low = yy.low();
  int yy_high = yy.high();
  int zz_low = zz.low();
  int zz_high = zz.high();
  int yv_low = yv.low();
  int yv_high = yv.high();
  int zw_low = zw.low();
  int zw_high = zw.high();
  Uintah::IntVector pcell_low = pcell.getWindow()->getOffset();
  Uintah::IntVector pcell_high = pcell.getWindow()->getData()->size() + pcell_low - Uintah::IntVector(1, 1, 1);
  int pcell_low_x = pcell_low.x();
  int pcell_high_x = pcell_high.x();
  int pcell_low_y = pcell_low.y();
  int pcell_high_y = pcell_high.y();
  int pcell_low_z = pcell_low.z();
  int pcell_high_z = pcell_high.z();
  momentum_exchange_cont_cc_( &sux_fcy_low_x, &sux_fcy_low_y, &sux_fcy_low_z, &sux_fcy_high_x, &sux_fcy_high_y, &sux_fcy_high_z, sux_fcy.getPointer(),
                              &spx_fcy_low_x, &spx_fcy_low_y, &spx_fcy_low_z, &spx_fcy_high_x, &spx_fcy_high_y, &spx_fcy_high_z, spx_fcy.getPointer(),
                              &sux_fcz_low_x, &sux_fcz_low_y, &sux_fcz_low_z, &sux_fcz_high_x, &sux_fcz_high_y, &sux_fcz_high_z, sux_fcz.getPointer(),
                              &spx_fcz_low_x, &spx_fcz_low_y, &spx_fcz_low_z, &spx_fcz_high_x, &spx_fcz_high_y, &spx_fcz_high_z, spx_fcz.getPointer(),
                              &sux_cc_low_x, &sux_cc_low_y, &sux_cc_low_z, &sux_cc_high_x, &sux_cc_high_y, &sux_cc_high_z, sux_cc.getPointer(),
                              &spx_cc_low_x, &spx_cc_low_y, &spx_cc_low_z, &spx_cc_high_x, &spx_cc_high_y, &spx_cc_high_z, spx_cc.getPointer(),
                              &kstabu_low_x, &kstabu_low_y, &kstabu_low_z, &kstabu_high_x, &kstabu_high_y, &kstabu_high_z, kstabu.getPointer(),
                              &dfx_fcy_low_x, &dfx_fcy_low_y, &dfx_fcy_low_z, &dfx_fcy_high_x, &dfx_fcy_high_y, &dfx_fcy_high_z, dfx_fcy.getPointer(),
                              &dfx_fcz_low_x, &dfx_fcz_low_y, &dfx_fcz_low_z, &dfx_fcz_high_x, &dfx_fcz_high_y, &dfx_fcz_high_z, dfx_fcz.getPointer(),
                              &dfx_cc_low_x, &dfx_cc_low_y, &dfx_cc_low_z, &dfx_cc_high_x, &dfx_cc_high_y, &dfx_cc_high_z, dfx_cc.getPointer(),
                              &ug_cc_low_x, &ug_cc_low_y, &ug_cc_low_z, &ug_cc_high_x, &ug_cc_high_y, &ug_cc_high_z, ug_cc.getPointer(),
                              &up_cc_low_x, &up_cc_low_y, &up_cc_low_z, &up_cc_high_x, &up_cc_high_y, &up_cc_high_z, const_cast<double*>(up_cc.getPointer()),
                              &up_fcy_low_x, &up_fcy_low_y, &up_fcy_low_z, &up_fcy_high_x, &up_fcy_high_y, &up_fcy_high_z, const_cast<double*>(up_fcy.getPointer()),
                              &up_fcz_low_x, &up_fcz_low_y, &up_fcz_low_z, &up_fcz_high_x, &up_fcz_high_y, &up_fcz_high_z, const_cast<double*>(up_fcz.getPointer()),
                              &epsg_low_x, &epsg_low_y, &epsg_low_z, &epsg_high_x, &epsg_high_y, &epsg_high_z, const_cast<double*>(epsg.getPointer()),
                              &den_low_x, &den_low_y, &den_low_z, &den_high_x, &den_high_y, &den_high_z, const_cast<double*>(den.getPointer()),
                              &denmicro_low_x, &denmicro_low_y, &denmicro_low_z, &denmicro_high_x, &denmicro_high_y, &denmicro_high_z, const_cast<double*>(denmicro.getPointer()),
                              &epss_low_x, &epss_low_y, &epss_low_z, &epss_high_x, &epss_high_y, &epss_high_z, const_cast<double*>(epss.getPointer()),
                              &viscos,
                              &csmag,
                              &sew_low, &sew_high, sew.get_objs(),
                              &sns_low, &sns_high, sns.get_objs(),
                              &stb_low, &stb_high, stb.get_objs(),
                              &yy_low, &yy_high, yy.get_objs(),
                              &zz_low, &zz_high, zz.get_objs(),
                              &yv_low, &yv_high, yv.get_objs(),
                              &zw_low, &zw_high, zw.get_objs(),
                              valid_lo.get_pointer(),
                              valid_hi.get_pointer(),
                              &ioff,
                              &joff,
                              &koff,
                              &indexflo,
                              &indext1,
                              &indext2,
                              &pcell_low_x, &pcell_low_y, &pcell_low_z, &pcell_high_x, &pcell_high_y, &pcell_high_z, const_cast<int*>(pcell.getPointer()),
                              &wall,
                              &ffield );
}

#else /* !__cplusplus */

C This is the FORTRAN code portion of the file:

      subroutine momentum_exchange_cont_cc(sux_fcy_low_x, sux_fcy_low_y
     & , sux_fcy_low_z, sux_fcy_high_x, sux_fcy_high_y, sux_fcy_high_z,
     &  sux_fcy, spx_fcy_low_x, spx_fcy_low_y, spx_fcy_low_z, 
     & spx_fcy_high_x, spx_fcy_high_y, spx_fcy_high_z, spx_fcy, 
     & sux_fcz_low_x, sux_fcz_low_y, sux_fcz_low_z, sux_fcz_high_x, 
     & sux_fcz_high_y, sux_fcz_high_z, sux_fcz, spx_fcz_low_x, 
     & spx_fcz_low_y, spx_fcz_low_z, spx_fcz_high_x, spx_fcz_high_y, 
     & spx_fcz_high_z, spx_fcz, sux_cc_low_x, sux_cc_low_y, 
     & sux_cc_low_z, sux_cc_high_x, sux_cc_high_y, sux_cc_high_z, 
     & sux_cc, spx_cc_low_x, spx_cc_low_y, spx_cc_low_z, spx_cc_high_x,
     &  spx_cc_high_y, spx_cc_high_z, spx_cc, kstabu_low_x, 
     & kstabu_low_y, kstabu_low_z, kstabu_high_x, kstabu_high_y, 
     & kstabu_high_z, kstabu, dfx_fcy_low_x, dfx_fcy_low_y, 
     & dfx_fcy_low_z, dfx_fcy_high_x, dfx_fcy_high_y, dfx_fcy_high_z, 
     & dfx_fcy, dfx_fcz_low_x, dfx_fcz_low_y, dfx_fcz_low_z, 
     & dfx_fcz_high_x, dfx_fcz_high_y, dfx_fcz_high_z, dfx_fcz, 
     & dfx_cc_low_x, dfx_cc_low_y, dfx_cc_low_z, dfx_cc_high_x, 
     & dfx_cc_high_y, dfx_cc_high_z, dfx_cc, ug_cc_low_x, ug_cc_low_y, 
     & ug_cc_low_z, ug_cc_high_x, ug_cc_high_y, ug_cc_high_z, ug_cc, 
     & up_cc_low_x, up_cc_low_y, up_cc_low_z, up_cc_high_x, 
     & up_cc_high_y, up_cc_high_z, up_cc, up_fcy_low_x, up_fcy_low_y, 
     & up_fcy_low_z, up_fcy_high_x, up_fcy_high_y, up_fcy_high_z, 
     & up_fcy, up_fcz_low_x, up_fcz_low_y, up_fcz_low_z, up_fcz_high_x,
     &  up_fcz_high_y, up_fcz_high_z, up_fcz, epsg_low_x, epsg_low_y, 
     & epsg_low_z, epsg_high_x, epsg_high_y, epsg_high_z, epsg, 
     & den_low_x, den_low_y, den_low_z, den_high_x, den_high_y, 
     & den_high_z, den, denmicro_low_x, denmicro_low_y, denmicro_low_z,
     &  denmicro_high_x, denmicro_high_y, denmicro_high_z, denmicro, 
     & epss_low_x, epss_low_y, epss_low_z, epss_high_x, epss_high_y, 
     & epss_high_z, epss, viscos, csmag, sew_low, sew_high, sew, 
     & sns_low, sns_high, sns, stb_low, stb_high, stb, yy_low, yy_high,
     &  yy, zz_low, zz_high, zz, yv_low, yv_high, yv, zw_low, zw_high, 
     & zw, valid_lo, valid_hi, ioff, joff, koff, indexflo, indext1, 
     & indext2, pcell_low_x, pcell_low_y, pcell_low_z, pcell_high_x, 
     & pcell_high_y, pcell_high_z, pcell, wall, ffield)

      implicit none
      integer sux_fcy_low_x, sux_fcy_low_y, sux_fcy_low_z, 
     & sux_fcy_high_x, sux_fcy_high_y, sux_fcy_high_z
      double precision sux_fcy(sux_fcy_low_x:sux_fcy_high_x, 
     & sux_fcy_low_y:sux_fcy_high_y, sux_fcy_low_z:sux_fcy_high_z)
      integer spx_fcy_low_x, spx_fcy_low_y, spx_fcy_low_z, 
     & spx_fcy_high_x, spx_fcy_high_y, spx_fcy_high_z
      double precision spx_fcy(spx_fcy_low_x:spx_fcy_high_x, 
     & spx_fcy_low_y:spx_fcy_high_y, spx_fcy_low_z:spx_fcy_high_z)
      integer sux_fcz_low_x, sux_fcz_low_y, sux_fcz_low_z, 
     & sux_fcz_high_x, sux_fcz_high_y, sux_fcz_high_z
      double precision sux_fcz(sux_fcz_low_x:sux_fcz_high_x, 
     & sux_fcz_low_y:sux_fcz_high_y, sux_fcz_low_z:sux_fcz_high_z)
      integer spx_fcz_low_x, spx_fcz_low_y, spx_fcz_low_z, 
     & spx_fcz_high_x, spx_fcz_high_y, spx_fcz_high_z
      double precision spx_fcz(spx_fcz_low_x:spx_fcz_high_x, 
     & spx_fcz_low_y:spx_fcz_high_y, spx_fcz_low_z:spx_fcz_high_z)
      integer sux_cc_low_x, sux_cc_low_y, sux_cc_low_z, sux_cc_high_x, 
     & sux_cc_high_y, sux_cc_high_z
      double precision sux_cc(sux_cc_low_x:sux_cc_high_x, sux_cc_low_y:
     & sux_cc_high_y, sux_cc_low_z:sux_cc_high_z)
      integer spx_cc_low_x, spx_cc_low_y, spx_cc_low_z, spx_cc_high_x, 
     & spx_cc_high_y, spx_cc_high_z
      double precision spx_cc(spx_cc_low_x:spx_cc_high_x, spx_cc_low_y:
     & spx_cc_high_y, spx_cc_low_z:spx_cc_high_z)
      integer kstabu_low_x, kstabu_low_y, kstabu_low_z, kstabu_high_x, 
     & kstabu_high_y, kstabu_high_z
      double precision kstabu(kstabu_low_x:kstabu_high_x, kstabu_low_y:
     & kstabu_high_y, kstabu_low_z:kstabu_high_z)
      integer dfx_fcy_low_x, dfx_fcy_low_y, dfx_fcy_low_z, 
     & dfx_fcy_high_x, dfx_fcy_high_y, dfx_fcy_high_z
      double precision dfx_fcy(dfx_fcy_low_x:dfx_fcy_high_x, 
     & dfx_fcy_low_y:dfx_fcy_high_y, dfx_fcy_low_z:dfx_fcy_high_z)
      integer dfx_fcz_low_x, dfx_fcz_low_y, dfx_fcz_low_z, 
     & dfx_fcz_high_x, dfx_fcz_high_y, dfx_fcz_high_z
      double precision dfx_fcz(dfx_fcz_low_x:dfx_fcz_high_x, 
     & dfx_fcz_low_y:dfx_fcz_high_y, dfx_fcz_low_z:dfx_fcz_high_z)
      integer dfx_cc_low_x, dfx_cc_low_y, dfx_cc_low_z, dfx_cc_high_x, 
     & dfx_cc_high_y, dfx_cc_high_z
      double precision dfx_cc(dfx_cc_low_x:dfx_cc_high_x, dfx_cc_low_y:
     & dfx_cc_high_y, dfx_cc_low_z:dfx_cc_high_z)
      integer ug_cc_low_x, ug_cc_low_y, ug_cc_low_z, ug_cc_high_x, 
     & ug_cc_high_y, ug_cc_high_z
      double precision ug_cc(ug_cc_low_x:ug_cc_high_x, ug_cc_low_y:
     & ug_cc_high_y, ug_cc_low_z:ug_cc_high_z)
      integer up_cc_low_x, up_cc_low_y, up_cc_low_z, up_cc_high_x, 
     & up_cc_high_y, up_cc_high_z
      double precision up_cc(up_cc_low_x:up_cc_high_x, up_cc_low_y:
     & up_cc_high_y, up_cc_low_z:up_cc_high_z)
      integer up_fcy_low_x, up_fcy_low_y, up_fcy_low_z, up_fcy_high_x, 
     & up_fcy_high_y, up_fcy_high_z
      double precision up_fcy(up_fcy_low_x:up_fcy_high_x, up_fcy_low_y:
     & up_fcy_high_y, up_fcy_low_z:up_fcy_high_z)
      integer up_fcz_low_x, up_fcz_low_y, up_fcz_low_z, up_fcz_high_x, 
     & up_fcz_high_y, up_fcz_high_z
      double precision up_fcz(up_fcz_low_x:up_fcz_high_x, up_fcz_low_y:
     & up_fcz_high_y, up_fcz_low_z:up_fcz_high_z)
      integer epsg_low_x, epsg_low_y, epsg_low_z, epsg_high_x, 
     & epsg_high_y, epsg_high_z
      double precision epsg(epsg_low_x:epsg_high_x, epsg_low_y:
     & epsg_high_y, epsg_low_z:epsg_high_z)
      integer den_low_x, den_low_y, den_low_z, den_high_x, den_high_y, 
     & den_high_z
      double precision den(den_low_x:den_high_x, den_low_y:den_high_y, 
     & den_low_z:den_high_z)
      integer denmicro_low_x, denmicro_low_y, denmicro_low_z, 
     & denmicro_high_x, denmicro_high_y, denmicro_high_z
      double precision denmicro(denmicro_low_x:denmicro_high_x, 
     & denmicro_low_y:denmicro_high_y, denmicro_low_z:denmicro_high_z)
      integer epss_low_x, epss_low_y, epss_low_z, epss_high_x, 
     & epss_high_y, epss_high_z
      double precision epss(epss_low_x:epss_high_x, epss_low_y:
     & epss_high_y, epss_low_z:epss_high_z)
      double precision viscos
      double precision csmag
      integer sew_low
      integer sew_high
      double precision sew(sew_low:sew_high)
      integer sns_low
      integer sns_high
      double precision sns(sns_low:sns_high)
      integer stb_low
      integer stb_high
      double precision stb(stb_low:stb_high)
      integer yy_low
      integer yy_high
      double precision yy(yy_low:yy_high)
      integer zz_low
      integer zz_high
      double precision zz(zz_low:zz_high)
      integer yv_low
      integer yv_high
      double precision yv(yv_low:yv_high)
      integer zw_low
      integer zw_high
      double precision zw(zw_low:zw_high)
      integer valid_lo(3)
      integer valid_hi(3)
      integer ioff
      integer joff
      integer koff
      integer indexflo
      integer indext1
      integer indext2
      integer pcell_low_x, pcell_low_y, pcell_low_z, pcell_high_x, 
     & pcell_high_y, pcell_high_z
      integer pcell(pcell_low_x:pcell_high_x, pcell_low_y:pcell_high_y,
     &  pcell_low_z:pcell_high_z)
      integer wall
      integer ffield
#endif /* __cplusplus */

#endif /* fspec_momentum_exchange_cont_cc */

#ifndef PASS1
#  define PASS1(x) x/**/_low, x/**/_high, x
#endif

#ifndef PASS3
#  define PASS3A(x) x/**/_low_x, x/**/_low_y, x/**/_low_z, 
#  define PASS3B(x) x/**/_high_x, x/**/_high_y, x/**/_high_z, x
#endif
