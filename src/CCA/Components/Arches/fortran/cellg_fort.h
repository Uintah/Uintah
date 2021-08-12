
#ifndef fspec_cellg
#define fspec_cellg

#ifdef __cplusplus

#include <CCA/Components/Arches/fortran/FortranNameMangle.h>

extern "C" void F_cellg(int* domainLow,
                      int* domainHigh,
                      int* indexLowU,
                      int* indexHighU,
                      int* indexLowV,
                      int* indexHighV,
                      int* indexLowW,
                      int* indexHighW,
                      int* indexLow,
                      int* indexHigh,
                      int* sew_low, int* sew_high, double* sew_ptr,
                      int* sns_low, int* sns_high, double* sns_ptr,
                      int* stb_low, int* stb_high, double* stb_ptr,
                      int* sewu_low, int* sewu_high, double* sewu_ptr,
                      int* snsv_low, int* snsv_high, double* snsv_ptr,
                      int* stbw_low, int* stbw_high, double* stbw_ptr,
                      int* dxep_low, int* dxep_high, double* dxep_ptr,
                      int* dynp_low, int* dynp_high, double* dynp_ptr,
                      int* dztp_low, int* dztp_high, double* dztp_ptr,
                      int* dxepu_low, int* dxepu_high, double* dxepu_ptr,
                      int* dynpv_low, int* dynpv_high, double* dynpv_ptr,
                      int* dztpw_low, int* dztpw_high, double* dztpw_ptr,
                      int* dxpw_low, int* dxpw_high, double* dxpw_ptr,
                      int* dyps_low, int* dyps_high, double* dyps_ptr,
                      int* dzpb_low, int* dzpb_high, double* dzpb_ptr,
                      int* dxpwu_low, int* dxpwu_high, double* dxpwu_ptr,
                      int* dypsv_low, int* dypsv_high, double* dypsv_ptr,
                      int* dzpbw_low, int* dzpbw_high, double* dzpbw_ptr,
                      int* cee_low, int* cee_high, double* cee_ptr,
                      int* cwe_low, int* cwe_high, double* cwe_ptr,
                      int* cww_low, int* cww_high, double* cww_ptr,
                      int* ceeu_low, int* ceeu_high, double* ceeu_ptr,
                      int* cweu_low, int* cweu_high, double* cweu_ptr,
                      int* cwwu_low, int* cwwu_high, double* cwwu_ptr,
                      int* cnn_low, int* cnn_high, double* cnn_ptr,
                      int* csn_low, int* csn_high, double* csn_ptr,
                      int* css_low, int* css_high, double* css_ptr,
                      int* cnnv_low, int* cnnv_high, double* cnnv_ptr,
                      int* csnv_low, int* csnv_high, double* csnv_ptr,
                      int* cssv_low, int* cssv_high, double* cssv_ptr,
                      int* ctt_low, int* ctt_high, double* ctt_ptr,
                      int* cbt_low, int* cbt_high, double* cbt_ptr,
                      int* cbb_low, int* cbb_high, double* cbb_ptr,
                      int* cttw_low, int* cttw_high, double* cttw_ptr,
                      int* cbtw_low, int* cbtw_high, double* cbtw_ptr,
                      int* cbbw_low, int* cbbw_high, double* cbbw_ptr,
                      int* xx_low, int* xx_high, double* xx_ptr,
                      int* xu_low, int* xu_high, double* xu_ptr,
                      int* yy_low, int* yy_high, double* yy_ptr,
                      int* yv_low, int* yv_high, double* yv_ptr,
                      int* zz_low, int* zz_high, double* zz_ptr,
                      int* zw_low, int* zw_high, double* zw_ptr,
                      int* efac_low, int* efac_high, double* efac_ptr,
                      int* wfac_low, int* wfac_high, double* wfac_ptr,
                      int* nfac_low, int* nfac_high, double* nfac_ptr,
                      int* sfac_low, int* sfac_high, double* sfac_ptr,
                      int* tfac_low, int* tfac_high, double* tfac_ptr,
                      int* bfac_low, int* bfac_high, double* bfac_ptr,
                      int* fac1u_low, int* fac1u_high, double* fac1u_ptr,
                      int* fac2u_low, int* fac2u_high, double* fac2u_ptr,
                      int* fac3u_low, int* fac3u_high, double* fac3u_ptr,
                      int* fac4u_low, int* fac4u_high, double* fac4u_ptr,
                      int* fac1v_low, int* fac1v_high, double* fac1v_ptr,
                      int* fac2v_low, int* fac2v_high, double* fac2v_ptr,
                      int* fac3v_low, int* fac3v_high, double* fac3v_ptr,
                      int* fac4v_low, int* fac4v_high, double* fac4v_ptr,
                      int* fac1w_low, int* fac1w_high, double* fac1w_ptr,
                      int* fac2w_low, int* fac2w_high, double* fac2w_ptr,
                      int* fac3w_low, int* fac3w_high, double* fac3w_ptr,
                      int* fac4w_low, int* fac4w_high, double* fac4w_ptr,
                      int* iesdu_low, int* iesdu_high, int* iesdu_ptr,
                      int* iwsdu_low, int* iwsdu_high, int* iwsdu_ptr,
                      int* jnsdv_low, int* jnsdv_high, int* jnsdv_ptr,
                      int* jssdv_low, int* jssdv_high, int* jssdv_ptr,
                      int* ktsdw_low, int* ktsdw_high, int* ktsdw_ptr,
                      int* kbsdw_low, int* kbsdw_high, int* kbsdw_ptr,
                      int* fac1ew_low, int* fac1ew_high, double* fac1ew_ptr,
                      int* fac2ew_low, int* fac2ew_high, double* fac2ew_ptr,
                      int* fac3ew_low, int* fac3ew_high, double* fac3ew_ptr,
                      int* fac4ew_low, int* fac4ew_high, double* fac4ew_ptr,
                      int* fac1ns_low, int* fac1ns_high, double* fac1ns_ptr,
                      int* fac2ns_low, int* fac2ns_high, double* fac2ns_ptr,
                      int* fac3ns_low, int* fac3ns_high, double* fac3ns_ptr,
                      int* fac4ns_low, int* fac4ns_high, double* fac4ns_ptr,
                      int* fac1tb_low, int* fac1tb_high, double* fac1tb_ptr,
                      int* fac2tb_low, int* fac2tb_high, double* fac2tb_ptr,
                      int* fac3tb_low, int* fac3tb_high, double* fac3tb_ptr,
                      int* fac4tb_low, int* fac4tb_high, double* fac4tb_ptr,
                      int* e_shift_low, int* e_shift_high, int* e_shift_ptr,
                      int* w_shift_low, int* w_shift_high, int* w_shift_ptr,
                      int* n_shift_low, int* n_shift_high, int* n_shift_ptr,
                      int* s_shift_low, int* s_shift_high, int* s_shift_ptr,
                      int* t_shift_low, int* t_shift_high, int* t_shift_ptr,
                      int* b_shift_low, int* b_shift_high, int* b_shift_ptr,
                      bool* xminus,
                      bool* xplus,
                      bool* yminus,
                      bool* yplus,
                      bool* zminus,
                      bool* zplus);

static void fort_cellg( Uintah::IntVector & domainLow,
                        Uintah::IntVector & domainHigh,
                        Uintah::IntVector & indexLowU,
                        Uintah::IntVector & indexHighU,
                        Uintah::IntVector & indexLowV,
                        Uintah::IntVector & indexHighV,
                        Uintah::IntVector & indexLowW,
                        Uintah::IntVector & indexHighW,
                        Uintah::IntVector & indexLow,
                        Uintah::IntVector & indexHigh,
                        Uintah::OffsetArray1<double> & sew,
                        Uintah::OffsetArray1<double> & sns,
                        Uintah::OffsetArray1<double> & stb,
                        Uintah::OffsetArray1<double> & sewu,
                        Uintah::OffsetArray1<double> & snsv,
                        Uintah::OffsetArray1<double> & stbw,
                        Uintah::OffsetArray1<double> & dxep,
                        Uintah::OffsetArray1<double> & dynp,
                        Uintah::OffsetArray1<double> & dztp,
                        Uintah::OffsetArray1<double> & dxepu,
                        Uintah::OffsetArray1<double> & dynpv,
                        Uintah::OffsetArray1<double> & dztpw,
                        Uintah::OffsetArray1<double> & dxpw,
                        Uintah::OffsetArray1<double> & dyps,
                        Uintah::OffsetArray1<double> & dzpb,
                        Uintah::OffsetArray1<double> & dxpwu,
                        Uintah::OffsetArray1<double> & dypsv,
                        Uintah::OffsetArray1<double> & dzpbw,
                        Uintah::OffsetArray1<double> & cee,
                        Uintah::OffsetArray1<double> & cwe,
                        Uintah::OffsetArray1<double> & cww,
                        Uintah::OffsetArray1<double> & ceeu,
                        Uintah::OffsetArray1<double> & cweu,
                        Uintah::OffsetArray1<double> & cwwu,
                        Uintah::OffsetArray1<double> & cnn,
                        Uintah::OffsetArray1<double> & csn,
                        Uintah::OffsetArray1<double> & css,
                        Uintah::OffsetArray1<double> & cnnv,
                        Uintah::OffsetArray1<double> & csnv,
                        Uintah::OffsetArray1<double> & cssv,
                        Uintah::OffsetArray1<double> & ctt,
                        Uintah::OffsetArray1<double> & cbt,
                        Uintah::OffsetArray1<double> & cbb,
                        Uintah::OffsetArray1<double> & cttw,
                        Uintah::OffsetArray1<double> & cbtw,
                        Uintah::OffsetArray1<double> & cbbw,
                        Uintah::OffsetArray1<double> & xx,
                        Uintah::OffsetArray1<double> & xu,
                        Uintah::OffsetArray1<double> & yy,
                        Uintah::OffsetArray1<double> & yv,
                        Uintah::OffsetArray1<double> & zz,
                        Uintah::OffsetArray1<double> & zw,
                        Uintah::OffsetArray1<double> & efac,
                        Uintah::OffsetArray1<double> & wfac,
                        Uintah::OffsetArray1<double> & nfac,
                        Uintah::OffsetArray1<double> & sfac,
                        Uintah::OffsetArray1<double> & tfac,
                        Uintah::OffsetArray1<double> & bfac,
                        Uintah::OffsetArray1<double> & fac1u,
                        Uintah::OffsetArray1<double> & fac2u,
                        Uintah::OffsetArray1<double> & fac3u,
                        Uintah::OffsetArray1<double> & fac4u,
                        Uintah::OffsetArray1<double> & fac1v,
                        Uintah::OffsetArray1<double> & fac2v,
                        Uintah::OffsetArray1<double> & fac3v,
                        Uintah::OffsetArray1<double> & fac4v,
                        Uintah::OffsetArray1<double> & fac1w,
                        Uintah::OffsetArray1<double> & fac2w,
                        Uintah::OffsetArray1<double> & fac3w,
                        Uintah::OffsetArray1<double> & fac4w,
                        Uintah::OffsetArray1<int> & iesdu,
                        Uintah::OffsetArray1<int> & iwsdu,
                        Uintah::OffsetArray1<int> & jnsdv,
                        Uintah::OffsetArray1<int> & jssdv,
                        Uintah::OffsetArray1<int> & ktsdw,
                        Uintah::OffsetArray1<int> & kbsdw,
                        Uintah::OffsetArray1<double> & fac1ew,
                        Uintah::OffsetArray1<double> & fac2ew,
                        Uintah::OffsetArray1<double> & fac3ew,
                        Uintah::OffsetArray1<double> & fac4ew,
                        Uintah::OffsetArray1<double> & fac1ns,
                        Uintah::OffsetArray1<double> & fac2ns,
                        Uintah::OffsetArray1<double> & fac3ns,
                        Uintah::OffsetArray1<double> & fac4ns,
                        Uintah::OffsetArray1<double> & fac1tb,
                        Uintah::OffsetArray1<double> & fac2tb,
                        Uintah::OffsetArray1<double> & fac3tb,
                        Uintah::OffsetArray1<double> & fac4tb,
                        Uintah::OffsetArray1<int> & e_shift,
                        Uintah::OffsetArray1<int> & w_shift,
                        Uintah::OffsetArray1<int> & n_shift,
                        Uintah::OffsetArray1<int> & s_shift,
                        Uintah::OffsetArray1<int> & t_shift,
                        Uintah::OffsetArray1<int> & b_shift,
                        bool & xminus,
                        bool & xplus,
                        bool & yminus,
                        bool & yplus,
                        bool & zminus,
                        bool & zplus )
{
  int sew_low = sew.low();
  int sew_high = sew.high();
  int sns_low = sns.low();
  int sns_high = sns.high();
  int stb_low = stb.low();
  int stb_high = stb.high();
  int sewu_low = sewu.low();
  int sewu_high = sewu.high();
  int snsv_low = snsv.low();
  int snsv_high = snsv.high();
  int stbw_low = stbw.low();
  int stbw_high = stbw.high();
  int dxep_low = dxep.low();
  int dxep_high = dxep.high();
  int dynp_low = dynp.low();
  int dynp_high = dynp.high();
  int dztp_low = dztp.low();
  int dztp_high = dztp.high();
  int dxepu_low = dxepu.low();
  int dxepu_high = dxepu.high();
  int dynpv_low = dynpv.low();
  int dynpv_high = dynpv.high();
  int dztpw_low = dztpw.low();
  int dztpw_high = dztpw.high();
  int dxpw_low = dxpw.low();
  int dxpw_high = dxpw.high();
  int dyps_low = dyps.low();
  int dyps_high = dyps.high();
  int dzpb_low = dzpb.low();
  int dzpb_high = dzpb.high();
  int dxpwu_low = dxpwu.low();
  int dxpwu_high = dxpwu.high();
  int dypsv_low = dypsv.low();
  int dypsv_high = dypsv.high();
  int dzpbw_low = dzpbw.low();
  int dzpbw_high = dzpbw.high();
  int cee_low = cee.low();
  int cee_high = cee.high();
  int cwe_low = cwe.low();
  int cwe_high = cwe.high();
  int cww_low = cww.low();
  int cww_high = cww.high();
  int ceeu_low = ceeu.low();
  int ceeu_high = ceeu.high();
  int cweu_low = cweu.low();
  int cweu_high = cweu.high();
  int cwwu_low = cwwu.low();
  int cwwu_high = cwwu.high();
  int cnn_low = cnn.low();
  int cnn_high = cnn.high();
  int csn_low = csn.low();
  int csn_high = csn.high();
  int css_low = css.low();
  int css_high = css.high();
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
  int cttw_low = cttw.low();
  int cttw_high = cttw.high();
  int cbtw_low = cbtw.low();
  int cbtw_high = cbtw.high();
  int cbbw_low = cbbw.low();
  int cbbw_high = cbbw.high();
  int xx_low = xx.low();
  int xx_high = xx.high();
  int xu_low = xu.low();
  int xu_high = xu.high();
  int yy_low = yy.low();
  int yy_high = yy.high();
  int yv_low = yv.low();
  int yv_high = yv.high();
  int zz_low = zz.low();
  int zz_high = zz.high();
  int zw_low = zw.low();
  int zw_high = zw.high();
  int efac_low = efac.low();
  int efac_high = efac.high();
  int wfac_low = wfac.low();
  int wfac_high = wfac.high();
  int nfac_low = nfac.low();
  int nfac_high = nfac.high();
  int sfac_low = sfac.low();
  int sfac_high = sfac.high();
  int tfac_low = tfac.low();
  int tfac_high = tfac.high();
  int bfac_low = bfac.low();
  int bfac_high = bfac.high();
  int fac1u_low = fac1u.low();
  int fac1u_high = fac1u.high();
  int fac2u_low = fac2u.low();
  int fac2u_high = fac2u.high();
  int fac3u_low = fac3u.low();
  int fac3u_high = fac3u.high();
  int fac4u_low = fac4u.low();
  int fac4u_high = fac4u.high();
  int fac1v_low = fac1v.low();
  int fac1v_high = fac1v.high();
  int fac2v_low = fac2v.low();
  int fac2v_high = fac2v.high();
  int fac3v_low = fac3v.low();
  int fac3v_high = fac3v.high();
  int fac4v_low = fac4v.low();
  int fac4v_high = fac4v.high();
  int fac1w_low = fac1w.low();
  int fac1w_high = fac1w.high();
  int fac2w_low = fac2w.low();
  int fac2w_high = fac2w.high();
  int fac3w_low = fac3w.low();
  int fac3w_high = fac3w.high();
  int fac4w_low = fac4w.low();
  int fac4w_high = fac4w.high();
  int iesdu_low = iesdu.low();
  int iesdu_high = iesdu.high();
  int iwsdu_low = iwsdu.low();
  int iwsdu_high = iwsdu.high();
  int jnsdv_low = jnsdv.low();
  int jnsdv_high = jnsdv.high();
  int jssdv_low = jssdv.low();
  int jssdv_high = jssdv.high();
  int ktsdw_low = ktsdw.low();
  int ktsdw_high = ktsdw.high();
  int kbsdw_low = kbsdw.low();
  int kbsdw_high = kbsdw.high();
  int fac1ew_low = fac1ew.low();
  int fac1ew_high = fac1ew.high();
  int fac2ew_low = fac2ew.low();
  int fac2ew_high = fac2ew.high();
  int fac3ew_low = fac3ew.low();
  int fac3ew_high = fac3ew.high();
  int fac4ew_low = fac4ew.low();
  int fac4ew_high = fac4ew.high();
  int fac1ns_low = fac1ns.low();
  int fac1ns_high = fac1ns.high();
  int fac2ns_low = fac2ns.low();
  int fac2ns_high = fac2ns.high();
  int fac3ns_low = fac3ns.low();
  int fac3ns_high = fac3ns.high();
  int fac4ns_low = fac4ns.low();
  int fac4ns_high = fac4ns.high();
  int fac1tb_low = fac1tb.low();
  int fac1tb_high = fac1tb.high();
  int fac2tb_low = fac2tb.low();
  int fac2tb_high = fac2tb.high();
  int fac3tb_low = fac3tb.low();
  int fac3tb_high = fac3tb.high();
  int fac4tb_low = fac4tb.low();
  int fac4tb_high = fac4tb.high();
  int e_shift_low = e_shift.low();
  int e_shift_high = e_shift.high();
  int w_shift_low = w_shift.low();
  int w_shift_high = w_shift.high();
  int n_shift_low = n_shift.low();
  int n_shift_high = n_shift.high();
  int s_shift_low = s_shift.low();
  int s_shift_high = s_shift.high();
  int t_shift_low = t_shift.low();
  int t_shift_high = t_shift.high();
  int b_shift_low = b_shift.low();
  int b_shift_high = b_shift.high();
  F_cellg( domainLow.get_pointer(),
         domainHigh.get_pointer(),
         indexLowU.get_pointer(),
         indexHighU.get_pointer(),
         indexLowV.get_pointer(),
         indexHighV.get_pointer(),
         indexLowW.get_pointer(),
         indexHighW.get_pointer(),
         indexLow.get_pointer(),
         indexHigh.get_pointer(),
         &sew_low, &sew_high, sew.get_objs(),
         &sns_low, &sns_high, sns.get_objs(),
         &stb_low, &stb_high, stb.get_objs(),
         &sewu_low, &sewu_high, sewu.get_objs(),
         &snsv_low, &snsv_high, snsv.get_objs(),
         &stbw_low, &stbw_high, stbw.get_objs(),
         &dxep_low, &dxep_high, dxep.get_objs(),
         &dynp_low, &dynp_high, dynp.get_objs(),
         &dztp_low, &dztp_high, dztp.get_objs(),
         &dxepu_low, &dxepu_high, dxepu.get_objs(),
         &dynpv_low, &dynpv_high, dynpv.get_objs(),
         &dztpw_low, &dztpw_high, dztpw.get_objs(),
         &dxpw_low, &dxpw_high, dxpw.get_objs(),
         &dyps_low, &dyps_high, dyps.get_objs(),
         &dzpb_low, &dzpb_high, dzpb.get_objs(),
         &dxpwu_low, &dxpwu_high, dxpwu.get_objs(),
         &dypsv_low, &dypsv_high, dypsv.get_objs(),
         &dzpbw_low, &dzpbw_high, dzpbw.get_objs(),
         &cee_low, &cee_high, cee.get_objs(),
         &cwe_low, &cwe_high, cwe.get_objs(),
         &cww_low, &cww_high, cww.get_objs(),
         &ceeu_low, &ceeu_high, ceeu.get_objs(),
         &cweu_low, &cweu_high, cweu.get_objs(),
         &cwwu_low, &cwwu_high, cwwu.get_objs(),
         &cnn_low, &cnn_high, cnn.get_objs(),
         &csn_low, &csn_high, csn.get_objs(),
         &css_low, &css_high, css.get_objs(),
         &cnnv_low, &cnnv_high, cnnv.get_objs(),
         &csnv_low, &csnv_high, csnv.get_objs(),
         &cssv_low, &cssv_high, cssv.get_objs(),
         &ctt_low, &ctt_high, ctt.get_objs(),
         &cbt_low, &cbt_high, cbt.get_objs(),
         &cbb_low, &cbb_high, cbb.get_objs(),
         &cttw_low, &cttw_high, cttw.get_objs(),
         &cbtw_low, &cbtw_high, cbtw.get_objs(),
         &cbbw_low, &cbbw_high, cbbw.get_objs(),
         &xx_low, &xx_high, xx.get_objs(),
         &xu_low, &xu_high, xu.get_objs(),
         &yy_low, &yy_high, yy.get_objs(),
         &yv_low, &yv_high, yv.get_objs(),
         &zz_low, &zz_high, zz.get_objs(),
         &zw_low, &zw_high, zw.get_objs(),
         &efac_low, &efac_high, efac.get_objs(),
         &wfac_low, &wfac_high, wfac.get_objs(),
         &nfac_low, &nfac_high, nfac.get_objs(),
         &sfac_low, &sfac_high, sfac.get_objs(),
         &tfac_low, &tfac_high, tfac.get_objs(),
         &bfac_low, &bfac_high, bfac.get_objs(),
         &fac1u_low, &fac1u_high, fac1u.get_objs(),
         &fac2u_low, &fac2u_high, fac2u.get_objs(),
         &fac3u_low, &fac3u_high, fac3u.get_objs(),
         &fac4u_low, &fac4u_high, fac4u.get_objs(),
         &fac1v_low, &fac1v_high, fac1v.get_objs(),
         &fac2v_low, &fac2v_high, fac2v.get_objs(),
         &fac3v_low, &fac3v_high, fac3v.get_objs(),
         &fac4v_low, &fac4v_high, fac4v.get_objs(),
         &fac1w_low, &fac1w_high, fac1w.get_objs(),
         &fac2w_low, &fac2w_high, fac2w.get_objs(),
         &fac3w_low, &fac3w_high, fac3w.get_objs(),
         &fac4w_low, &fac4w_high, fac4w.get_objs(),
         &iesdu_low, &iesdu_high, iesdu.get_objs(),
         &iwsdu_low, &iwsdu_high, iwsdu.get_objs(),
         &jnsdv_low, &jnsdv_high, jnsdv.get_objs(),
         &jssdv_low, &jssdv_high, jssdv.get_objs(),
         &ktsdw_low, &ktsdw_high, ktsdw.get_objs(),
         &kbsdw_low, &kbsdw_high, kbsdw.get_objs(),
         &fac1ew_low, &fac1ew_high, fac1ew.get_objs(),
         &fac2ew_low, &fac2ew_high, fac2ew.get_objs(),
         &fac3ew_low, &fac3ew_high, fac3ew.get_objs(),
         &fac4ew_low, &fac4ew_high, fac4ew.get_objs(),
         &fac1ns_low, &fac1ns_high, fac1ns.get_objs(),
         &fac2ns_low, &fac2ns_high, fac2ns.get_objs(),
         &fac3ns_low, &fac3ns_high, fac3ns.get_objs(),
         &fac4ns_low, &fac4ns_high, fac4ns.get_objs(),
         &fac1tb_low, &fac1tb_high, fac1tb.get_objs(),
         &fac2tb_low, &fac2tb_high, fac2tb.get_objs(),
         &fac3tb_low, &fac3tb_high, fac3tb.get_objs(),
         &fac4tb_low, &fac4tb_high, fac4tb.get_objs(),
         &e_shift_low, &e_shift_high, e_shift.get_objs(),
         &w_shift_low, &w_shift_high, w_shift.get_objs(),
         &n_shift_low, &n_shift_high, n_shift.get_objs(),
         &s_shift_low, &s_shift_high, s_shift.get_objs(),
         &t_shift_low, &t_shift_high, t_shift.get_objs(),
         &b_shift_low, &b_shift_high, b_shift.get_objs(),
         &xminus,
         &xplus,
         &yminus,
         &yplus,
         &zminus,
         &zplus );
}

#else /* !__cplusplus */

C This is the FORTRAN code portion of the file:

      subroutine CELLG(domainLow, domainHigh, indexLowU, indexHighU,
     & indexLowV, indexHighV, indexLowW, indexHighW, indexLow, 
     & indexHigh, sew_low, sew_high, sew, sns_low, sns_high, sns, 
     & stb_low, stb_high, stb, sewu_low, sewu_high, sewu, snsv_low, 
     & snsv_high, snsv, stbw_low, stbw_high, stbw, dxep_low, dxep_high,
     &  dxep, dynp_low, dynp_high, dynp, dztp_low, dztp_high, dztp, 
     & dxepu_low, dxepu_high, dxepu, dynpv_low, dynpv_high, dynpv, 
     & dztpw_low, dztpw_high, dztpw, dxpw_low, dxpw_high, dxpw, 
     & dyps_low, dyps_high, dyps, dzpb_low, dzpb_high, dzpb, dxpwu_low,
     &  dxpwu_high, dxpwu, dypsv_low, dypsv_high, dypsv, dzpbw_low, 
     & dzpbw_high, dzpbw, cee_low, cee_high, cee, cwe_low, cwe_high, 
     & cwe, cww_low, cww_high, cww, ceeu_low, ceeu_high, ceeu, cweu_low
     & , cweu_high, cweu, cwwu_low, cwwu_high, cwwu, cnn_low, cnn_high,
     &  cnn, csn_low, csn_high, csn, css_low, css_high, css, cnnv_low, 
     & cnnv_high, cnnv, csnv_low, csnv_high, csnv, cssv_low, cssv_high,
     &  cssv, ctt_low, ctt_high, ctt, cbt_low, cbt_high, cbt, cbb_low, 
     & cbb_high, cbb, cttw_low, cttw_high, cttw, cbtw_low, cbtw_high, 
     & cbtw, cbbw_low, cbbw_high, cbbw, xx_low, xx_high, xx, xu_low, 
     & xu_high, xu, yy_low, yy_high, yy, yv_low, yv_high, yv, zz_low, 
     & zz_high, zz, zw_low, zw_high, zw, efac_low, efac_high, efac, 
     & wfac_low, wfac_high, wfac, nfac_low, nfac_high, nfac, sfac_low, 
     & sfac_high, sfac, tfac_low, tfac_high, tfac, bfac_low, bfac_high,
     &  bfac, fac1u_low, fac1u_high, fac1u, fac2u_low, fac2u_high, 
     & fac2u, fac3u_low, fac3u_high, fac3u, fac4u_low, fac4u_high, 
     & fac4u, fac1v_low, fac1v_high, fac1v, fac2v_low, fac2v_high, 
     & fac2v, fac3v_low, fac3v_high, fac3v, fac4v_low, fac4v_high, 
     & fac4v, fac1w_low, fac1w_high, fac1w, fac2w_low, fac2w_high, 
     & fac2w, fac3w_low, fac3w_high, fac3w, fac4w_low, fac4w_high, 
     & fac4w, iesdu_low, iesdu_high, iesdu, iwsdu_low, iwsdu_high, 
     & iwsdu, jnsdv_low, jnsdv_high, jnsdv, jssdv_low, jssdv_high, 
     & jssdv, ktsdw_low, ktsdw_high, ktsdw, kbsdw_low, kbsdw_high, 
     & kbsdw, fac1ew_low, fac1ew_high, fac1ew, fac2ew_low, fac2ew_high,
     &  fac2ew, fac3ew_low, fac3ew_high, fac3ew, fac4ew_low, 
     & fac4ew_high, fac4ew, fac1ns_low, fac1ns_high, fac1ns, fac2ns_low
     & , fac2ns_high, fac2ns, fac3ns_low, fac3ns_high, fac3ns, 
     & fac4ns_low, fac4ns_high, fac4ns, fac1tb_low, fac1tb_high, fac1tb
     & , fac2tb_low, fac2tb_high, fac2tb, fac3tb_low, fac3tb_high, 
     & fac3tb, fac4tb_low, fac4tb_high, fac4tb, e_shift_low, 
     & e_shift_high, e_shift, w_shift_low, w_shift_high, w_shift, 
     & n_shift_low, n_shift_high, n_shift, s_shift_low, s_shift_high, 
     & s_shift, t_shift_low, t_shift_high, t_shift, b_shift_low, 
     & b_shift_high, b_shift, xminus, xplus, yminus, yplus, zminus, 
     & zplus)

      implicit none
      integer domainLow(3)
      integer domainHigh(3)
      integer indexLowU(3)
      integer indexHighU(3)
      integer indexLowV(3)
      integer indexHighV(3)
      integer indexLowW(3)
      integer indexHighW(3)
      integer indexLow(3)
      integer indexHigh(3)
      integer sew_low
      integer sew_high
      double precision sew(sew_low:sew_high)
      integer sns_low
      integer sns_high
      double precision sns(sns_low:sns_high)
      integer stb_low
      integer stb_high
      double precision stb(stb_low:stb_high)
      integer sewu_low
      integer sewu_high
      double precision sewu(sewu_low:sewu_high)
      integer snsv_low
      integer snsv_high
      double precision snsv(snsv_low:snsv_high)
      integer stbw_low
      integer stbw_high
      double precision stbw(stbw_low:stbw_high)
      integer dxep_low
      integer dxep_high
      double precision dxep(dxep_low:dxep_high)
      integer dynp_low
      integer dynp_high
      double precision dynp(dynp_low:dynp_high)
      integer dztp_low
      integer dztp_high
      double precision dztp(dztp_low:dztp_high)
      integer dxepu_low
      integer dxepu_high
      double precision dxepu(dxepu_low:dxepu_high)
      integer dynpv_low
      integer dynpv_high
      double precision dynpv(dynpv_low:dynpv_high)
      integer dztpw_low
      integer dztpw_high
      double precision dztpw(dztpw_low:dztpw_high)
      integer dxpw_low
      integer dxpw_high
      double precision dxpw(dxpw_low:dxpw_high)
      integer dyps_low
      integer dyps_high
      double precision dyps(dyps_low:dyps_high)
      integer dzpb_low
      integer dzpb_high
      double precision dzpb(dzpb_low:dzpb_high)
      integer dxpwu_low
      integer dxpwu_high
      double precision dxpwu(dxpwu_low:dxpwu_high)
      integer dypsv_low
      integer dypsv_high
      double precision dypsv(dypsv_low:dypsv_high)
      integer dzpbw_low
      integer dzpbw_high
      double precision dzpbw(dzpbw_low:dzpbw_high)
      integer cee_low
      integer cee_high
      double precision cee(cee_low:cee_high)
      integer cwe_low
      integer cwe_high
      double precision cwe(cwe_low:cwe_high)
      integer cww_low
      integer cww_high
      double precision cww(cww_low:cww_high)
      integer ceeu_low
      integer ceeu_high
      double precision ceeu(ceeu_low:ceeu_high)
      integer cweu_low
      integer cweu_high
      double precision cweu(cweu_low:cweu_high)
      integer cwwu_low
      integer cwwu_high
      double precision cwwu(cwwu_low:cwwu_high)
      integer cnn_low
      integer cnn_high
      double precision cnn(cnn_low:cnn_high)
      integer csn_low
      integer csn_high
      double precision csn(csn_low:csn_high)
      integer css_low
      integer css_high
      double precision css(css_low:css_high)
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
      integer cttw_low
      integer cttw_high
      double precision cttw(cttw_low:cttw_high)
      integer cbtw_low
      integer cbtw_high
      double precision cbtw(cbtw_low:cbtw_high)
      integer cbbw_low
      integer cbbw_high
      double precision cbbw(cbbw_low:cbbw_high)
      integer xx_low
      integer xx_high
      double precision xx(xx_low:xx_high)
      integer xu_low
      integer xu_high
      double precision xu(xu_low:xu_high)
      integer yy_low
      integer yy_high
      double precision yy(yy_low:yy_high)
      integer yv_low
      integer yv_high
      double precision yv(yv_low:yv_high)
      integer zz_low
      integer zz_high
      double precision zz(zz_low:zz_high)
      integer zw_low
      integer zw_high
      double precision zw(zw_low:zw_high)
      integer efac_low
      integer efac_high
      double precision efac(efac_low:efac_high)
      integer wfac_low
      integer wfac_high
      double precision wfac(wfac_low:wfac_high)
      integer nfac_low
      integer nfac_high
      double precision nfac(nfac_low:nfac_high)
      integer sfac_low
      integer sfac_high
      double precision sfac(sfac_low:sfac_high)
      integer tfac_low
      integer tfac_high
      double precision tfac(tfac_low:tfac_high)
      integer bfac_low
      integer bfac_high
      double precision bfac(bfac_low:bfac_high)
      integer fac1u_low
      integer fac1u_high
      double precision fac1u(fac1u_low:fac1u_high)
      integer fac2u_low
      integer fac2u_high
      double precision fac2u(fac2u_low:fac2u_high)
      integer fac3u_low
      integer fac3u_high
      double precision fac3u(fac3u_low:fac3u_high)
      integer fac4u_low
      integer fac4u_high
      double precision fac4u(fac4u_low:fac4u_high)
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
      integer fac1w_low
      integer fac1w_high
      double precision fac1w(fac1w_low:fac1w_high)
      integer fac2w_low
      integer fac2w_high
      double precision fac2w(fac2w_low:fac2w_high)
      integer fac3w_low
      integer fac3w_high
      double precision fac3w(fac3w_low:fac3w_high)
      integer fac4w_low
      integer fac4w_high
      double precision fac4w(fac4w_low:fac4w_high)
      integer iesdu_low
      integer iesdu_high
      integer iesdu(iesdu_low:iesdu_high)
      integer iwsdu_low
      integer iwsdu_high
      integer iwsdu(iwsdu_low:iwsdu_high)
      integer jnsdv_low
      integer jnsdv_high
      integer jnsdv(jnsdv_low:jnsdv_high)
      integer jssdv_low
      integer jssdv_high
      integer jssdv(jssdv_low:jssdv_high)
      integer ktsdw_low
      integer ktsdw_high
      integer ktsdw(ktsdw_low:ktsdw_high)
      integer kbsdw_low
      integer kbsdw_high
      integer kbsdw(kbsdw_low:kbsdw_high)
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
      integer fac1ns_low
      integer fac1ns_high
      double precision fac1ns(fac1ns_low:fac1ns_high)
      integer fac2ns_low
      integer fac2ns_high
      double precision fac2ns(fac2ns_low:fac2ns_high)
      integer fac3ns_low
      integer fac3ns_high
      double precision fac3ns(fac3ns_low:fac3ns_high)
      integer fac4ns_low
      integer fac4ns_high
      double precision fac4ns(fac4ns_low:fac4ns_high)
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
      integer e_shift_low
      integer e_shift_high
      integer e_shift(e_shift_low:e_shift_high)
      integer w_shift_low
      integer w_shift_high
      integer w_shift(w_shift_low:w_shift_high)
      integer n_shift_low
      integer n_shift_high
      integer n_shift(n_shift_low:n_shift_high)
      integer s_shift_low
      integer s_shift_high
      integer s_shift(s_shift_low:s_shift_high)
      integer t_shift_low
      integer t_shift_high
      integer t_shift(t_shift_low:t_shift_high)
      integer b_shift_low
      integer b_shift_high
      integer b_shift(b_shift_low:b_shift_high)
      logical*1 xminus
      logical*1 xplus
      logical*1 yminus
      logical*1 yplus
      logical*1 zminus
      logical*1 zplus
#endif /* __cplusplus */

#endif /* fspec_cellg */
