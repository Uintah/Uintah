
#ifndef fspec_uvelsrc
#define fspec_uvelsrc

#ifdef __cplusplus

#include <CCA/Components/Arches/fortran/FortranNameMangle.h>

extern "C" void F_uvelsrc(int* idxLoU,
                         int* idxHiU,
                         int* uu_low_x, int* uu_low_y, int* uu_low_z, int* uu_high_x, int* uu_high_y, int* uu_high_z, double* uu_ptr,
                         int* old_uu_low_x, int* old_uu_low_y, int* old_uu_low_z, int* old_uu_high_x, int* old_uu_high_y, int* old_uu_high_z, double* old_uu_ptr,
                         int* su_low_x, int* su_low_y, int* su_low_z, int* su_high_x, int* su_high_y, int* su_high_z, double* su_ptr,
                         int* sp_low_x, int* sp_low_y, int* sp_low_z, int* sp_high_x, int* sp_high_y, int* sp_high_z, double* sp_ptr,
                         int* vv_low_x, int* vv_low_y, int* vv_low_z, int* vv_high_x, int* vv_high_y, int* vv_high_z, double* vv_ptr,
                         int* ww_low_x, int* ww_low_y, int* ww_low_z, int* ww_high_x, int* ww_high_y, int* ww_high_z, double* ww_ptr,
                         int* den_low_x, int* den_low_y, int* den_low_z, int* den_high_x, int* den_high_y, int* den_high_z, double* den_ptr,
                         int* vis_low_x, int* vis_low_y, int* vis_low_z, int* vis_high_x, int* vis_high_y, int* vis_high_z, double* vis_ptr,
                         int* old_den_low_x, int* old_den_low_y, int* old_den_low_z, int* old_den_high_x, int* old_den_high_y, int* old_den_high_z, double* old_den_ptr,
                         int* den_ref_low_x, int* den_ref_low_y, int* den_ref_low_z, int* den_ref_high_x, int* den_ref_high_y, int* den_ref_high_z, double* den_ref_ptr,
                         double* grav,
                         double* deltat,
                         int* ceeu_low, int* ceeu_high, double* ceeu_ptr,
                         int* cweu_low, int* cweu_high, double* cweu_ptr,
                         int* cwwu_low, int* cwwu_high, double* cwwu_ptr,
                         int* cnn_low, int* cnn_high, double* cnn_ptr,
                         int* csn_low, int* csn_high, double* csn_ptr,
                         int* css_low, int* css_high, double* css_ptr,
                         int* ctt_low, int* ctt_high, double* ctt_ptr,
                         int* cbt_low, int* cbt_high, double* cbt_ptr,
                         int* cbb_low, int* cbb_high, double* cbb_ptr,
                         int* sewu_low, int* sewu_high, double* sewu_ptr,
                         int* sew_low, int* sew_high, double* sew_ptr,
                         int* sns_low, int* sns_high, double* sns_ptr,
                         int* stb_low, int* stb_high, double* stb_ptr,
                         int* dxpw_low, int* dxpw_high, double* dxpw_ptr,
                         int* fac1u_low, int* fac1u_high, double* fac1u_ptr,
                         int* fac2u_low, int* fac2u_high, double* fac2u_ptr,
                         int* fac3u_low, int* fac3u_high, double* fac3u_ptr,
                         int* fac4u_low, int* fac4u_high, double* fac4u_ptr,
                         int* iesdu_low, int* iesdu_high, int* iesdu_ptr,
                         int* iwsdu_low, int* iwsdu_high, int* iwsdu_ptr);

static void fort_uvelsrc( Uintah::IntVector & idxLoU,
                          Uintah::IntVector & idxHiU,
                          Uintah::constSFCXVariable<double> & uu,
                          Uintah::constSFCXVariable<double> & old_uu,
                          Uintah::SFCXVariable<double> & su,
                          Uintah::SFCXVariable<double> & sp,
                          Uintah::constSFCYVariable<double> & vv,
                          Uintah::constSFCZVariable<double> & ww,
                          Uintah::constCCVariable<double> & den,
                          Uintah::constCCVariable<double> & vis,
                          Uintah::constCCVariable<double> & old_den,
                          Uintah::constCCVariable<double> & den_ref,
                          double & grav,
                          double & deltat,
                          Uintah::OffsetArray1<double> & ceeu,
                          Uintah::OffsetArray1<double> & cweu,
                          Uintah::OffsetArray1<double> & cwwu,
                          Uintah::OffsetArray1<double> & cnn,
                          Uintah::OffsetArray1<double> & csn,
                          Uintah::OffsetArray1<double> & css,
                          Uintah::OffsetArray1<double> & ctt,
                          Uintah::OffsetArray1<double> & cbt,
                          Uintah::OffsetArray1<double> & cbb,
                          Uintah::OffsetArray1<double> & sewu,
                          Uintah::OffsetArray1<double> & sew,
                          Uintah::OffsetArray1<double> & sns,
                          Uintah::OffsetArray1<double> & stb,
                          Uintah::OffsetArray1<double> & dxpw,
                          Uintah::OffsetArray1<double> & fac1u,
                          Uintah::OffsetArray1<double> & fac2u,
                          Uintah::OffsetArray1<double> & fac3u,
                          Uintah::OffsetArray1<double> & fac4u,
                          Uintah::OffsetArray1<int> & iesdu,
                          Uintah::OffsetArray1<int> & iwsdu )
{
  Uintah::IntVector uu_low = uu.getWindow()->getOffset();
  Uintah::IntVector uu_high = uu.getWindow()->getData()->size() + uu_low - Uintah::IntVector(1, 1, 1);
  int uu_low_x = uu_low.x();
  int uu_high_x = uu_high.x();
  int uu_low_y = uu_low.y();
  int uu_high_y = uu_high.y();
  int uu_low_z = uu_low.z();
  int uu_high_z = uu_high.z();
  Uintah::IntVector old_uu_low = old_uu.getWindow()->getOffset();
  Uintah::IntVector old_uu_high = old_uu.getWindow()->getData()->size() + old_uu_low - Uintah::IntVector(1, 1, 1);
  int old_uu_low_x = old_uu_low.x();
  int old_uu_high_x = old_uu_high.x();
  int old_uu_low_y = old_uu_low.y();
  int old_uu_high_y = old_uu_high.y();
  int old_uu_low_z = old_uu_low.z();
  int old_uu_high_z = old_uu_high.z();
  Uintah::IntVector su_low = su.getWindow()->getOffset();
  Uintah::IntVector su_high = su.getWindow()->getData()->size() + su_low - Uintah::IntVector(1, 1, 1);
  int su_low_x = su_low.x();
  int su_high_x = su_high.x();
  int su_low_y = su_low.y();
  int su_high_y = su_high.y();
  int su_low_z = su_low.z();
  int su_high_z = su_high.z();
  Uintah::IntVector sp_low = sp.getWindow()->getOffset();
  Uintah::IntVector sp_high = sp.getWindow()->getData()->size() + sp_low - Uintah::IntVector(1, 1, 1);
  int sp_low_x = sp_low.x();
  int sp_high_x = sp_high.x();
  int sp_low_y = sp_low.y();
  int sp_high_y = sp_high.y();
  int sp_low_z = sp_low.z();
  int sp_high_z = sp_high.z();
  Uintah::IntVector vv_low = vv.getWindow()->getOffset();
  Uintah::IntVector vv_high = vv.getWindow()->getData()->size() + vv_low - Uintah::IntVector(1, 1, 1);
  int vv_low_x = vv_low.x();
  int vv_high_x = vv_high.x();
  int vv_low_y = vv_low.y();
  int vv_high_y = vv_high.y();
  int vv_low_z = vv_low.z();
  int vv_high_z = vv_high.z();
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
  Uintah::IntVector old_den_low = old_den.getWindow()->getOffset();
  Uintah::IntVector old_den_high = old_den.getWindow()->getData()->size() + old_den_low - Uintah::IntVector(1, 1, 1);
  int old_den_low_x = old_den_low.x();
  int old_den_high_x = old_den_high.x();
  int old_den_low_y = old_den_low.y();
  int old_den_high_y = old_den_high.y();
  int old_den_low_z = old_den_low.z();
  int old_den_high_z = old_den_high.z();
  Uintah::IntVector den_ref_low = den_ref.getWindow()->getOffset();
  Uintah::IntVector den_ref_high = den_ref.getWindow()->getData()->size() + den_ref_low - Uintah::IntVector(1, 1, 1);
  int den_ref_low_x = den_ref_low.x();
  int den_ref_high_x = den_ref_high.x();
  int den_ref_low_y = den_ref_low.y();
  int den_ref_high_y = den_ref_high.y();
  int den_ref_low_z = den_ref_low.z();
  int den_ref_high_z = den_ref_high.z();
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
  int ctt_low = ctt.low();
  int ctt_high = ctt.high();
  int cbt_low = cbt.low();
  int cbt_high = cbt.high();
  int cbb_low = cbb.low();
  int cbb_high = cbb.high();
  int sewu_low = sewu.low();
  int sewu_high = sewu.high();
  int sew_low = sew.low();
  int sew_high = sew.high();
  int sns_low = sns.low();
  int sns_high = sns.high();
  int stb_low = stb.low();
  int stb_high = stb.high();
  int dxpw_low = dxpw.low();
  int dxpw_high = dxpw.high();
  int fac1u_low = fac1u.low();
  int fac1u_high = fac1u.high();
  int fac2u_low = fac2u.low();
  int fac2u_high = fac2u.high();
  int fac3u_low = fac3u.low();
  int fac3u_high = fac3u.high();
  int fac4u_low = fac4u.low();
  int fac4u_high = fac4u.high();
  int iesdu_low = iesdu.low();
  int iesdu_high = iesdu.high();
  int iwsdu_low = iwsdu.low();
  int iwsdu_high = iwsdu.high();
  F_uvelsrc( idxLoU.get_pointer(),
            idxHiU.get_pointer(),
            &uu_low_x, &uu_low_y, &uu_low_z, &uu_high_x, &uu_high_y, &uu_high_z, const_cast<double*>(uu.getPointer()),
            &old_uu_low_x, &old_uu_low_y, &old_uu_low_z, &old_uu_high_x, &old_uu_high_y, &old_uu_high_z, const_cast<double*>(old_uu.getPointer()),
            &su_low_x, &su_low_y, &su_low_z, &su_high_x, &su_high_y, &su_high_z, su.getPointer(),
            &sp_low_x, &sp_low_y, &sp_low_z, &sp_high_x, &sp_high_y, &sp_high_z, sp.getPointer(),
            &vv_low_x, &vv_low_y, &vv_low_z, &vv_high_x, &vv_high_y, &vv_high_z, const_cast<double*>(vv.getPointer()),
            &ww_low_x, &ww_low_y, &ww_low_z, &ww_high_x, &ww_high_y, &ww_high_z, const_cast<double*>(ww.getPointer()),
            &den_low_x, &den_low_y, &den_low_z, &den_high_x, &den_high_y, &den_high_z, const_cast<double*>(den.getPointer()),
            &vis_low_x, &vis_low_y, &vis_low_z, &vis_high_x, &vis_high_y, &vis_high_z, const_cast<double*>(vis.getPointer()),
            &old_den_low_x, &old_den_low_y, &old_den_low_z, &old_den_high_x, &old_den_high_y, &old_den_high_z, const_cast<double*>(old_den.getPointer()),
            &den_ref_low_x, &den_ref_low_y, &den_ref_low_z, &den_ref_high_x, &den_ref_high_y, &den_ref_high_z, const_cast<double*>(den_ref.getPointer()),
            &grav,
            &deltat,
            &ceeu_low, &ceeu_high, ceeu.get_objs(),
            &cweu_low, &cweu_high, cweu.get_objs(),
            &cwwu_low, &cwwu_high, cwwu.get_objs(),
            &cnn_low, &cnn_high, cnn.get_objs(),
            &csn_low, &csn_high, csn.get_objs(),
            &css_low, &css_high, css.get_objs(),
            &ctt_low, &ctt_high, ctt.get_objs(),
            &cbt_low, &cbt_high, cbt.get_objs(),
            &cbb_low, &cbb_high, cbb.get_objs(),
            &sewu_low, &sewu_high, sewu.get_objs(),
            &sew_low, &sew_high, sew.get_objs(),
            &sns_low, &sns_high, sns.get_objs(),
            &stb_low, &stb_high, stb.get_objs(),
            &dxpw_low, &dxpw_high, dxpw.get_objs(),
            &fac1u_low, &fac1u_high, fac1u.get_objs(),
            &fac2u_low, &fac2u_high, fac2u.get_objs(),
            &fac3u_low, &fac3u_high, fac3u.get_objs(),
            &fac4u_low, &fac4u_high, fac4u.get_objs(),
            &iesdu_low, &iesdu_high, iesdu.get_objs(),
            &iwsdu_low, &iwsdu_high, iwsdu.get_objs() );
}

#else /* !__cplusplus */

C This is the FORTRAN code portion of the file:

      subroutine UVELSRC(idxLoU, idxHiU, uu_low_x, uu_low_y, uu_low_z,
     & uu_high_x, uu_high_y, uu_high_z, uu, old_uu_low_x, old_uu_low_y,
     &  old_uu_low_z, old_uu_high_x, old_uu_high_y, old_uu_high_z, 
     & old_uu, su_low_x, su_low_y, su_low_z, su_high_x, su_high_y, 
     & su_high_z, su, sp_low_x, sp_low_y, sp_low_z, sp_high_x, 
     & sp_high_y, sp_high_z, sp, vv_low_x, vv_low_y, vv_low_z, 
     & vv_high_x, vv_high_y, vv_high_z, vv, ww_low_x, ww_low_y, 
     & ww_low_z, ww_high_x, ww_high_y, ww_high_z, ww, den_low_x, 
     & den_low_y, den_low_z, den_high_x, den_high_y, den_high_z, den, 
     & vis_low_x, vis_low_y, vis_low_z, vis_high_x, vis_high_y, 
     & vis_high_z, vis, old_den_low_x, old_den_low_y, old_den_low_z, 
     & old_den_high_x, old_den_high_y, old_den_high_z, old_den, 
     & den_ref_low_x, den_ref_low_y, den_ref_low_z, den_ref_high_x, 
     & den_ref_high_y, den_ref_high_z, den_ref, grav, deltat, ceeu_low,
     &  ceeu_high, ceeu, cweu_low, cweu_high, cweu, cwwu_low, cwwu_high
     & , cwwu, cnn_low, cnn_high, cnn, csn_low, csn_high, csn, css_low,
     &  css_high, css, ctt_low, ctt_high, ctt, cbt_low, cbt_high, cbt, 
     & cbb_low, cbb_high, cbb, sewu_low, sewu_high, sewu, sew_low, 
     & sew_high, sew, sns_low, sns_high, sns, stb_low, stb_high, stb, 
     & dxpw_low, dxpw_high, dxpw, fac1u_low, fac1u_high, fac1u, 
     & fac2u_low, fac2u_high, fac2u, fac3u_low, fac3u_high, fac3u, 
     & fac4u_low, fac4u_high, fac4u, iesdu_low, iesdu_high, iesdu, 
     & iwsdu_low, iwsdu_high, iwsdu)

      implicit none
      integer idxLoU(3)
      integer idxHiU(3)
      integer uu_low_x, uu_low_y, uu_low_z, uu_high_x, uu_high_y, 
     & uu_high_z
      double precision uu(uu_low_x:uu_high_x, uu_low_y:uu_high_y, 
     & uu_low_z:uu_high_z)
      integer old_uu_low_x, old_uu_low_y, old_uu_low_z, old_uu_high_x, 
     & old_uu_high_y, old_uu_high_z
      double precision old_uu(old_uu_low_x:old_uu_high_x, old_uu_low_y:
     & old_uu_high_y, old_uu_low_z:old_uu_high_z)
      integer su_low_x, su_low_y, su_low_z, su_high_x, su_high_y, 
     & su_high_z
      double precision su(su_low_x:su_high_x, su_low_y:su_high_y, 
     & su_low_z:su_high_z)
      integer sp_low_x, sp_low_y, sp_low_z, sp_high_x, sp_high_y, 
     & sp_high_z
      double precision sp(sp_low_x:sp_high_x, sp_low_y:sp_high_y, 
     & sp_low_z:sp_high_z)
      integer vv_low_x, vv_low_y, vv_low_z, vv_high_x, vv_high_y, 
     & vv_high_z
      double precision vv(vv_low_x:vv_high_x, vv_low_y:vv_high_y, 
     & vv_low_z:vv_high_z)
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
      integer old_den_low_x, old_den_low_y, old_den_low_z, 
     & old_den_high_x, old_den_high_y, old_den_high_z
      double precision old_den(old_den_low_x:old_den_high_x, 
     & old_den_low_y:old_den_high_y, old_den_low_z:old_den_high_z)
      integer den_ref_low_x, den_ref_low_y, den_ref_low_z, 
     & den_ref_high_x, den_ref_high_y, den_ref_high_z
      double precision den_ref(den_ref_low_x:den_ref_high_x, 
     & den_ref_low_y:den_ref_high_y, den_ref_low_z:den_ref_high_z)
      double precision grav
      double precision deltat
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
      integer ctt_low
      integer ctt_high
      double precision ctt(ctt_low:ctt_high)
      integer cbt_low
      integer cbt_high
      double precision cbt(cbt_low:cbt_high)
      integer cbb_low
      integer cbb_high
      double precision cbb(cbb_low:cbb_high)
      integer sewu_low
      integer sewu_high
      double precision sewu(sewu_low:sewu_high)
      integer sew_low
      integer sew_high
      double precision sew(sew_low:sew_high)
      integer sns_low
      integer sns_high
      double precision sns(sns_low:sns_high)
      integer stb_low
      integer stb_high
      double precision stb(stb_low:stb_high)
      integer dxpw_low
      integer dxpw_high
      double precision dxpw(dxpw_low:dxpw_high)
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
      integer iesdu_low
      integer iesdu_high
      integer iesdu(iesdu_low:iesdu_high)
      integer iwsdu_low
      integer iwsdu_high
      integer iwsdu(iwsdu_low:iwsdu_high)
#endif /* __cplusplus */

#endif /* fspec_uvelsrc */

#ifndef PASS1
#  define PASS1(x) x/**/_low, x/**/_high, x
#endif

#ifndef PASS3
#  define PASS3A(x) x/**/_low_x, x/**/_low_y, x/**/_low_z, 
#  define PASS3B(x) x/**/_high_x, x/**/_high_y, x/**/_high_z, x
#endif
