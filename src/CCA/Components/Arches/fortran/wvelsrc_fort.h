
#ifndef fspec_wvelsrc
#define fspec_wvelsrc

#ifdef __cplusplus

#include <CCA/Components/Arches/fortran/FortranNameMangle.h>

extern "C" void F_wvelsrc(int* idxLoW,
                        int* idxHiW,
                        int* ww_low_x, int* ww_low_y, int* ww_low_z, int* ww_high_x, int* ww_high_y, int* ww_high_z, double* ww_ptr,
                        int* old_ww_low_x, int* old_ww_low_y, int* old_ww_low_z, int* old_ww_high_x, int* old_ww_high_y, int* old_ww_high_z, double* old_ww_ptr,
                        int* su_low_x, int* su_low_y, int* su_low_z, int* su_high_x, int* su_high_y, int* su_high_z, double* su_ptr,
                        int* sp_low_x, int* sp_low_y, int* sp_low_z, int* sp_high_x, int* sp_high_y, int* sp_high_z, double* sp_ptr,
                        int* uu_low_x, int* uu_low_y, int* uu_low_z, int* uu_high_x, int* uu_high_y, int* uu_high_z, double* uu_ptr,
                        int* vv_low_x, int* vv_low_y, int* vv_low_z, int* vv_high_x, int* vv_high_y, int* vv_high_z, double* vv_ptr,
                        int* den_low_x, int* den_low_y, int* den_low_z, int* den_high_x, int* den_high_y, int* den_high_z, double* den_ptr,
                        int* vis_low_x, int* vis_low_y, int* vis_low_z, int* vis_high_x, int* vis_high_y, int* vis_high_z, double* vis_ptr,
                        int* old_den_low_x, int* old_den_low_y, int* old_den_low_z, int* old_den_high_x, int* old_den_high_y, int* old_den_high_z, double* old_den_ptr,
                        int* den_ref_low_x, int* den_ref_low_y, int* den_ref_low_z, int* den_ref_high_x, int* den_ref_high_y, int* den_ref_high_z, double* den_ref_ptr,
                        double* grav,
                        double* deltat,
                        int* cee_low, int* cee_high, double* cee_ptr,
                        int* cwe_low, int* cwe_high, double* cwe_ptr,
                        int* cww_low, int* cww_high, double* cww_ptr,
                        int* cnn_low, int* cnn_high, double* cnn_ptr,
                        int* csn_low, int* csn_high, double* csn_ptr,
                        int* css_low, int* css_high, double* css_ptr,
                        int* cttw_low, int* cttw_high, double* cttw_ptr,
                        int* cbtw_low, int* cbtw_high, double* cbtw_ptr,
                        int* cbbw_low, int* cbbw_high, double* cbbw_ptr,
                        int* sew_low, int* sew_high, double* sew_ptr,
                        int* sns_low, int* sns_high, double* sns_ptr,
                        int* stbw_low, int* stbw_high, double* stbw_ptr,
                        int* stb_low, int* stb_high, double* stb_ptr,
                        int* dzpb_low, int* dzpb_high, double* dzpb_ptr,
                        int* fac1w_low, int* fac1w_high, double* fac1w_ptr,
                        int* fac2w_low, int* fac2w_high, double* fac2w_ptr,
                        int* fac3w_low, int* fac3w_high, double* fac3w_ptr,
                        int* fac4w_low, int* fac4w_high, double* fac4w_ptr,
                        int* ktsdw_low, int* ktsdw_high, int* ktsdw_ptr,
                        int* kbsdw_low, int* kbsdw_high, int* kbsdw_ptr);

static void fort_wvelsrc( Uintah::IntVector & idxLoW,
                          Uintah::IntVector & idxHiW,
                          Uintah::constSFCZVariable<double> & ww,
                          Uintah::constSFCZVariable<double> & old_ww,
                          Uintah::SFCZVariable<double> & su,
                          Uintah::SFCZVariable<double> & sp,
                          Uintah::constSFCXVariable<double> & uu,
                          Uintah::constSFCYVariable<double> & vv,
                          Uintah::constCCVariable<double> & den,
                          Uintah::constCCVariable<double> & vis,
                          Uintah::constCCVariable<double> & old_den,
                          Uintah::constCCVariable<double> & den_ref,
                          double & grav,
                          double & deltat,
                          Uintah::OffsetArray1<double> & cee,
                          Uintah::OffsetArray1<double> & cwe,
                          Uintah::OffsetArray1<double> & cww,
                          Uintah::OffsetArray1<double> & cnn,
                          Uintah::OffsetArray1<double> & csn,
                          Uintah::OffsetArray1<double> & css,
                          Uintah::OffsetArray1<double> & cttw,
                          Uintah::OffsetArray1<double> & cbtw,
                          Uintah::OffsetArray1<double> & cbbw,
                          Uintah::OffsetArray1<double> & sew,
                          Uintah::OffsetArray1<double> & sns,
                          Uintah::OffsetArray1<double> & stbw,
                          Uintah::OffsetArray1<double> & stb,
                          Uintah::OffsetArray1<double> & dzpb,
                          Uintah::OffsetArray1<double> & fac1w,
                          Uintah::OffsetArray1<double> & fac2w,
                          Uintah::OffsetArray1<double> & fac3w,
                          Uintah::OffsetArray1<double> & fac4w,
                          Uintah::OffsetArray1<int> & ktsdw,
                          Uintah::OffsetArray1<int> & kbsdw )
{
  Uintah::IntVector ww_low = ww.getWindow()->getOffset();
  Uintah::IntVector ww_high = ww.getWindow()->getData()->size() + ww_low - Uintah::IntVector(1, 1, 1);
  int ww_low_x = ww_low.x();
  int ww_high_x = ww_high.x();
  int ww_low_y = ww_low.y();
  int ww_high_y = ww_high.y();
  int ww_low_z = ww_low.z();
  int ww_high_z = ww_high.z();
  Uintah::IntVector old_ww_low = old_ww.getWindow()->getOffset();
  Uintah::IntVector old_ww_high = old_ww.getWindow()->getData()->size() + old_ww_low - Uintah::IntVector(1, 1, 1);
  int old_ww_low_x = old_ww_low.x();
  int old_ww_high_x = old_ww_high.x();
  int old_ww_low_y = old_ww_low.y();
  int old_ww_high_y = old_ww_high.y();
  int old_ww_low_z = old_ww_low.z();
  int old_ww_high_z = old_ww_high.z();
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
  int cee_low = cee.low();
  int cee_high = cee.high();
  int cwe_low = cwe.low();
  int cwe_high = cwe.high();
  int cww_low = cww.low();
  int cww_high = cww.high();
  int cnn_low = cnn.low();
  int cnn_high = cnn.high();
  int csn_low = csn.low();
  int csn_high = csn.high();
  int css_low = css.low();
  int css_high = css.high();
  int cttw_low = cttw.low();
  int cttw_high = cttw.high();
  int cbtw_low = cbtw.low();
  int cbtw_high = cbtw.high();
  int cbbw_low = cbbw.low();
  int cbbw_high = cbbw.high();
  int sew_low = sew.low();
  int sew_high = sew.high();
  int sns_low = sns.low();
  int sns_high = sns.high();
  int stbw_low = stbw.low();
  int stbw_high = stbw.high();
  int stb_low = stb.low();
  int stb_high = stb.high();
  int dzpb_low = dzpb.low();
  int dzpb_high = dzpb.high();
  int fac1w_low = fac1w.low();
  int fac1w_high = fac1w.high();
  int fac2w_low = fac2w.low();
  int fac2w_high = fac2w.high();
  int fac3w_low = fac3w.low();
  int fac3w_high = fac3w.high();
  int fac4w_low = fac4w.low();
  int fac4w_high = fac4w.high();
  int ktsdw_low = ktsdw.low();
  int ktsdw_high = ktsdw.high();
  int kbsdw_low = kbsdw.low();
  int kbsdw_high = kbsdw.high();
  F_wvelsrc( idxLoW.get_pointer(),
           idxHiW.get_pointer(),
           &ww_low_x, &ww_low_y, &ww_low_z, &ww_high_x, &ww_high_y, &ww_high_z, const_cast<double*>(ww.getPointer()),
           &old_ww_low_x, &old_ww_low_y, &old_ww_low_z, &old_ww_high_x, &old_ww_high_y, &old_ww_high_z, const_cast<double*>(old_ww.getPointer()),
           &su_low_x, &su_low_y, &su_low_z, &su_high_x, &su_high_y, &su_high_z, su.getPointer(),
           &sp_low_x, &sp_low_y, &sp_low_z, &sp_high_x, &sp_high_y, &sp_high_z, sp.getPointer(),
           &uu_low_x, &uu_low_y, &uu_low_z, &uu_high_x, &uu_high_y, &uu_high_z, const_cast<double*>(uu.getPointer()),
           &vv_low_x, &vv_low_y, &vv_low_z, &vv_high_x, &vv_high_y, &vv_high_z, const_cast<double*>(vv.getPointer()),
           &den_low_x, &den_low_y, &den_low_z, &den_high_x, &den_high_y, &den_high_z, const_cast<double*>(den.getPointer()),
           &vis_low_x, &vis_low_y, &vis_low_z, &vis_high_x, &vis_high_y, &vis_high_z, const_cast<double*>(vis.getPointer()),
           &old_den_low_x, &old_den_low_y, &old_den_low_z, &old_den_high_x, &old_den_high_y, &old_den_high_z, const_cast<double*>(old_den.getPointer()),
           &den_ref_low_x, &den_ref_low_y, &den_ref_low_z, &den_ref_high_x, &den_ref_high_y, &den_ref_high_z, const_cast<double*>(den_ref.getPointer()),
           &grav,
           &deltat,
           &cee_low, &cee_high, cee.get_objs(),
           &cwe_low, &cwe_high, cwe.get_objs(),
           &cww_low, &cww_high, cww.get_objs(),
           &cnn_low, &cnn_high, cnn.get_objs(),
           &csn_low, &csn_high, csn.get_objs(),
           &css_low, &css_high, css.get_objs(),
           &cttw_low, &cttw_high, cttw.get_objs(),
           &cbtw_low, &cbtw_high, cbtw.get_objs(),
           &cbbw_low, &cbbw_high, cbbw.get_objs(),
           &sew_low, &sew_high, sew.get_objs(),
           &sns_low, &sns_high, sns.get_objs(),
           &stbw_low, &stbw_high, stbw.get_objs(),
           &stb_low, &stb_high, stb.get_objs(),
           &dzpb_low, &dzpb_high, dzpb.get_objs(),
           &fac1w_low, &fac1w_high, fac1w.get_objs(),
           &fac2w_low, &fac2w_high, fac2w.get_objs(),
           &fac3w_low, &fac3w_high, fac3w.get_objs(),
           &fac4w_low, &fac4w_high, fac4w.get_objs(),
           &ktsdw_low, &ktsdw_high, ktsdw.get_objs(),
           &kbsdw_low, &kbsdw_high, kbsdw.get_objs() );
}

#else /* !__cplusplus */

C This is the FORTRAN code portion of the file:

      subroutine WVELSRC(idxLoW, idxHiW, ww_low_x, ww_low_y, ww_low_z,
     & ww_high_x, ww_high_y, ww_high_z, ww, old_ww_low_x, old_ww_low_y,
     &  old_ww_low_z, old_ww_high_x, old_ww_high_y, old_ww_high_z, 
     & old_ww, su_low_x, su_low_y, su_low_z, su_high_x, su_high_y, 
     & su_high_z, su, sp_low_x, sp_low_y, sp_low_z, sp_high_x, 
     & sp_high_y, sp_high_z, sp, uu_low_x, uu_low_y, uu_low_z, 
     & uu_high_x, uu_high_y, uu_high_z, uu, vv_low_x, vv_low_y, 
     & vv_low_z, vv_high_x, vv_high_y, vv_high_z, vv, den_low_x, 
     & den_low_y, den_low_z, den_high_x, den_high_y, den_high_z, den, 
     & vis_low_x, vis_low_y, vis_low_z, vis_high_x, vis_high_y, 
     & vis_high_z, vis, old_den_low_x, old_den_low_y, old_den_low_z, 
     & old_den_high_x, old_den_high_y, old_den_high_z, old_den, 
     & den_ref_low_x, den_ref_low_y, den_ref_low_z, den_ref_high_x, 
     & den_ref_high_y, den_ref_high_z, den_ref, grav, deltat, cee_low, 
     & cee_high, cee, cwe_low, cwe_high, cwe, cww_low, cww_high, cww, 
     & cnn_low, cnn_high, cnn, csn_low, csn_high, csn, css_low, 
     & css_high, css, cttw_low, cttw_high, cttw, cbtw_low, cbtw_high, 
     & cbtw, cbbw_low, cbbw_high, cbbw, sew_low, sew_high, sew, sns_low
     & , sns_high, sns, stbw_low, stbw_high, stbw, stb_low, stb_high, 
     & stb, dzpb_low, dzpb_high, dzpb, fac1w_low, fac1w_high, fac1w, 
     & fac2w_low, fac2w_high, fac2w, fac3w_low, fac3w_high, fac3w, 
     & fac4w_low, fac4w_high, fac4w, ktsdw_low, ktsdw_high, ktsdw, 
     & kbsdw_low, kbsdw_high, kbsdw)

      implicit none
      integer idxLoW(3)
      integer idxHiW(3)
      integer ww_low_x, ww_low_y, ww_low_z, ww_high_x, ww_high_y, 
     & ww_high_z
      double precision ww(ww_low_x:ww_high_x, ww_low_y:ww_high_y, 
     & ww_low_z:ww_high_z)
      integer old_ww_low_x, old_ww_low_y, old_ww_low_z, old_ww_high_x, 
     & old_ww_high_y, old_ww_high_z
      double precision old_ww(old_ww_low_x:old_ww_high_x, old_ww_low_y:
     & old_ww_high_y, old_ww_low_z:old_ww_high_z)
      integer su_low_x, su_low_y, su_low_z, su_high_x, su_high_y, 
     & su_high_z
      double precision su(su_low_x:su_high_x, su_low_y:su_high_y, 
     & su_low_z:su_high_z)
      integer sp_low_x, sp_low_y, sp_low_z, sp_high_x, sp_high_y, 
     & sp_high_z
      double precision sp(sp_low_x:sp_high_x, sp_low_y:sp_high_y, 
     & sp_low_z:sp_high_z)
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
      integer cee_low
      integer cee_high
      double precision cee(cee_low:cee_high)
      integer cwe_low
      integer cwe_high
      double precision cwe(cwe_low:cwe_high)
      integer cww_low
      integer cww_high
      double precision cww(cww_low:cww_high)
      integer cnn_low
      integer cnn_high
      double precision cnn(cnn_low:cnn_high)
      integer csn_low
      integer csn_high
      double precision csn(csn_low:csn_high)
      integer css_low
      integer css_high
      double precision css(css_low:css_high)
      integer cttw_low
      integer cttw_high
      double precision cttw(cttw_low:cttw_high)
      integer cbtw_low
      integer cbtw_high
      double precision cbtw(cbtw_low:cbtw_high)
      integer cbbw_low
      integer cbbw_high
      double precision cbbw(cbbw_low:cbbw_high)
      integer sew_low
      integer sew_high
      double precision sew(sew_low:sew_high)
      integer sns_low
      integer sns_high
      double precision sns(sns_low:sns_high)
      integer stbw_low
      integer stbw_high
      double precision stbw(stbw_low:stbw_high)
      integer stb_low
      integer stb_high
      double precision stb(stb_low:stb_high)
      integer dzpb_low
      integer dzpb_high
      double precision dzpb(dzpb_low:dzpb_high)
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
      integer ktsdw_low
      integer ktsdw_high
      integer ktsdw(ktsdw_low:ktsdw_high)
      integer kbsdw_low
      integer kbsdw_high
      integer kbsdw(kbsdw_low:kbsdw_high)
#endif /* __cplusplus */

#ifndef PASS1
#  define PASS1(x) x/**/_low, x/**/_high, x
#endif

#ifndef PASS3
#  define PASS3A(x) x/**/_low_x, x/**/_low_y, x/**/_low_z,
#  define PASS3B(x) x/**/_high_x, x/**/_high_y, x/**/_high_z, x
#endif

#endif /* fspec_wvelsrc */
