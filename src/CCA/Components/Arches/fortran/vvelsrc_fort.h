
#ifndef fspec_vvelsrc
#define fspec_vvelsrc

#ifdef __cplusplus

#include <CCA/Components/Arches/fortran/FortranNameMangle.h>

extern "C" void F_vvelsrc(int* idxLoV,
                        int* idxHiV,
                        int* vv_low_x, int* vv_low_y, int* vv_low_z, int* vv_high_x, int* vv_high_y, int* vv_high_z, double* vv_ptr,
                        int* old_vv_low_x, int* old_vv_low_y, int* old_vv_low_z, int* old_vv_high_x, int* old_vv_high_y, int* old_vv_high_z, double* old_vv_ptr,
                        int* su_low_x, int* su_low_y, int* su_low_z, int* su_high_x, int* su_high_y, int* su_high_z, double* su_ptr,
                        int* sp_low_x, int* sp_low_y, int* sp_low_z, int* sp_high_x, int* sp_high_y, int* sp_high_z, double* sp_ptr,
                        int* uu_low_x, int* uu_low_y, int* uu_low_z, int* uu_high_x, int* uu_high_y, int* uu_high_z, double* uu_ptr,
                        int* ww_low_x, int* ww_low_y, int* ww_low_z, int* ww_high_x, int* ww_high_y, int* ww_high_z, double* ww_ptr,
                        int* den_low_x, int* den_low_y, int* den_low_z, int* den_high_x, int* den_high_y, int* den_high_z, double* den_ptr,
                        int* vis_low_x, int* vis_low_y, int* vis_low_z, int* vis_high_x, int* vis_high_y, int* vis_high_z, double* vis_ptr,
                        int* old_den_low_x, int* old_den_low_y, int* old_den_low_z, int* old_den_high_x, int* old_den_high_y, int* old_den_high_z, double* old_den_ptr,
                        int* den_ref_low_x, int* den_ref_low_y, int* den_ref_low_z, int* den_ref_high_x, int* den_ref_high_y, int* den_ref_high_z, double* den_ref_ptr,
                        double* grav,
                        double* deltat,
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
                        int* dyps_low, int* dyps_high, double* dyps_ptr,
                        int* fac1v_low, int* fac1v_high, double* fac1v_ptr,
                        int* fac2v_low, int* fac2v_high, double* fac2v_ptr,
                        int* fac3v_low, int* fac3v_high, double* fac3v_ptr,
                        int* fac4v_low, int* fac4v_high, double* fac4v_ptr,
                        int* jnsdv_low, int* jnsdv_high, int* jnsdv_ptr,
                        int* jssdv_low, int* jssdv_high, int* jssdv_ptr);

static void fort_vvelsrc( Uintah::IntVector & idxLoV,
                          Uintah::IntVector & idxHiV,
                          Uintah::constSFCYVariable<double> & vv,
                          Uintah::constSFCYVariable<double> & old_vv,
                          Uintah::SFCYVariable<double> & su,
                          Uintah::SFCYVariable<double> & sp,
                          Uintah::constSFCXVariable<double> & uu,
                          Uintah::constSFCZVariable<double> & ww,
                          Uintah::constCCVariable<double> & den,
                          Uintah::constCCVariable<double> & vis,
                          Uintah::constCCVariable<double> & old_den,
                          Uintah::constCCVariable<double> & den_ref,
                          double & grav,
                          double & deltat,
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
                          Uintah::OffsetArray1<double> & dyps,
                          Uintah::OffsetArray1<double> & fac1v,
                          Uintah::OffsetArray1<double> & fac2v,
                          Uintah::OffsetArray1<double> & fac3v,
                          Uintah::OffsetArray1<double> & fac4v,
                          Uintah::OffsetArray1<int> & jnsdv,
                          Uintah::OffsetArray1<int> & jssdv )
{
  Uintah::IntVector vv_low = vv.getWindow()->getOffset();
  Uintah::IntVector vv_high = vv.getWindow()->getData()->size() + vv_low - Uintah::IntVector(1, 1, 1);
  int vv_low_x = vv_low.x();
  int vv_high_x = vv_high.x();
  int vv_low_y = vv_low.y();
  int vv_high_y = vv_high.y();
  int vv_low_z = vv_low.z();
  int vv_high_z = vv_high.z();
  Uintah::IntVector old_vv_low = old_vv.getWindow()->getOffset();
  Uintah::IntVector old_vv_high = old_vv.getWindow()->getData()->size() + old_vv_low - Uintah::IntVector(1, 1, 1);
  int old_vv_low_x = old_vv_low.x();
  int old_vv_high_x = old_vv_high.x();
  int old_vv_low_y = old_vv_low.y();
  int old_vv_high_y = old_vv_high.y();
  int old_vv_low_z = old_vv_low.z();
  int old_vv_high_z = old_vv_high.z();
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
  int dyps_low = dyps.low();
  int dyps_high = dyps.high();
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
  F_vvelsrc( idxLoV.get_pointer(),
           idxHiV.get_pointer(),
           &vv_low_x, &vv_low_y, &vv_low_z, &vv_high_x, &vv_high_y, &vv_high_z, const_cast<double*>(vv.getPointer()),
           &old_vv_low_x, &old_vv_low_y, &old_vv_low_z, &old_vv_high_x, &old_vv_high_y, &old_vv_high_z, const_cast<double*>(old_vv.getPointer()),
           &su_low_x, &su_low_y, &su_low_z, &su_high_x, &su_high_y, &su_high_z, su.getPointer(),
           &sp_low_x, &sp_low_y, &sp_low_z, &sp_high_x, &sp_high_y, &sp_high_z, sp.getPointer(),
           &uu_low_x, &uu_low_y, &uu_low_z, &uu_high_x, &uu_high_y, &uu_high_z, const_cast<double*>(uu.getPointer()),
           &ww_low_x, &ww_low_y, &ww_low_z, &ww_high_x, &ww_high_y, &ww_high_z, const_cast<double*>(ww.getPointer()),
           &den_low_x, &den_low_y, &den_low_z, &den_high_x, &den_high_y, &den_high_z, const_cast<double*>(den.getPointer()),
           &vis_low_x, &vis_low_y, &vis_low_z, &vis_high_x, &vis_high_y, &vis_high_z, const_cast<double*>(vis.getPointer()),
           &old_den_low_x, &old_den_low_y, &old_den_low_z, &old_den_high_x, &old_den_high_y, &old_den_high_z, const_cast<double*>(old_den.getPointer()),
           &den_ref_low_x, &den_ref_low_y, &den_ref_low_z, &den_ref_high_x, &den_ref_high_y, &den_ref_high_z, const_cast<double*>(den_ref.getPointer()),
           &grav,
           &deltat,
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
           &dyps_low, &dyps_high, dyps.get_objs(),
           &fac1v_low, &fac1v_high, fac1v.get_objs(),
           &fac2v_low, &fac2v_high, fac2v.get_objs(),
           &fac3v_low, &fac3v_high, fac3v.get_objs(),
           &fac4v_low, &fac4v_high, fac4v.get_objs(),
           &jnsdv_low, &jnsdv_high, jnsdv.get_objs(),
           &jssdv_low, &jssdv_high, jssdv.get_objs() );
}

#else /* !__cplusplus */

C This is the FORTRAN code portion of the file:

      subroutine VVELSRC(idxLoV, idxHiV, vv_low_x, vv_low_y, vv_low_z,
     & vv_high_x, vv_high_y, vv_high_z, vv, old_vv_low_x, old_vv_low_y,
     &  old_vv_low_z, old_vv_high_x, old_vv_high_y, old_vv_high_z, 
     & old_vv, su_low_x, su_low_y, su_low_z, su_high_x, su_high_y, 
     & su_high_z, su, sp_low_x, sp_low_y, sp_low_z, sp_high_x, 
     & sp_high_y, sp_high_z, sp, uu_low_x, uu_low_y, uu_low_z, 
     & uu_high_x, uu_high_y, uu_high_z, uu, ww_low_x, ww_low_y, 
     & ww_low_z, ww_high_x, ww_high_y, ww_high_z, ww, den_low_x, 
     & den_low_y, den_low_z, den_high_x, den_high_y, den_high_z, den, 
     & vis_low_x, vis_low_y, vis_low_z, vis_high_x, vis_high_y, 
     & vis_high_z, vis, old_den_low_x, old_den_low_y, old_den_low_z, 
     & old_den_high_x, old_den_high_y, old_den_high_z, old_den, 
     & den_ref_low_x, den_ref_low_y, den_ref_low_z, den_ref_high_x, 
     & den_ref_high_y, den_ref_high_z, den_ref, grav, deltat, cee_low, 
     & cee_high, cee, cwe_low, cwe_high, cwe, cww_low, cww_high, cww, 
     & cnnv_low, cnnv_high, cnnv, csnv_low, csnv_high, csnv, cssv_low, 
     & cssv_high, cssv, ctt_low, ctt_high, ctt, cbt_low, cbt_high, cbt,
     &  cbb_low, cbb_high, cbb, sew_low, sew_high, sew, snsv_low, 
     & snsv_high, snsv, sns_low, sns_high, sns, stb_low, stb_high, stb,
     &  dyps_low, dyps_high, dyps, fac1v_low, fac1v_high, fac1v, 
     & fac2v_low, fac2v_high, fac2v, fac3v_low, fac3v_high, fac3v, 
     & fac4v_low, fac4v_high, fac4v, jnsdv_low, jnsdv_high, jnsdv, 
     & jssdv_low, jssdv_high, jssdv)

      implicit none
      integer idxLoV(3)
      integer idxHiV(3)
      integer vv_low_x, vv_low_y, vv_low_z, vv_high_x, vv_high_y, 
     & vv_high_z
      double precision vv(vv_low_x:vv_high_x, vv_low_y:vv_high_y, 
     & vv_low_z:vv_high_z)
      integer old_vv_low_x, old_vv_low_y, old_vv_low_z, old_vv_high_x, 
     & old_vv_high_y, old_vv_high_z
      double precision old_vv(old_vv_low_x:old_vv_high_x, old_vv_low_y:
     & old_vv_high_y, old_vv_low_z:old_vv_high_z)
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
      integer dyps_low
      integer dyps_high
      double precision dyps(dyps_low:dyps_high)
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
#endif /* __cplusplus */

#ifndef PASS1
#  define PASS1(x) x/**/_low, x/**/_high, x
#endif

#ifndef PASS3
#  define PASS3A(x) x/**/_low_x, x/**/_low_y, x/**/_low_z,
#  define PASS3B(x) x/**/_high_x, x/**/_high_y, x/**/_high_z, x
#endif

#endif /* fspec_vvelsrc */
