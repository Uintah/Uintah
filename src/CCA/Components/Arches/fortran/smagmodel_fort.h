
#ifndef fspec_smagmodel
#define fspec_smagmodel

#ifdef __cplusplus

#include <CCA/Components/Arches/fortran/FortranNameMangle.h>

extern "C" void F_smagmodel(int* uu_low_x, int* uu_low_y, int* uu_low_z, int* uu_high_x, int* uu_high_y, int* uu_high_z, double* uu_ptr,
                           int* vv_low_x, int* vv_low_y, int* vv_low_z, int* vv_high_x, int* vv_high_y, int* vv_high_z, double* vv_ptr,
                           int* ww_low_x, int* ww_low_y, int* ww_low_z, int* ww_high_x, int* ww_high_y, int* ww_high_z, double* ww_ptr,
                           int* ucc_low_x, int* ucc_low_y, int* ucc_low_z, int* ucc_high_x, int* ucc_high_y, int* ucc_high_z, double* ucc_ptr,
                           int* vcc_low_x, int* vcc_low_y, int* vcc_low_z, int* vcc_high_x, int* vcc_high_y, int* vcc_high_z, double* vcc_ptr,
                           int* wcc_low_x, int* wcc_low_y, int* wcc_low_z, int* wcc_high_x, int* wcc_high_y, int* wcc_high_z, double* wcc_ptr,
                           int* den_low_x, int* den_low_y, int* den_low_z, int* den_high_x, int* den_high_y, int* den_high_z, double* den_ptr,
                           int* vis_low_x, int* vis_low_y, int* vis_low_z, int* vis_high_x, int* vis_high_y, int* vis_high_z, double* vis_ptr,
                           int* domlo,
                           int* domhi,
                           int* sew_low, int* sew_high, double* sew_ptr,
                           int* sns_low, int* sns_high, double* sns_ptr,
                           int* stb_low, int* stb_high, double* stb_ptr,
                           double* viscos,
                           double* CF,
                           double* fac_mesh,
                           double* filterl);

static void fort_smagmodel( Uintah::constSFCXVariable<double> & uu,
                            Uintah::constSFCYVariable<double> & vv,
                            Uintah::constSFCZVariable<double> & ww,
                            Uintah::constCCVariable<double> & ucc,
                            Uintah::constCCVariable<double> & vcc,
                            Uintah::constCCVariable<double> & wcc,
                            Uintah::constCCVariable<double> & den,
                            Uintah::CCVariable<double> & vis,
                            Uintah::IntVector & domlo,
                            Uintah::IntVector & domhi,
                            Uintah::OffsetArray1<double> & sew,
                            Uintah::OffsetArray1<double> & sns,
                            Uintah::OffsetArray1<double> & stb,
                            double & viscos,
                            double & CF,
                            double & fac_mesh,
                            double & filterl )
{
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
  Uintah::IntVector ww_low = ww.getWindow()->getOffset();
  Uintah::IntVector ww_high = ww.getWindow()->getData()->size() + ww_low - Uintah::IntVector(1, 1, 1);
  int ww_low_x = ww_low.x();
  int ww_high_x = ww_high.x();
  int ww_low_y = ww_low.y();
  int ww_high_y = ww_high.y();
  int ww_low_z = ww_low.z();
  int ww_high_z = ww_high.z();
  Uintah::IntVector ucc_low = ucc.getWindow()->getOffset();
  Uintah::IntVector ucc_high = ucc.getWindow()->getData()->size() + ucc_low - Uintah::IntVector(1, 1, 1);
  int ucc_low_x = ucc_low.x();
  int ucc_high_x = ucc_high.x();
  int ucc_low_y = ucc_low.y();
  int ucc_high_y = ucc_high.y();
  int ucc_low_z = ucc_low.z();
  int ucc_high_z = ucc_high.z();
  Uintah::IntVector vcc_low = vcc.getWindow()->getOffset();
  Uintah::IntVector vcc_high = vcc.getWindow()->getData()->size() + vcc_low - Uintah::IntVector(1, 1, 1);
  int vcc_low_x = vcc_low.x();
  int vcc_high_x = vcc_high.x();
  int vcc_low_y = vcc_low.y();
  int vcc_high_y = vcc_high.y();
  int vcc_low_z = vcc_low.z();
  int vcc_high_z = vcc_high.z();
  Uintah::IntVector wcc_low = wcc.getWindow()->getOffset();
  Uintah::IntVector wcc_high = wcc.getWindow()->getData()->size() + wcc_low - Uintah::IntVector(1, 1, 1);
  int wcc_low_x = wcc_low.x();
  int wcc_high_x = wcc_high.x();
  int wcc_low_y = wcc_low.y();
  int wcc_high_y = wcc_high.y();
  int wcc_low_z = wcc_low.z();
  int wcc_high_z = wcc_high.z();
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
  int sew_low = sew.low();
  int sew_high = sew.high();
  int sns_low = sns.low();
  int sns_high = sns.high();
  int stb_low = stb.low();
  int stb_high = stb.high();
  F_smagmodel( &uu_low_x, &uu_low_y, &uu_low_z, &uu_high_x, &uu_high_y, &uu_high_z, const_cast<double*>(uu.getPointer()),
              &vv_low_x, &vv_low_y, &vv_low_z, &vv_high_x, &vv_high_y, &vv_high_z, const_cast<double*>(vv.getPointer()),
              &ww_low_x, &ww_low_y, &ww_low_z, &ww_high_x, &ww_high_y, &ww_high_z, const_cast<double*>(ww.getPointer()),
              &ucc_low_x, &ucc_low_y, &ucc_low_z, &ucc_high_x, &ucc_high_y, &ucc_high_z, const_cast<double*>(ucc.getPointer()),
              &vcc_low_x, &vcc_low_y, &vcc_low_z, &vcc_high_x, &vcc_high_y, &vcc_high_z, const_cast<double*>(vcc.getPointer()),
              &wcc_low_x, &wcc_low_y, &wcc_low_z, &wcc_high_x, &wcc_high_y, &wcc_high_z, const_cast<double*>(wcc.getPointer()),
              &den_low_x, &den_low_y, &den_low_z, &den_high_x, &den_high_y, &den_high_z, const_cast<double*>(den.getPointer()),
              &vis_low_x, &vis_low_y, &vis_low_z, &vis_high_x, &vis_high_y, &vis_high_z, vis.getPointer(),
              domlo.get_pointer(),
              domhi.get_pointer(),
              &sew_low, &sew_high, sew.get_objs(),
              &sns_low, &sns_high, sns.get_objs(),
              &stb_low, &stb_high, stb.get_objs(),
              &viscos,
              &CF,
              &fac_mesh,
              &filterl );
}

#else /* !__cplusplus */

C This is the FORTRAN code portion of the file:

      subroutine SMAGMODEL(uu_low_x, uu_low_y, uu_low_z, uu_high_x,
     & uu_high_y, uu_high_z, uu, vv_low_x, vv_low_y, vv_low_z, 
     & vv_high_x, vv_high_y, vv_high_z, vv, ww_low_x, ww_low_y, 
     & ww_low_z, ww_high_x, ww_high_y, ww_high_z, ww, ucc_low_x, 
     & ucc_low_y, ucc_low_z, ucc_high_x, ucc_high_y, ucc_high_z, ucc, 
     & vcc_low_x, vcc_low_y, vcc_low_z, vcc_high_x, vcc_high_y, 
     & vcc_high_z, vcc, wcc_low_x, wcc_low_y, wcc_low_z, wcc_high_x, 
     & wcc_high_y, wcc_high_z, wcc, den_low_x, den_low_y, den_low_z, 
     & den_high_x, den_high_y, den_high_z, den, vis_low_x, vis_low_y, 
     & vis_low_z, vis_high_x, vis_high_y, vis_high_z, vis, domlo, domhi
     & , sew_low, sew_high, sew, sns_low, sns_high, sns, stb_low, 
     & stb_high, stb, viscos, CF, fac_mesh, filterl)

      implicit none
      integer uu_low_x, uu_low_y, uu_low_z, uu_high_x, uu_high_y, 
     & uu_high_z
      double precision uu(uu_low_x:uu_high_x, uu_low_y:uu_high_y, 
     & uu_low_z:uu_high_z)
      integer vv_low_x, vv_low_y, vv_low_z, vv_high_x, vv_high_y, 
     & vv_high_z
      double precision vv(vv_low_x:vv_high_x, vv_low_y:vv_high_y, 
     & vv_low_z:vv_high_z)
      integer ww_low_x, ww_low_y, ww_low_z, ww_high_x, ww_high_y, 
     & ww_high_z
      double precision ww(ww_low_x:ww_high_x, ww_low_y:ww_high_y, 
     & ww_low_z:ww_high_z)
      integer ucc_low_x, ucc_low_y, ucc_low_z, ucc_high_x, ucc_high_y, 
     & ucc_high_z
      double precision ucc(ucc_low_x:ucc_high_x, ucc_low_y:ucc_high_y, 
     & ucc_low_z:ucc_high_z)
      integer vcc_low_x, vcc_low_y, vcc_low_z, vcc_high_x, vcc_high_y, 
     & vcc_high_z
      double precision vcc(vcc_low_x:vcc_high_x, vcc_low_y:vcc_high_y, 
     & vcc_low_z:vcc_high_z)
      integer wcc_low_x, wcc_low_y, wcc_low_z, wcc_high_x, wcc_high_y, 
     & wcc_high_z
      double precision wcc(wcc_low_x:wcc_high_x, wcc_low_y:wcc_high_y, 
     & wcc_low_z:wcc_high_z)
      integer den_low_x, den_low_y, den_low_z, den_high_x, den_high_y, 
     & den_high_z
      double precision den(den_low_x:den_high_x, den_low_y:den_high_y, 
     & den_low_z:den_high_z)
      integer vis_low_x, vis_low_y, vis_low_z, vis_high_x, vis_high_y, 
     & vis_high_z
      double precision vis(vis_low_x:vis_high_x, vis_low_y:vis_high_y, 
     & vis_low_z:vis_high_z)
      integer domlo(3)
      integer domhi(3)
      integer sew_low
      integer sew_high
      double precision sew(sew_low:sew_high)
      integer sns_low
      integer sns_high
      double precision sns(sns_low:sns_high)
      integer stb_low
      integer stb_high
      double precision stb(stb_low:stb_high)
      double precision viscos
      double precision CF
      double precision fac_mesh
      double precision filterl
#endif /* __cplusplus */

#endif /* fspec_smagmodel */

#ifndef PASS1
#  define PASS1(x) x/**/_low, x/**/_high, x
#endif

#ifndef PASS3
#  define PASS3A(x) x/**/_low_x, x/**/_low_y, x/**/_low_z, 
#  define PASS3B(x) x/**/_high_x, x/**/_high_y, x/**/_high_z, x
#endif
