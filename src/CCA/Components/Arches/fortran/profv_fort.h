
#ifndef fspec_profv
#define fspec_profv

#ifdef __cplusplus

extern "C" void F_profv(int* uu_low_x, int* uu_low_y, int* uu_low_z, int* uu_high_x, int* uu_high_y, int* uu_high_z, double* uu_ptr,
                       int* vv_low_x, int* vv_low_y, int* vv_low_z, int* vv_high_x, int* vv_high_y, int* vv_high_z, double* vv_ptr,
                       int* ww_low_x, int* ww_low_y, int* ww_low_z, int* ww_high_x, int* ww_high_y, int* ww_high_z, double* ww_ptr,
                       int* idxLo,
                       int* idxHi,
                       int* pcell_low_x, int* pcell_low_y, int* pcell_low_z, int* pcell_high_x, int* pcell_high_y, int* pcell_high_z, int* pcell_ptr,
                       double* areapr,
                       int* pfield,
                       double* flowpr,
                       double* inletvel,
                       double* denpr,
                       bool* xminus,
                       bool* xplus,
                       bool* yminus,
                       bool* yplus,
                       bool* zminus,
                       bool* zplus,
                       double* time,
                       bool* ramping,
                       double* actual_flow_rate);

static void fort_profv( Uintah::SFCXVariable<double> & uu,
                        Uintah::SFCYVariable<double> & vv,
                        Uintah::SFCZVariable<double> & ww,
                        Uintah::IntVector & idxLo,
                        Uintah::IntVector & idxHi,
                        Uintah::constCCVariable<int> & pcell,
                        double & areapr,
                        int & pfield,
                        double & flowpr,
                        double & inletvel,
                        double & denpr,
                        bool & xminus,
                        bool & xplus,
                        bool & yminus,
                        bool & yplus,
                        bool & zminus,
                        bool & zplus,
                        double & time,
                        bool & ramping,
                        double & actual_flow_rate )
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
  Uintah::IntVector pcell_low = pcell.getWindow()->getOffset();
  Uintah::IntVector pcell_high = pcell.getWindow()->getData()->size() + pcell_low - Uintah::IntVector(1, 1, 1);
  int pcell_low_x = pcell_low.x();
  int pcell_high_x = pcell_high.x();
  int pcell_low_y = pcell_low.y();
  int pcell_high_y = pcell_high.y();
  int pcell_low_z = pcell_low.z();
  int pcell_high_z = pcell_high.z();
  F_profv( &uu_low_x, &uu_low_y, &uu_low_z, &uu_high_x, &uu_high_y, &uu_high_z, uu.getPointer(),
          &vv_low_x, &vv_low_y, &vv_low_z, &vv_high_x, &vv_high_y, &vv_high_z, vv.getPointer(),
          &ww_low_x, &ww_low_y, &ww_low_z, &ww_high_x, &ww_high_y, &ww_high_z, ww.getPointer(),
          idxLo.get_pointer(),
          idxHi.get_pointer(),
          &pcell_low_x, &pcell_low_y, &pcell_low_z, &pcell_high_x, &pcell_high_y, &pcell_high_z, const_cast<int*>(pcell.getPointer()),
          &areapr,
          &pfield,
          &flowpr,
          &inletvel,
          &denpr,
          &xminus,
          &xplus,
          &yminus,
          &yplus,
          &zminus,
          &zplus,
          &time,
          &ramping,
          &actual_flow_rate );
}

#else /* !__cplusplus */

C This is the FORTRAN code portion of the file:

      subroutine profv(uu_low_x, uu_low_y, uu_low_z, uu_high_x, 
     & uu_high_y, uu_high_z, uu, vv_low_x, vv_low_y, vv_low_z, 
     & vv_high_x, vv_high_y, vv_high_z, vv, ww_low_x, ww_low_y, 
     & ww_low_z, ww_high_x, ww_high_y, ww_high_z, ww, idxLo, idxHi, 
     & pcell_low_x, pcell_low_y, pcell_low_z, pcell_high_x, 
     & pcell_high_y, pcell_high_z, pcell, areapr, pfield, flowpr, 
     & inletvel, denpr, xminus, xplus, yminus, yplus, zminus, zplus, 
     & time, ramping, actual_flow_rate)

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
      integer idxLo(3)
      integer idxHi(3)
      integer pcell_low_x, pcell_low_y, pcell_low_z, pcell_high_x, 
     & pcell_high_y, pcell_high_z
      integer pcell(pcell_low_x:pcell_high_x, pcell_low_y:pcell_high_y,
     &  pcell_low_z:pcell_high_z)
      double precision areapr
      integer pfield
      double precision flowpr
      double precision inletvel
      double precision denpr
      logical*1 xminus
      logical*1 xplus
      logical*1 yminus
      logical*1 yplus
      logical*1 zminus
      logical*1 zplus
      double precision time
      logical*1 ramping
      double precision actual_flow_rate
#endif /* __cplusplus */

#endif /* fspec_profv */

#ifndef PASS1
#  define PASS1(x) x/**/_low, x/**/_high, x
#endif

#ifndef PASS3
#  define PASS3A(x) x/**/_low_x, x/**/_low_y, x/**/_low_z, 
#  define PASS3B(x) x/**/_high_x, x/**/_high_y, x/**/_high_z, x
#endif
