
#ifndef fspec_mm_computevel
#define fspec_mm_computevel

#ifdef __cplusplus

#include <CCA/Components/Arches/fortran/FortranNameMangle.h>

extern "C" void F_mm_computevel(int* uu_low_x, int* uu_low_y, int* uu_low_z, int* uu_high_x, int* uu_high_y, int* uu_high_z, double* uu_ptr,
                              int* press_low_x, int* press_low_y, int* press_low_z, int* press_high_x, int* press_high_y, int* press_high_z, double* press_ptr,
                              int* den_low_x, int* den_low_y, int* den_low_z, int* den_high_x, int* den_high_y, int* den_high_z, double* den_ptr,
                              int* epsg_low_x, int* epsg_low_y, int* epsg_low_z, int* epsg_high_x, int* epsg_high_y, int* epsg_high_z, double* epsg_ptr,
                              int* dxpw_low, int* dxpw_high, double* dxpw_ptr,
                              double* deltat,
                              int* ioff,
                              int* joff,
                              int* koff,
                              int* pcell_low_x, int* pcell_low_y, int* pcell_low_z, int* pcell_high_x, int* pcell_high_y, int* pcell_high_z, int* pcell_ptr,
                              int* idxlo_u,
                              int* idxhi_u,
                              int* mmwallid);

static void fort_mm_computevel( Uintah::Array3<double> & uu,
                                Uintah::constCCVariable<double> & press,
                                Uintah::constCCVariable<double> & den,
                                Uintah::constCCVariable<double> & epsg,
                                Uintah::OffsetArray1<double> & dxpw,
                                double & deltat,
                                int & ioff,
                                int & joff,
                                int & koff,
                                Uintah::constCCVariable<int> & pcell,
                                Uintah::IntVector & idxlo_u,
                                Uintah::IntVector & idxhi_u,
                                int & mmwallid )
{
  Uintah::IntVector uu_low = uu.getWindow()->getOffset();
  Uintah::IntVector uu_high = uu.getWindow()->getData()->size() + uu_low - Uintah::IntVector(1, 1, 1);
  int uu_low_x = uu_low.x();
  int uu_high_x = uu_high.x();
  int uu_low_y = uu_low.y();
  int uu_high_y = uu_high.y();
  int uu_low_z = uu_low.z();
  int uu_high_z = uu_high.z();
  Uintah::IntVector press_low = press.getWindow()->getOffset();
  Uintah::IntVector press_high = press.getWindow()->getData()->size() + press_low - Uintah::IntVector(1, 1, 1);
  int press_low_x = press_low.x();
  int press_high_x = press_high.x();
  int press_low_y = press_low.y();
  int press_high_y = press_high.y();
  int press_low_z = press_low.z();
  int press_high_z = press_high.z();
  Uintah::IntVector den_low = den.getWindow()->getOffset();
  Uintah::IntVector den_high = den.getWindow()->getData()->size() + den_low - Uintah::IntVector(1, 1, 1);
  int den_low_x = den_low.x();
  int den_high_x = den_high.x();
  int den_low_y = den_low.y();
  int den_high_y = den_high.y();
  int den_low_z = den_low.z();
  int den_high_z = den_high.z();
  Uintah::IntVector epsg_low = epsg.getWindow()->getOffset();
  Uintah::IntVector epsg_high = epsg.getWindow()->getData()->size() + epsg_low - Uintah::IntVector(1, 1, 1);
  int epsg_low_x = epsg_low.x();
  int epsg_high_x = epsg_high.x();
  int epsg_low_y = epsg_low.y();
  int epsg_high_y = epsg_high.y();
  int epsg_low_z = epsg_low.z();
  int epsg_high_z = epsg_high.z();
  int dxpw_low = dxpw.low();
  int dxpw_high = dxpw.high();
  Uintah::IntVector pcell_low = pcell.getWindow()->getOffset();
  Uintah::IntVector pcell_high = pcell.getWindow()->getData()->size() + pcell_low - Uintah::IntVector(1, 1, 1);
  int pcell_low_x = pcell_low.x();
  int pcell_high_x = pcell_high.x();
  int pcell_low_y = pcell_low.y();
  int pcell_high_y = pcell_high.y();
  int pcell_low_z = pcell_low.z();
  int pcell_high_z = pcell_high.z();
  F_mm_computevel( &uu_low_x, &uu_low_y, &uu_low_z, &uu_high_x, &uu_high_y, &uu_high_z, uu.getPointer(),
                 &press_low_x, &press_low_y, &press_low_z, &press_high_x, &press_high_y, &press_high_z, const_cast<double*>(press.getPointer()),
                 &den_low_x, &den_low_y, &den_low_z, &den_high_x, &den_high_y, &den_high_z, const_cast<double*>(den.getPointer()),
                 &epsg_low_x, &epsg_low_y, &epsg_low_z, &epsg_high_x, &epsg_high_y, &epsg_high_z, const_cast<double*>(epsg.getPointer()),
                 &dxpw_low, &dxpw_high, dxpw.get_objs(),
                 &deltat,
                 &ioff,
                 &joff,
                 &koff,
                 &pcell_low_x, &pcell_low_y, &pcell_low_z, &pcell_high_x, &pcell_high_y, &pcell_high_z, const_cast<int*>(pcell.getPointer()),
                 idxlo_u.get_pointer(),
                 idxhi_u.get_pointer(),
                 &mmwallid );
}

#else /* !__cplusplus */

C This is the FORTRAN code portion of the file:

      subroutine MM_COMPUTEVEL(uu_low_x, uu_low_y, uu_low_z, uu_high_x,
     &  uu_high_y, uu_high_z, uu, press_low_x, press_low_y, press_low_z
     & , press_high_x, press_high_y, press_high_z, press, den_low_x, 
     & den_low_y, den_low_z, den_high_x, den_high_y, den_high_z, den, 
     & epsg_low_x, epsg_low_y, epsg_low_z, epsg_high_x, epsg_high_y, 
     & epsg_high_z, epsg, dxpw_low, dxpw_high, dxpw, deltat, ioff, joff
     & , koff, pcell_low_x, pcell_low_y, pcell_low_z, pcell_high_x, 
     & pcell_high_y, pcell_high_z, pcell, idxlo_u, idxhi_u, mmwallid)

      implicit none
      integer uu_low_x, uu_low_y, uu_low_z, uu_high_x, uu_high_y, 
     & uu_high_z
      double precision uu(uu_low_x:uu_high_x, uu_low_y:uu_high_y, 
     & uu_low_z:uu_high_z)
      integer press_low_x, press_low_y, press_low_z, press_high_x, 
     & press_high_y, press_high_z
      double precision press(press_low_x:press_high_x, press_low_y:
     & press_high_y, press_low_z:press_high_z)
      integer den_low_x, den_low_y, den_low_z, den_high_x, den_high_y, 
     & den_high_z
      double precision den(den_low_x:den_high_x, den_low_y:den_high_y, 
     & den_low_z:den_high_z)
      integer epsg_low_x, epsg_low_y, epsg_low_z, epsg_high_x, 
     & epsg_high_y, epsg_high_z
      double precision epsg(epsg_low_x:epsg_high_x, epsg_low_y:
     & epsg_high_y, epsg_low_z:epsg_high_z)
      integer dxpw_low
      integer dxpw_high
      double precision dxpw(dxpw_low:dxpw_high)
      double precision deltat
      integer ioff
      integer joff
      integer koff
      integer pcell_low_x, pcell_low_y, pcell_low_z, pcell_high_x, 
     & pcell_high_y, pcell_high_z
      integer pcell(pcell_low_x:pcell_high_x, pcell_low_y:pcell_high_y,
     &  pcell_low_z:pcell_high_z)
      integer idxlo_u(3)
      integer idxhi_u(3)
      integer mmwallid
#endif /* __cplusplus */

#endif /* fspec_mm_computevel */
