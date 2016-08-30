#ifndef Uintah_Component_Arches_GRIDTOOLS_h
#define Uintah_Component_Arches_GRIDTOOLS_h

#include <Core/Exceptions/InvalidValue.h>

/** @class GridTools
    @author J. Thornock
    @date Dec, 2015

    @brief Provides some basic, commonly used/shared functionality for differencing
    grid variables.

**/

namespace Uintah{ namespace ArchesCore{

  enum DIR {XDIR, YDIR, ZDIR};

#define STENCIL3_1D( dir ) \
  const int ip  = dir == 0 ? i+1 : i; \
  const int im  = dir == 0 ? i-1 : i; \
  const int jp  = dir == 1 ? j+1 : j; \
  const int jm  = dir == 1 ? j-1 : j; \
  const int kp  = dir == 2 ? k+1 : k; \
  const int km  = dir == 2 ? k-1 : k; \
  (void)ip; (void)im; \
  (void)jp; (void)jm; \
  (void)kp; (void)km;

#define STENCIL5_1D( dir ) \
  const int ip  = dir == 0 ? i+1 : i; \
  const int ipp = dir == 0 ? i+2 : i; \
  const int im  = dir == 0 ? i-1 : i; \
  const int imm = dir == 0 ? i-2 : i; \
  const int jp  = dir == 1 ? j+1 : j; \
  const int jpp = dir == 1 ? j+2 : j; \
  const int jm  = dir == 1 ? j-1 : j; \
  const int jmm = dir == 1 ? j-2 : j; \
  const int kp  = dir == 2 ? k+1 : k; \
  const int kpp = dir == 2 ? k+2 : k; \
  const int km  = dir == 2 ? k-1 : k; \
  const int kmm = dir == 2 ? k-2 : k; \
  (void)ip; (void)ipp; (void)im; (void)imm; \
  (void)jp; (void)jpp; (void)jm; (void)jmm; \
  (void)kp; (void)kpp; (void)km; (void)kmm;

#define C_    i,   j,   k
#define CP_   ip,  jp,  kp
#define CPP_  ipp, jpp, kpp
#define CM_   im,  jm,  km
#define CMM_  imm, jmm, kmm

//Staggered rotation
#define CE_ i+ioff, j+joff, k+koff
#define CW_ i-ioff, j-joff, k-koff
#define CN_ i+koff, j+ioff, k+joff
#define CS_ i-koff, j-ioff, k-joff
#define CT_ i+joff, j+koff, k+ioff
#define CB_ i-joff, j-koff, k-ioff
#define CNE_ i+idt1,j+jdt1,k+kdt1
#define CNW_ i+inw,j+jnw,k+knw
#define CSE_ i-idt1,j-jdt1,k-kdt1
#define CSW_ i-isw,j-jsw,k-ksw
#define CTE_ i+idt2,j+jdt2,k+kdt2
#define CTW_ i+itw,j+jtw,k+ktw
#define CBE_ i-idt2,j-jdt2,k-kdt2
#define CBW_ i-ibw,j-jbw,k-kbw
#define C2E_ i+i2off, j+j2off, k+k2off
#define C2W_ i-i2off, j-j2off, k-k2off

#define PRINT_CURR_REFERENCE(i,j,k,string) \
  std::cout << "Location " << string << " = (" << i << "," << j << "," << k << ")" << std::endl;

#define STAGGERED_INDEX(dir) \
  const int ioff = dir == 0 ? 1 : 0;

#define GET_BUFFERED_PATCH_RANGE(buffer_low, buffer_high) \
  Uintah::IntVector low_x_adjust = Uintah::IntVector(0,0,0); Uintah::IntVector high_x_adjust = Uintah::IntVector(0,0,0);  \
  Uintah::IntVector low_y_adjust = Uintah::IntVector(0,0,0); Uintah::IntVector high_y_adjust = Uintah::IntVector(0,0,0);  \
  Uintah::IntVector low_z_adjust = Uintah::IntVector(0,0,0); Uintah::IntVector high_z_adjust = Uintah::IntVector(0,0,0);  \
  if ( patch->getBCType(Patch::xminus) != Patch::Neighbor ) low_x_adjust = Uintah::IntVector(buffer_low,0,0); \
  if ( patch->getBCType(Patch::yminus) != Patch::Neighbor ) low_y_adjust = Uintah::IntVector(0,buffer_low,0); \
  if ( patch->getBCType(Patch::zminus) != Patch::Neighbor ) low_z_adjust = Uintah::IntVector(0,0,buffer_low); \
  if ( patch->getBCType(Patch::xplus)  != Patch::Neighbor ) high_x_adjust = Uintah::IntVector(buffer_high,0,0); \
  if ( patch->getBCType(Patch::yplus)  != Patch::Neighbor ) high_y_adjust = Uintah::IntVector(0,buffer_high,0); \
  if ( patch->getBCType(Patch::zplus)  != Patch::Neighbor ) high_z_adjust = Uintah::IntVector(0,0,buffer_high); \
  Uintah::IntVector low_patch_range = patch->getCellLowIndex()+low_x_adjust+low_y_adjust+low_z_adjust;      \
  Uintah::IntVector high_patch_range = patch->getCellHighIndex()+high_x_adjust+high_y_adjust+high_z_adjust;

#define GET_FX_BUFFERED_PATCH_RANGE(buffer_low, buffer_high) \
    Uintah::IntVector low_fx_patch_range = patch->getCellLowIndex(); \
    Uintah::IntVector high_fx_patch_range = patch->getCellHighIndex(); \
    if ( patch->getBCType(Patch::xminus) != Patch::Neighbor ){ \
      low_fx_patch_range += Uintah::IntVector(buffer_low,0,0); \
    } \
    if ( patch->getBCType(Patch::xplus) != Patch::Neighbor ){ \
      high_fx_patch_range += Uintah::IntVector(buffer_high,0,0);\
    }

#define GET_FY_BUFFERED_PATCH_RANGE(buffer_low, buffer_high) \
    Uintah::IntVector low_fy_patch_range = patch->getCellLowIndex(); \
    Uintah::IntVector high_fy_patch_range = patch->getCellHighIndex(); \
    if ( patch->getBCType(Patch::yminus) != Patch::Neighbor ){ \
      low_fy_patch_range += Uintah::IntVector(0,buffer_low,0); \
    } \
    if ( patch->getBCType(Patch::yplus) != Patch::Neighbor ){ \
      high_fy_patch_range += Uintah::IntVector(0,buffer_high,0); \
    }

#define GET_FZ_BUFFERED_PATCH_RANGE(buffer_low, buffer_high) \
    Uintah::IntVector low_fz_patch_range = patch->getCellLowIndex(); \
    Uintah::IntVector high_fz_patch_range = patch->getCellHighIndex(); \
    if ( patch->getBCType(Patch::zminus) != Patch::Neighbor ){ \
      low_fz_patch_range += Uintah::IntVector(0,0,buffer_low); \
    } \
    if ( patch->getBCType(Patch::zplus) != Patch::Neighbor ){ \
      high_fz_patch_range += Uintah::IntVector(0,0,buffer_high); \
    }

  /**
      @struct VariableHelper
      @details Useful for reconciling the const and non-const types for a specific
      template parameter. Also useful for deducing face types.
  **/
  template <typename T>
  struct VariableHelper{
  };

  //Helper specialization:
  template <>
  struct VariableHelper<Uintah::CCVariable<double> >{
    typedef Uintah::constCCVariable<double> ConstType;
    typedef Uintah::CCVariable<double> Type;
    typedef Uintah::SFCXVariable<double> XFaceType;
    typedef Uintah::SFCYVariable<double> YFaceType;
    typedef Uintah::SFCZVariable<double> ZFaceType;
    typedef Uintah::constSFCXVariable<double> ConstXFaceType;
    typedef Uintah::constSFCYVariable<double> ConstYFaceType;
    typedef Uintah::constSFCZVariable<double> ConstZFaceType;
    int dir;
    VariableHelper():dir(999){}
  };

  template <>
  struct VariableHelper<Uintah::SFCXVariable<double> >{
    typedef Uintah::constSFCXVariable<double> ConstType;
    typedef Uintah::SFCXVariable<double> Type;
    typedef Uintah::CCVariable<double> XFaceType;
    typedef Uintah::SFCXVariable<double> YFaceType;
    typedef Uintah::SFCXVariable<double> ZFaceType;
    typedef Uintah::constCCVariable<double> ConstXFaceType;
    typedef Uintah::constSFCXVariable<double> ConstYFaceType;
    typedef Uintah::constSFCXVariable<double> ConstZFaceType;
    DIR dir;
    const int ioff;
    const int joff;
    const int koff;
    const int idt1;
    const int idt2;
    const int jdt1;
    const int jdt2;
    const int kdt1;
    const int kdt2;
    VariableHelper():dir(XDIR), ioff(1), joff(0), koff(0),
    idt1(koff), idt2(joff), jdt1(ioff), jdt2(koff), kdt1(joff), kdt2(ioff){}
  };

  template <>
  struct VariableHelper<Uintah::SFCYVariable<double> >{
    typedef Uintah::constSFCYVariable<double> ConstType;
    typedef Uintah::SFCYVariable<double> Type;
    typedef Uintah::SFCYVariable<double> XFaceType;
    typedef Uintah::CCVariable<double> YFaceType;
    typedef Uintah::SFCYVariable<double> ZFaceType;
    typedef Uintah::constSFCYVariable<double> ConstXFaceType;
    typedef Uintah::constCCVariable<double> ConstYFaceType;
    typedef Uintah::constSFCYVariable<double> ConstZFaceType;
    DIR dir;
    const int ioff;
    const int joff;
    const int koff;
    const int idt1;
    const int idt2;
    const int jdt1;
    const int jdt2;
    const int kdt1;
    const int kdt2;
    VariableHelper():dir(XDIR), ioff(1), joff(0), koff(0),
    idt1(koff), idt2(joff), jdt1(ioff), jdt2(koff), kdt1(joff), kdt2(ioff){}
  };

  template <>
  struct VariableHelper<Uintah::SFCZVariable<double> >{
    typedef Uintah::constSFCZVariable<double> ConstType;
    typedef Uintah::SFCZVariable<double> Type;
    typedef Uintah::SFCZVariable<double> XFaceType;
    typedef Uintah::SFCZVariable<double> YFaceType;
    typedef Uintah::CCVariable<double> ZFaceType;
    typedef Uintah::constSFCZVariable<double> ConstXFaceType;
    typedef Uintah::constSFCZVariable<double> ConstYFaceType;
    typedef Uintah::constCCVariable<double> ConstZFaceType;
    DIR dir;
    const int ioff;
    const int joff;
    const int koff;
    const int idt1;
    const int idt2;
    const int jdt1;
    const int jdt2;
    const int kdt1;
    const int kdt2;
    VariableHelper():dir(XDIR), ioff(1), joff(0), koff(0),
    idt1(koff), idt2(joff), jdt1(ioff), jdt2(koff), kdt1(joff), kdt2(ioff){}
  };

  //discretization
  class GridTools{
  public:

  private:

  };
}} //namespace Uintah::Arches
#endif
