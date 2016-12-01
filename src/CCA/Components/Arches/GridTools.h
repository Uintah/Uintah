#ifndef Uintah_Component_Arches_GRIDTOOLS_h
#define Uintah_Component_Arches_GRIDTOOLS_h

/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <Core/Exceptions/InvalidValue.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <CCA/Components/Arches/UPSHelper.h>

/** @class GridTools
    @author J. Thornock
    @date Dec, 2015

    @brief Provides some basic, commonly used/shared functionality for differencing
    grid variables.

**/

namespace Uintah{ namespace ArchesCore{

  enum DIR {XDIR, YDIR, ZDIR, NODIR};

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
    DIR dir;
    const int ioff;
    const int joff;
    const int koff;
    VariableHelper():dir(NODIR), ioff(0), joff(0), koff(0){}
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
    VariableHelper():dir(YDIR), ioff(1), joff(0), koff(0),
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
    VariableHelper():dir(ZDIR), ioff(1), joff(0), koff(0),
    idt1(koff), idt2(joff), jdt1(ioff), jdt2(koff), kdt1(joff), kdt2(ioff){}
  };

  /// @brief Map specific ARCHES variables based on type
  template <typename T>
  struct GridVarMap {
    std::string vol_frac_name = "NOT_AVAILABLE";
  };
  template <>
  struct GridVarMap<CCVariable<double> >{
    void problemSetup( ProblemSpecP db ){
      uvel_name = parse_ups_for_role( UVELOCITY, db, "NotSet" );
      vvel_name = parse_ups_for_role( VVELOCITY, db, "NotSet" );
      wvel_name = parse_ups_for_role( WVELOCITY, db, "NotSet" );
    }
    std::string vol_frac_name = "cc_volume_fraction";
    std::string uvel_name;
    std::string vvel_name;
    std::string wvel_name;
  };
  template <>
  struct GridVarMap<SFCXVariable<double> >{
    void problemSetup( ProblemSpecP db ){
    }
    std::string vol_frac_name = "fx_volume_fraction";
    std::string uvel_name = "ucell_xvel";
    std::string vvel_name = "ucell_yvel";
    std::string wvel_name = "ucell_zvel";
  };
  template <>
  struct GridVarMap<SFCYVariable<double> >{
    void problemSetup( ProblemSpecP db ){
    }
    std::string vol_frac_name = "fy_volume_fraction";
    std::string uvel_name = "vcell_xvel";
    std::string vvel_name = "vcell_yvel";
    std::string wvel_name = "vcell_zvel";
  };
  template <>
  struct GridVarMap<SFCZVariable<double> >{
    void problemSetup( ProblemSpecP db ){
    }
    std::string vol_frac_name = "fz_volume_fraction";
    std::string uvel_name = "wcell_xvel";
    std::string vvel_name = "wcell_yvel";
    std::string wvel_name = "wcell_zvel";
  };

  /// @brief Returns a weight for interpolation
  /// @TODO This doesn't cover all cases correctly. Fix it.
  ///       For example, consider SFCX, SFCY
  template <typename DT, typename IT>
  struct oneDInterp {
    double get_central_weight(){ return 0; }
    int dir=99;
  };

  template <>
  struct oneDInterp<SFCXVariable<double>, constCCVariable<double> >{
    double get_central_weight(){ return 0.5; }
    int dir=0;
  };

  template <>
  struct oneDInterp<SFCYVariable<double>, constCCVariable<double> >{
    double get_central_weight(){ return 0.5; }
    int dir=1;
  };

  template <>
  struct oneDInterp<SFCZVariable<double>, constCCVariable<double> >{
    double get_central_weight(){ return 0.5; }
    int dir=2;
  };

  class GridTools{

  public:

    GridTools(){}
    ~GridTools(){}

  private:

  };
}} //namespace Uintah::Arches
#endif
