#ifndef Uintah_Component_Arches_GRIDTOOLS_h
#define Uintah_Component_Arches_GRIDTOOLS_h

/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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
    @file

    @brief Provides some basic, commonly used/shared functionality for differencing
    grid variables.

**/

namespace Uintah{ namespace ArchesCore{

  enum DIR { XDIR, YDIR, ZDIR, NODIR };
  enum INTERPOLANT {SECONDCENTRAL, FOURTHCENTRAL};

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

#define IJK_    i,   j,   k
#define IJK_P_   ip,  jp,  kp
#define IJK_PP_  ipp, jpp, kpp
#define IJK_M_   im,  jm,  km
#define IJK_MM_  imm, jmm, kmm

#define PRINT_CURR_REFERENCE(i,j,k,string) \
  std::cout << "Location " << string << " = (" << i << "," << j << "," << k << ")" << std::endl;

#define STAGGERED_INDEX(dir) \
  const int ioff = dir == 0 ? 1 : 0;

/** @brief Get a low and high index for the patch with the option of adding
           a buffer to the range. Note that the buffer will adjust for the
           presence of domain edges for the patch, in that if the buffer is
           greater than 1, it will be reasigned to 1 so as not to go beyond
           the extra cell. buffer_low and buffer_high in this case are
           Uintah::IntVectors.
*/
#define GET_BUFFERED_PATCH_RANGE(buffer_low, buffer_high, low_patch_range, high_patch_range) \
  if ( buffer_low[0] < 0 ){ \
    buffer_low[0] = ( patch->getBCType(Patch::xminus) != Patch::Neighbor) ? -1 : 0; \
  } \
  if ( buffer_low[1] > 0 ){ \
    buffer_low[1] = ( patch->getBCType(Patch::yminus) != Patch::Neighbor) ? -1 : 0; \
  } \
  if ( buffer_low[2] > 0 ){ \
    buffer_low[2] = ( patch->getBCType(Patch::zminus) != Patch::Neighbor) ? -1 : 0; \
  } \
  if ( buffer_high[0] > 0 ){ \
    buffer_high[0] = ( patch->getBCType(Patch::xplus) != Patch::Neighbor) ? 1 : 0; \
  } \
  if ( buffer_high[1] > 0 ){ \
    buffer_high[1] = ( patch->getBCType(Patch::yplus) != Patch::Neighbor) ? 1 : 0; \
  } \
  if ( buffer_high[2] > 0 ){ \
    buffer_high[2] = ( patch->getBCType(Patch::zplus) != Patch::Neighbor) ? 1 : 0; \
  } \
  low_patch_range = patch->getCellLowIndex()+buffer_low;      \
  high_patch_range = patch->getCellHighIndex()+buffer_high;

/** @brief Get a low and high index for the patch where the buffer is added ONLY in the
           case that the domain edge appears on that side of the patch.
           a buffer to the range. Note that the buffer will adjust for the
           presence of domain edges for the patch. buffer_low and buffer_high in this case
           are std::int.
*/
#define GET_EXTRACELL_BUFFERED_PATCH_RANGE(buffer_low, buffer_high) \
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

/** @brief Get a low and high index for the patch where the buffer is added ONLY in the
           case that the domain edge appears on that side of the patch AND only applied to the
           X-direction. Note that the buffer will adjust for the
           presence of domain edges for the patch.  buffer_low and buffer_high in this case
           are std::int.
*/
#define GET_EXTRACELL_FX_BUFFERED_PATCH_RANGE(buffer_low, buffer_high) \
    \
    Uintah::IntVector low_fx_patch_range = patch->getCellLowIndex(); \
    Uintah::IntVector high_fx_patch_range = patch->getCellHighIndex(); \
    if ( patch->getBCType(Patch::xminus) != Patch::Neighbor ){ \
      low_fx_patch_range += Uintah::IntVector(buffer_low,0,0); \
    } \
    if ( patch->getBCType(Patch::xplus) != Patch::Neighbor ){ \
      high_fx_patch_range += Uintah::IntVector(buffer_high,0,0);\
    }

#define GET_WALL_BUFFERED_PATCH_RANGE(low_patch_range, high_patch_range,\
                                      buffer_low_x,buffer_high_x,\
                                      buffer_low_y,buffer_high_y,\
                                      buffer_low_z,buffer_high_z) \
    \
    if ( patch->getBCType(Patch::xminus) != Patch::Neighbor ){ \
      low_patch_range += Uintah::IntVector(buffer_low_x,0,0); \
    } \
    if ( patch->getBCType(Patch::xplus) != Patch::Neighbor ){ \
      high_patch_range += Uintah::IntVector(buffer_high_x,0,0);\
    } \
    if ( patch->getBCType(Patch::yminus) != Patch::Neighbor ){ \
      low_patch_range += Uintah::IntVector(0,buffer_low_y,0); \
    } \
    if ( patch->getBCType(Patch::yplus) != Patch::Neighbor ){ \
      high_patch_range += Uintah::IntVector(0,buffer_high_y,0);\
    } \
    if ( patch->getBCType(Patch::zminus) != Patch::Neighbor ){ \
      low_patch_range += Uintah::IntVector(0,0,buffer_low_z); \
    } \
    if ( patch->getBCType(Patch::zplus) != Patch::Neighbor ){ \
      high_patch_range += Uintah::IntVector(0,0,buffer_high_z);\
    }


/** @brief Get a low and high index for the patch where the buffer is added ONLY in the
           case that the domain edge appears on that side of the patch AND only applied to the
           y-direction. Note that the buffer will adjust for the
           presence of domain edges for the patch.  buffer_low and buffer_high in this case
           are std::int.
*/
#define GET_EXTRACELL_FY_BUFFERED_PATCH_RANGE(buffer_low, buffer_high) \
    Uintah::IntVector low_fy_patch_range = patch->getCellLowIndex(); \
    Uintah::IntVector high_fy_patch_range = patch->getCellHighIndex(); \
    if ( patch->getBCType(Patch::yminus) != Patch::Neighbor ){ \
      low_fy_patch_range += Uintah::IntVector(0,buffer_low,0); \
    } \
    if ( patch->getBCType(Patch::yplus) != Patch::Neighbor ){ \
      high_fy_patch_range += Uintah::IntVector(0,buffer_high,0); \
    }

/** @brief Get a low and high index for the patch where the buffer is added ONLY in the
           case that the domain edge appears on that side of the patch AND only applied to the
           z-direction. Note that the buffer will adjust for the
           presence of domain edges for the patch. buffer_low and buffer_high in this case
           are std::int.
*/
#define GET_EXTRACELL_FZ_BUFFERED_PATCH_RANGE(buffer_low, buffer_high) \
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
    typedef double PODType;
    typedef Uintah::constCCVariable<double> ConstType;
    typedef Uintah::SFCXVariable<double> XFaceType;
    typedef Uintah::SFCYVariable<double> YFaceType;
    typedef Uintah::SFCZVariable<double> ZFaceType;
    DIR dir;
    int ioff, joff, koff;
    VariableHelper():dir(NODIR), ioff(0), joff(0), koff(0){}
  };

  template <>
  struct VariableHelper<Uintah::constCCVariable<double> >{
    typedef const double PODType;
    typedef Uintah::constCCVariable<double> ConstType;
    typedef Uintah::constSFCXVariable<double> XFaceType;
    typedef Uintah::constSFCYVariable<double> YFaceType;
    typedef Uintah::constSFCZVariable<double> ZFaceType;
    DIR dir;
    int ioff, joff, koff;
    VariableHelper():dir(NODIR), ioff(0), joff(0), koff(0){}
  };

  template <>
  struct VariableHelper<Uintah::SFCXVariable<double> >{
    typedef double PODType;
    typedef Uintah::constSFCXVariable<double> ConstType;
    typedef Uintah::SFCXVariable<double> XFaceType;
    typedef Uintah::SFCXVariable<double> YFaceType;
    typedef Uintah::SFCXVariable<double> ZFaceType;
    DIR dir;
    int ioff, joff, koff;
    VariableHelper():dir(XDIR), ioff(1), joff(0), koff(0){}
  };

  template <>
  struct VariableHelper<Uintah::constSFCXVariable<double> >{
    typedef const double PODType;
    typedef Uintah::constSFCXVariable<double> ConstType;
    typedef Uintah::constSFCXVariable<double> XFaceType;
    typedef Uintah::constSFCXVariable<double> YFaceType;
    typedef Uintah::constSFCXVariable<double> ZFaceType;
    DIR dir;
    int ioff, joff, koff;
    VariableHelper():dir(XDIR), ioff(1), joff(0), koff(0){}
  };

  template <>
  struct VariableHelper<Uintah::SFCYVariable<double> >{
    typedef double PODType;
    typedef Uintah::constSFCYVariable<double> ConstType;
    typedef Uintah::SFCYVariable<double> XFaceType;
    typedef Uintah::SFCYVariable<double> YFaceType;
    typedef Uintah::SFCYVariable<double> ZFaceType;
    DIR dir;
    int ioff, joff, koff;
    VariableHelper():dir(YDIR), ioff(0), joff(1), koff(0){}
  };

  template <>
  struct VariableHelper<Uintah::constSFCYVariable<double> >{
    typedef const double PODType;
    typedef Uintah::constSFCYVariable<double> ConstType;
    typedef Uintah::constSFCYVariable<double> XFaceType;
    typedef Uintah::constSFCYVariable<double> YFaceType;
    typedef Uintah::constSFCYVariable<double> ZFaceType;
    DIR dir;
    int ioff, joff, koff;
    VariableHelper():dir(YDIR), ioff(0), joff(1), koff(0){}
  };

  template <>
  struct VariableHelper<Uintah::SFCZVariable<double> >{
    typedef double PODType;
    typedef Uintah::constSFCZVariable<double> ConstType;
    typedef Uintah::SFCZVariable<double> XFaceType;
    typedef Uintah::SFCZVariable<double> YFaceType;
    typedef Uintah::SFCZVariable<double> ZFaceType;
    DIR dir;
    int ioff, joff, koff;
    VariableHelper():dir(ZDIR), ioff(0), joff(0), koff(1){}
  };

  template <>
  struct VariableHelper<Uintah::constSFCZVariable<double> >{
    typedef const double PODType;
    typedef Uintah::constSFCZVariable<double> ConstType;
    typedef Uintah::constSFCZVariable<double> XFaceType;
    typedef Uintah::constSFCZVariable<double> YFaceType;
    typedef Uintah::constSFCZVariable<double> ZFaceType;
    DIR dir;
    int ioff, joff, koff;
    VariableHelper():dir(ZDIR), ioff(0), joff(0), koff(1){}
  };

  /// @brief Map specific ARCHES variables based on type
  template <typename T>
  struct GridVarMap {
    std::string vol_frac_name = "NOT_AVAILABLE";
  };

  template <>
  struct GridVarMap<CCVariable<double> >{
    void problemSetup( ProblemSpecP db ){
      uvel_name = parse_ups_for_role( UVELOCITY_ROLE, db, ArchesCore::default_uVel_name );
      vvel_name = parse_ups_for_role( VVELOCITY_ROLE, db, ArchesCore::default_vVel_name );
      wvel_name = parse_ups_for_role( WVELOCITY_ROLE, db, ArchesCore::default_wVel_name );
      mu_name = parse_ups_for_role( TOTAL_VISCOSITY_ROLE, db, ArchesCore::default_viscosity_name );
    }
    std::string vol_frac_name = "volFraction";
    std::string mu_name;
    std::string uvel_name;
    std::string vvel_name;
    std::string wvel_name;
  };
  template <>
  struct GridVarMap<SFCXVariable<double> >{
    void problemSetup( ProblemSpecP db ){
      mu_name = parse_ups_for_role( TOTAL_VISCOSITY_ROLE, db, ArchesCore::default_viscosity_name );
    }
    std::string vol_frac_name = "volFractionX";
    std::string uvel_name = "ucell_xvel";
    std::string vvel_name = "ucell_yvel";
    std::string wvel_name = "ucell_zvel";
    std::string mu_name;
    std::string sigmax_name = "sigma11";
    std::string sigmay_name = "sigma12";
    std::string sigmaz_name = "sigma13";
  };
  template <>
  struct GridVarMap<SFCYVariable<double> >{
    void problemSetup( ProblemSpecP db ){
      mu_name = parse_ups_for_role( TOTAL_VISCOSITY_ROLE, db, ArchesCore::default_viscosity_name );
    }
    std::string vol_frac_name = "volFractionY";
    std::string uvel_name = "vcell_xvel";
    std::string vvel_name = "vcell_yvel";
    std::string wvel_name = "vcell_zvel";
    std::string mu_name;
    std::string sigmax_name = "sigma12";
    std::string sigmay_name = "sigma22";
    std::string sigmaz_name = "sigma23";
  };
  template <>
  struct GridVarMap<SFCZVariable<double> >{
    void problemSetup( ProblemSpecP db ){
      mu_name = parse_ups_for_role( TOTAL_VISCOSITY_ROLE, db, ArchesCore::default_viscosity_name );
    }
    std::string vol_frac_name = "volFractionZ";
    std::string uvel_name = "wcell_xvel";
    std::string vvel_name = "wcell_yvel";
    std::string wvel_name = "wcell_zvel";
    std::string mu_name;
    std::string sigmax_name = "sigma13";
    std::string sigmay_name = "sigma23";
    std::string sigmaz_name = "sigma33";
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


  ///  @brief Generic interface to grid interpolators.
  template <typename ExecSpace, typename MemSpace, typename grid_T, typename grid_CT>
  void doInterpolation( ExecutionObject<ExecSpace, MemSpace> execObj,
                        Uintah::BlockRange& range, grid_T& v_i, grid_CT& v,
                        const int &ioff, const int &joff, const int &koff,
                        unsigned int interpScheme ){

    if (interpScheme == FOURTHCENTRAL ){

      Uintah::parallel_for(execObj, range, KOKKOS_LAMBDA(int i, int j, int k) {

        v_i(i,j,k) = (9./16.)*(v(i,j,k) + v(i+ioff,j+joff,k+koff))
                   - (1./16.)*(v(i+2*ioff,j+2*joff,k+2*koff) + v(i-ioff,j-joff,k-koff)) ;

      });

    } else if ( interpScheme == SECONDCENTRAL ){

      Uintah::parallel_for(execObj, range, KOKKOS_LAMBDA(int i, int j, int k) {

        v_i(i,j,k) = 0.5 * ( v(i,j,k) + v(i+ioff,j+joff,k+koff) );

      });

    } else {

      throw InvalidValue("Error: Interpolator scheme not valid.", __FILE__, __LINE__);

    }
  }

  /**
      @brief Returns the value (currently 0 or 1) of the volume/area fraction of gas on
             at the location specified.
      @param eps        The CC volume fraction.
      @param i,j,k      Current cell location of interest
      @param idir       Array of the direction of interest
      @param vdir       Array of the variable direction  ([0,0,0] in the case of CC variables)
  **/
  inline double get_eps( const Array3<double>& eps, const int i, const int j, const int k,
                         const int* idir, const int* vdir )
  {

    const int i1 = i - vdir[0];
    const int j1 = j - vdir[1];
    const int k1 = k - vdir[2];
    const int i2 = i - idir[0];
    const int j2 = j - idir[1];
    const int k2 = k - idir[2];
    const int i3 = i - idir[0] - vdir[0];
    const int j3 = j - idir[1] - vdir[1];
    const int k3 = k - idir[2] - vdir[2];

    return eps(i,j,k)*eps(i1,j1,k1)*eps(i2,j2,k2)*eps(i3,j3,k3);

   }

  INTERPOLANT get_interpolant_from_string(const std::string value);

  class GridTools{

  public:

    GridTools(){}
    ~GridTools(){}

  private:

  };
}} //namespace Uintah::ArchesCore
#endif
