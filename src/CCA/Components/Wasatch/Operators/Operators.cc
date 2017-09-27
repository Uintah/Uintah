/*
 * The MIT License
 *
 * Copyright (c) 2012-2017 The University of Utah
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

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/Operators/Operators.h>
#include <CCA/Components/Wasatch/BCHelper.h>
#include <CCA/Components/Wasatch/WasatchBCHelper.h>

#include <spatialops/particles/ParticleOperators.h>
#include <CCA/Components/Wasatch/Operators/Extrapolant.h>
#include <CCA/Components/Wasatch/Operators/UpwindInterpolant.h>
#include <CCA/Components/Wasatch/Operators/FluxLimiterInterpolant.h>

//-- Uintah includes --//
#include <Core/Grid/Patch.h>

using namespace SpatialOps;

namespace WasatchCore{
  
  template<typename FieldT>
  const Uintah::Point get_low_position(const Uintah::Patch& patch);
  
  template<>
  const Uintah::Point get_low_position<SVolField>(const Uintah::Patch& patch)
  {
    return patch.getCellPosition(patch.getCellLowIndex());
  }
    
  template<>
  const Uintah::Point get_low_position<XVolField>(const Uintah::Patch& patch)
  {
    const Uintah::Vector spacing = patch.dCell();
    Uintah::Point low = patch.getCellPosition(patch.getCellLowIndex());
    low.x( low.x() - spacing[0]/2.0 );
    return low;
  }
  
  template<>
  const Uintah::Point get_low_position<YVolField>(const Uintah::Patch& patch)
  {
    const Uintah::Vector spacing = patch.dCell();
    Uintah::Point low = patch.getCellPosition(patch.getCellLowIndex());
    low.y( low.y() - spacing[1]/2.0 );
    return low;
  }
  
  template<>
  const Uintah::Point get_low_position<ZVolField>(const Uintah::Patch& patch)
  {
    const Uintah::Vector spacing = patch.dCell();
    Uintah::Point low = patch.getCellPosition(patch.getCellLowIndex());
    low.z( low.z() - spacing[2]/2.0 );
    return low;
  }
  
#define BUILD_PARTICLE_OPS( VOLT )                                                                     \
{                                                                                                       \
  typedef SpatialOps::Particle::CellToParticle<VOLT> C2P;                                               \
  typedef SpatialOps::Particle::ParticleToCell<VOLT> P2C;                                               \
  typedef SpatialOps::Particle::ParticlesPerCell<VOLT> PPerC;                                               \
  const Uintah::Point low = get_low_position<VOLT>(patch);                                               \
  opDB.register_new_operator<C2P>(scinew C2P(spacing[0], low.x(), spacing[1], low.y(), spacing[2], low.z()) );\
  opDB.register_new_operator<P2C>(scinew P2C(spacing[0], low.x(), spacing[1], low.y(), spacing[2], low.z()) );\
  opDB.register_new_operator<PPerC>(scinew PPerC(spacing[0], low.x(), spacing[1], low.y(), spacing[2], low.z()) );\
}

#define BUILD_UPWIND( VOLT )                                            \
{                                                                     \
  typedef UpwindInterpolant<VOLT,FaceTypes<VOLT>::XFace> OpX;         \
  typedef UpwindInterpolant<VOLT,FaceTypes<VOLT>::YFace> OpY;         \
  typedef UpwindInterpolant<VOLT,FaceTypes<VOLT>::ZFace> OpZ;         \
  opDB.register_new_operator<OpX>( scinew OpX() );                    \
  opDB.register_new_operator<OpY>( scinew OpY() );                    \
  opDB.register_new_operator<OpZ>( scinew OpZ() );                    \
}
  
#define BUILD_UPWIND_LIMITER( VOLT )                                  \
{                                                                     \
  typedef FluxLimiterInterpolant<VOLT,FaceTypes<VOLT>::XFace> OpX;    \
  typedef FluxLimiterInterpolant<VOLT,FaceTypes<VOLT>::YFace> OpY;    \
  typedef FluxLimiterInterpolant<VOLT,FaceTypes<VOLT>::ZFace> OpZ;    \
  opDB.register_new_operator<OpX>( scinew OpX(dim,bcPlus,bcMinus) );  \
  opDB.register_new_operator<OpY>( scinew OpY(dim,bcPlus,bcMinus) );  \
  opDB.register_new_operator<OpZ>( scinew OpZ(dim,bcPlus,bcMinus) );  \
}
  
#define BUILD_EXTRAPOLANT( VOLT )                                     \
{                                                                     \
  typedef WasatchCore::Extrapolant<VOLT> OpVol;                       \
  opDB.register_new_operator<OpVol>( scinew OpVol(bcMinus, bcPlus) ); \
}

  void build_operators( const Uintah::Patch& patch,
                        SpatialOps::OperatorDatabase& opDB )
  {
    const Uintah::IntVector udim = patch.getCellHighIndex() - patch.getCellLowIndex();

    std::vector<int> dim(3,1);
    for( size_t i=0; i<3; ++i ){ dim[i] = udim[i];}
    
    const Uintah::Vector spacing = patch.dCell();
    
    std::vector<bool> bcPlus(3,false);
    bcPlus[0] = patch.getBCType(Uintah::Patch::xplus) != Uintah::Patch::Neighbor;
    bcPlus[1] = patch.getBCType(Uintah::Patch::yplus) != Uintah::Patch::Neighbor;
    bcPlus[2] = patch.getBCType(Uintah::Patch::zplus) != Uintah::Patch::Neighbor;
    
    // check if there are any physical boundaries present on the minus side of the patch
    std::vector<bool> bcMinus(3,false);
    bcMinus[0] = patch.getBCType(Uintah::Patch::xminus) != Uintah::Patch::Neighbor;
    bcMinus[1] = patch.getBCType(Uintah::Patch::yminus) != Uintah::Patch::Neighbor;
    bcMinus[2] = patch.getBCType(Uintah::Patch::zminus) != Uintah::Patch::Neighbor;
    
    // build all of the stencils defined in SpatialOps
    SpatialOps::build_stencils( udim[0], udim[1], udim[2],
                                udim[0]*spacing[0], udim[1]*spacing[1], udim[2]*spacing[2],
                                opDB );
    
    //--------------------------------------------------------
    // UPWIND interpolants - phi volume to phi surface
    //--------------------------------------------------------
    BUILD_UPWIND( SVolField )
    BUILD_UPWIND( XVolField )
    BUILD_UPWIND( YVolField )
    BUILD_UPWIND( ZVolField )
    
    //--------------------------------------------------------
    // FLUX LIMITER interpolants - phi volume to phi surface
    //--------------------------------------------------------
    BUILD_UPWIND_LIMITER( SVolField )
    BUILD_UPWIND_LIMITER( XVolField )
    BUILD_UPWIND_LIMITER( YVolField )
    BUILD_UPWIND_LIMITER( ZVolField )
    
    //--------------------------------------------------------
    // Extrapolants
    //--------------------------------------------------------
    BUILD_EXTRAPOLANT( SVolField )
    BUILD_EXTRAPOLANT( XVolField )
    BUILD_EXTRAPOLANT( YVolField )
    BUILD_EXTRAPOLANT( ZVolField )

    //--------------------------------------------------------
    // Particles
    //--------------------------------------------------------
    BUILD_PARTICLE_OPS( SVolField );
    BUILD_PARTICLE_OPS( XVolField );
    BUILD_PARTICLE_OPS( YVolField );
    BUILD_PARTICLE_OPS( ZVolField );
  }
  
} // namespace WasatchCore
