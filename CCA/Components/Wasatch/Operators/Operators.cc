//-- SpatialOps includes --//
#include <spatialops/structured/FVStaggered.h>

//-- Wasatch includes --//
#include "Operators.h"
#include <CCA/Components/Wasatch/Operators/UpwindInterpolant.h>
#include <CCA/Components/Wasatch/Operators/FluxLimiterInterpolant.h>

//-- Uintah includes --//
#include <Core/Grid/Patch.h>

using namespace SpatialOps;
using namespace structured;

namespace Wasatch{

#define BUILD_UPWIND( VOLT )                                            \
  {                                                                     \
    typedef UpwindInterpolant<VOLT,FaceTypes<VOLT>::XFace> OpX;         \
    typedef UpwindInterpolant<VOLT,FaceTypes<VOLT>::YFace> OpY;         \
    typedef UpwindInterpolant<VOLT,FaceTypes<VOLT>::ZFace> OpZ;         \
    opDB.register_new_operator<OpX>( scinew OpX() );                    \
    opDB.register_new_operator<OpY>( scinew OpY() );                    \
    opDB.register_new_operator<OpZ>( scinew OpZ() );                    \
  }

#define BUILD_UPWIND_LIMITER( VOLT )                                    \
  {                                                                     \
    typedef FluxLimiterInterpolant<VOLT,FaceTypes<VOLT>::XFace> OpX;    \
    typedef FluxLimiterInterpolant<VOLT,FaceTypes<VOLT>::YFace> OpY;    \
    typedef FluxLimiterInterpolant<VOLT,FaceTypes<VOLT>::ZFace> OpZ;    \
    opDB.register_new_operator<OpX>( scinew OpX(dim,bcPlus) );          \
    opDB.register_new_operator<OpY>( scinew OpY(dim,bcPlus) );          \
    opDB.register_new_operator<OpZ>( scinew OpZ(dim,bcPlus) );          \
  }


  void build_operators( const Uintah::Patch& patch,
                        SpatialOps::OperatorDatabase& opDB )
  {
    using namespace SpatialOps;
    using namespace structured;

    const SCIRun::IntVector udim = patch.getCellHighIndex() - patch.getCellLowIndex();
    std::vector<int> dim(3,1);
    for( size_t i=0; i<3; ++i ){ dim[i] = udim[i]; }

    const Uintah::Vector spacing = patch.dCell();
    std::vector<double> area(3,1);
    area[0] = spacing[1]*spacing[2];
    area[1] = spacing[0]*spacing[2];
    area[2] = spacing[0]*spacing[1];

    std::vector<bool> bcPlus(3,false);
    bcPlus[0] = patch.getBCType(Uintah::Patch::xplus) != Uintah::Patch::Neighbor;
    bcPlus[1] = patch.getBCType(Uintah::Patch::yplus) != Uintah::Patch::Neighbor;
    bcPlus[2] = patch.getBCType(Uintah::Patch::zplus) != Uintah::Patch::Neighbor;

    // build all of the stencils defined in SpatialOps
    build_stencils( udim[0], udim[1], udim[2],
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
  }

} // namespace Wasatch
