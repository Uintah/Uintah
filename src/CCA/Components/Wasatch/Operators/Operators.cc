/*
 * The MIT License
 *
 * Copyright (c) 2012 The University of Utah
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
opDB.register_new_operator<OpX>( scinew OpX(dim,bcPlus,hasMinusBoundary) );          \
opDB.register_new_operator<OpY>( scinew OpY(dim,bcPlus,hasMinusBoundary) );          \
opDB.register_new_operator<OpZ>( scinew OpZ(dim,bcPlus,hasMinusBoundary) );          \
}
  
  
  void build_operators( const Uintah::Patch& patch,
                       SpatialOps::OperatorDatabase& opDB )
  {
    const SCIRun::IntVector udim = patch.getCellHighIndex() - patch.getCellLowIndex();
    std::vector<int> dim(3,1);
    for( size_t i=0; i<3; ++i ){ dim[i] = udim[i];}
    
    const Uintah::Vector spacing = patch.dCell();
    std::vector<double> area(3,1);
    area[0] = spacing[1]*spacing[2];
    area[1] = spacing[0]*spacing[2];
    area[2] = spacing[0]*spacing[1];
    
    std::vector<bool> bcPlus(3,false);
    bcPlus[0] = patch.getBCType(Uintah::Patch::xplus) != Uintah::Patch::Neighbor;
    bcPlus[1] = patch.getBCType(Uintah::Patch::yplus) != Uintah::Patch::Neighbor;
    bcPlus[2] = patch.getBCType(Uintah::Patch::zplus) != Uintah::Patch::Neighbor;
    
    // check if there are any physical boundaries present on the minus side of the patch
    std::vector<bool> hasMinusBoundary(3,false);
    hasMinusBoundary[0] = patch.getBCType(Uintah::Patch::xminus) != Uintah::Patch::Neighbor;
    hasMinusBoundary[1] = patch.getBCType(Uintah::Patch::yminus) != Uintah::Patch::Neighbor;
    hasMinusBoundary[2] = patch.getBCType(Uintah::Patch::zminus) != Uintah::Patch::Neighbor;
    
    // build all of the stencils defined in SpatialOps
    SpatialOps::structured::build_stencils( udim[0], udim[1], udim[2],
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
