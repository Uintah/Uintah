#ifndef Wasatch_OperatorTypes_h
#define Wasatch_OperatorTypes_h

#include "../FieldTypes.h"
#include "UpwindInterpolant.h"
#include "SuperbeeInterpolant.h"

#include <spatialops/structured/FVStaggered.h>

/**
 *  \file OperatorTypes.h
 */


using SpatialOps::Divergence;
using SpatialOps::Gradient;
using SpatialOps::Interpolant;

using SpatialOps::structured::OperatorTypeBuilder;

namespace Wasatch{

  /**
   *  \ingroup WasatchOperators
   *  \ingroup WasatchCore
   *  \struct OpTypes
   *  \brief provides typedefs for various operators related to the given cell type
   *
   *  Note: this extends the BasicOpTypes definitions in SpatialOps.
   *  The full set of available operator types provided is:
   *
   *    - \b GradX - x-gradient of cell-centered quantities produced at cell x-faces
   *    - \b GradY - y-gradient of cell-centered quantities produced at cell y-faces
   *    - \b GradZ - z-gradient of cell-centered quantities produced at cell z-faces
   *
   *    - \b DivX - x-divergence of cell-centered quantities produced at cell x-faces
   *    - \b DivY - y-divergence of cell-centered quantities produced at cell y-faces
   *    - \b DivZ - z-divergence of cell-centered quantities produced at cell z-faces
   *
   *    - \b InterpC2FX - Interpolate cell-centered quantities to x-faces
   *    - \b InterpC2FY - Interpolate cell-centered quantities to y-faces
   *    - \b InterpC2FZ - Interpolate cell-centered quantities to z-faces
   *
   *    - \b InterpF2CX - Interpolate x-face quantities to cell-centered
   *    - \b InterpF2CY - Interpolate y-face quantities to cell-centered
   *    - \b InterpF2CZ - Interpolate z-face quantities to cell-centered
   *
   *    - \b InterpC2FXUpwind - upwind or limited interpolants in x-dir
   *    - \b InterpC2FYUpwind - upwind or limited interpolants in y-dir
   *    - \b InterpC2FZUpwind - upwind or limited interpolants in z-dir
   *
   *  Example:
   *  \code
   *    typedef OpTypes< SVolField >  VolOps;
   *    typedef VolOps::GradX         GradX;
   *    typedef VolOps::DivX          DivX;
   *  \endcode
   */
  template< typename CellT > struct OpTypes
    : public SpatialOps::structured::BasicOpTypes<CellT>
  {
    typedef UpwindInterpolant< CellT, typename FaceTypes<CellT>::XFace >    InterpC2FXUpwind;
    typedef UpwindInterpolant< CellT, typename FaceTypes<CellT>::YFace >    InterpC2FYUpwind;
    typedef UpwindInterpolant< CellT, typename FaceTypes<CellT>::ZFace >    InterpC2FZUpwind;
    
    typedef SuperbeeInterpolant< CellT, typename FaceTypes<CellT>::XFace >  InterpC2FXSuperbee;
    typedef SuperbeeInterpolant< CellT, typename FaceTypes<CellT>::YFace >  InterpC2FYSuperbee;
    typedef SuperbeeInterpolant< CellT, typename FaceTypes<CellT>::ZFace >  InterpC2FZSuperbee;    
  };
}

#endif // Wasatch_OperatorTypes_h
