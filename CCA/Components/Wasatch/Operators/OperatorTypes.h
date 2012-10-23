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

#ifndef Wasatch_OperatorTypes_h
#define Wasatch_OperatorTypes_h

#include <CCA/Components/Wasatch/FieldTypes.h>
#include "UpwindInterpolant.h"
#include "FluxLimiterInterpolant.h"

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
   *    - \b InterpC2FXLimiter - upwind or limited interpolants in x-dir
   *    - \b InterpC2FYLimiter - upwind or limited interpolants in y-dir
   *    - \b InterpC2FZLimiter - upwind or limited interpolants in z-dir
   *
   *  Example:
   *  \code
   *    typedef OpTypes< SVolField >  VolOps;
   *    typedef VolOps::GradX         GradX;
   *    typedef VolOps::DivX          DivX;
   *  \endcode
   *
   *  Operator types may also be determined by the helpful struct OperatorTypeBuilder.
   *  \code
   *    typedef OperatorTypeBuilder< Interpolant, SrcFieldT, DestFieldT >::type  InterpOperator;
   *    typedef OperatorTypeBuilder< Interpolant, XVolField, YSurfXField >::type InterpT;
   *  \endcode
   *  Note that if you have trouble compiling after defining one of
   *  these operator types, it may not be supported.
   */
  template< typename CellT > struct OpTypes
    : public SpatialOps::structured::BasicOpTypes<CellT>
  {
    typedef UpwindInterpolant< CellT, typename FaceTypes<CellT>::XFace >    InterpC2FXUpwind;
    typedef UpwindInterpolant< CellT, typename FaceTypes<CellT>::YFace >    InterpC2FYUpwind;
    typedef UpwindInterpolant< CellT, typename FaceTypes<CellT>::ZFace >    InterpC2FZUpwind;

    typedef FluxLimiterInterpolant< CellT, typename FaceTypes<CellT>::XFace >  InterpC2FXLimiter;
    typedef FluxLimiterInterpolant< CellT, typename FaceTypes<CellT>::YFace >  InterpC2FYLimiter;
    typedef FluxLimiterInterpolant< CellT, typename FaceTypes<CellT>::ZFace >  InterpC2FZLimiter;
  };
}

#endif // Wasatch_OperatorTypes_h
