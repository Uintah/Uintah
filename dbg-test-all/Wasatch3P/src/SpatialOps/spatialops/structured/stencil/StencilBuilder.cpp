/*
 * Copyright (c) 2014 The University of Utah
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

#include "StencilBuilder.h"
#include "FVStaggeredOperatorTypes.h"

#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/OperatorDatabase.h>

#include <spatialops/structured/Grid.h>

namespace SpatialOps{

#define REG_BASIC_OP_TYPES( VOL )                                       \
  {                                                                     \
    typedef BasicOpTypes<VOL>  OpTypes;                                 \
    opdb.register_new_operator( new OpTypes::InterpC2FX( coefHalf ) );  \
    opdb.register_new_operator( new OpTypes::InterpC2FY( coefHalf ) );  \
    opdb.register_new_operator( new OpTypes::InterpC2FZ( coefHalf ) );  \
    opdb.register_new_operator( new OpTypes::InterpF2CX( coefHalf ) );  \
    opdb.register_new_operator( new OpTypes::InterpF2CY( coefHalf ) );  \
    opdb.register_new_operator( new OpTypes::InterpF2CZ( coefHalf ) );  \
    opdb.register_new_operator( new OpTypes::GradX( coefDx ) );         \
    opdb.register_new_operator( new OpTypes::GradY( coefDy ) );         \
    opdb.register_new_operator( new OpTypes::GradZ( coefDz ) );         \
    opdb.register_new_operator( new OpTypes::DivX ( coefDx ) );         \
    opdb.register_new_operator( new OpTypes::DivY ( coefDy ) );         \
    opdb.register_new_operator( new OpTypes::DivZ ( coefDz ) );         \
  }

  //------------------------------------------------------------------

  void build_stencils( const unsigned int nx,
                       const unsigned int ny,
                       const unsigned int nz,
                       const double Lx,
                       const double Ly,
                       const double Lz,
                       OperatorDatabase& opdb )
  {
    const double dx = Lx/nx;
    const double dy = Ly/ny;
    const double dz = Lz/nz;

    //Coefficients:
    NeboStencilCoefCollection<2> coefHalf    = build_two_point_coef_collection( 0.5, 0.5 );
    NeboStencilCoefCollection<2> coefDx      = build_two_point_coef_collection( -1.0/dx, 1.0/dx );
    NeboStencilCoefCollection<2> coefDy      = build_two_point_coef_collection( -1.0/dy, 1.0/dy );
    NeboStencilCoefCollection<2> coefDz      = build_two_point_coef_collection( -1.0/dz, 1.0/dz );
    NeboStencilCoefCollection<4> coefQuarter = build_four_point_coef_collection( 0.25, 0.25, 0.25, 0.25 );
    NeboStencilCoefCollection<2> coefHalfDx  = build_two_point_coef_collection( -0.5/dx, 0.5/dx );
    NeboStencilCoefCollection<2> coefHalfDy  = build_two_point_coef_collection( -0.5/dy, 0.5/dy );
    NeboStencilCoefCollection<2> coefHalfDz  = build_two_point_coef_collection( -0.5/dz, 0.5/dz );

    //___________________________________________________________________
    // stencil2:
    //
    REG_BASIC_OP_TYPES( SVolField )  // basic operator types on a scalar volume
    REG_BASIC_OP_TYPES( XVolField )  // basic operator types on a x volume
    REG_BASIC_OP_TYPES( YVolField )  // basic operator types on a y volume
    REG_BASIC_OP_TYPES( ZVolField )  // basic operator types on a z volume

    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,XVolField,YSurfXField>::type( coefHalf ) );
    opdb.register_new_operator( new OperatorTypeBuilder<Gradient,   XVolField,YSurfXField>::type( coefDy   ) );
    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,XVolField,ZSurfXField>::type( coefHalf ) );
    opdb.register_new_operator( new OperatorTypeBuilder<Gradient,   XVolField,ZSurfXField>::type( coefDz   ) );

    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,YVolField,XSurfYField>::type( coefHalf ) );
    opdb.register_new_operator( new OperatorTypeBuilder<Gradient,   YVolField,XSurfYField>::type( coefDx   ) );
    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,YVolField,ZSurfYField>::type( coefHalf ) );
    opdb.register_new_operator( new OperatorTypeBuilder<Gradient,   YVolField,ZSurfYField>::type( coefDz   ) );

    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,ZVolField,XSurfZField>::type( coefHalf ) );
    opdb.register_new_operator( new OperatorTypeBuilder<Gradient   ,ZVolField,XSurfZField>::type( coefDx   ) );
    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,ZVolField,YSurfZField>::type( coefHalf ) );
    opdb.register_new_operator( new OperatorTypeBuilder<Gradient   ,ZVolField,YSurfZField>::type( coefDy   ) );

    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,SVolField,  XVolField>::type( coefHalf ) );
    opdb.register_new_operator( new OperatorTypeBuilder<Gradient   ,SVolField,  XVolField>::type( coefDx   ) );

    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,SVolField,  YVolField>::type( coefHalf ) );
    opdb.register_new_operator( new OperatorTypeBuilder<Gradient   ,SVolField,  YVolField>::type( coefDy   ) );

    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,SVolField,  ZVolField>::type( coefHalf ) );
    opdb.register_new_operator( new OperatorTypeBuilder<Gradient   ,SVolField,  ZVolField>::type( coefDz   ) );

    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,XVolField,  SVolField>::type( coefHalf ) );
    opdb.register_new_operator( new OperatorTypeBuilder<Gradient   ,XVolField,  SVolField>::type( coefDx   ) );

    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,YVolField,  SVolField>::type( coefHalf ) );
    opdb.register_new_operator( new OperatorTypeBuilder<Gradient   ,YVolField,  SVolField>::type( coefDy   ) );

    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,ZVolField,  SVolField>::type( coefHalf ) );
    opdb.register_new_operator( new OperatorTypeBuilder<Gradient   ,ZVolField,  SVolField>::type( coefDz   ) );

    //___________________________________________________________________
    // NullStencil:
    //
    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,SVolField,SVolField>::type() );
    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,XVolField,XVolField>::type() );
    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,YVolField,YVolField>::type() );
    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,ZVolField,ZVolField>::type() );

    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,XVolField,SSurfXField>::type() );
    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,YVolField,SSurfYField>::type() );
    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,ZVolField,SSurfZField>::type() );

    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,SVolField,XSurfXField>::type() );
    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,SVolField,YSurfYField>::type() );
    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,SVolField,ZSurfZField>::type() );

    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,XSurfXField,SVolField>::type() );
    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,YSurfYField,SVolField>::type() );
    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,ZSurfZField,SVolField>::type() );

    //___________________________________________________________________
    // stencil4:
    //
    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,SVolField,XSurfYField>::type( coefQuarter ) );
    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,SVolField,XSurfZField>::type( coefQuarter ) );

    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,SVolField,YSurfXField>::type( coefQuarter ) );
    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,SVolField,YSurfZField>::type( coefQuarter ) );

    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,SVolField,ZSurfXField>::type( coefQuarter ) );
    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,SVolField,ZSurfYField>::type( coefQuarter ) );

    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,XSurfYField,SVolField>::type( coefQuarter ) );
    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,XSurfZField,SVolField>::type( coefQuarter ) );
    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,YSurfXField,SVolField>::type( coefQuarter ) );
    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,YSurfZField,SVolField>::type( coefQuarter ) );
    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,ZSurfXField,SVolField>::type( coefQuarter ) );
    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,ZSurfYField,SVolField>::type( coefQuarter ) );

    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,XVolField,YVolField>::type( coefQuarter ) );
    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,XVolField,ZVolField>::type( coefQuarter ) );
    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,YVolField,XVolField>::type( coefQuarter ) );
    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,YVolField,ZVolField>::type( coefQuarter ) );
    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,ZVolField,XVolField>::type( coefQuarter ) );
    opdb.register_new_operator( new OperatorTypeBuilder<Interpolant,ZVolField,YVolField>::type( coefQuarter ) );

    //___________________________________________________________________
    // Box filter:
    //
    opdb.register_new_operator( new OperatorTypeBuilder<Filter,SVolField,SVolField>::type() );
    opdb.register_new_operator( new OperatorTypeBuilder<Filter,XVolField,XVolField>::type() );
    opdb.register_new_operator( new OperatorTypeBuilder<Filter,YVolField,YVolField>::type() );
    opdb.register_new_operator( new OperatorTypeBuilder<Filter,ZVolField,ZVolField>::type() );

    opdb.register_new_operator( new OperatorTypeBuilder<Filter,XSurfXField,XSurfXField>::type() );
    opdb.register_new_operator( new OperatorTypeBuilder<Filter,XSurfYField,XSurfYField>::type() );
    opdb.register_new_operator( new OperatorTypeBuilder<Filter,XSurfZField,XSurfZField>::type() );

    opdb.register_new_operator( new OperatorTypeBuilder<Filter,YSurfXField,YSurfXField>::type() );
    opdb.register_new_operator( new OperatorTypeBuilder<Filter,YSurfYField,YSurfYField>::type() );
    opdb.register_new_operator( new OperatorTypeBuilder<Filter,YSurfZField,YSurfZField>::type() );

    opdb.register_new_operator( new OperatorTypeBuilder<Filter,ZSurfXField,ZSurfXField>::type() );
    opdb.register_new_operator( new OperatorTypeBuilder<Filter,ZSurfYField,ZSurfYField>::type() );
    opdb.register_new_operator( new OperatorTypeBuilder<Filter,ZSurfZField,ZSurfZField>::type() );

    //___________________________________________________________________
    // Finite Difference stencils:
    //
    opdb.register_new_operator( new OperatorTypeBuilder<InterpolantX,SVolField,SVolField>::type( coefHalf ) );
    opdb.register_new_operator( new OperatorTypeBuilder<InterpolantY,SVolField,SVolField>::type( coefHalf ) );
    opdb.register_new_operator( new OperatorTypeBuilder<InterpolantZ,SVolField,SVolField>::type( coefHalf ) );
    opdb.register_new_operator( new OperatorTypeBuilder<GradientX,   SVolField,SVolField>::type( coefHalfDx ) );
    opdb.register_new_operator( new OperatorTypeBuilder<GradientY,   SVolField,SVolField>::type( coefHalfDy ) );
    opdb.register_new_operator( new OperatorTypeBuilder<GradientZ,   SVolField,SVolField>::type( coefHalfDz ) );

    opdb.register_new_operator( new OperatorTypeBuilder<InterpolantX,XVolField,XVolField>::type( coefHalf ) );
    opdb.register_new_operator( new OperatorTypeBuilder<InterpolantY,XVolField,XVolField>::type( coefHalf ) );
    opdb.register_new_operator( new OperatorTypeBuilder<InterpolantZ,XVolField,XVolField>::type( coefHalf ) );
    opdb.register_new_operator( new OperatorTypeBuilder<GradientX,   XVolField,XVolField>::type( coefHalfDx ) );
    opdb.register_new_operator( new OperatorTypeBuilder<GradientY,   XVolField,XVolField>::type( coefHalfDy ) );
    opdb.register_new_operator( new OperatorTypeBuilder<GradientZ,   XVolField,XVolField>::type( coefHalfDz ) );

    opdb.register_new_operator( new OperatorTypeBuilder<InterpolantX,YVolField,YVolField>::type( coefHalf ) );
    opdb.register_new_operator( new OperatorTypeBuilder<InterpolantY,YVolField,YVolField>::type( coefHalf ) );
    opdb.register_new_operator( new OperatorTypeBuilder<InterpolantZ,YVolField,YVolField>::type( coefHalf ) );
    opdb.register_new_operator( new OperatorTypeBuilder<GradientX,   YVolField,YVolField>::type( coefHalfDx ) );
    opdb.register_new_operator( new OperatorTypeBuilder<GradientY,   YVolField,YVolField>::type( coefHalfDy ) );
    opdb.register_new_operator( new OperatorTypeBuilder<GradientZ,   YVolField,YVolField>::type( coefHalfDz ) );

    opdb.register_new_operator( new OperatorTypeBuilder<InterpolantX,ZVolField,ZVolField>::type( coefHalf ) );
    opdb.register_new_operator( new OperatorTypeBuilder<InterpolantY,ZVolField,ZVolField>::type( coefHalf ) );
    opdb.register_new_operator( new OperatorTypeBuilder<InterpolantZ,ZVolField,ZVolField>::type( coefHalf ) );
    opdb.register_new_operator( new OperatorTypeBuilder<GradientX,   ZVolField,ZVolField>::type( coefHalfDx ) );
    opdb.register_new_operator( new OperatorTypeBuilder<GradientY,   ZVolField,ZVolField>::type( coefHalfDy ) );
    opdb.register_new_operator( new OperatorTypeBuilder<GradientZ,   ZVolField,ZVolField>::type( coefHalfDz ) );
}

  //------------------------------------------------------------------

  void build_stencils( const Grid& grid, OperatorDatabase& opDB )
  {
    build_stencils( grid.extent(0), grid.extent(1), grid.extent(2),
                    grid.length(0), grid.length(1), grid.length(2),
                    opDB );
  }

} // namespace SpatialOps
