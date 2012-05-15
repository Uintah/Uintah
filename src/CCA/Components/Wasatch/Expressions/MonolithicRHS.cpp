/**
 *  \file   MonolithicRHS.cpp
 *
 *  \date   Apr 10, 2012
 *  \author James C. Sutherland
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

#include "MonolithicRHS.h"

#include <spatialops/structured/FVStaggered.h>

template< typename FieldT >
MonolithicRHS<FieldT>::
MonolithicRHS( const Expr::Tag& dCoefTag,
               const Expr::Tag& xconvFluxTag,
               const Expr::Tag& yconvFluxTag,
               const Expr::Tag& zconvFluxTag,
               const Expr::Tag& phiTag,
               const Expr::Tag& srcTag )
  : Expr::Expression<FieldT>(),
    dCoefTag_    ( dCoefTag     ),
    xconvFluxTag_( xconvFluxTag ),
    yconvFluxTag_( yconvFluxTag ),
    zconvFluxTag_( zconvFluxTag ),
    phiTag_      ( phiTag       ),
    srcTag_      ( srcTag       )
{
  // right now we must have all directions active with convection and diffusion.
  assert( xconvFluxTag != Expr::Tag() );
  assert( yconvFluxTag != Expr::Tag() );
  assert( zconvFluxTag != Expr::Tag() );
  assert( dCoefTag     != Expr::Tag() );
}

//--------------------------------------------------------------------

template< typename FieldT >
MonolithicRHS<FieldT>::
~MonolithicRHS()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
MonolithicRHS<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( dCoefTag_    );
  exprDeps.requires_expression( phiTag_      );
  if( xconvFluxTag_ != Expr::Tag() ) exprDeps.requires_expression( xconvFluxTag_ );
  if( yconvFluxTag_ != Expr::Tag() ) exprDeps.requires_expression( yconvFluxTag_ );
  if( zconvFluxTag_ != Expr::Tag() ) exprDeps.requires_expression( zconvFluxTag_ );
  if( srcTag_       != Expr::Tag() ) exprDeps.requires_expression( srcTag_       );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
MonolithicRHS<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<FieldT>& fm = fml.template field_manager<FieldT>();
  dCoef_ = &fm.field_ref( dCoefTag_ );
  phi_   = &fm.field_ref( phiTag_   );

  if( srcTag_ != Expr::Tag() ) src_ = &fm.field_ref( srcTag_ );

  if( xconvFluxTag_ != Expr::Tag() ) convFluxX_ = &fml.template field_manager<XFaceT>().field_ref( xconvFluxTag_ );
  if( yconvFluxTag_ != Expr::Tag() ) convFluxY_ = &fml.template field_manager<YFaceT>().field_ref( yconvFluxTag_ );
  if( zconvFluxTag_ != Expr::Tag() ) convFluxZ_ = &fml.template field_manager<ZFaceT>().field_ref( zconvFluxTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
MonolithicRHS<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  interpX_ = opDB.retrieve_operator<InterpX>();
  gradX_   = opDB.retrieve_operator<GradX  >();
  divX_    = opDB.retrieve_operator<DivX   >();
  interpY_ = opDB.retrieve_operator<InterpY>();
  gradY_   = opDB.retrieve_operator<GradY  >();
  divY_    = opDB.retrieve_operator<DivY   >();
  interpZ_ = opDB.retrieve_operator<InterpZ>();
  gradZ_   = opDB.retrieve_operator<GradZ  >();
  divZ_    = opDB.retrieve_operator<DivZ   >();
}

//--------------------------------------------------------------------

#define build_field(f, theoffset)                                                       \
    (FieldT(structured::MemoryWindow(f.window_without_ghost().glob_dim(),               \
                                     f.window_without_ghost().offset() + theoffset,     \
                                     f.window_without_ghost().extent(),                 \
                                     f.window_without_ghost().has_bc(0),                \
                                     f.window_without_ghost().has_bc(1),                \
                                     f.window_without_ghost().has_bc(2)),               \
            f.field_values(),                                                           \
            structured::ExternalStorage))

template< typename FieldT >
void
MonolithicRHS<FieldT>::
evaluate()
{
  FieldT& result = this->value();

  using namespace SpatialOps;
  using SpatialOps::structured::IntVec;

  const IntVec neutral( 0, 0, 0 );
  const IntVec neg_X  (-1, 0, 0 );
  const IntVec pos_X  ( 1, 0, 0 );
  const IntVec neg_Y  ( 0,-1, 0 );
  const IntVec pos_Y  ( 0, 1, 0 );
  const IntVec neg_Z  ( 0, 0,-1 );
  const IntVec pos_Z  ( 0, 0, 1 );

  FieldT r = build_field(result, neutral);

  const FieldT phi_xminus = build_field( (*phi_), neg_X   );
  const FieldT phi_x0_1   = build_field( (*phi_), neutral );
  const FieldT phi_x0_2   = build_field( (*phi_), neutral );
  const FieldT phi_xplus  = build_field( (*phi_), pos_X   );
  const FieldT phi_yminus = build_field( (*phi_), neg_Y   );
  const FieldT phi_y0_1   = build_field( (*phi_), neutral );
  const FieldT phi_y0_2   = build_field( (*phi_), neutral );
  const FieldT phi_yplus  = build_field( (*phi_), pos_Y   );
  const FieldT phi_zminus = build_field( (*phi_), neg_Z   );
  const FieldT phi_z0_1   = build_field( (*phi_), neutral );
  const FieldT phi_z0_2   = build_field( (*phi_), neutral );
  const FieldT phi_zplus  = build_field( (*phi_), pos_Z   );

//  const FieldT cflux_xm   = build_field( (*convFluxX_), neutral );
//  const FieldT cflux_xp   = build_field( (*convFluxX_), pos_X   );
//  const FieldT cflux_ym   = build_field( (*convFluxY_), neutral );
//  const FieldT cflux_yp   = build_field( (*convFluxY_), pos_Y   );
//  const FieldT cflux_zm   = build_field( (*convFluxZ_), neutral );
//  const FieldT cflux_zp   = build_field( (*convFluxZ_), pos_Z   );
  const double cflux_xm=0, cflux_xp=0, cflux_ym=0, cflux_yp=0, cflux_zm=0, cflux_zp=0;

  const FieldT dCoef_xminus = build_field( (*dCoef_), neg_X   );
  const FieldT dCoef_x0_1   = build_field( (*dCoef_), neutral );
  const FieldT dCoef_x0_2   = build_field( (*dCoef_), neutral );
  const FieldT dCoef_xplus  = build_field( (*dCoef_), pos_X   );
  const FieldT dCoef_yminus = build_field( (*dCoef_), neg_Y   );
  const FieldT dCoef_y0_1   = build_field( (*dCoef_), neutral );
  const FieldT dCoef_y0_2   = build_field( (*dCoef_), neutral );
  const FieldT dCoef_yplus  = build_field( (*dCoef_), pos_Y   );
  const FieldT dCoef_zminus = build_field( (*dCoef_), neg_Z   );
  const FieldT dCoef_z0_1   = build_field( (*dCoef_), neutral );
  const FieldT dCoef_z0_2   = build_field( (*dCoef_), neutral );
  const FieldT dCoef_zplus  = build_field( (*dCoef_), pos_Z   );

  // get the stencil coefficients from the operators
  const double gXl = gradX_  ->get_minus_coef();
  const double gXh = gradX_  ->get_plus_coef();
  const double iXl = interpX_->get_minus_coef();
  const double iXh = interpX_->get_plus_coef();
  const double dXl = divX_   ->get_minus_coef();
  const double dXh = divX_   ->get_plus_coef();

  const double gYl = gradY_  ->get_minus_coef();
  const double gYh = gradY_  ->get_plus_coef();
  const double iYl = interpY_->get_minus_coef();
  const double iYh = interpY_->get_plus_coef();
  const double dYl = divY_   ->get_minus_coef();
  const double dYh = divY_   ->get_plus_coef();

  const double gZl = gradZ_  ->get_minus_coef();
  const double gZh = gradZ_  ->get_plus_coef();
  const double iZl = interpZ_->get_minus_coef();
  const double iZh = interpZ_->get_plus_coef();
  const double dZl = divZ_   ->get_minus_coef();
  const double dZh = divZ_   ->get_plus_coef();

  // build the full RHS including diffusive & reactive terms
  r <<= (
      ( /* x-direction convective and diffusive flux contributions - building diffusive flux inline */
          dXl * ( -cflux_xm + (gXl * phi_xminus + gXh * phi_x0_1 ) * (iXl * dCoef_xminus + iXh * dCoef_x0_1 ) ) +
          dXh * ( -cflux_xp + (gXl * phi_x0_2   + gXh * phi_xplus) * (iXl * dCoef_x0_2   + iXh * dCoef_xplus) )
      )
      +
      ( /* y-direction convective and diffusive flux contributions - building diffusive flux inline */
          dYl * ( -cflux_ym + (gYl * phi_yminus + gYh * phi_y0_1 ) * (iYl * dCoef_yminus + iYh * dCoef_y0_1 ) ) +
          dYh * ( -cflux_yp + (gYl * phi_y0_2   + gYh * phi_yplus) * (iYl * dCoef_y0_2   + iYh * dCoef_yplus) )
      )
      +
      ( /* z-direction convective and diffusive flux contributions - building diffusive flux inline */
          dZl * ( -cflux_zm + (gZl * phi_zminus + gZh * phi_z0_1 ) * (iZl * dCoef_zminus + iZh * dCoef_z0_1 ) ) +
          dZh * ( -cflux_zp + (gZl * phi_z0_2   + gZh * phi_zplus) * (iZl * dCoef_z0_2   + iZh * dCoef_zplus) )
      )
  );

  if( srcTag_ != Expr::Tag() ) result <<= result + *src_;
}

//--------------------------------------------------------------------

template< typename FieldT >
MonolithicRHS<FieldT>::
Builder::Builder( const Expr::Tag& resultTag,
                  const Expr::Tag& dCoefTag,
//                  const Expr::Tag& xconvFluxTag,
//                  const Expr::Tag& yconvFluxTag,
//                  const Expr::Tag& zconvFluxTag,
                  const Expr::Tag& phiTag,
                  const Expr::Tag& srcTag )
  : ExpressionBuilder( resultTag ),
    dCoefTag_   ( dCoefTag    ),
    xconvFluxTag_( Expr::Tag() ), //xconvFluxTag ),
    yconvFluxTag_( Expr::Tag() ), //yconvFluxTag ),
    zconvFluxTag_( Expr::Tag() ), //zconvFluxTag ),
    phiTag_     ( phiTag      ),
    srcTag_     ( srcTag      )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
MonolithicRHS<FieldT>::
Builder::build() const
{
  return new MonolithicRHS<FieldT>( dCoefTag_, xconvFluxTag_, yconvFluxTag_, zconvFluxTag_, phiTag_, srcTag_ );
}

//--------------------------------------------------------------------



//--- Explicit template instantiations ---
template class MonolithicRHS<SpatialOps::structured::SVolField>;
template class MonolithicRHS<SpatialOps::structured::XVolField>;
template class MonolithicRHS<SpatialOps::structured::YVolField>;
template class MonolithicRHS<SpatialOps::structured::ZVolField>;
