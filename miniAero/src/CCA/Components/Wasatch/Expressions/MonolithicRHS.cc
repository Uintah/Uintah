/**
 *  \file   MonolithicRHS.cc
 *
 *  \date   Apr 10, 2012
 *  \author James C. Sutherland
 *
 * The MIT License
 *
 * Copyright (c) 2012-2015 The University of Utah
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

#include <CCA/Components/Wasatch/Expressions/MonolithicRHS.h>

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
    srcTag_      ( srcTag       ),
    is3d_( xconvFluxTag != Expr::Tag() && yconvFluxTag != Expr::Tag() && zconvFluxTag != Expr::Tag() && dCoefTag != Expr::Tag() )
{
  assert( dCoefTag != Expr::Tag() );
  this->set_gpu_runnable( true );
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
  const typename Expr::FieldMgrSelector<FieldT>::type& fm = fml.template field_manager<FieldT>();
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
    (FieldT(MemoryWindow(f.window_without_ghost().glob_dim(),               \
                                     f.window_without_ghost().offset() + theoffset,     \
                                     f.window_without_ghost().extent(),                 \
                                     f.window_without_ghost().has_bc(0),                \
                                     f.window_without_ghost().has_bc(1),                \
                                     f.window_without_ghost().has_bc(2)),               \
            f))

template< typename FieldT >
void
MonolithicRHS<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  const Expr::Tag nullTag = Expr::Tag();

  if( is3d_ ){ // inline everything for speed:
    if( srcTag_ == nullTag )
      result <<= -(*divX_)( -(*interpX_)(*dCoef_) * (*gradX_)(*phi_) + *convFluxX_ )
                 -(*divY_)( -(*interpY_)(*dCoef_) * (*gradY_)(*phi_) + *convFluxY_ )
                 -(*divZ_)( -(*interpZ_)(*dCoef_) * (*gradZ_)(*phi_) + *convFluxZ_ );
    else
      result <<= -(*divX_)( -(*interpX_)(*dCoef_) * (*gradX_)(*phi_) + *convFluxX_ )
                 -(*divY_)( -(*interpY_)(*dCoef_) * (*gradY_)(*phi_) + *convFluxY_ )
                 -(*divZ_)( -(*interpZ_)(*dCoef_) * (*gradZ_)(*phi_) + *convFluxZ_ )
                 + *src_;
  }
  else{
    if( xconvFluxTag_ == nullTag ) result <<=        -(*divX_)( -(*interpX_)(*dCoef_) * (*gradX_)(*phi_) );
    else                           result <<=        -(*divX_)( -(*interpX_)(*dCoef_) * (*gradX_)(*phi_) + *convFluxX_ );
    if( yconvFluxTag_ == nullTag ) result <<= result -(*divY_)( -(*interpY_)(*dCoef_) * (*gradY_)(*phi_) );
    else                           result <<= result -(*divY_)( -(*interpY_)(*dCoef_) * (*gradY_)(*phi_) + *convFluxY_ );
    if( zconvFluxTag_ == nullTag ) result <<= result -(*divZ_)( -(*interpZ_)(*dCoef_) * (*gradZ_)(*phi_) );
    else                           result <<= result -(*divZ_)( -(*interpZ_)(*dCoef_) * (*gradZ_)(*phi_) + *convFluxZ_ );
    if( srcTag_ != nullTag ) result <<= result + *src_;
  }

}

//--------------------------------------------------------------------

template< typename FieldT >
MonolithicRHS<FieldT>::
Builder::Builder( const Expr::Tag& resultTag,
                  const Expr::Tag& dCoefTag,
                  const Expr::Tag& xconvFluxTag,
                  const Expr::Tag& yconvFluxTag,
                  const Expr::Tag& zconvFluxTag,
                  const Expr::Tag& phiTag,
                  const Expr::Tag& srcTag )
  : ExpressionBuilder( resultTag ),
    dCoefTag_    ( dCoefTag     ),
    xconvFluxTag_( xconvFluxTag ),
    yconvFluxTag_( yconvFluxTag ),
    zconvFluxTag_( zconvFluxTag ),
    phiTag_      ( phiTag       ),
    srcTag_      ( srcTag       )
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
template class MonolithicRHS<SpatialOps::SVolField>;
template class MonolithicRHS<SpatialOps::XVolField>;
template class MonolithicRHS<SpatialOps::YVolField>;
template class MonolithicRHS<SpatialOps::ZVolField>;
