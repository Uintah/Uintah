/**
 *  \file   MonolithicRHS.cc
 *
 *  \date   Apr 10, 2012
 *  \author James C. Sutherland
 *
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
    doX_( xconvFluxTag != Expr::Tag()),
    doY_( yconvFluxTag != Expr::Tag()),
    doZ_( zconvFluxTag != Expr::Tag()),
    doSrc_      ( srcTag != Expr::Tag() ),
    is3d_( doX_ && doY_ && doZ_ )
{
  assert( dCoefTag != Expr::Tag() );
  this->set_gpu_runnable( true );
   dCoef_ = this->template create_field_request<FieldT>(dCoefTag);
  if (doX_)  convFluxX_ = this->template create_field_request<XFaceT>(xconvFluxTag);
  if (doY_)  convFluxY_ = this->template create_field_request<YFaceT>(xconvFluxTag);
  if (doZ_)  convFluxZ_ = this->template create_field_request<ZFaceT>(xconvFluxTag);
  if (doSrc_)  src_ = this->template create_field_request<FieldT>(srcTag);
   phi_ = this->template create_field_request<FieldT>(phiTag);
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
  const FieldT& coef = dCoef_->field_ref();
  const FieldT& phi = phi_->field_ref();

  if( is3d_ ){ // inline everything for speed:
    const XFaceT& cfx = convFluxX_->field_ref();
    const YFaceT& cfy = convFluxY_->field_ref();
    const ZFaceT& cfz = convFluxZ_->field_ref();
    
    if( !doSrc_ ) {
      result <<= -(*divX_)( -(*interpX_)(coef) * (*gradX_)(phi) + cfx )
                 -(*divY_)( -(*interpY_)(coef) * (*gradY_)(phi) + cfy )
                 -(*divZ_)( -(*interpZ_)(coef) * (*gradZ_)(phi) + cfz );
    } else {
      const FieldT& src = src_->field_ref();
      result <<= -(*divX_)( -(*interpX_)(coef) * (*gradX_)(phi) + cfx )
                 -(*divY_)( -(*interpY_)(coef) * (*gradY_)(phi) + cfy )
                 -(*divZ_)( -(*interpZ_)(coef) * (*gradZ_)(phi) + cfz )
                  + src;
    }
  }
  else{
    if( !doX_ ) result <<=        -(*divX_)( -(*interpX_)(coef) * (*gradX_)(phi) );
    else        result <<=        -(*divX_)( -(*interpX_)(coef) * (*gradX_)(phi) + convFluxX_->field_ref() );
    if( !doY_ ) result <<= result -(*divY_)( -(*interpY_)(coef) * (*gradY_)(phi) );
    else        result <<= result -(*divY_)( -(*interpY_)(coef) * (*gradY_)(phi) + convFluxY_->field_ref() );
    if( !doZ_ ) result <<= result -(*divZ_)( -(*interpZ_)(coef) * (*gradZ_)(phi) );
    else        result <<= result -(*divZ_)( -(*interpZ_)(coef) * (*gradZ_)(phi) + convFluxZ_->field_ref() );
    if( doSrc_ ) result <<= result + src_->field_ref();
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
