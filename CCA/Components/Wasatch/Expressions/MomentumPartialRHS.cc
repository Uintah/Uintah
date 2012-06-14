/*
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

#include "MomentumPartialRHS.h"

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>



template< typename FieldT >
MomRHSPart<FieldT>::
MomRHSPart( const Expr::Tag& convFluxX,
            const Expr::Tag& convFluxY,
            const Expr::Tag& convFluxZ,
            const Expr::Tag& tauX,
            const Expr::Tag& tauY,
            const Expr::Tag& tauZ,
            const Expr::Tag& densityTag,
            const Expr::Tag& bodyForce )
  : Expr::Expression<FieldT>(),
    cfluxXt_   ( convFluxX   ),
    cfluxYt_   ( convFluxY   ),
    cfluxZt_   ( convFluxZ   ),
    tauXt_     ( tauX        ),
    tauYt_     ( tauY        ),
    tauZt_     ( tauZ        ),
    densityt_  ( densityTag  ),
    bodyForcet_( bodyForce   ),
    emptyTag_  ( Expr::Tag() )
{}

//--------------------------------------------------------------------

template< typename FieldT >
MomRHSPart<FieldT>::
~MomRHSPart()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
MomRHSPart<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  if( cfluxXt_ != emptyTag_ )  exprDeps.requires_expression( cfluxXt_ );
  if( cfluxYt_ != emptyTag_ )  exprDeps.requires_expression( cfluxYt_ );
  if( cfluxZt_ != emptyTag_ )  exprDeps.requires_expression( cfluxZt_ );
  if( tauXt_   != emptyTag_ )  exprDeps.requires_expression( tauXt_   );
  if( tauYt_   != emptyTag_ )  exprDeps.requires_expression( tauYt_   );
  if( tauZt_   != emptyTag_ )  exprDeps.requires_expression( tauZt_   );
  exprDeps.requires_expression( densityt_);
  if( bodyForcet_!=emptyTag_)  exprDeps.requires_expression( bodyForcet_);
}

//--------------------------------------------------------------------

template< typename FieldT >
void
MomRHSPart<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<XFluxT>::type& xfm = fml.template field_manager<XFluxT>();
  const typename Expr::FieldMgrSelector<YFluxT>::type& yfm = fml.template field_manager<YFluxT>();
  const typename Expr::FieldMgrSelector<ZFluxT>::type& zfm = fml.template field_manager<ZFluxT>();

  if( cfluxXt_ != emptyTag_ )  cFluxX_ = &xfm.field_ref(cfluxXt_);
  if( cfluxYt_ != emptyTag_ )  cFluxY_ = &yfm.field_ref(cfluxYt_);
  if( cfluxZt_ != emptyTag_ )  cFluxZ_ = &zfm.field_ref(cfluxZt_);

  if( tauXt_ != emptyTag_ )  tauX_ = &xfm.field_ref(tauXt_);
  if( tauYt_ != emptyTag_ )  tauY_ = &yfm.field_ref(tauYt_);
  if( tauZt_ != emptyTag_ )  tauZ_ = &zfm.field_ref(tauZt_);

  density_ = &fml.field_manager<SVolField>().field_ref( densityt_ );

  if( bodyForcet_ != emptyTag_ )  bodyForce_ = &fml.field_manager<FieldT>().field_ref( bodyForcet_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
MomRHSPart<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  if( cfluxXt_ != emptyTag_ || tauXt_ != emptyTag_ )  divXOp_ = opDB.retrieve_operator<DivX>();
  if( cfluxYt_ != emptyTag_ || tauYt_ != emptyTag_ )  divYOp_ = opDB.retrieve_operator<DivY>();
  if( cfluxZt_ != emptyTag_ || tauZt_ != emptyTag_ )  divZOp_ = opDB.retrieve_operator<DivZ>();
  densityInterpOp_                                            = opDB.retrieve_operator<DensityInterpT>();
}

//--------------------------------------------------------------------

template< typename FieldT >
void
MomRHSPart<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  result <<= 0.0;

  SpatialOps::SpatFldPtr<FieldT> tmp = SpatialOps::SpatialFieldStore::get<FieldT>( result );

  if( cfluxXt_ != emptyTag_ ){
    divXOp_->apply_to_field( *cFluxX_, *tmp );
    result <<= result - *tmp;
  }

  if( cfluxYt_ != emptyTag_ ){
    divYOp_->apply_to_field( *cFluxY_, *tmp );
    result <<= result - *tmp;
  }

  if( cfluxZt_ != emptyTag_ ){
    divZOp_->apply_to_field( *cFluxZ_, *tmp );
    result <<= result - *tmp;
  }

  if( tauXt_ != emptyTag_ ){
    divXOp_->apply_to_field( *tauX_, *tmp );
    result <<= result - *tmp;
  }

  if( tauYt_ != emptyTag_ ){
    divYOp_->apply_to_field( *tauY_, *tmp );
    result <<= result - *tmp;
  }

  if( tauZt_ != emptyTag_ ){
    divZOp_->apply_to_field( *tauZ_, *tmp );
    result <<= result - *tmp;
  }

  if( bodyForcet_ != emptyTag_ ){
    densityInterpOp_->apply_to_field( *density_, *tmp );
    result <<= result + *tmp * *bodyForce_;
  }

}

//--------------------------------------------------------------------

template< typename FieldT >
MomRHSPart<FieldT>::
Builder::Builder( const Expr::Tag& result,
                  const Expr::Tag& convFluxX,
                  const Expr::Tag& convFluxY,
                  const Expr::Tag& convFluxZ,
                  const Expr::Tag& tauX,
                  const Expr::Tag& tauY,
                  const Expr::Tag& tauZ,
                  const Expr::Tag& densityTag,
                  const Expr::Tag& bodyForce )
  : ExpressionBuilder(result),
    cfluxXt_   ( convFluxX  ),
    cfluxYt_   ( convFluxY  ),
    cfluxZt_   ( convFluxZ  ),
    tauXt_     ( tauX       ),
    tauYt_     ( tauY       ),
    tauZt_     ( tauZ       ),
    densityt_  ( densityTag ),
    bodyForcet_( bodyForce  )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
MomRHSPart<FieldT>::Builder::build() const
{
  return new MomRHSPart<FieldT>( cfluxXt_, cfluxYt_, cfluxZt_, tauXt_, tauYt_, tauZt_, densityt_, bodyForcet_ );
}

//--------------------------------------------------------------------

//==================================================================
// Explicit template instantiation
template class MomRHSPart< SpatialOps::structured::XVolField >;
template class MomRHSPart< SpatialOps::structured::YVolField >;
template class MomRHSPart< SpatialOps::structured::ZVolField >;
//==================================================================
