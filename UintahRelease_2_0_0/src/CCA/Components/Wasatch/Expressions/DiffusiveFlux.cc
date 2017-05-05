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

#include <CCA/Components/Wasatch/Expressions/DiffusiveFlux.h>

//-- ExprLib includes --//
#include <expression/ExprLib.h>

//-- SpatialOps includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

//===================================================================

template< typename FluxT >
DiffusiveFlux<FluxT>::
DiffusiveFlux( const Expr::Tag& rhoTag,
               const Expr::Tag& turbDiffTag,
               const Expr::Tag& phiTag,
               const Expr::Tag& coefTag )
  : Expr::Expression<FluxT>(),
    isTurbulent_( turbDiffTag != Expr::Tag() ),
    isConstCoef_( false ),
    coefVal_    ( 0.0   )
{
   phi_ = this->template create_field_request<ScalarT>(phiTag);
   rho_ = this->template create_field_request<ScalarT>(rhoTag);
  if(!isConstCoef_ )  coef_ = this->template create_field_request<ScalarT>(coefTag);
  if( isTurbulent_ )  turbDiff_ = this->template create_field_request<ScalarT>(turbDiffTag);

  this->set_gpu_runnable( true );
}

template< typename FluxT >
DiffusiveFlux<FluxT>::
DiffusiveFlux( const Expr::Tag& rhoTag,
               const Expr::Tag& turbDiffTag,
               const Expr::Tag& phiTag,
               const double coefVal )
  : Expr::Expression<FluxT>(),
    isTurbulent_( turbDiffTag != Expr::Tag() ),
    isConstCoef_( true    ),
    coefVal_    ( coefVal )
{
   phi_ = this->template create_field_request<ScalarT>(phiTag);
   rho_ = this->template create_field_request<ScalarT>(rhoTag);
  if( isTurbulent_ )  turbDiff_ = this->template create_field_request<ScalarT>(turbDiffTag);

  this->set_gpu_runnable( true );
}

//--------------------------------------------------------------------

template< typename FluxT >
DiffusiveFlux<FluxT>::
~DiffusiveFlux()
{}


//--------------------------------------------------------------------

template< typename FluxT >
void
DiffusiveFlux<FluxT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  gradOp_   = opDB.retrieve_operator<GradT  >();
  interpOp_ = opDB.retrieve_operator<InterpT>();
}

//--------------------------------------------------------------------

template< typename FluxT >
void
DiffusiveFlux<FluxT>::
evaluate()
{
  using namespace SpatialOps;
  FluxT& result = this->value();
  if( isTurbulent_ ){
    if( isConstCoef_ ) result <<= - (*interpOp_)( rho_->field_ref() * (coefVal_ + turbDiff_->field_ref())) * (*gradOp_)(phi_->field_ref());
    else               result <<= - (*interpOp_)( rho_->field_ref() * (coef_->field_ref() + turbDiff_->field_ref())) * (*gradOp_)(phi_->field_ref());
  }
  else{
    if( isConstCoef_ ) result <<= - (*interpOp_)( rho_->field_ref() * coefVal_ ) * (*gradOp_)(phi_->field_ref());
    else               result <<= - (*interpOp_)( rho_->field_ref() * coef_->field_ref() ) * (*gradOp_)(phi_->field_ref());
  }
}

//--------------------------------------------------------------------

template< typename FluxT >
Expr::ExpressionBase*
DiffusiveFlux<FluxT>::Builder::build() const
{
  if( coeft_ == Expr::Tag() ) return new DiffusiveFlux<FluxT>( rhot_, turbDifft_, phit_, coefVal_ );
  else                        return new DiffusiveFlux<FluxT>( rhot_, turbDifft_, phit_, coeft_   );
}

//--------------------------------------------------------------------


//==========================================================================
// Explicit template instantiation for supported versions of this expression
//
#include <spatialops/structured/FVStaggered.h>

#define DECLARE_DIFF_FLUX( VOL )                                     \
  template class DiffusiveFlux< SpatialOps::FaceTypes<VOL>::XFace >; \
  template class DiffusiveFlux< SpatialOps::FaceTypes<VOL>::YFace >; \
  template class DiffusiveFlux< SpatialOps::FaceTypes<VOL>::ZFace >;

DECLARE_DIFF_FLUX( SpatialOps::SVolField );
//
//==========================================================================
