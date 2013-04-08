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

#include "DiffusiveFlux.h"

//-- ExprLib includes --//
#include <expression/ExprLib.h>

//-- SpatialOps includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

template< typename ScalarT, typename FluxT >
DiffusiveFlux<ScalarT, FluxT>::DiffusiveFlux( const Expr::Tag& rhoTag,
                                              const Expr::Tag& turbDiffTag,
                                              const Expr::Tag& phiTag,
                                              const Expr::Tag& coefTag )
  : Expr::Expression<FluxT>(),
    isConstCoef_( false ),
    isTurbulent_( turbDiffTag != Expr::Tag() ),
    phiTag_     ( phiTag      ),
    coefTag_    ( coefTag     ),
    rhoTag_     ( rhoTag      ),
    turbDiffTag_( turbDiffTag ),
    coefVal_    ( 0.0         )
{}

//--------------------------------------------------------------------

template< typename ScalarT, typename FluxT >
DiffusiveFlux<ScalarT, FluxT>::DiffusiveFlux( const Expr::Tag& rhoTag,
                                              const Expr::Tag& turbDiffTag,
                                              const Expr::Tag& phiTag,
                                              const double coef )
  : Expr::Expression<FluxT>(),
    isConstCoef_( true        ),
    isTurbulent_( turbDiffTag != Expr::Tag()    ),
    phiTag_     ( phiTag      ),
    coefTag_    ( "NULL", Expr::INVALID_CONTEXT ),
    rhoTag_     ( rhoTag      ),
    turbDiffTag_( turbDiffTag ),
    coefVal_    ( coef        )
{}

//--------------------------------------------------------------------

template< typename ScalarT, typename FluxT >
DiffusiveFlux<ScalarT, FluxT>::
~DiffusiveFlux()
{}

//--------------------------------------------------------------------

template< typename ScalarT, typename FluxT >
void
DiffusiveFlux<ScalarT, FluxT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( phiTag_ );
  if( isTurbulent_  ) exprDeps.requires_expression( turbDiffTag_ );
  if( !isConstCoef_ ) exprDeps.requires_expression( coefTag_     );
  exprDeps.requires_expression( rhoTag_  );
}

//--------------------------------------------------------------------

template< typename ScalarT, typename FluxT >
void
DiffusiveFlux<ScalarT, FluxT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  phi_ = &fml.template field_manager<ScalarT  >().field_ref( phiTag_ );
  rho_ = &fml.template field_manager<SVolField>().field_ref( rhoTag_ );
  if( isTurbulent_  ) turbDiff_ = &fml.template field_manager<SVolField>().field_ref( turbDiffTag_ );
  if( !isConstCoef_ ) coef_     = &fml.template field_manager<FluxT>().field_ref  ( coefTag_  );
}

//--------------------------------------------------------------------

template< typename ScalarT, typename FluxT >
void
DiffusiveFlux<ScalarT, FluxT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  gradOp_          = opDB.retrieve_operator<GradT>();
  sVolInterpOp_ = opDB.retrieve_operator<SVolInterpT>();
}

//--------------------------------------------------------------------

template< typename ScalarT, typename FluxT >
void
DiffusiveFlux<ScalarT, FluxT>::
evaluate()
{
  using namespace SpatialOps;
  FluxT& result = this->value();

//  
//  
////  gradOp_->apply_to_field( *phi_, result );  // J = grad(phi)
//
//  SpatFldPtr<FluxT> gammaTotal = SpatialFieldStore::get<FluxT>( result );
//  *gammaTotal <<= 0.0;
//  
//  if (isTurbulent_) {
//    sVolInterpOp_->apply_to_field( *turbDiff_, *gammaTotal );
//  }
//  
//  *gammaTotal <<= *gammaTotal + coefVal_;
////  if( isConstCoef_ ){
////    *gammaTotal <<= *gammaTotal + coefVal_;     // gamma_mix = gamma + gamma_T
////  }
////  else{
////    *gammaTotal <<= *gammaTotal + *coef_;       // gamma_mix = gamma + gamma_T
////  }
//  
//  result <<= - *gammaTotal * (*sVolInterpOp_)(*rho_) * (*gradOp_)(*phi_);
//  
//  
////  result <<= -result * *tmp;      // J =  - gamma * grad(phi)
////  
////  SpatFldPtr<FluxT> interpRho = SpatialFieldStore::get<FluxT>(result);
////  sVolInterpOp_->apply_to_field( *rho_, *interpRho );
////  result <<= result * *interpRho;               // J = - rho * gamma * grad(phi)

  if (isTurbulent_) {
    result <<= - (*sVolInterpOp_)(*rho_) * (coefVal_ + (*sVolInterpOp_)(*turbDiff_)) * (*gradOp_)(*phi_);
  } else {
    result <<= - (*sVolInterpOp_)(*rho_) * coefVal_ * (*gradOp_)(*phi_);
  }

}


//====================================================================


template< typename ScalarT, typename FluxT >
DiffusiveFlux2<ScalarT, FluxT>::
DiffusiveFlux2( const Expr::Tag& rhoTag,
                const Expr::Tag& turbDiffTag,
                const Expr::Tag& phiTag,
                const Expr::Tag& coefTag )
  : Expr::Expression<FluxT>(),
    isTurbulent_( turbDiffTag != Expr::Tag() ),
    phiTag_     ( phiTag      ),
    coefTag_    ( coefTag     ),
    rhoTag_     ( rhoTag      ),
    turbDiffTag_( turbDiffTag )
{}

//--------------------------------------------------------------------

template< typename ScalarT, typename FluxT >
DiffusiveFlux2<ScalarT, FluxT>::
~DiffusiveFlux2()
{}

//--------------------------------------------------------------------

template< typename ScalarT, typename FluxT >
void
DiffusiveFlux2<ScalarT, FluxT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( phiTag_ );
  exprDeps.requires_expression( coefTag_ );
  exprDeps.requires_expression( rhoTag_ );
  if (isTurbulent_) exprDeps.requires_expression( turbDiffTag_ );
}

//--------------------------------------------------------------------

template< typename ScalarT, typename FluxT >
void
DiffusiveFlux2<ScalarT, FluxT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<ScalarT>::type& scalarFM = fml.template field_manager<ScalarT>();
  phi_  = &scalarFM.field_ref( phiTag_  );
  coef_ = &scalarFM.field_ref( coefTag_ );
  rho_ = &fml.template field_manager<SVolField>().field_ref( rhoTag_ );
  if (isTurbulent_) turbDiff_  = &fml.template field_manager<SVolField>().field_ref( turbDiffTag_  );
}

//--------------------------------------------------------------------

template< typename ScalarT, typename FluxT >
void
DiffusiveFlux2<ScalarT, FluxT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  gradOp_       = opDB.retrieve_operator<GradT  >();
  sVolInterpOp_ = opDB.retrieve_operator<SVolInterpT>();
  interpOp_     = opDB.retrieve_operator<InterpT>();
}

//--------------------------------------------------------------------

template< typename ScalarT, typename FluxT >
void
DiffusiveFlux2<ScalarT, FluxT>::
evaluate()
{
  using namespace SpatialOps;
  FluxT& result = this->value();

//  SpatFldPtr<FluxT> fluxTmp = SpatialFieldStore::get<FluxT>( result );
//
//  gradOp_  ->apply_to_field( *phi_, result );  // J = grad(phi)  
//  interpOp_->apply_to_field( *coef_, *fluxTmp  );
//  
//  SpatFldPtr<FluxT> tmp = SpatialFieldStore::get<FluxT>( result );
//  *tmp <<= 0.0;
//  if (isTurbulent_) {
//    sVolInterpOp_->apply_to_field( *turbDiff_, *tmp );
//    *fluxTmp <<= *fluxTmp + *tmp;                // gamma_mix = gamma + gamma_T
//  }
//  
//  result <<= -result * *fluxTmp;                 // J = - gamma * grad(phi)
//
//  sVolInterpOp_->apply_to_field( *rho_, *fluxTmp );
//  result <<= result * *fluxTmp;               // J = - rho * gamma * grad(phi)
//  
//  
  if (isTurbulent_) {
    result <<= - (*sVolInterpOp_)(*rho_) * ((*interpOp_)(*coef_) + (*interpOp_)(*turbDiff_)) * (*gradOp_)(*phi_);
  } else {
    result <<= - (*sVolInterpOp_)(*rho_) * (*interpOp_)(*coef_) * (*gradOp_)(*phi_);
  }
}

//--------------------------------------------------------------------


//==========================================================================
// Explicit template instantiation for supported versions of this expression
//
#include <spatialops/structured/FVStaggered.h>

#define DECLARE_DIFF_FLUX( VOL )                                                       \
  template class DiffusiveFlux < VOL, SpatialOps::structured::FaceTypes<VOL>::XFace >; \
  template class DiffusiveFlux < VOL, SpatialOps::structured::FaceTypes<VOL>::YFace >; \
  template class DiffusiveFlux < VOL, SpatialOps::structured::FaceTypes<VOL>::ZFace >; \
  template class DiffusiveFlux2< VOL, SpatialOps::structured::FaceTypes<VOL>::XFace >; \
  template class DiffusiveFlux2< VOL, SpatialOps::structured::FaceTypes<VOL>::YFace >; \
  template class DiffusiveFlux2< VOL, SpatialOps::structured::FaceTypes<VOL>::ZFace >;

DECLARE_DIFF_FLUX( SpatialOps::structured::SVolField );
//
//==========================================================================
