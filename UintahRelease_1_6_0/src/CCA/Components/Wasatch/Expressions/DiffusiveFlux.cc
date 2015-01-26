/*
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
    isConstCoef_( false       ),
    phiTag_     ( phiTag      ),
    coefTag_    ( coefTag     ),
    rhoTag_     ( rhoTag      ),
    turbDiffTag_( turbDiffTag ),
    coefVal_    ( 0.0         )
{
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
    isConstCoef_( true        ),
    phiTag_     ( phiTag      ),
    coefTag_    (             ),
    rhoTag_     ( rhoTag      ),
    turbDiffTag_( turbDiffTag ),
    coefVal_    ( coefVal     )
{
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
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( phiTag_ );
  exprDeps.requires_expression( rhoTag_ );
  if(!isConstCoef_ ) exprDeps.requires_expression( coefTag_     );
  if( isTurbulent_ ) exprDeps.requires_expression( turbDiffTag_ );
}

//--------------------------------------------------------------------

template< typename FluxT >
void
DiffusiveFlux<FluxT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<ScalarT>::type& scalarFM = fml.template field_manager<ScalarT  >();
  phi_  = &scalarFM.field_ref( phiTag_  );
  rho_  = &scalarFM.field_ref( rhoTag_  );
  if(!isConstCoef_ ) coef_     = &scalarFM.field_ref( coefTag_ );
  if( isTurbulent_ ) turbDiff_ = &scalarFM.field_ref( turbDiffTag_ );
}

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
    if( isConstCoef_ ) result <<= - (*interpOp_)( *rho_ * (coefVal_ + *turbDiff_)) * (*gradOp_)(*phi_);
    else               result <<= - (*interpOp_)( *rho_ * (*coef_   + *turbDiff_)) * (*gradOp_)(*phi_);
  }
  else{
    if( isConstCoef_ ) result <<= - (*interpOp_)( *rho_ * coefVal_ ) * (*gradOp_)(*phi_);
    else               result <<= - (*interpOp_)( *rho_ * *coef_   ) * (*gradOp_)(*phi_);
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

#define DECLARE_DIFF_FLUX( VOL )                                                 \
  template class DiffusiveFlux< SpatialOps::FaceTypes<VOL>::XFace >; \
  template class DiffusiveFlux< SpatialOps::FaceTypes<VOL>::YFace >; \
  template class DiffusiveFlux< SpatialOps::FaceTypes<VOL>::ZFace >;

DECLARE_DIFF_FLUX( SpatialOps::SVolField );
//
//==========================================================================
