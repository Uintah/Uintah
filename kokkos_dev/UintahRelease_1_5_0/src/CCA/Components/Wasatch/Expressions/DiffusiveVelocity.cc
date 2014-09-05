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

#include "DiffusiveVelocity.h"

//-- ExprLib includes --//
#include <expression/ExprLib.h>

//-- SpatialOps includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>  // jcs need to rework spatialops install structure

template< typename GradT >
DiffusiveVelocity<GradT>::DiffusiveVelocity( const Expr::Tag& turbDiffTag,
                                             const Expr::Tag phiTag,
                                             const Expr::Tag coefTag )
  : Expr::Expression<VelT>(),
    isConstCoef_( false ),
    isTurbulent_( turbDiffTag != Expr::Tag() ),
    phiTag_     ( phiTag      ),
    coefTag_    ( coefTag     ),
    turbDiffTag_( turbDiffTag ),
    coefVal_    ( 0.0         )
{}

//--------------------------------------------------------------------

template< typename GradT >
DiffusiveVelocity<GradT>::DiffusiveVelocity( const Expr::Tag& turbDiffTag,
                                             const Expr::Tag phiTag,
                                             const double coef )
  : Expr::Expression<VelT>(),
    isConstCoef_( true ),
    isTurbulent_( turbDiffTag != Expr::Tag()    ),
    phiTag_ ( phiTag ),
    coefTag_( Expr::Tag() ),
    turbDiffTag_( turbDiffTag ),
    coefVal_( coef )
{}

//--------------------------------------------------------------------

template< typename GradT >
DiffusiveVelocity<GradT>::
~DiffusiveVelocity()
{}

//--------------------------------------------------------------------

template< typename GradT >
void
DiffusiveVelocity<GradT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( phiTag_ );
  if( isTurbulent_  ) exprDeps.requires_expression( turbDiffTag_ );
  if( !isConstCoef_ ) exprDeps.requires_expression( coefTag_     );
}

//--------------------------------------------------------------------

template< typename GradT >
void
DiffusiveVelocity<GradT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<VelT   >::type& velFM    = fml.template field_manager<VelT   >();
  const typename Expr::FieldMgrSelector<ScalarT>::type& scalarFM = fml.template field_manager<ScalarT>();

  phi_ = &scalarFM.field_ref( phiTag_ );
  if( isTurbulent_  ) turbDiff_ = &fml.template field_manager<SVolField>().field_ref( turbDiffTag_ );
  if( !isConstCoef_ ) coef_ = &velFM.field_ref( coefTag_ );
}

//--------------------------------------------------------------------

template< typename GradT >
void
DiffusiveVelocity<GradT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  gradOp_ = opDB.retrieve_operator<GradT>();
  if( isTurbulent_ ) sVolInterpOp_ = opDB.retrieve_operator<SVolInterpT>();
}

//--------------------------------------------------------------------

template< typename GradT >
void
DiffusiveVelocity<GradT>::
evaluate()
{
  using namespace SpatialOps;
  VelT& result = this->value();

  gradOp_->apply_to_field( *phi_, result );  // V = grad(phi)
  
  SpatFldPtr<VelT> tmp = SpatialFieldStore::get<VelT>( result );
  *tmp <<= 0.0;
  
  if (isTurbulent_) {
    sVolInterpOp_->apply_to_field( *turbDiff_, *tmp );
  }
  
  if( isConstCoef_ ){
    *tmp <<= *tmp + coefVal_;     // gamma_mix = gamma + gamma_T
  }
  else{
    *tmp <<= *tmp + *coef_;       // gamma_mix = gamma + gamma_T
  }
  
  result <<= -result * *tmp;      // J =  - gamma * grad(phi)

}


//====================================================================


template< typename GradT, typename InterpT >
DiffusiveVelocity2<GradT,InterpT>::
DiffusiveVelocity2( const Expr::Tag& turbDiffTag,
                    const Expr::Tag& phiTag,
                    const Expr::Tag& coefTag )
  : Expr::Expression<VelT>(),
    isTurbulent_( turbDiffTag != Expr::Tag() ),
    phiTag_     ( phiTag      ),
    coefTag_    ( coefTag     ),
    turbDiffTag_( turbDiffTag )

{}

//--------------------------------------------------------------------

template< typename GradT, typename InterpT >
DiffusiveVelocity2<GradT,InterpT>::
~DiffusiveVelocity2()
{}

//--------------------------------------------------------------------

template< typename GradT, typename InterpT >
void
DiffusiveVelocity2<GradT,InterpT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( phiTag_ );
  exprDeps.requires_expression( coefTag_ );
  if (isTurbulent_) exprDeps.requires_expression( turbDiffTag_ );  
}

//--------------------------------------------------------------------

template< typename GradT, typename InterpT >
void
DiffusiveVelocity2<GradT,InterpT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<ScalarT>::type& scalarFM = fml.template field_manager<ScalarT>();
  phi_  = &scalarFM.field_ref( phiTag_  );
  coef_ = &scalarFM.field_ref( coefTag_ );
  if (isTurbulent_) turbDiff_  = &fml.template field_manager<SVolField>().field_ref( turbDiffTag_  );

}

//--------------------------------------------------------------------

template< typename GradT, typename InterpT >
void
DiffusiveVelocity2<GradT,InterpT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  gradOp_       = opDB.retrieve_operator<GradT  >();
  interpOp_     = opDB.retrieve_operator<InterpT>();
  sVolInterpOp_ = opDB.retrieve_operator<SVolInterpT>();
}

//--------------------------------------------------------------------

template< typename GradT, typename InterpT >
void
DiffusiveVelocity2<GradT,InterpT>::
evaluate()
{
  using namespace SpatialOps;
  VelT& result = this->value();

  SpatFldPtr<VelT> velTmp = SpatialFieldStore::get<VelT>( result );

  gradOp_  ->apply_to_field( *phi_, result );  // V = grad(phi)
  interpOp_->apply_to_field( *coef_, *velTmp  );

  SpatFldPtr<VelT> tmp = SpatialFieldStore::get<VelT>( result );
  *tmp <<= 0.0;
  if (isTurbulent_) {
    sVolInterpOp_->apply_to_field( *turbDiff_, *tmp );
    *velTmp <<= *velTmp + *tmp;                // gamma_mix = gamma + gamma_T
  }
  
  result <<= -result * *velTmp;                 // V = - gamma * grad(phi)
}

//--------------------------------------------------------------------


//==========================================================================
// Explicit template instantiation for supported versions of this expression
//
#include <spatialops/structured/FVStaggered.h>

#define DECLARE_DIFF_VELOCITY( VOL )                                                            \
  template class DiffusiveVelocity< SpatialOps::structured::BasicOpTypes<VOL>::GradX >;         \
  template class DiffusiveVelocity< SpatialOps::structured::BasicOpTypes<VOL>::GradY >;	        \
  template class DiffusiveVelocity< SpatialOps::structured::BasicOpTypes<VOL>::GradZ >;	        \
  template class DiffusiveVelocity2< SpatialOps::structured::BasicOpTypes<VOL>::GradX,	        \
                                     SpatialOps::structured::BasicOpTypes<VOL>::InterpC2FX >;   \
  template class DiffusiveVelocity2< SpatialOps::structured::BasicOpTypes<VOL>::GradY,          \
                                     SpatialOps::structured::BasicOpTypes<VOL>::InterpC2FY >;   \
  template class DiffusiveVelocity2< SpatialOps::structured::BasicOpTypes<VOL>::GradZ,          \
                                     SpatialOps::structured::BasicOpTypes<VOL>::InterpC2FZ >;

DECLARE_DIFF_VELOCITY( SpatialOps::structured::SVolField );
//
//==========================================================================
