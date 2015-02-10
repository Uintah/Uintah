#include <CCA/Components/Wasatch/Expressions/SolnVarEst.h>

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

//====================================================================

template< typename FieldT >
SolnVarEst<FieldT>::SolnVarEst( const Expr::Tag& solnVarOldTag,
                                const Expr::Tag& solnVarRHSTag,
                                const Expr::Tag& timeStepTag )
  : Expr::Expression<FieldT>()
{
  this->set_gpu_runnable( true );
  
  this->template create_field_request(solnVarOldTag, fOld_);
  this->template create_field_request(solnVarRHSTag, rhs_);
  this->template create_field_request(timeStepTag, dt_);
}

//------------------------------------------------------------------

template< typename FieldT >
SolnVarEst<FieldT>::~SolnVarEst()
{}

//------------------------------------------------------------------

template< typename FieldT >
void SolnVarEst<FieldT>::evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  const FieldT& fOld = fOld_->field_ref();
  const FieldT& rhs  = rhs_->field_ref();
  const TimeField& dt = dt_->field_ref();
  result <<= fOld + dt * rhs;
}

//------------------------------------------------------------------

template< typename FieldT >
SolnVarEst<FieldT>::Builder::Builder( const Expr::Tag& result,
                                      const Expr::Tag& solnVarOldTag,
                                      const Expr::Tag& solnVarRHSTag,
                                      const Expr::Tag& timeStepTag )
    : ExpressionBuilder(result),
      solnVarOldt_( solnVarOldTag ),
      solnVarRHSt_( solnVarRHSTag ),
      tstpt_      ( timeStepTag   )
{}

//------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
SolnVarEst<FieldT>::Builder::build() const
{
  return new SolnVarEst<FieldT>( solnVarOldt_, solnVarRHSt_, tstpt_ );
}
//------------------------------------------------------------------

//==========================================================================
// Explicit template instantiation for supported versions of this expression
#include <spatialops/structured/FVStaggered.h>
template class SolnVarEst< SpatialOps::SVolField >;
//==========================================================================
