#include <CCA/Components/Wasatch/Expressions/SolnVarEst.h>

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

//====================================================================

template< typename FieldT >
SolnVarEst<FieldT>::SolnVarEst( const Expr::Tag& solnVarOldTag,
                                const Expr::Tag& solnVarRHSTag,
                                const Expr::Tag& timeStepTag )
  : Expr::Expression<FieldT>(),
    solnVarOldt_ ( solnVarOldTag ),
    solnVarRHSt_ ( solnVarRHSTag ),
    tStept_      ( timeStepTag   )
{
  this->set_gpu_runnable( true );
}

//------------------------------------------------------------------

template< typename FieldT >
SolnVarEst<FieldT>::~SolnVarEst()
{}

//------------------------------------------------------------------

template< typename FieldT >
void SolnVarEst<FieldT>::advertise_dependents( Expr::ExprDeps& exprDeps )
{  
  exprDeps.requires_expression( solnVarOldt_ );
  exprDeps.requires_expression( solnVarRHSt_ );
  exprDeps.requires_expression( tStept_      );  
}

//------------------------------------------------------------------

template< typename FieldT >
void SolnVarEst<FieldT>::bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<FieldT>::type& FM = fml.field_manager<FieldT>();
  
  solnVarOld_ = &FM.field_ref ( solnVarOldt_ );    
  solnVarRHS_ = &FM.field_ref ( solnVarRHSt_ );    

  tStep_ = &fml.field_ref<TimeField>( tStept_ );
}

//------------------------------------------------------------------

template< typename FieldT >
void SolnVarEst<FieldT>::evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  result <<= *solnVarOld_ + *tStep_ * *solnVarRHS_ ;
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
template class SolnVarEst< SpatialOps::structured::SVolField >;
//==========================================================================
