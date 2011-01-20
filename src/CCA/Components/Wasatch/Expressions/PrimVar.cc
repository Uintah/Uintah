#ifndef PrimVar_h
#define PrimVar_h

#include "PrimVar.h"

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

//====================================================================

template< typename FieldT, typename DensT >
PrimVar<FieldT,DensT>::
PrimVar( const Expr::Tag rhoPhiTag,
         const Expr::Tag rhoTag,
         const Expr::ExpressionID& id,
         const Expr::ExpressionRegistry& reg  )
  : Expr::Expression<FieldT>(id,reg),
    rhophit_( rhoPhiTag ),
    rhot_( rhoTag )
{}

//--------------------------------------------------------------------

template< typename FieldT, typename DensT >
PrimVar<FieldT,DensT>::
~PrimVar()
{}

//--------------------------------------------------------------------

template< typename FieldT, typename DensT >
void
PrimVar<FieldT,DensT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( rhophit_ );
  exprDeps.requires_expression( rhot_    );
}

//--------------------------------------------------------------------

template< typename FieldT, typename DensT >
void
PrimVar<FieldT,DensT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<FieldT>& phifm = fml.template field_manager<FieldT>();
  const Expr::FieldManager<DensT >& denfm = fml.template field_manager<DensT >();

  rhophi_ = &phifm.field_ref( rhophit_ );
  rho_    = &denfm.field_ref( rhot_    );
}

//--------------------------------------------------------------------

template< typename FieldT, typename DensT >
void
PrimVar<FieldT,DensT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  // bind operators as follows:
  interpOp_ = opDB.retrieve_operator<InterpT>();
}

template<>
void
PrimVar<SVolField,SVolField>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//--------------------------------------------------------------------

template< typename FieldT, typename DensT >
void
PrimVar<FieldT,DensT>::
evaluate()
{
  FieldT& phi = this->value();

  SpatialOps::SpatFldPtr<FieldT> tmp = SpatialOps::SpatialFieldStore<FieldT>::self().get( phi );

  interpOp_->apply_to_field( *rho_, *tmp );
  phi <<= *rhophi_ / *tmp;
}

template<>
void
PrimVar<SVolField,SVolField>::
evaluate()
{
  SVolField& phi = this->value();
  phi <<= *rhophi_ / *rho_;
}

//--------------------------------------------------------------------

template< typename FieldT, typename DensT >
PrimVar<FieldT,DensT>::
Builder::Builder( const Expr::Tag rhoPhiTag,
                  const Expr::Tag rhoTag )
  : rhophit_( rhoPhiTag ),
    rhot_   ( rhoTag    )
{}

//--------------------------------------------------------------------

template< typename FieldT, typename DensT >
Expr::ExpressionBase*
PrimVar<FieldT,DensT>::
Builder::build( const Expr::ExpressionID& id,
                const Expr::ExpressionRegistry& reg ) const
{
  return new PrimVar<FieldT,DensT>( rhophit_, rhot_, id, reg );
}

//====================================================================
//  Explicit template instantiation
#include <CCA/Components/Wasatch/FieldTypes.h>
template class PrimVar<SVolField,SVolField>;
template class PrimVar<XVolField,SVolField>;
template class PrimVar<YVolField,SVolField>;
template class PrimVar<ZVolField,SVolField>;
//====================================================================


#endif // PrimVar_h
