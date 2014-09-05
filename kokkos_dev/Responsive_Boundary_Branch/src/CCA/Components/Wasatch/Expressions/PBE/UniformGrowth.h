#ifndef UniformGrowth_Expr_h
#define UniformGrowth_Expr_h

#include <expression/Expr_Expression.h>
#include <spatialops/structured/SpatialFieldStore.h>

/**
 *  \class UniformGrowth
 *  \author Tony Saad
 *  \todo add documentation
 */
template< typename FieldT >
class UniformGrowth
 : public Expr::Expression<FieldT>
{
  /* declare private variables such as fields, operators, etc. here */
  const FieldT* phi_;
  const Expr::Tag phiTag_;
  const double growthRateVal_;
  UniformGrowth( const Expr::Tag phiTag,
                const double growthRateVal,
                const Expr::ExpressionID& id,
                const Expr::ExpressionRegistry& reg  );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag phiTag,
            const double growthRateVal)
    : phit_(phiTag),
    growthrateval_(growthRateVal)
    {}

    Expr::ExpressionBase*
    build( const Expr::ExpressionID& id,
          const Expr::ExpressionRegistry& reg ) const 
    {
      return new UniformGrowth<FieldT>(phit_, growthrateval_, id, reg);
    }

  private:
    const Expr::Tag phit_;
    const double growthrateval_;
  };

  ~UniformGrowth();

  void advertise_dependents( Expr::ExprDeps& exprDeps );

  void bind_fields( const Expr::FieldManagerList& fml );

  void bind_operators( const SpatialOps::OperatorDatabase& opDB );

  void evaluate();

};



// ###################################################################
//
//                          Implementation
//
// ###################################################################



template< typename FieldT >
UniformGrowth<FieldT>::
UniformGrowth( const Expr::Tag phiTag,
              const double growthRateVal,
                const Expr::ExpressionID& id,
                const Expr::ExpressionRegistry& reg  )
  : Expr::Expression<FieldT>(id,reg),
		phiTag_(phiTag),
		growthRateVal_(growthRateVal)
{}

//--------------------------------------------------------------------

template< typename FieldT >
UniformGrowth<FieldT>::
~UniformGrowth()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
UniformGrowth<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( phiTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
UniformGrowth<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<FieldT>& fm = fml.template field_manager<FieldT>();
  phi_ = &fm.field_ref( phiTag_ );  
}

//--------------------------------------------------------------------

template< typename FieldT >
void
UniformGrowth<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
UniformGrowth<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  result <<= - growthRateVal_* *phi_;
}

//--------------------------------------------------------------------

#endif // UniformGrowth_Expr_h
