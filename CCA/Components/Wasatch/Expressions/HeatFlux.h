#ifndef HeatFlux_Expr_h
#define HeatFlux_Expr_h

//-- ExprLib includes --//
#include <expression/ExprLib.h>

//-- SpatialOps includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/SpatialFieldStore.h>


/**
 *  \class HeatFlux
 *
 *  \brief Calculates a component of the Fourier heat flux,
 *         \f$q=-\lambda \frac{\partial T}{\partial x}\f$
 */
template< typename GradT,
          typename InterpT >
class HeatFlux
  : public Expr::Expression< typename GradT::DestFieldType >
{
  typedef typename GradT::DestFieldType FluxT;
  typedef typename GradT::SrcFieldType  CellT;

  const Expr::Tag lambdaTag_, temperatureTag_;

  const CellT* lambda_;
  const CellT* temperature_;
  const GradT* gradOp_;
  const InterpT* interpOp_;

  HeatFlux( const Expr::Tag lambdaTag,
            const Expr::Tag tempTag,
            const Expr::ExpressionID& id,
            const Expr::ExpressionRegistry& reg );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag lambdaTag,
             const Expr::Tag tempTag )
      : lambda_( lambdaTag ), temp_( tempTag ) {}

    Expr::ExpressionBase*
    build( const Expr::ExpressionID& id,
           const Expr::ExpressionRegistry& reg ) const
    {
      return new HeatFlux<GradT,InterpT>( lambda_, temp_, id, reg );
    }

  private:
    const Expr::Tag lambda_, temp_;
  };

  ~HeatFlux();

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



template< typename GradT, typename InterpT >
HeatFlux<GradT,InterpT>::
HeatFlux( const Expr::Tag thermCondTag,
          const Expr::Tag temperatureTag,
          const Expr::ExpressionID& id,
          const Expr::ExpressionRegistry& reg  )
  : Expr::Expression<FluxT>( id, reg ),
    lambdaTag_     ( thermCondTag   ),
    temperatureTag_( temperatureTag )
{}

//--------------------------------------------------------------------

template< typename GradT, typename InterpT >
HeatFlux<GradT,InterpT>::
~HeatFlux()
{}

//--------------------------------------------------------------------

template< typename GradT, typename InterpT >
void
HeatFlux<GradT,InterpT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( lambdaTag_        );
  exprDeps.requires_expression( temperatureTag_   );
}

//--------------------------------------------------------------------

template< typename GradT, typename InterpT >
void
HeatFlux<GradT,InterpT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<CellT>& cellFM = fml.template field_manager<CellT>();

  lambda_      = &cellFM.field_ref( lambdaTag_      );
  temperature_ = &cellFM.field_ref( temperatureTag_ );
}

//--------------------------------------------------------------------

template< typename GradT, typename InterpT >
void
HeatFlux<GradT,InterpT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  gradOp_   = opDB.retrieve_operator<GradT  >();
  interpOp_ = opDB.retrieve_operator<InterpT>();
}

//--------------------------------------------------------------------

template< typename GradT, typename InterpT >
void
HeatFlux<GradT,InterpT>::
evaluate()
{
  FluxT& result = this->value();

  SpatialOps::SpatFldPtr<FluxT> tmp = SpatialOps::SpatialFieldStore<FluxT>::self().get(result);

  gradOp_  ->apply_to_field( *temperature_, result );
  interpOp_->apply_to_field( *lambda_,      *tmp   );

  result *= *tmp;
  result *= -1.0;
}

//--------------------------------------------------------------------

#endif // HeatFlux_Expr_h
