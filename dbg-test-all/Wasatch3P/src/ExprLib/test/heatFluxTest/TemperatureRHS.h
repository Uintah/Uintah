#ifndef TemperatureRHS_h
#define TemperatureRHS_h

#include <expression/Expression.h>

template< typename DivOp >
class TemperatureRHS
  : public Expr::Expression< typename DivOp::DestFieldType >
{
  typedef typename Expr::FieldID  FieldID;

  typedef typename DivOp::SrcFieldType  FluxFieldT;
  typedef typename DivOp::DestFieldType ScalarFieldT;

  TemperatureRHS( const Expr::Tag& heatFluxTag );

  const Expr::Tag heatFluxT_;
  const DivOp* divOp_;
  const FluxFieldT* heatFlux_;

public:

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();

  virtual ~TemperatureRHS(){}

  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Expr::ExpressionBase* build() const;
    Builder( const Expr::Tag& trhsTag,
             const Expr::Tag& heatFluxTag );
    ~Builder(){}
  private:
    const Expr::Tag hft_;
  };
};






//--------------------------------------------------------------------
template<typename DivOp>
TemperatureRHS<DivOp>::
TemperatureRHS( const Expr::Tag& heatFluxTag )
  : Expr::Expression<ScalarFieldT>(),
    heatFluxT_( heatFluxTag )
{
# ifdef ENABLE_CUDA
  this->set_gpu_runnable( true );
# endif
  heatFlux_ = NULL;
}
//--------------------------------------------------------------------
template<typename DivOp>
void
TemperatureRHS<DivOp>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( heatFluxT_ );
}
//--------------------------------------------------------------------
template<typename DivOp>
void
TemperatureRHS<DivOp>::
bind_fields( const Expr::FieldManagerList& fml )
{
  heatFlux_ = &fml.template field_ref<FluxFieldT>( heatFluxT_ );
}
//--------------------------------------------------------------------
template<typename DivOp>
void
TemperatureRHS<DivOp>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  divOp_ = opDB.retrieve_operator<DivOp>();
}
//--------------------------------------------------------------------
template<typename DivOp>
void
TemperatureRHS<DivOp>::
evaluate()
{
  using namespace SpatialOps;
  ScalarFieldT& result = this->value();
  result <<= -(*divOp_)( *heatFlux_ );
}
//--------------------------------------------------------------------

//--------------------------------------------------------------------
template<typename DivOp>
Expr::ExpressionBase*
TemperatureRHS<DivOp>::Builder::
build() const
{
  return new TemperatureRHS( hft_ );
}
//--------------------------------------------------------------------
template<typename DivOp>
TemperatureRHS<DivOp>::Builder::
Builder( const Expr::Tag& trhsTag,
         const Expr::Tag& heatFluxTag )
  : Expr::ExpressionBuilder(trhsTag),
    hft_( heatFluxTag )
{}
//--------------------------------------------------------------------

#endif
