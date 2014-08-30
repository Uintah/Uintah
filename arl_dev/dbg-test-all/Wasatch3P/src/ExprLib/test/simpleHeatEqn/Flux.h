#ifndef Flux_Expr_h
#define Flux_Expr_h

#include <expression/Expression.h>

template< typename GradT,
          typename InterpT >
class Flux
  : public Expr::Expression< typename GradT::DestFieldType >
{
  typedef typename GradT::DestFieldType FluxT;
  typedef typename GradT::SrcFieldType ScalarT;

  const Expr::Tag tt_, dct_;
  const ScalarT *temp_, *dCoef_;

  const GradT* grad_;
  const InterpT* interp_;

  Flux( const Expr::Tag& tempTag,
        const Expr::Tag& diffCoefTag );

public:
  class Builder : public Expr::ExpressionBuilder
  {
    const Expr::Tag tt_, dct_;
  public:
    Builder( const Expr::Tag& fluxTag,
             const Expr::Tag& tempTag,
             const Expr::Tag& diffCoefTag )
      : ExpressionBuilder(fluxTag),
        tt_ ( tempTag     ),
        dct_( diffCoefTag )
    {}
    ~Builder(){}
    Expr::ExpressionBase*
    build() const
    {
      return new Flux<GradT,InterpT>( tt_, dct_ );
    }
  };

  ~Flux(){}

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
Flux<GradT,InterpT>::
Flux( const Expr::Tag& tempTag,
      const Expr::Tag& diffCoefTag )
  : Expr::Expression<FluxT>(),
    tt_( tempTag ),
    dct_( diffCoefTag )
{
# ifdef ENABLE_CUDA
  this->set_gpu_runnable( true );
# endif
}

//--------------------------------------------------------------------

template< typename GradT, typename InterpT >
void
Flux<GradT,InterpT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( dct_ );
  exprDeps.requires_expression( tt_  );
}

//--------------------------------------------------------------------

template< typename GradT, typename InterpT >
void
Flux<GradT,InterpT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<ScalarT>::type& fm = fml.template field_manager<ScalarT>();
  temp_  = &fm.field_ref( tt_  );
  dCoef_ = &fm.field_ref( dct_ );
}

//--------------------------------------------------------------------

template< typename GradT, typename InterpT >
void
Flux<GradT,InterpT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  grad_   = opDB.retrieve_operator<GradT  >();
  interp_ = opDB.retrieve_operator<InterpT>();
}

//--------------------------------------------------------------------

template< typename GradT, typename InterpT >
void
Flux<GradT,InterpT>::
evaluate()
{
  using namespace SpatialOps;
  FluxT& flux = this->value();
  flux <<= - (*interp_)(*dCoef_) * (*grad_)(*temp_);
}

//--------------------------------------------------------------------

#endif // Flux_Expr_h
