#ifndef DiffusiveFlux_Expr_h
#define DiffusiveFlux_Expr_h

#include <expression/ExprLib.h>


/**
 *  \class DiffusiveFlux
 */
template< typename GradT, >
class DiffusiveFlux
  : public Expr::Expression<GradT::DestFieldType>
{
  typedef GradT::DestFieldType FluxT;
  typedef GradT::SrcFieldType  ScalarT;

  const Expr::Tag phiTag_, coefTag_;

  const GradT* gradOp_;
  const ScalarT* phi_;
  const FluxT* coef_;

  DiffusiveFlux( const Expr::Tag phiTag,
                 const Expr::Tag coefTag,
                 const Expr::ExpressionID& id,
                 const Expr::ExpressionRegistry& reg  );

public:
  /**
   *  \brief Builder for a diffusive flux \f$ J = -\Gamma \frac{\partial
   *         \phi}{\partial x} \f$ where \f$\Gamma\f$ is stored at the
   *         same location as \f$J\f$.
   */
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \param phiTag   the Expr::Tag for the scalar field
     *
     *  \param coefTag the Expr::Tag for the diffusion coefficient
     *         (located at same points as the flux field).
     */
    Builder( const Expr::Tag phiTag, conts Expr::Tag coefTag )
      : phit_(phiTag), coeft_( coefTag )
    {}

    Expr::ExpressionBase*
    build( const Expr::ExpressionID& id,
           const Expr::ExpressionRegistry& reg ) const
    {
      return new DiffusiveFlux<GradT>( phit_, coeft_, id, reg, p );
    }
  private:
    const Expr::Tag phit_,coeft_;
  };

  ~DiffusiveFlux();

  void advertise_dependents( Expr::ExprDeps& exprDeps,
                             Expr::FieldDeps& fieldDeps );

  void bind_fields( const Expr::FieldManagerList& fml );

  void bind_operators( const SpatialOps::OperatorDatabase& opDB );

  void evaluate();

};



// ###################################################################
//
//                          Implementation
//
// ###################################################################



template< typename GradT >
DiffusiveFlux<GradT>::
DiffusiveFlux( const Expr::Tag phiTag,
               const Expr::Tag coefTag,
               const Expr::ExpressionID& id,
               const Expr::ExpressionRegistry& reg  )
  : Expr::Expression<FluxT>( id, reg.get_label(id) ),
    phiTag_ ( phiTag  ),
    coefTag_( coefTag )
{}

//--------------------------------------------------------------------

template< typename GradT >
DiffusiveFlux<GradT>::
~DiffusiveFlux()
{
}

//--------------------------------------------------------------------

template< typename GradT >
void
DiffusiveFlux<GradT>::
advertise_dependents( Expr::ExprDeps& exprDeps,
                      Expr::FieldDeps& fieldDeps )
{
  fieldDeps.requires_field<FluxT>( this->name() );

  exprDeps.requires_expression( phiTag_ );
  exprDeps.requires_expression( coefTag_ );
}

//--------------------------------------------------------------------

template< typename GradT >
void
DiffusiveFlux<GradT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<FluxT  >& fluxFM   = fml.template field_manager<FluxT  >();
  const Expr::FieldManager<ScalarT>& scalarFM = fml.template field_manager<ScalarT>();

  this->exprValue_ = &fluxFM.field_ref( this->name() );

  phi_  = &scalarFM.field_ref( phiTag_  );
  coef_ = &fluxFM  .field_ref( coefTag_ );
}

//--------------------------------------------------------------------

template< typename GradT >
void
DiffusiveFlux<GradT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  gradOp_ = opDB.retrive_operator<GradT>();
}

//--------------------------------------------------------------------

template< typename GradT >
void
DiffusiveFlux<GradT>::
evaluate()
{
  FieldT& result = this->value();

  gradOp_->apply_to_field( *phi_, result );
  result *= fcoef_;
  result *= -1.0;
}

//--------------------------------------------------------------------

#endif // DiffusiveFlux_Expr_h
