#ifndef DiffusiveFlux_Expr_h
#define DiffusiveFlux_Expr_h

//-- ExprLib includes --//
#include <expression/ExprLib.h>

//-- SpatialOps includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/SpatialFieldStore.h>


/**
 *  \class  DiffusiveFlux
 *  \author James C. Sutherland
 *  \date   June, 2010
 *
 *  \brief Calculates a simple diffusive flux of the form
 *         \f$ J = -\Gamma \frac{\partial \phi}{\partial x} \f$
 *
 *  Note that this requires the diffusion coefficient, \f$\Gamma\f$,
 *  to be evaluated at the same location as \f$J\f$ and \f$\nabla
 *  \phi\f$.
 */
template< typename GradT >
class DiffusiveFlux
  : public Expr::Expression< typename GradT::DestFieldType >
{
  typedef typename GradT::DestFieldType FluxT;
  typedef typename GradT::SrcFieldType  ScalarT;

  const bool isConstCoef_;
  const Expr::Tag phiTag_, coefTag_;
  const double coefVal_;

  const GradT* gradOp_;
  const ScalarT* phi_;
  const FluxT* coef_;

  DiffusiveFlux( const Expr::Tag phiTag,
                 const Expr::Tag coefTag,
                 const Expr::ExpressionID& id,
                 const Expr::ExpressionRegistry& reg );

  DiffusiveFlux( const Expr::Tag phiTag,
                 const double coefTag,
                 const Expr::ExpressionID& id,
                 const Expr::ExpressionRegistry& reg );

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
     *  \brief Construct a diffusive flux given expressions for
     *         \f$\phi\f$ and \f$\Gamma$\f
     *
     *  \param phiTag  the Expr::Tag for the scalar field
     *
     *  \param coefTag the Expr::Tag for the diffusion coefficient
     *         (located at same points as the flux field).
     */
    Builder( const Expr::Tag phiTag, const Expr::Tag coefTag )
      : isConstCoef_( false ),
        phit_(phiTag),
        coeft_(coefTag),
        coef_(0.0)
    {}

    /**
     *  \brief Construct a diffusive flux given an expression for
     *         \f$\phi\f$ and a constant value for \f$\Gamma\f$.
     *
     *  \param phiTag  the Expr::Tag for the scalar field
     *
     *  \param coef the value (constant in space and time) for the
     *         diffusion coefficient.
     */
    Builder( const Expr::Tag phiTag, const double coef )
      : isConstCoef_( true ),
        phit_(phiTag),
        coef_(coef)
    {}

    Expr::ExpressionBase*
    build( const Expr::ExpressionID& id,
           const Expr::ExpressionRegistry& reg ) const
    {
      if( isConstCoef_ ) return new DiffusiveFlux<GradT>( phit_, coef_,  id, reg );
      else               return new DiffusiveFlux<GradT>( phit_, coeft_, id, reg );
    }
  private:
        const bool isConstCoef_;
    const Expr::Tag phit_,coeft_;
    const double coef_;
  };

  ~DiffusiveFlux();
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};




/**
 *  \class  DiffusiveFlux2
 *  \author James C. Sutherland
 *  \date   June, 2010
 *
 *  \brief Calculates a generic diffusive flux, \f$J = -\Gamma
 *         \frac{\partial \phi}{\partial x}\f$, where \f$\Gamma\f$ is
 *         located at the same location as \f$\phi\f$.
 *
 *  \par Template Parameters
 *  <ul>
 *  <li> \b GradT The type of operator used in forming
 *       \f$\frac{\partial \phi}{\partial x}\f$
 *  <li> \b InterpT The type of operator used in interpolating
 *       \f$\Gamma\f$ from the location of \f$\phi\f$ to the location
 *       of \f$\frac{\partial \phi}{\partial x}\f$
 *  </ul>
 */
template< typename GradT,
          typename InterpT >
class DiffusiveFlux2
  : public Expr::Expression< typename GradT::DestFieldType >
{
  typedef typename GradT::DestFieldType FluxT;
  typedef typename GradT::SrcFieldType  ScalarT;

  const Expr::Tag phiTag_, coefTag_;

  const GradT* gradOp_;
  const InterpT* interpOp_;

  const ScalarT* phi_;
  const ScalarT* coef_;

  DiffusiveFlux2( const Expr::Tag phiTag,
                  const Expr::Tag coefTag,
                  const Expr::ExpressionID& id,
                  const Expr::ExpressionRegistry& reg );

public:
  /**
   *  \brief Builder for a diffusive flux \f$ J = -\Gamma \frac{\partial
   *         \phi}{\partial x} \f$ where \f$\Gamma\f$ is stored at the
   *         same location as \f$\phi\f$.
   */
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \brief Construct a DiffusiveFlux2::Builder object for
     *         registration with an Expr::ExpressionFactory.
     *
     *  \param phiTag the Expr::Tag for the scalar field.
     *
     *  \param coefTag the Expr::Tag for the diffusion coefficient
     *         (located at same points as the scalar field).
     */
    Builder( const Expr::Tag phiTag, const Expr::Tag coefTag )
      : phit_(phiTag), coeft_( coefTag )
    {}

    Expr::ExpressionBase*
    build( const Expr::ExpressionID& id,
           const Expr::ExpressionRegistry& reg ) const    {
      return new DiffusiveFlux2<GradT,InterpT>( phit_, coeft_, id, reg );
    }
  private:
    const Expr::Tag phit_,coeft_;
  };

  ~DiffusiveFlux2();
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



template< typename GradT >
DiffusiveFlux<GradT>::
DiffusiveFlux( const Expr::Tag phiTag,
               const Expr::Tag coefTag,
               const Expr::ExpressionID& id,
               const Expr::ExpressionRegistry& reg  )
  : Expr::Expression<FluxT>( id, reg ),
    isConstCoef_( false ),
    phiTag_ ( phiTag  ),
    coefTag_( coefTag ),
    coefVal_( 0.0 )
{}

//--------------------------------------------------------------------

template< typename GradT >
DiffusiveFlux<GradT>::
DiffusiveFlux( const Expr::Tag phiTag,
               const double coef,
               const Expr::ExpressionID& id,
               const Expr::ExpressionRegistry& reg  )
  : Expr::Expression<FluxT>( id, reg ),
    isConstCoef_( true ),
    phiTag_ ( phiTag ),
    coefTag_( "NULL", Expr::INVALID_CONTEXT ),
    coefVal_( coef )
{}

//--------------------------------------------------------------------

template< typename GradT >
DiffusiveFlux<GradT>::
~DiffusiveFlux()
{}

//--------------------------------------------------------------------

template< typename GradT >
void
DiffusiveFlux<GradT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( phiTag_ );
  if( !isConstCoef_ ) exprDeps.requires_expression( coefTag_ );
}

//--------------------------------------------------------------------

template< typename GradT >
void
DiffusiveFlux<GradT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<FluxT  >& fluxFM   = fml.template field_manager<FluxT  >();
  const Expr::FieldManager<ScalarT>& scalarFM = fml.template field_manager<ScalarT>();

  phi_ = &scalarFM.field_ref( phiTag_ );
  if( !isConstCoef_ ) coef_ = &fluxFM.field_ref( coefTag_ );
}

//--------------------------------------------------------------------

template< typename GradT >
void
DiffusiveFlux<GradT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  gradOp_ = opDB.retrieve_operator<GradT>();
}

//--------------------------------------------------------------------

template< typename GradT >
void
DiffusiveFlux<GradT>::
evaluate()
{
  FluxT& result = this->value();

  gradOp_->apply_to_field( *phi_, result );  // J = grad(phi)
  if( isConstCoef_ ){
    result *= -coefVal_;  // J = -gamma * grad(phi)
  }
  else{
    result *= *coef_;  // J =  gamma * grad(phi)
    result *= -1.0;    // J = -gamma * grad(phi)
  }
}


//====================================================================


template< typename GradT, typename InterpT >
DiffusiveFlux2<GradT,InterpT>::
DiffusiveFlux2( const Expr::Tag phiTag,
                const Expr::Tag coefTag,
                const Expr::ExpressionID& id,
                const Expr::ExpressionRegistry& reg  )
  : Expr::Expression<FluxT>(id,reg),
    phiTag_ ( phiTag  ),
    coefTag_( coefTag )
{}

//--------------------------------------------------------------------

template< typename GradT, typename InterpT >
DiffusiveFlux2<GradT,InterpT>::
~DiffusiveFlux2()
{}

//--------------------------------------------------------------------

template< typename GradT, typename InterpT >
void
DiffusiveFlux2<GradT,InterpT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( phiTag_ );
  exprDeps.requires_expression( coefTag_ );
}

//--------------------------------------------------------------------

template< typename GradT, typename InterpT >
void
DiffusiveFlux2<GradT,InterpT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<ScalarT>& scalarFM = fml.template field_manager<ScalarT>();
  phi_  = &scalarFM.field_ref( phiTag_  );
  coef_ = &scalarFM.field_ref( coefTag_ );
}

//--------------------------------------------------------------------

template< typename GradT, typename InterpT >
void
DiffusiveFlux2<GradT,InterpT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  gradOp_   = opDB.retrieve_operator<GradT  >();
  interpOp_ = opDB.retrieve_operator<InterpT>();
}

//--------------------------------------------------------------------

template< typename GradT, typename InterpT >
void
DiffusiveFlux2<GradT,InterpT>::
evaluate()
{
  FluxT& result = this->value();

  SpatialOps::SpatFldPtr<FluxT> fluxTmp = SpatialOps::SpatialFieldStore<FluxT>::self().get( result );

  gradOp_  ->apply_to_field( *phi_, *fluxTmp );  // J = grad(phi)
  interpOp_->apply_to_field( *coef_, result  );
  result *= *fluxTmp;                            // J =   gamma * grad(phi)
  result *= -1.0;                                // J = - gamma * grad(phi)
}

//--------------------------------------------------------------------

#endif // DiffusiveFlux_Expr_h
