#ifndef NormalStress_Expr_h
#define NormalStress_Expr_h

#include <expression/Expr_Expression.h>

/**
 *  \class NormalStress
 *  \author James C. Sutherland
 *  \date December, 2010
 *
 *  \brief Calculates a component of the stress tensor.
 *
 *  The stress tensor is given as
 *  \[ \tau_{ij} = -\mu \left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right) - \frac{2}{3} \mu \delta_{ij} \frac{\partial u_k}{\partial x_k} \]
 *
 *  This expression calculates only the normal stresses
 *  \[ \tau_{ij} = -2\mu \frac{\partial u_i}{\partial x_i}  - \frac{2}{3} \mu \delta_{ij} \frac{\partial u_k}{\partial x_k} \]
 *
 *  Here we assume that the dilatation, \f$\frac{\partial u_k}{\partial x_k}$\f,
 *  is known and is stored at the same location as the viscosity (scalar cell centroids).
 *
 *  \par Template Parameters
 *  <ul>
 *  <li> \b StressT The type of field for this stress component.
 *  <li> \b VelT    The type of field for the velocity component forming the stress: \f$u_i$\f.
 *  <li> \b ViscT   The type of field for the viscosity and dilatation.
 *  </ul>
 */
template< typename StressT,
          typename VelT,
          typename ViscT >
class NormalStress
 : public Expr::Expression<StressT>
{
  const Expr::Tag visct_, velt_, dilt_;

  typedef typename OperatorTypeBuilder< Interpolant, ViscT, StressT >::type  InterpT;
  typedef typename OperatorTypeBuilder< Gradient, VelT, StressT >::type GradT;

  InterpT* const interpOp_; ///< Interpolate viscosity and dilatation to the face where we are building the stress
  GradT  * const gradOp_;   ///< Calculate the velocity gradient at the stress face.

  ViscT* const visc_;
  ViscT* const dil_;
  VelT * const vel_;

  
  NormalStress( const Expr::Tag viscTag,
                const Expr::Tag velTag,
                const Expr::Tag dilTag,
                const Expr::ExpressionID& id,
                const Expr::ExpressionRegistry& reg  );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag viscTag,
             const Expr::Tag velTag,
             const Expr::Tag dilTag );
                
    Expr::ExpressionBase*
    build( const Expr::ExpressionID& id,
           const Expr::ExpressionRegistry& reg ) const;

  private:
    const Expr::Tag visct_, vel1t_, vel2t_;
  };

  ~NormalStress();

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



template< typename StressT, typename VelT, typename ViscT >
NormalStress<StressT,VelT,ViscT>::
NormalStress( const Expr::Tag viscTag,
              const Expr::Tag velTag,
              const Expr::Tag dilTag,
              const Expr::ExpressionID& id,
              const Expr::ExpressionRegistry& reg  )
  : Expr::Expression<StressT>(id,reg),
    visct_( viscTag ),
    velt_( velTag ),
    dilt_( dilTag )
{}

//--------------------------------------------------------------------

template< typename StressT, typename VelT, typename ViscT >
NormalStress<StressT,VelT,ViscT>::
~NormalStress()
{}

//--------------------------------------------------------------------

template< typename StressT, typename VelT, typename ViscT >
void
NormalStress<StressT,VelT,ViscT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( visct_ );
  exprDeps.requires_expression( velt_ );
  exprDeps.requires_expression( dilt_ );
}

//--------------------------------------------------------------------

template< typename StressT, typename VelT, typename ViscT >
void
NormalStress<StressT,VelT,ViscT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<ViscT>& viscfm = fml.template field_manager<ViscT>();
  const Expr::FieldManager<VelT >& velfm  = fml.template field_manager<VelT >();
  visc_ = &viscfm.field_ref( visct_ );
  dil_  = &viscfm.field_ref( dilt_  );
  vel_  = &velfm.field_ref( velt_ );
}

//--------------------------------------------------------------------

template< typename StressT, typename VelT, typename ViscT >
void
NormalStress<StressT,VelT,ViscT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  interpOp_ = opDB.retrieve_operator<InterpT>();
  gradOp_   = opDB.retrieve_operator<GradT  >();
}

//--------------------------------------------------------------------

template< typename StressT, typename VelT, typename ViscT >
void
NormalStress<StressT,VelT,ViscT>::
evaluate()
{
  StressT& stress = this->value();
  SpatialOps::SpatFldPtr<StressT> tmp1 = SpatialOps::SpatialFieldStore<StressT>::self().get( stress );
  SpatialOps::SpatFldPtr<StressT> tmp2 = SpatialOps::SpatialFieldStore<StressT>::self().get( stress );

  gradOp_  ->apply_to_field( *vel_, stress );
  interpOp_->apply_to_field( *visc_, *tmp1 );
  stress += *tmp;
  stress *= -2.0;

  interpOp_->apply_to_field( *dil_, *tmp2 );
  *tmp2 *= *tmp1;
  *tmp2 *= -2.0/3.0;

  stress += *tmp2;
}

//--------------------------------------------------------------------

template< typename StressT, typename VelT, typename ViscT >
NormalStress<StressT,VelT,ViscT>::
Builder::Builder( const Expr::Tag viscTag,
                  const Expr::Tag velTag,
                  const Expr::Tag dilTag )
  : visct_( viscTag ),
    velt_( velTag ),
    dilt_( dilTag )
{}

//--------------------------------------------------------------------

template< typename StressT, typename VelT, typename ViscT >
Expr::ExpressionBase*
NormalStress<StressT,VelT,ViscT>::
Builder::build( const Expr::ExpressionID& id,
                const Expr::ExpressionRegistry& reg ) const
{
  return new NormalStress<StressT,VelT,ViscT>( visct_, velt_, dilt_, id, reg );
}


#endif // NormalStress_Expr_h
