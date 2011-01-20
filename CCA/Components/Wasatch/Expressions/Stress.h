#ifndef Stress_Expr_h
#define Stress_Expr_h

#include <expression/Expr_Expression.h>

/**
 *  \class Stress
 *  \author James C. Sutherland
 *  \date December, 2010
 *
 *  \brief Calculates a component of the stress tensor.
 *
 *  The stress tensor is given as
 *  \[ \tau_{ij} = -\mu \left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right) - \frac{2}{3} \mu \delta_{ij} \frac{\partial u_k}{\partial x_k} \]
 *  This expression calculates only the off-diagonal components of the stress tensor,
 *  \[ \tau_{ij} = -\mu \left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right) \]
 *
 *  \par Template Parameters
 *  <ul>
 *  <li> \b StressT The type of field for this stress component.
 *  <li> \b Vel1T   The type of field for the first velocity component.
 *  <li> \b Vel2T   The type of field for the second velocity component.
 *  <li> \b ViscT   The type of field for the viscosity.
 *  </ul>
 */
template< typename StressT,
          typename Vel1T,
          typename Vel2T,
          typename ViscT >
class Stress
 : public Expr::Expression<StressT>
{
  const Expr::Tag visct_, vel1t_, vel2t_;

  typedef typename OperatorTypeBuilder< Interpolant, ViscT, StressT >::type  ViscInterpT;
  typedef typename OperatorTypeBuilder< Gradient,    Vel1T, StressT >::type  Vel1GradT;
  typedef typename OperatorTypeBuilder< Gradient,    Vel2T, StressT >::type  Vel2GradT;

  ViscInterpT* const viscInterpOp_; ///< Interpolate viscosity to the face where we are building the stress
  Vel1GradT*   const vel1GradOp_;   ///< Calculate the velocity gradient dui/dxj at the stress face
  Vel2GradT*   const vel2GradOp_;   ///< Calculate the velocity gradient duj/dxi at the stress face

  ViscT* const visc_;
  Vel1T* const vel1_;
  Vel2T* const vel2_;

  Stress( const Expr::Tag viscTag,
          const Expr::Tag vel1Tag,
          const Expr::Tag vel2Tag,
          const Expr::ExpressionID& id,
          const Expr::ExpressionRegistry& reg );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag viscTag,
             const Expr::Tag vel1Tag,
             const Expr::Tag vel2Tag );

    Expr::ExpressionBase*
    build( const Expr::ExpressionID& id,
           const Expr::ExpressionRegistry& reg ) const;

  private:
    const Expr::Tag visct_, vel1t_, vel2t_;
  };

  ~Stress();

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



template< typename StressT, typename Vel1T, typename Vel2T, typename ViscT >
Stress<StressT,Vel1T,Vel2T,ViscT>::
Stress( const Expr::Tag viscTag,
        const Expr::Tag vel1Tag,
        const Expr::Tag vel2Tag,
        const Expr::ExpressionID& id,
        const Expr::ExpressionRegistry& reg  )
  : Expr::Expression<StressT>(id,reg),
    visct_( viscTag ),
    vel1t_( vel1Tag ),
    vel2t_( vel2Tag )
{}

//--------------------------------------------------------------------

template< typename StressT, typename Vel1T, typename Vel2T, typename ViscT >
Stress<StressT,Vel1T,Vel2T,ViscT>::
~Stress()
{}

//--------------------------------------------------------------------

template< typename StressT, typename Vel1T, typename Vel2T, typename ViscT >
void
Stress<StressT,Vel1T,Vel2T,ViscT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( visct_ );
  exprDeps.requires_expression( vel1t_ );
  exprDeps.requires_expression( vel2t_ );
}

//--------------------------------------------------------------------

template< typename StressT, typename Vel1T, typename Vel2T, typename ViscT >
void
Stress<StressT,Vel1T,Vel2T,ViscT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<ViscT>& viscfm = fml.template field_manager<ViscT>();
  const Expr::FieldManager<Vel1T>& vel1fm = fml.template field_manager<Vel1T>();
  const Expr::FieldManager<Vel2T>& vel2fm = fml.template field_manager<Vel2T>();

  visc_ = &viscfm.field_ref( visct_ );
  vel1_ = &vel1fm.field_ref( vel1t_ );
  vel2_ = &vel1fm.field_ref( vel2t_ );
}

//--------------------------------------------------------------------

template< typename StressT, typename Vel1T, typename Vel2T, typename ViscT >
void
Stress<StressT,Vel1T,Vel2T,ViscT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  viscInterpOp_ = opDB.retrieve_operator<ViscInterpT>();
  vel1GradOp_   = opDB.retrieve_operator<Vel1GradT  >();
  vel2GradOp_   = opDB.retrieve_operator<Vel2GradT  >();
}

//--------------------------------------------------------------------

template< typename StressT, typename Vel1T, typename Vel2T, typename ViscT >
void
Stress<StressT,Vel1T,Vel2T,ViscT>::
evaluate()
{
  StressT& stress = this->value();
  SpatialOps::SpatFldPtr<StressT> tmp = SpatialOps::SpatialFieldStore<StressT>::self().get( stress );

  vel1GradOp_->apply_to_field( *vel1_, stress );
  vel2GradOp_->apply_to_field( *vel2_, *tmp   );

  stress += *tmp;

  viscInterpOp_->apply_to_field( *visc_, *tmp );
  stress *= *tmp;
  stress *= -1.0;
}

//--------------------------------------------------------------------

template< typename StressT, typename Vel1T, typename Vel2T, typename ViscT >
Stress<StressT,Vel1T,Vel2T,ViscT>::
Builder::Builder( const Expr::Tag viscTag,
             const Expr::Tag vel1Tag,
             const Expr::Tag vel2Tag )
  : visct_( viscTag ),
    vel1t_( vel1Tag ),
    vel2t_( vel2Tag )
{}

//--------------------------------------------------------------------

template< typename StressT, typename Vel1T, typename Vel2T, typename ViscT >
Expr::ExpressionBase*
Stress<StressT,Vel1T,Vel2T,ViscT>::
Builder::build( const Expr::ExpressionID& id,
                const Expr::ExpressionRegistry& reg ) const
{
  return new Stress<StressT,Vel1T,Vel2T,ViscT>( visct_, vel1t_, vel2t, id, reg );
}


#endif // Stress_Expr_h
