#ifndef Stress_Expr_h
#define Stress_Expr_h

#include <expression/Expression.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>

/**
 *  \class 	Stress
 *  \author 	James C. Sutherland
 *  \date 	December, 2010
 *  \ingroup	Expressions
 *
 *  \brief Calculates a component of the stress tensor.
 *
 *  The stress tensor is given as
 *  \f[ \tau_{ij} = -\mu \left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right) + \frac{2}{3} \mu \delta_{ij} \frac{\partial u_k}{\partial x_k} \f]
 *
 *  \tparam StressT The type of field for this stress component.
 *  \tparam Vel1T   The type of field for the first velocity component.
 *  \tparam Vel2T   The type of field for the second velocity component.
 *  \tparam ViscT   The type of field for the viscosity.
 */
template< typename StressT,
          typename Vel1T,
          typename Vel2T,
          typename ViscT >
class Stress
 : public Expr::Expression<StressT>
{
  const Expr::Tag visct_, vel1t_, vel2t_, dilt_;

  typedef typename SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, ViscT, StressT >::type  ViscInterpT;
  typedef typename SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Gradient,    Vel1T, StressT >::type  Vel1GradT;  // jcs this will likely be insufficient
  typedef typename SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Gradient,    Vel2T, StressT >::type  Vel2GradT;  // jcs this will likely be insufficient

  const ViscInterpT* viscInterpOp_; ///< Interpolate viscosity to the face where we are building the stress
  const Vel1GradT*   vel1GradOp_;   ///< Calculate the velocity gradient dui/dxj at the stress face
  const Vel2GradT*   vel2GradOp_;   ///< Calculate the velocity gradient duj/dxi at the stress face

  const ViscT* visc_;
  const Vel1T* vel1_;
  const Vel2T* vel2_;

  Stress( const Expr::Tag& viscTag,
          const Expr::Tag& vel1Tag,
          const Expr::Tag& vel2Tag );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \param result the stress component being calculated
     *  \param viscTag the viscosity
     *  \param vel1Tag the first velocity component
     *  \param vel2Tag the second velocity component
     *
     *  \param dilTag the dilatation tag. If supplied, the
     *         dilatational stress term will be added.
     */
    Builder( const Expr::Tag& result,
             const Expr::Tag& viscTag,
             const Expr::Tag& vel1Tag,
             const Expr::Tag& vel2Tag,
             const Expr::Tag& dilTag );
    ~Builder(){}
    Expr::ExpressionBase* build() const;

  private:
    const Expr::Tag visct_, vel1t_, vel2t_;
  };

  ~Stress();

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();

};



/**
 *  specialized version of the stress tensor for normal stresses.
 *
 */
template< typename StressT,
          typename VelT,
          typename ViscT >
class Stress< StressT, VelT, VelT, ViscT >
 : public Expr::Expression<StressT>
{
  const Expr::Tag visct_, velt_, dilt_;

  typedef typename SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, ViscT, StressT >::type  ViscInterpT;
  typedef typename SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Gradient,    VelT,  StressT >::type  VelGradT;

  const ViscInterpT* viscInterpOp_; ///< Interpolate viscosity to the face where we are building the stress
  const VelGradT*    velGradOp_;    ///< Calculate the velocity gradient dui/dxj at the stress face

  const ViscT* visc_;
  const VelT*  vel_;
  const ViscT* dil_;

  Stress( const Expr::Tag& viscTag,
          const Expr::Tag& velTag,
          const Expr::Tag& dilTag );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \param result the stress component calculated here
     *  \param viscTag the viscosity
     *  \param vel1Tag the first velocity component
     *  \param vel2Tag the second velocity component
     *
     *  \param dilTag the dilatation tag. If supplied, the
     *         dilatational stress term will be added.
     *
     *  note that in this case, the second velocity component will be
     *  ignored.  It is kept for consistency with the off-diagonal
     *  stress builder.
     */
    Builder( const Expr::Tag& result,
             const Expr::Tag& viscTag,
             const Expr::Tag& vel1Tag,
             const Expr::Tag& vel2Tag,
             const Expr::Tag& dilTag );
    ~Builder(){}
    Expr::ExpressionBase* build() const;

  private:
    const Expr::Tag visct_, velt_, dilt_;
  };

  ~Stress();

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};

#endif // Stress_Expr_h
