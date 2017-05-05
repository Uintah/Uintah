#ifndef StableTimestep_Expr_h
#define StableTimestep_Expr_h

#include <expression/Expression.h>
#include <spatialops/structured/FVStaggered.h>
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
/**
 *  \class 	StableTimestep
 *  \author 	Tony Saad
 *  \date 	 June 2013
 *  \ingroup	Expressions
 *
 *  \brief Calculates a stable timestep based on the momemntum equations using
 common CFD criteria.
 *
 */
template< typename Vel1T, typename Vel2T, typename Vel3T >
class StableTimestep
 : public Expr::Expression<SpatialOps::SingleValueField>
{
  typedef SVolField FieldT;
  DECLARE_FIELDS(SVolField, rho_, visc_, csound_)
  DECLARE_FIELD(Vel1T, u_)
  DECLARE_FIELD(Vel2T, v_)
  DECLARE_FIELD(Vel3T, w_)
  DECLARE_FIELDS(ParticleField, pu_, pv_, pw_)

  double invDx_, invDy_, invDz_; // 1/dx, 1/dy, 1/dz
  const bool doX_, doY_, doZ_, isViscous_, doParticles_, isCompressible_;
  const bool is3dconvdiff_;
  
  
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, Vel1T, SVolField >::type X2SOpT;
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, Vel2T, SVolField >::type Y2SOpT;
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, Vel3T, SVolField >::type Z2SOpT;
  const X2SOpT* x2SInterp_;
  const Y2SOpT* y2SInterp_;
  const Z2SOpT* z2SInterp_;

  // gradient operators are declared here to extract grid-spacing information
  typedef typename SpatialOps::OperatorTypeBuilder< typename WasatchCore::GradOpSelector<Vel1T, SpatialOps::XDIR>::Gradient, Vel1T, FieldT >::type GradXT;
  typedef typename SpatialOps::OperatorTypeBuilder< typename WasatchCore::GradOpSelector<Vel2T, SpatialOps::YDIR>::Gradient, Vel2T, FieldT >::type GradYT;
  typedef typename SpatialOps::OperatorTypeBuilder< typename WasatchCore::GradOpSelector<Vel3T, SpatialOps::ZDIR>::Gradient, Vel3T, FieldT >::type GradZT;
  const GradXT* gradXOp_;
  const GradYT* gradYOp_;
  const GradZT* gradZOp_;

  StableTimestep( const Expr::Tag& rhoTag,
              const Expr::Tag& viscTag,
              const Expr::Tag& uTag,
              const Expr::Tag& vTag,
              const Expr::Tag& wTag,
              const Expr::Tag& puTag,
              const Expr::Tag& pvTag,
              const Expr::Tag& pwTag,
              const Expr::Tag& csoundTag);
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a StableTimestep expression
     *  @param resultTag the tag for the value that this expression computes
     */
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& rhoTag,
             const Expr::Tag& viscTag,
             const Expr::Tag& uTag,
             const Expr::Tag& vTag,
             const Expr::Tag& wTag,
             const Expr::Tag& puTag,
             const Expr::Tag& pvTag,
             const Expr::Tag& pwTag,
             const Expr::Tag& csoundTag);

    Expr::ExpressionBase* build() const;

  private:
    const Expr::Tag rhoTag_, viscTag_, uTag_, vTag_, wTag_, puTag_, pvTag_, pwTag_, csoundTag_;
  };

  ~StableTimestep();
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};

#endif // StableTimestep_Expr_h
