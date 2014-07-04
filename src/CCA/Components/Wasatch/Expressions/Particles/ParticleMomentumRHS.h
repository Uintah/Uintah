#ifndef ParticleMomentumRHS_Expr_h
#define ParticleMomentumRHS_Expr_h

#include <expression/Expression.h>

#include <spatialops/structured/SpatialFieldStore.h>
#include <spatialops/particles/ParticleFieldTypes.h>

/**
 *  \class ParticleMomentumRHS
 *  \ingroup WasatchParticles
 *  \brief Evaluates the RHS of the particle momentum equations.
 *
 *  \author James C. Sutherland, Tony Saad
 *
 *  The particle momentum equation can be written as
 *   \f[
 *      m_p \frac{\mathrm{d} \mathbf{u}_{p}}{\mathrm{d} t} =
 *      m_{p}\mathbf{f}_{p}
 *      -\int_{\mathsf{S}} \left( \tau_{p}\cdot\mathbf{a}+p\mathbf{a} \right) \mathrm{d}\mathsf{S}
 *   \f]
 *
 *  Invoking a drag model for the stress term and assuming that the
 *  particle is sub-grid so that there is no resolved pressure
 *  distribution around the surface we can write we have
 *
 *  \f[
 *    \frac{\mathrm{d} m_{p}\mathbf{u}_{p}}{\mathrm{d} t} =
 *    c_{d}\frac{A_{p}}{2}\rho_{p}\left(\mathbf{v}-\mathbf{u}_{p}\right)^{2}
 *    +m_{p}\mathbf{f}_{p}
 *    -\mathsf{V}_p \nabla p,
 *  \f]
 *
 *  which is the working equation.
 *
 */
class ParticleMomentumRHS
: public Expr::Expression<SpatialOps::Particle::ParticleField>
{
  const Expr::Tag pBodyForceTag_, pDragTag_;
  const bool doBodyForce_, doDragForce_;
  const ParticleField *pg_, *pdrag_;
  
  ParticleMomentumRHS( const Expr::Tag& particleBodyForceTag,
                       const Expr::Tag& ParticleDragForceTag );
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& particleBodyForceTag,
             const Expr::Tag& ParticleDragForceTag );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const Expr::Tag pBodyForceTag_, pDragTag_;
  };
  
  ~ParticleMomentumRHS();
  
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




ParticleMomentumRHS::
ParticleMomentumRHS( const Expr::Tag& particleBodyForceTag,
                     const Expr::Tag& ParticleDragForceTag )
: Expr::Expression<ParticleField>(),
  pBodyForceTag_( particleBodyForceTag ),
  pDragTag_     ( ParticleDragForceTag ),
  doBodyForce_  ( pBodyForceTag_ != Expr::Tag() ),
  doDragForce_  ( pDragTag_ != Expr::Tag() )
{
  this->set_gpu_runnable(true);
}

//------------------------------------------------------------------

ParticleMomentumRHS::
~ParticleMomentumRHS()
{}

//------------------------------------------------------------------


void
ParticleMomentumRHS::
advertise_dependents( Expr::ExprDeps& exprDeps)
{
  if( doDragForce_ ) exprDeps.requires_expression( pDragTag_      );
  if( doBodyForce_ ) exprDeps.requires_expression( pBodyForceTag_ );
}

//------------------------------------------------------------------

void
ParticleMomentumRHS::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldMgrSelector<ParticleField>::type& pfm = fml.field_manager<ParticleField>();
  
  if( doBodyForce_ ) pg_   = &pfm.field_ref( pBodyForceTag_ );
  if( doDragForce_ ) pdrag_= &pfm.field_ref( pDragTag_      );
}

//------------------------------------------------------------------

void
ParticleMomentumRHS::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//------------------------------------------------------------------

void
ParticleMomentumRHS::
evaluate()
{
  using namespace SpatialOps;
  
  ParticleField& result = this->value();
  if( doBodyForce_ && doDragForce_ ){
    result <<= *pdrag_ + *pg_;
  }
  else if( doDragForce_ ){
    result <<= *pdrag_;
  }
  else if( doBodyForce_ ){
    result <<= *pg_;
  }
  else{
    result <<= 0.0;
  }
}

//------------------------------------------------------------------

ParticleMomentumRHS::
Builder::Builder( const Expr::Tag& resultTag,
                  const Expr::Tag& particleBodyForceTag,
                  const Expr::Tag& ParticleDragForceTag )
: ExpressionBuilder( resultTag ),
  pBodyForceTag_( particleBodyForceTag ),
  pDragTag_     ( ParticleDragForceTag )
{}

//------------------------------------------------------------------


Expr::ExpressionBase*
ParticleMomentumRHS::Builder::build() const
{
  return new ParticleMomentumRHS( pBodyForceTag_, pDragTag_ );
}

//------------------------------------------------------------------

#endif // ParticleParticleMomentumRHS_Expr_h
