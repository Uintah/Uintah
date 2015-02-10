#ifndef ParticlePositionRHS_h
#define ParticlePositionRHS_h

#include <expression/ExprLib.h>

//==================================================================
/**
 *  \class ParticlePositionRHS
 *  \ingroup WasatchParticles
 *  \brief Calculates the change in particle position according to
 *         \f$\frac{d x}{d t}=u_p\f$, where \f$u_p\f$ is the
 *         particle velocity.
 *
 *  \author James C. Sutherland, Tony Saad
 */
class ParticlePositionRHS :
public Expr::Expression<ParticleField>
{
  DECLARE_FIELD(ParticleField, pvel_);
  
  ParticlePositionRHS( const Expr::Tag& particleVelocity );
  
public:
  /**
   *  \class Builder
   *  \ingroup Particle
   *  \brief constructs ParticlePositionRHS objects
   */
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /** \brief Construct a ParticlePositionRHS::Builder
     *  \param positionRHSTag the position RHS tag (value that this expression evaluates)
     *  \param particleVelocity the advecting velocity for the particle
     */
    Builder( const Expr::Tag& positionRHSTag,
             const Expr::Tag& particleVelocity );
    ~Builder(){}
    Expr::ExpressionBase* build() const{ return new ParticlePositionRHS( pvel_ ); }
    
  private:
    const Expr::Tag pvel_;
  };
  
  ~ParticlePositionRHS();
  
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
  
};

//==================================================================

ParticlePositionRHS::
ParticlePositionRHS( const Expr::Tag& particleVelocityTag )
: Expr::Expression<ParticleField>()
{
  this->set_gpu_runnable( true );
   pvel_ = create_field_request<ParticleField>(particleVelocityTag);
}

//------------------------------------------------------------------

ParticlePositionRHS::~ParticlePositionRHS(){}

//------------------------------------------------------------------

void
ParticlePositionRHS::bind_operators( const SpatialOps::OperatorDatabase& opDB ){}

//------------------------------------------------------------------

void
ParticlePositionRHS::evaluate()
{
  using SpatialOps::operator<<=;
  ParticleField& rhs = this->value();
  rhs <<= pvel_->field_ref();
}

//------------------------------------------------------------------

ParticlePositionRHS::
Builder::Builder( const Expr::Tag& positionRHSTag,
                  const Expr::Tag& particleVelocity )
: ExpressionBuilder( positionRHSTag ),
  pvel_( particleVelocity )
{}


//==================================================================



#endif // ParticlePositionEquation_h
