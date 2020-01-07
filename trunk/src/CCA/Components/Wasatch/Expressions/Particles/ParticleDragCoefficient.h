#ifndef ParticleDragCoefficient_Expr_h
#define ParticleDragCoefficient_Expr_h

#include <expression/Expression.h>
#include <cmath>

//==================================================================
/**
 *  \class  ParticleDensity
 *  \ingroup WasatchParticles
 *  \author Tony Saad, ODT
 *  \date   June, 2014
 *  \brief  Calculates the drag coefficient.
 */
class ParticleDragCoefficient
 : public Expr::Expression<ParticleField>
{
  ParticleDragCoefficient( const Expr::Tag& pReTag )
    : Expr::Expression<ParticleField>()
  {
    this->set_gpu_runnable(true);
    pRe_ = create_field_request<ParticleField>(pReTag);
  }

  DECLARE_FIELD(ParticleField, pRe_)

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     * @param dragTag tag for the drag coefficient
     * @param pReTag tag for the particle reynolds number
     */
    Builder( const Expr::Tag& dragTag,
             const Expr::Tag& pReTag )
    : ExpressionBuilder(dragTag),
      pRet_(pReTag)
    {}

    ~Builder(){}
    Expr::ExpressionBase* build() const{ return new ParticleDragCoefficient(pRet_); }
  private:
    const Expr::Tag pRet_;
  };

  void evaluate()
  {
    using namespace SpatialOps;
    ParticleField& result = this->value();
    const ParticleField& pRe = pRe_->field_ref();

    result <<= cond( pRe <= 1.0,  1.0 )
                   ( pRe <= 1000, 1.0 + 0.15 * pow( pRe , 0.687 ) )
                   ( 0.0183 * pRe );
  }

};

#endif // ParticleDragCoefficient_Expr_h
