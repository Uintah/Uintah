#ifndef ParticleDensity_Expr_h
#define ParticleDensity_Expr_h

#include <expression/Expression.h>
#include <spatialops/particles/ParticleFieldTypes.h>


/**
 *  \class  ParticleDensity
 *  \ingroup WasatchParticles
 *  \author Tony Saad, Naveen Punati
 *  \date   June, 2014
 *  \brief  Calculates the particle density \f$ \rho_\text{p} = \frac{m_\text{p}}{\mathcal{V}_\text{p}}\f$
 */
class ParticleDensity
: public Expr::Expression<ParticleField>
{
  DECLARE_FIELDS(ParticleField, pmass_, psize_)
  
  ParticleDensity( const Expr::Tag& pmassTag,
                   const Expr::Tag& psizeTag )
    : Expr::Expression<ParticleField>()
  {
    this->set_gpu_runnable(true);
    pmass_ = create_field_request<ParticleField>(pmassTag);
    psize_ = create_field_request<ParticleField>(psizeTag);
  }
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     * @param resultTag the particle density
     * @param pmassTag the particle mass
     * @param psizeTag the particle size
     */
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& pmassTag,
             const Expr::Tag& psizeTag )
    : ExpressionBuilder(resultTag),
      pmassTag_( pmassTag ),
      psizeTag_( psizeTag )
    {}
    ~Builder(){}
    Expr::ExpressionBase* build() const{ return new ParticleDensity(pmassTag_,psizeTag_); }
  private:
    const Expr::Tag pmassTag_, psizeTag_;
  };
  
  ~ParticleDensity(){}
  
  void evaluate()
  {
    using namespace SpatialOps;
    ParticleField& result = this->value();
    const ParticleField& pmass = pmass_->field_ref();
    const ParticleField& psize = psize_->field_ref();

    result <<= pmass / ( (22.0/42.0) * pow( psize, 3.0 )) ;
  }

};

//------------------------------------------------------------------

#endif // ParticleDensity_Expr_h
