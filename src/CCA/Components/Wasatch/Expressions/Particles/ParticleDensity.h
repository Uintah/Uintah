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
  const Expr::Tag pmassTag_,psizeTag_;
  const ParticleField *pmass_, *psize_;
  
  ParticleDensity( const Expr::Tag& pmassTag,
                   const Expr::Tag& psizeTag );
  
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
             const Expr::Tag& psizeTag );
    ~Builder(){}
    Expr::ExpressionBase* build() const{ return new ParticleDensity(pmassTag_,psizeTag_); }
  private:
    const Expr::Tag pmassTag_, psizeTag_;
  };
  
  ~ParticleDensity();
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();
};

// ###################################################################
//
//                          Implementation
//
// ###################################################################

ParticleDensity::
ParticleDensity( const Expr::Tag& pmassTag,
                 const Expr::Tag& psizeTag )
: Expr::Expression<ParticleField>(),
  pmassTag_( pmassTag ),
  psizeTag_( psizeTag )
{
  this->set_gpu_runnable(true);
}

//------------------------------------------------------------------

ParticleDensity::~ParticleDensity(){}

//------------------------------------------------------------------

void
ParticleDensity::advertise_dependents( Expr::ExprDeps& exprDeps)
{
  exprDeps.requires_expression( pmassTag_  );
  exprDeps.requires_expression( psizeTag_  );
}

//------------------------------------------------------------------

void
ParticleDensity::bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldMgrSelector<ParticleField>::type& pfm = fml.field_manager<ParticleField>();
  pmass_ = &pfm.field_ref( pmassTag_ );
  psize_ = &pfm.field_ref( psizeTag_ );
}

//------------------------------------------------------------------

void
ParticleDensity::evaluate()
{
  using namespace SpatialOps;
  ParticleField& result = this->value();
  result <<= *pmass_ / ( (22.0/42.0) * pow( *psize_, 3.0 )) ;
}

//------------------------------------------------------------------

ParticleDensity::
Builder::Builder( const Expr::Tag& resultTag,
                  const Expr::Tag& pmassTag,
                  const Expr::Tag& psizeTag )
: ExpressionBuilder(resultTag),
  pmassTag_( pmassTag ),
  psizeTag_( psizeTag )
{}

//------------------------------------------------------------------

#endif // ParticleDensity_Expr_h
