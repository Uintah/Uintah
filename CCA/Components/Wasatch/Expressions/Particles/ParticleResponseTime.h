#ifndef ParticleResponseTime_Expr_h
#define ParticleResponseTime_Expr_h

#include <expression/Expression.h>

//==================================================================

/**
 *  \class ParticleResponseTime
 *  \ingroup WasatchParticles
 *  \author Tony Saad, ODT
 *  \date June 2014
 *  \brief Calculates the particle Response time \f$\tau_\text{p}\f$. 
 *  \f[
 *    \tau_\text{p} \equiv \frac{ \rho_\text{p} }{ 18 \mu_\text{g} }
 *   \f]
 *  \tparam Field type for the gas viscosity.
 */
template< typename ViscT >
class ParticleResponseTime
 : public Expr::Expression<ParticleField>
{
  DECLARE_FIELDS(ParticleField, pdensity_, psize_, px_, py_, pz_);
  DECLARE_FIELD(ViscT, gVisc_);

  typedef typename SpatialOps::Particle::CellToParticle<ViscT> Scal2POpT;
  Scal2POpT* s2pOp_;

  ParticleResponseTime( const Expr::Tag& particleDensityTag,
                        const Expr::Tag& particleSizeTag,
                        const Expr::Tag& gasViscosityTag,
                        const Expr::TagList& particlePositionTags );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     * @brief Builder for ParticleResponseTime
     * @param resultTag the particle response time tag
     * @param particleDensityTag tag for particle density
     * @param particleSizeTag tag for particle size
     * @param gasViscosityTag tag for gas phase viscosity
     * @param particlePositionTags tag list of particle coordinates
     */
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& particleDensityTag,
             const Expr::Tag& particleSizeTag,
             const Expr::Tag& gasViscosityTag,
             const Expr::TagList& particlePositionTags );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const Expr::Tag pDensityTag_, pSizeTag_, gViscTag_;
    const Expr::TagList pPosTags_;
  };

  ~ParticleResponseTime();
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};

 // ###################################################################
  //
  //                          Implementation
  //
  // ###################################################################

template< typename ViscT >
ParticleResponseTime<ViscT>::
ParticleResponseTime( const Expr::Tag& particleDensityTag,
                      const Expr::Tag& particleSizeTag,
                      const Expr::Tag& gasViscosityTag,
                      const Expr::TagList& particlePositionTags )
  : Expr::Expression<ParticleField>()
{
  this->set_gpu_runnable( false );  // not until we get particle interpolants GPU ready
  this->template create_field_request(particleDensityTag, pdensity_);
  this->template create_field_request(particleSizeTag, psize_);
  this->template create_field_request(gasViscosityTag, gVisc_);
  this->template create_field_request(particlePositionTags[0], px_);
  this->template create_field_request(particlePositionTags[1], py_);
  this->template create_field_request(particlePositionTags[2], pz_);
}

//--------------------------------------------------------------------

template< typename ViscT >
ParticleResponseTime<ViscT>::~ParticleResponseTime()
{}

//--------------------------------------------------------------------

template< typename ViscT >
void
ParticleResponseTime<ViscT>::bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  s2pOp_ = opDB.retrieve_operator<Scal2POpT>();
}

//--------------------------------------------------------------------

template< typename ViscT >
void
ParticleResponseTime<ViscT>::evaluate()
{
  using namespace SpatialOps;
  ParticleField& result = this->value();
  const ParticleField& psize = psize_->field_ref();
  const ParticleField& px = px_->field_ref();
  const ParticleField& py = py_->field_ref();
  const ParticleField& pz = pz_->field_ref();
  const ParticleField& pdensity = pdensity_->field_ref();
  const ViscT& gVisc = gVisc_->field_ref();
  
  SpatFldPtr<ParticleField> tmpvisc = SpatialFieldStore::get<ParticleField>( result );
  
  s2pOp_->set_coordinate_information(&px,&py,&pz,&psize);
  s2pOp_->apply_to_field( gVisc, *tmpvisc );

  result <<= pdensity * psize * psize / ( 18.0 * *tmpvisc );
}

//--------------------------------------------------------------------

template< typename ViscT >
ParticleResponseTime<ViscT>::
Builder::Builder( const Expr::Tag& resultTag,
                  const Expr::Tag& particleDensityTag,
                  const Expr::Tag& particleSizeTag,
                  const Expr::Tag& gasViscosityTag,
                  const Expr::TagList& particlePositionTags )
  : ExpressionBuilder( resultTag ),
    pDensityTag_( particleDensityTag ),
    pSizeTag_   ( particleSizeTag    ),
    gViscTag_   ( gasViscosityTag    ),
    pPosTags_   (particlePositionTags)
{}

//--------------------------------------------------------------------

template< typename ViscT >
Expr::ExpressionBase*
ParticleResponseTime<ViscT>::Builder::build() const
{
  return new ParticleResponseTime( pDensityTag_, pSizeTag_, gViscTag_, pPosTags_ );
}

#endif // ParticleResponseTime_Expr_h
