#ifndef ParticleDragForce_Expr_h
#define ParticleDragForce_Expr_h

#include <expression/Expression.h>
#include <spatialops/particles/ParticleFieldTypes.h>

//==================================================================

/**
 *  \class  ParticleDragForce
 *  \ingroup WasatchParticles
 *  \author Tony Saad
 *  \date   June, 2014
 *  \brief  Calculates the particle drag force.
 *
 *  The drag force is given by
 *  \f[
 *    F_\text{Drag} =
 *    \tfrac{1}{2}c_\text{d} \frac{A_{p}} \rho_{p}  \left(\mathbf{u}_\text{g} -\mathbf{u}_{p}\right) \left|\mathbf{u}_\text{g} - \mathbf{u}_{p}\right|
 *  \f]
 *  The drag coefficient is written as
 *  \f[
 *     c_\text{d} \equiv \frac{24}{\text{Re}_\text{p}} f_\text{d}
 *  \f]
 *  where \f$ f_\text{d} \f$ is a unified drag coefficient and \f$ \text{Re}_\text{p} \equiv \frac{ \rho_\text{p} \left|\mathbf{u}_text{g} - \mathbf{u}_{p} \right| d_\text{p} }{\mu_\text{g}} \f$ is the particle
 *  Reynolds number. Upon substitution and simplification, we get
 *    \f[
 *      \frac{ F_\text{Drag} } { m_\text{p} } =
 *      f_\text{d} \frac{ \left(\mathbf{u}_\text{g} -\mathbf{u}_{p}\right) } { \tau_\text{p} }
 *    \f]
 *  where \f$ \tau_\text{p} \equiv \frac{ \rho_\text{p} }{ 18 \mu_\text{g} } \f$ is the particle response time.
 */
template< typename GasVelT >
class ParticleDragForce : public Expr::Expression<ParticleField>
{
  
  DECLARE_FIELDS(ParticleField, pfd_, px_, py_, pz_, ptau_, pvel_, psize_)
  DECLARE_FIELD (GasVelT, gvel_)
  
  typedef typename SpatialOps::Particle::CellToParticle<GasVelT> GVel2POpT;
  GVel2POpT* gvOp_;
  
  ParticleDragForce( const Expr::Tag& gasVelTag,
                     const Expr::Tag& ParticleDragForceCoefTag,
                     const Expr::Tag& particleTauTag,
                     const Expr::Tag& particleVelTag,
                     const Expr::Tag& particleSizeTag,
                     const Expr::TagList& particlePositionTags );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \param resultTag The particle drag force
     *  \param gasVelTag The gas-phase velocity in the direction of this drag force
     *  \param ParticleDragForceCoefTag The particle drag coefficient
     *  \param particleTauTag The particle response time expression
     *  \param particleVelTag The particle velocity in the same direction as this drag force
     *  \param particleSizeTag The particle size
     *  \param particlePositionTags Particle positions: x, y, and z - respectively
     */
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& gasVelTag,
             const Expr::Tag& ParticleDragForceCoefTag,
             const Expr::Tag& particleTauTag,
             const Expr::Tag& particleVelTag,
             const Expr::Tag& particleSizeTag,
             const Expr::TagList& particlePositionTags )
      : ExpressionBuilder(resultTag),
        gVelTag_      ( gasVelTag            ),
        pDragCoefTag_ ( ParticleDragForceCoefTag  ),
        pTauTag_      ( particleTauTag       ),
        pVelTag_      ( particleVelTag       ),
        pSizeTag_     ( particleSizeTag      ),
        pPosTags_     ( particlePositionTags )
    {}

    ~Builder(){}

    Expr::ExpressionBase* build() const{
      return new ParticleDragForce<GasVelT>( gVelTag_, pDragCoefTag_, pTauTag_, pVelTag_, pSizeTag_, pPosTags_);
    }

  private:
    const Expr::Tag gVelTag_, pDragCoefTag_, pTauTag_, pVelTag_, pSizeTag_;
    const Expr::TagList pPosTags_;
  };
  
  ~ParticleDragForce(){}

  void bind_operators( const SpatialOps::OperatorDatabase& opDB ){
    gvOp_ = opDB.retrieve_operator<GVel2POpT>();
  }

  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################



template<typename GasVelT>
ParticleDragForce<GasVelT>::
ParticleDragForce( const Expr::Tag& gasVelTag,
                   const Expr::Tag& ParticleDragForceCoefTag,
                   const Expr::Tag& particleTauTag,
                   const Expr::Tag& particleVelTag,
                   const Expr::Tag& particleSizeTag,
                   const Expr::TagList& particlePositionTags )
: Expr::Expression<ParticleField>()
{
  this->set_gpu_runnable(false);  // waiting for GPU-enabled particle interpolants

  px_    = this->template create_field_request<ParticleField>(particlePositionTags[0]);
  py_    = this->template create_field_request<ParticleField>(particlePositionTags[1]);
  pz_    = this->template create_field_request<ParticleField>(particlePositionTags[2]);

  gvel_  = this->template create_field_request<GasVelT>(gasVelTag);
  pfd_   = this->template create_field_request<ParticleField>(ParticleDragForceCoefTag);
  ptau_  = this->template create_field_request<ParticleField>(particleTauTag);
  pvel_  = this->template create_field_request<ParticleField>(particleVelTag);
  psize_ = this->template create_field_request<ParticleField>(particleSizeTag);
}

//------------------------------------------------------------------

template<typename GasVelT>
void
ParticleDragForce<GasVelT>::
evaluate()
{
  ParticleField& result = this->value();
  
  using namespace SpatialOps;
  
  const ParticleField& px = px_->field_ref();
  const ParticleField& py = py_->field_ref();
  const ParticleField& pz = pz_->field_ref();
  const ParticleField& psize = psize_->field_ref();
  const ParticleField& pvel = pvel_->field_ref();
  const ParticleField& pfd = pfd_->field_ref();
  const ParticleField& ptau = ptau_->field_ref();
  const GasVelT& gvel = gvel_->field_ref();

  SpatFldPtr<ParticleField> tmpu = SpatialFieldStore::get<ParticleField>( result );
  
  // assemble drag term: cd * A/2*rho*(v-up) = cd * 3/(4r) * m * (v-up)  (assumes a spherical particle)
  gvOp_->set_coordinate_information(&px,&py,&pz,&psize);
  gvOp_->apply_to_field( gvel, *tmpu );
  
  result <<= ( *tmpu - pvel ) * pfd / ptau;
}

//------------------------------------------------------------------

#endif // ParticleDragForce_Expr_h
