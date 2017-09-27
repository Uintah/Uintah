#ifndef ParticleBodyForce_Expr_h
#define ParticleBodyForce_Expr_h

#include <expression/Expression.h>
#include <spatialops/particles/ParticleFieldTypes.h>
#include <spatialops/particles/ParticleOperators.h>
#include <spatialops/OperatorDatabase.h>

//==================================================================

/**
 *  \class  ParticleBodyForce
 *  \ingroup WasatchParticles
 *  \author Tony Saad, ODT
 *  \date   June, 2014
 *  \brief  Calculates particle body force which includes weight and buoyancy forces.
 *
 *  The buoyancy force can be written as
 *  \f[ F = w_\text{p} - f_\text{b} \f]
 *  where \f$ F \f$ is the total body force experience by the particle, \f$ w_\text{p} \f$ is the particle
 *  weight, and \f$ f_\text{b} \f$ is the buoyancy force experience by the particle as exerted by the fluid.
 *  The particle weight is easily computed as
 *   \f[ w_\text{p} = \rho_\text{p} \mathcal{V}_\text{p} \f]
 *  where \f$\mathcal{V}_\text{p}\f$ is the particle volume. The buoyancy force is equal to the weight
 *  of the fluid displaced by the particle. The volume of the fluid displaced by the particle is assumed
 *  to be \f$ \mathcal{V}_\text{p} \f$ and, if the fluid density is given by \f$ \rho_\text{f} \f$, then
 *  the buoyancy force is given by
 *   \f[ f_\text{b} = \rho_\text{f} \mathcal{V}_\text{p} g \f].
 *  At the outset, the total body force is divided by the particle mass
 *  \f$ m_\text{p} \equiv \rho_\text{p} \mathcal{V}_\text{p} \f$, then, upon substitution and simplification
 *   \f[ \frac{F}{m_\text{p}} = g - \frac{\rho_\text{f}}{\rho_\text{p}} g \f].
 *
 *  \tparam ScalarT the field type for the gas phase density
 */
template< typename ScalarT >
class ParticleBodyForce
: public Expr::Expression<ParticleField>
{
  DECLARE_FIELDS(ParticleField, prho_, px_, py_, pz_, psize_)
  DECLARE_FIELD (ScalarT, grho_)
  
  typedef typename SpatialOps::Particle::CellToParticle<ScalarT> S2POpT;
  S2POpT* sOp_;
  
  ParticleBodyForce( const Expr::Tag& gasDensityTag,
                     const Expr::Tag& particleDensityTag,
                     const Expr::Tag& particleSizeTag,
                     const Expr::TagList& particlePositionTags );
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \param resultTag The buoyancy force
     *  \param gasDensityTag The gas-phase velocity
     *  \param particleDensityTag The particle density Tag
     *  \param particleSizeTag The particle Size tag
     *  \param particlePositionTags the particle coordinates - x, y, and z, respectively
     */
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& gasDensityTag,
             const Expr::Tag& particleDensityTag,
             const Expr::Tag& particleSizeTag,
             const Expr::TagList& particlePositionTags )
    : ExpressionBuilder(resultTag),
      gDensityTag_  ( gasDensityTag        ),
      pDensityTag_  ( particleDensityTag   ),
      pSizeTag_     ( particleSizeTag      ),
      pPosTags_     ( particlePositionTags )
    {}

    ~Builder(){}
    Expr::ExpressionBase* build() const{
      return new ParticleBodyForce<ScalarT>( gDensityTag_, pDensityTag_, pSizeTag_, pPosTags_);
    }

  private:
    const Expr::Tag gDensityTag_, pDensityTag_, pSizeTag_;
    const Expr::TagList pPosTags_;
  };
  
  ~ParticleBodyForce(){}
  
  void bind_operators( const SpatialOps::OperatorDatabase& opDB ){
    sOp_ = opDB.retrieve_operator<S2POpT>();
  }

  void evaluate();
  
};

// ###################################################################
//
//                          Implementation
//
// ###################################################################

template<typename ScalarT>
ParticleBodyForce<ScalarT>::
ParticleBodyForce( const Expr::Tag& gasDensityTag,
                   const Expr::Tag& particleDensityTag,
                   const Expr::Tag& particleSizeTag,
                   const Expr::TagList& particlePositionTags )
: Expr::Expression<ParticleField>()
{
  this->set_gpu_runnable(false);  // waiting for GPU-enabled particle interpolants

  px_    = this->template create_field_request<ParticleField>(particlePositionTags[0]);
  py_    = this->template create_field_request<ParticleField>(particlePositionTags[1]);
  pz_    = this->template create_field_request<ParticleField>(particlePositionTags[2]);
  psize_ = this->template create_field_request<ParticleField>(particleSizeTag);
  prho_  = this->template create_field_request<ParticleField>(particleDensityTag);
  grho_  = this->template create_field_request<ScalarT>(gasDensityTag);
}

//------------------------------------------------------------------

template<typename ScalarT>
void
ParticleBodyForce<ScalarT>::
evaluate()
{
  using namespace SpatialOps;
  ParticleField& result = this->value();
  
  const ParticleField& px = px_->field_ref();
  const ParticleField& py = py_->field_ref();
  const ParticleField& pz = pz_->field_ref();
  const ParticleField& prho = prho_->field_ref();
  const ParticleField& psize = psize_->field_ref();
  const ScalarT& grho = grho_->field_ref();
  
  SpatFldPtr<ParticleField> tmprho = SpatialFieldStore::get<ParticleField>( result );
  
  sOp_->set_coordinate_information(&px,&py,&pz,&psize);
  sOp_->apply_to_field( grho, *tmprho );
  
  // jcs this hard-codes the acceleration constant in SI units and in a particular direction.  We shouldn't do this.
  result <<= -9.81 * (prho - *tmprho) / prho;
}

//------------------------------------------------------------------

#endif // ParticleBodyForce_Expr_h
