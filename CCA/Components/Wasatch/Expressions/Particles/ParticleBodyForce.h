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
  const Expr::Tag gDensityTag_, pDensityTag_,pSizeTag_;
  const Expr::TagList pPosTags_;

  const ParticleField *prho_, *px_, *py_, *pz_, *psize_;
  const ScalarT *grho_;
  
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
             const Expr::TagList& particlePositionTags );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const Expr::Tag gDensityTag_, pDensityTag_, pSizeTag_;
    const Expr::TagList pPosTags_;
  };
  
  ~ParticleBodyForce();
  
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

template<typename ScalarT>
ParticleBodyForce<ScalarT>::
ParticleBodyForce( const Expr::Tag& gasDensityTag,
                   const Expr::Tag& particleDensityTag,
                   const Expr::Tag& particleSizeTag,
                   const Expr::TagList& particlePositionTags )
: Expr::Expression<ParticleField>(),
  gDensityTag_ ( gasDensityTag        ),
  pDensityTag_ ( particleDensityTag   ),
  pSizeTag_    ( particleSizeTag      ),
  pPosTags_    ( particlePositionTags )
{
  this->set_gpu_runnable(false);  // need new particle operators...
}

//------------------------------------------------------------------

template<typename ScalarT>
ParticleBodyForce<ScalarT>::
~ParticleBodyForce()
{}

//------------------------------------------------------------------

template<typename ScalarT>
void
ParticleBodyForce<ScalarT>::
advertise_dependents( Expr::ExprDeps& exprDeps)
{
  exprDeps.requires_expression( gDensityTag_);
  exprDeps.requires_expression( pPosTags_   );
  exprDeps.requires_expression( pSizeTag_   );
  exprDeps.requires_expression( pDensityTag_);
}

//------------------------------------------------------------------

template<typename ScalarT>
void
ParticleBodyForce<ScalarT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<ParticleField>::type& pfm = fml.template field_manager<ParticleField>();
  px_    = &pfm.field_ref( pPosTags_[0] );
  py_    = &pfm.field_ref( pPosTags_[1] );
  pz_    = &pfm.field_ref( pPosTags_[2] );
  psize_ = &pfm.field_ref( pSizeTag_    );
  prho_  = &pfm.field_ref( pDensityTag_ );

  grho_  = &fml.field_ref<ScalarT>( gDensityTag_ );
}

//------------------------------------------------------------------

template<typename ScalarT>
void
ParticleBodyForce<ScalarT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  sOp_ = opDB.retrieve_operator<S2POpT>();
}

//------------------------------------------------------------------

template<typename ScalarT>
void
ParticleBodyForce<ScalarT>::
evaluate()
{
  using namespace SpatialOps;
  ParticleField& result = this->value();
  
  SpatFldPtr<ParticleField> tmprho = SpatialFieldStore::get<ParticleField>( result );
  
  sOp_->set_coordinate_information(px_,py_,pz_,psize_);
  sOp_->apply_to_field( *grho_, *tmprho );
  
  result <<= -9.81 * (*prho_ - *tmprho) / *prho_;
}

//------------------------------------------------------------------

template<typename ScalarT>
ParticleBodyForce<ScalarT>::
Builder::Builder( const Expr::Tag& resultTag,
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

//------------------------------------------------------------------

template<typename ScalarT>
Expr::ExpressionBase*
ParticleBodyForce<ScalarT>::Builder::build() const
{
  return new ParticleBodyForce<ScalarT>( gDensityTag_, pDensityTag_, pSizeTag_, pPosTags_);
}

//------------------------------------------------------------------

#endif // ParticleBodyForce_Expr_h
