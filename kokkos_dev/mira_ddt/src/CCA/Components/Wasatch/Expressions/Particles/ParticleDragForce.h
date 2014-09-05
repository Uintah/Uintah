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
  const Expr::Tag gVelTag_, pDragCoefTag_, pTauTag_, pVelTag_, pSizeTag_;
  const Expr::TagList pPosTags_;
  
  const ParticleField *pfd_, *px_, *py_, *pz_, *ptau_, *pvel_, *psize_;
  const GasVelT *gvel_;
  
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
             const Expr::TagList& particlePositionTags );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const Expr::Tag gVelTag_, pDragCoefTag_, pTauTag_, pVelTag_, pSizeTag_;
    const Expr::TagList pPosTags_;
  };
  
  ~ParticleDragForce();
  
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



template<typename GasVelT>
ParticleDragForce<GasVelT>::
ParticleDragForce( const Expr::Tag& gasVelTag,
                   const Expr::Tag& ParticleDragForceCoefTag,
                   const Expr::Tag& particleTauTag,
                   const Expr::Tag& particleVelTag,
                   const Expr::Tag& particleSizeTag,
                   const Expr::TagList& particlePositionTags )
: Expr::Expression<ParticleField>(),
  gVelTag_     ( gasVelTag            ),
  pDragCoefTag_( ParticleDragForceCoefTag ),
  pTauTag_     ( particleTauTag       ),
  pVelTag_     ( particleVelTag       ),
  pSizeTag_    ( particleSizeTag      ),
  pPosTags_    ( particlePositionTags )
{
  this->set_gpu_runnable(false);  // need new particle operators...
}

//------------------------------------------------------------------

template<typename GasVelT>
ParticleDragForce<GasVelT>::
~ParticleDragForce()
{}

//------------------------------------------------------------------

template<typename GasVelT>
void
ParticleDragForce<GasVelT>::
advertise_dependents( Expr::ExprDeps& exprDeps)
{
  exprDeps.requires_expression( pDragCoefTag_ );
  exprDeps.requires_expression( pPosTags_     );
  exprDeps.requires_expression( pTauTag_      );
  exprDeps.requires_expression( pVelTag_      );
  exprDeps.requires_expression( pSizeTag_     );
  
  exprDeps.requires_expression( gVelTag_);
}

//------------------------------------------------------------------

template<typename GasVelT>
void
ParticleDragForce<GasVelT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<ParticleField>::type& pfm = fml.template field_manager<ParticleField>();
  pfd_   = &pfm.field_ref( pDragCoefTag_ );
  px_    = &pfm.field_ref( pPosTags_[0]  );
  py_    = &pfm.field_ref( pPosTags_[1]  );
  pz_    = &pfm.field_ref( pPosTags_[2]  );
  ptau_  = &pfm.field_ref( pTauTag_      );
  pvel_  = &pfm.field_ref( pVelTag_      );
  psize_ = &pfm.field_ref( pSizeTag_     );
  
  gvel_ = &fml.field_ref<GasVelT>( gVelTag_ );
}

//------------------------------------------------------------------

template<typename GasVelT>
void
ParticleDragForce<GasVelT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  gvOp_ = opDB.retrieve_operator<GVel2POpT>();
}

//------------------------------------------------------------------

template<typename GasVelT>
void
ParticleDragForce<GasVelT>::
evaluate()
{
  ParticleField& result = this->value();
  
  using namespace SpatialOps;
  SpatFldPtr<ParticleField> tmpu = SpatialFieldStore::get<ParticleField>( result );
  
  // assemble drag term: cd * A/2*rho*(v-up) = cd * 3/(4r) * m * (v-up)  (assumes a spherical particle)
  gvOp_->set_coordinate_information( px_, py_, pz_, psize_ );
  gvOp_->apply_to_field( *gvel_, *tmpu );
  
  result <<= ( *tmpu - *pvel_ ) * *pfd_ / *ptau_;
}

//------------------------------------------------------------------

template<typename GasVelT>
ParticleDragForce<GasVelT>::
Builder::Builder( const Expr::Tag& resultTag,
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

//------------------------------------------------------------------

template<typename GasVelT>
Expr::ExpressionBase*
ParticleDragForce<GasVelT>::Builder::build() const
{
  return new ParticleDragForce<GasVelT>( gVelTag_, pDragCoefTag_, pTauTag_, pVelTag_, pSizeTag_, pPosTags_);
}

//------------------------------------------------------------------

#endif // ParticleDragForce_Expr_h
