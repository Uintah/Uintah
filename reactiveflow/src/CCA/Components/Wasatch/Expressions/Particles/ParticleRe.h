#ifndef ParticleRe_Expr_h
#define ParticleRe_Expr_h

#include <expression/Expression.h>

#include <spatialops/particles/ParticleFieldTypes.h>
#include <spatialops/particles/ParticleOperators.h>
#include <spatialops/OperatorDatabase.h>

#include<cmath>

//==================================================================
/**
 *  \class ParticleRe
 *  \ingroup WasatchParticles
 *  \author Tony Saad, ODT
 *  \date June 2014
 *  \brief Calculates the particle Reynolds number. 
 *  \f[
 *    \text{Re}_\text{p} \equiv \frac{ \rho_\text{p} \left|\mathbf{u}_text{g} - \mathbf{u}_{p} \right| d_\text{p} }{\mu_\text{g}}
 *   \f]
 *   Based on the ODT ParticleRe class.
 *  \tparam GVel1T type for the first  velocity component in the gas phase
 *  \tparam GVel2T type for the second velocity component in the gas phase
 *  \tparam GVel3T type for the third  velocity component in the gas phase
 */
template< typename GVel1T, typename GVel2T, typename GVel3T, typename ScalarT >
class ParticleRe
 : public Expr::Expression<ParticleField>
{
  const Expr::Tag pSizeTag_, gDensityTag_, gViscTag_;
  const Expr::TagList pPosTags_, pVelTags_, gVelTags_;
  const ParticleField   *psize_, *px_, *py_, *pz_, *pu_, *pv_, *pw_ ;
  const GVel1T *gU_;
  const GVel2T *gV_;
  const GVel3T *gW_;
  const ScalarT *gVisc_, *gDensity_;

  typedef typename SpatialOps::Particle::CellToParticle<GVel1T> GVel1OpT;
  typedef typename SpatialOps::Particle::CellToParticle<GVel2T> GVel2OpT;
  typedef typename SpatialOps::Particle::CellToParticle<GVel3T> GVel3OpT;
  GVel1OpT* gv1Op_;
  GVel2OpT* gv2Op_;
  GVel3OpT* gv3Op_;

  typedef typename SpatialOps::Particle::CellToParticle<ScalarT> Scal2POpT;
  Scal2POpT* sOp_;

  ParticleRe( const Expr::Tag& particleSizeTag,
              const Expr::Tag& gasDensityTag,
              const Expr::Tag& gasViscosityTag,
              const Expr::TagList& particlePositionTags,
              const Expr::TagList& particleVelocityTags,
              const Expr::TagList& gasVelocityTags );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \brief Create a ParticleRe::Builder
     *  \param resultTag            The tag for the particle reynolds number
     *  \param particleSizeTag      The particle mass density
     *  \param gasDensityTag        The gas density
     *  \param gasViscosityTag      The gas viscosity
     *  \param particlePositionTags The particle positions - x, y, and z, respectively
     *  \param particleVelocityTags The local particle velocities - up, vp, and wp, respectively
     *  \param gasVelocityTags      The local gas velocities - u, v, and w, respectively
     */
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& particleSizeTag,
             const Expr::Tag& gasDensityTag,
             const Expr::Tag& gasViscosityTag,
             const Expr::TagList& particlePositionTags,
             const Expr::TagList& particleVelocityTags,
             const Expr::TagList& gasVelocityTags );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const Expr::Tag pSizeTag_, gDensityTag_, gViscTag_;
    const Expr::TagList pPosTags_, pVelTags_, gVelTags_;
  };

  ~ParticleRe();

  void advertise_dependents( Expr::ExprDeps& exprDeps);
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();

};

// ###################################################################
  //
  //                          Implementation
  //
  // ###################################################################

template< typename GVel1T, typename GVel2T, typename GVel3T, typename ScalarT >
ParticleRe<GVel1T, GVel2T, GVel3T, ScalarT>::
ParticleRe( const Expr::Tag& particleSizeTag,
            const Expr::Tag& gasDensityTag,
            const Expr::Tag& gasViscosityTag,
            const Expr::TagList& particlePositionTags,
            const Expr::TagList& particleVelocityTags,
            const Expr::TagList& gasVelocityTags )
  : Expr::Expression<ParticleField>(),
    pSizeTag_   ( particleSizeTag      ),
    gDensityTag_( gasDensityTag        ),
    gViscTag_   ( gasViscosityTag      ),
    pPosTags_   ( particlePositionTags ),
    pVelTags_   ( particleVelocityTags ),
    gVelTags_   ( gasVelocityTags      )
{
  this->set_gpu_runnable( false );  // not until we get particle interpolants GPU ready
}

//--------------------------------------------------------------------

template< typename GVel1T, typename GVel2T, typename GVel3T, typename ScalarT >
ParticleRe<GVel1T, GVel2T, GVel3T, ScalarT>::~ParticleRe()
{}

//--------------------------------------------------------------------

template< typename GVel1T, typename GVel2T, typename GVel3T, typename ScalarT >
void
ParticleRe<GVel1T, GVel2T, GVel3T, ScalarT>::advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( pSizeTag_    );
  exprDeps.requires_expression( pPosTags_    );
  exprDeps.requires_expression( pVelTags_    );
  exprDeps.requires_expression( gDensityTag_ );
  exprDeps.requires_expression( gViscTag_    );
  exprDeps.requires_expression( gVelTags_    );
  
}

//--------------------------------------------------------------------
template< typename GVel1T, typename GVel2T, typename GVel3T, typename ScalarT >
void
ParticleRe<GVel1T, GVel2T, GVel3T, ScalarT>::bind_fields( const Expr::FieldManagerList& fml )
{
  using namespace Expr;

  // particle fields
  const typename FieldMgrSelector<ParticleField>::type& fm = fml.field_manager<ParticleField>();
  psize_  = &fm.field_ref( pSizeTag_ );
  px_     = &fm.field_ref( pPosTags_[0]  );
  py_     = &fm.field_ref( pPosTags_[1]  );
  pz_     = &fm.field_ref( pPosTags_[2]  );
  pu_     = &fm.field_ref( pVelTags_[0]  );
  pv_     = &fm.field_ref( pVelTags_[1]  );
  pw_     = &fm.field_ref( pVelTags_[2]  );

  // gas fields
  const typename FieldMgrSelector<ScalarT>::type& scalfm = fml.field_manager<ScalarT>();
  gDensity_ = &scalfm.field_ref( gDensityTag_ );
  gVisc_    = &scalfm.field_ref( gViscTag_    );

  // gas fields for velocity components
  gU_ = &fml.field_ref<GVel1T>( gVelTags_[0] );
  gV_ = &fml.field_ref<GVel2T>( gVelTags_[1] );
  gW_ = &fml.field_ref<GVel3T>( gVelTags_[2] );
}

//--------------------------------------------------------------------

template< typename GVel1T, typename GVel2T, typename GVel3T, typename ScalarT >
void
ParticleRe<GVel1T, GVel2T, GVel3T, ScalarT>::bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  gv1Op_ = opDB.retrieve_operator<GVel1OpT >();
  gv2Op_ = opDB.retrieve_operator<GVel2OpT >();
  gv3Op_ = opDB.retrieve_operator<GVel3OpT >();
  sOp_   = opDB.retrieve_operator<Scal2POpT>();
}

//--------------------------------------------------------------------

template< typename GVel1T, typename GVel2T, typename GVel3T, typename ScalarT >
void
ParticleRe<GVel1T, GVel2T, GVel3T, ScalarT>::evaluate()
{
  using namespace SpatialOps;
  ParticleField& result = this->value();
  
  SpatFldPtr<ParticleField> relativeVelMag = SpatialFieldStore::get<ParticleField>( result );

  //--------------------------------------------------------------
  // interpolate gas velocity components to the particle locations
  {
    SpatFldPtr<ParticleField> tmpu = SpatialFieldStore::get<ParticleField>( result );
    SpatFldPtr<ParticleField> tmpv = SpatialFieldStore::get<ParticleField>( result );
    SpatFldPtr<ParticleField> tmpw = SpatialFieldStore::get<ParticleField>( result );
    gv1Op_->set_coordinate_information(px_,py_,pz_,psize_);
    gv1Op_->apply_to_field(*gU_, *tmpu);
    *tmpu <<= *pu_ - *tmpu;

    gv2Op_->set_coordinate_information(px_,py_,pz_,psize_);
    gv2Op_->apply_to_field(*gV_, *tmpv);
    *tmpv <<= *pv_ - *tmpv;

    gv3Op_->set_coordinate_information(px_,py_,pz_,psize_);
    gv3Op_->apply_to_field(*gW_, *tmpw);
    *tmpw <<= *pw_ - *tmpw;

    *relativeVelMag <<= sqrt(*tmpu * *tmpu + *tmpv * *tmpv + *tmpw * *tmpw);
  }
  //--------------------------------------------------------------
  
  
  SpatFldPtr<ParticleField> tmpvisc = SpatialFieldStore::get<ParticleField>( result );
  SpatFldPtr<ParticleField> tmpden  = SpatialFieldStore::get<ParticleField>( result );
  sOp_->set_coordinate_information( px_, py_, pz_, psize_ );
  sOp_->apply_to_field( *gVisc_,    *tmpvisc );
  sOp_->apply_to_field( *gDensity_, *tmpden  );

  result <<= *tmpden * *psize_ * *relativeVelMag / *tmpvisc;
}

//--------------------------------------------------------------------

template< typename GVel1T, typename GVel2T, typename GVel3T, typename ScalarT >
ParticleRe<GVel1T, GVel2T, GVel3T, ScalarT>::Builder::
Builder( const Expr::Tag& resultTag,
         const Expr::Tag& particleSizeTag,
         const Expr::Tag& gasDensityTag,
         const Expr::Tag& gasViscosityTag,
         const Expr::TagList& particlePositionTags,
         const Expr::TagList& particleVelocityTags,
         const Expr::TagList& gasVelocityTags )
  : ExpressionBuilder( resultTag ),
    pSizeTag_   ( particleSizeTag      ),
    gDensityTag_( gasDensityTag        ),
    gViscTag_   ( gasViscosityTag      ),
    pPosTags_   ( particlePositionTags ),
    pVelTags_   ( particleVelocityTags ),
    gVelTags_   ( gasVelocityTags      )
{}

//--------------------------------------------------------------------
template< typename GVel1T, typename GVel2T, typename GVel3T, typename ScalarT >
Expr::ExpressionBase*
ParticleRe<GVel1T, GVel2T, GVel3T, ScalarT>::Builder::build() const
{
  return new ParticleRe( pSizeTag_,  gDensityTag_, gViscTag_, pPosTags_, pVelTags_, gVelTags_);
}

#endif // ParticleRe_Expr_h
