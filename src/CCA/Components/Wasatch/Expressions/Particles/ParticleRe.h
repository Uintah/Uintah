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
  DECLARE_FIELDS( ParticleField, psize_, px_, py_, pz_, pu_, pv_, pw_ )
  DECLARE_FIELDS( ScalarT, gVisc_, gDensity_ )
  DECLARE_FIELD( GVel1T, gu_ )
  DECLARE_FIELD( GVel2T, gv_ )
  DECLARE_FIELD( GVel3T, gw_ )

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
             const Expr::TagList& gasVelocityTags )
      : ExpressionBuilder( resultTag ),
        pSizeTag_   ( particleSizeTag      ),
        gDensityTag_( gasDensityTag        ),
        gViscTag_   ( gasViscosityTag      ),
        pPosTags_   ( particlePositionTags ),
        pVelTags_   ( particleVelocityTags ),
        gVelTags_   ( gasVelocityTags      )
    {}

    ~Builder(){}
    Expr::ExpressionBase* build() const{
      return new ParticleRe( pSizeTag_,  gDensityTag_, gViscTag_, pPosTags_, pVelTags_, gVelTags_);
    }
  private:
    const Expr::Tag pSizeTag_, gDensityTag_, gViscTag_;
    const Expr::TagList pPosTags_, pVelTags_, gVelTags_;
  };

  ~ParticleRe(){}
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
  : Expr::Expression<ParticleField>()
{
  this->set_gpu_runnable(false);  // waiting for GPU-enabled particle interpolants
  
  gDensity_ = this->template create_field_request<ScalarT>(gasDensityTag);
  psize_    = this->template create_field_request<ParticleField>(particleSizeTag);
  gVisc_    = this->template create_field_request<ScalarT>(gasViscosityTag);

  px_ = this->template create_field_request<ParticleField>(particlePositionTags[0]);
  py_ = this->template create_field_request<ParticleField>(particlePositionTags[1]);
  pz_ = this->template create_field_request<ParticleField>(particlePositionTags[2]);

  pu_ = this->template create_field_request<ParticleField>(particleVelocityTags[0]);
  pv_ = this->template create_field_request<ParticleField>(particleVelocityTags[1]);
  pw_ = this->template create_field_request<ParticleField>(particleVelocityTags[2]);

  gu_ = this->template create_field_request<GVel1T>(gasVelocityTags[0]);
  gv_ = this->template create_field_request<GVel2T>(gasVelocityTags[1]);
  gw_ = this->template create_field_request<GVel3T>(gasVelocityTags[2]);
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
  
  const ParticleField& psize = psize_->field_ref();
  
  const ParticleField& px = px_->field_ref();
  const ParticleField& py = py_->field_ref();
  const ParticleField& pz = pz_->field_ref();
  
  const ParticleField& pu = pu_->field_ref();
  const ParticleField& pv = pv_->field_ref();
  const ParticleField& pw = pw_->field_ref();

  const GVel1T& gu = gu_->field_ref();
  const GVel2T& gv = gv_->field_ref();
  const GVel3T& gw = gw_->field_ref();

  const ScalarT& grho = gDensity_->field_ref();
  const ScalarT& gmu  = gVisc_->field_ref();
  
  SpatFldPtr<ParticleField> relativeVelMag = SpatialFieldStore::get<ParticleField>( result );

  //--------------------------------------------------------------
  // interpolate gas velocity components to the particle locations
  {
    SpatFldPtr<ParticleField> tmpu = SpatialFieldStore::get<ParticleField>( result );
    SpatFldPtr<ParticleField> tmpv = SpatialFieldStore::get<ParticleField>( result );
    SpatFldPtr<ParticleField> tmpw = SpatialFieldStore::get<ParticleField>( result );
    
    gv1Op_->set_coordinate_information(&px,&py,&pz,&psize);
    gv1Op_->apply_to_field(gu, *tmpu);
    *tmpu <<= pu - *tmpu;

    gv2Op_->set_coordinate_information(&px,&py,&pz,&psize);
    gv2Op_->apply_to_field(gv, *tmpv);
    *tmpv <<= pv - *tmpv;

    gv3Op_->set_coordinate_information(&px,&py,&pz,&psize);
    gv3Op_->apply_to_field(gw, *tmpw);
    *tmpw <<= pw - *tmpw;

    *relativeVelMag <<= sqrt(*tmpu * *tmpu + *tmpv * *tmpv + *tmpw * *tmpw);
  }
  //--------------------------------------------------------------
  
  
  SpatFldPtr<ParticleField> tmpvisc = SpatialFieldStore::get<ParticleField>( result );
  SpatFldPtr<ParticleField> tmpden  = SpatialFieldStore::get<ParticleField>( result );
  sOp_->set_coordinate_information(&px,&py,&pz,&psize);
  sOp_->apply_to_field( gmu,    *tmpvisc );
  sOp_->apply_to_field( grho, *tmpden  );

  result <<= *tmpden * psize * *relativeVelMag / *tmpvisc;
}

//--------------------------------------------------------------------

#endif // ParticleRe_Expr_h
