#ifndef ParticleGasMomentumSrc_h
#define ParticleGasMomentumSrc_h

#include <expression/Expression.h>
#include <spatialops/particles/ParticleFieldTypes.h>
#include <spatialops/particles/ParticleOperators.h>
//====================================================================


/**
 *  @class ParticleGasMomentumSrc
 *  @author Tony Saad
 *  @date   July 2014
 *
 *  @brief Calculates the gas-phase momentum source term caused by the presence of particles
 */
template< typename GasVelT >
class ParticleGasMomentumSrc
: public Expr::Expression<GasVelT>
{
  
public:
  
  class Builder : public Expr::ExpressionBuilder
  {
    const Expr::Tag     pDragTag_, pMassTag_, pSizeTag_;
    const Expr::TagList pPosTags_;
    
  public:
    
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& particleDragTag,
             const Expr::Tag& particleMassTag,
             const Expr::Tag& particleSizeTag,
             const Expr::TagList& particlePositionTags )
    : ExpressionBuilder(resultTag),
      pDragTag_( particleDragTag ),
      pMassTag_(particleMassTag),
      pSizeTag_(particleSizeTag),
      pPosTags_(particlePositionTags)
    {}

    ~Builder(){}
    Expr::ExpressionBase* build() const{
      return new ParticleGasMomentumSrc<GasVelT> (pDragTag_, pMassTag_, pSizeTag_, pPosTags_ );
    }

  };
  
  ~ParticleGasMomentumSrc(){}

  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
  
private:
  
  ParticleGasMomentumSrc( const Expr::Tag& particleDragTag,
                          const Expr::Tag& particleMassTag,
                          const Expr::Tag& particleSizeTag,
                          const Expr::TagList& particlePositionTags);
  
  DECLARE_FIELDS(ParticleField, px_, py_, pz_, pDrag_, pSize_, pMass_)
  double vol_; // cell volume
  
  typedef typename SpatialOps::Particle::ParticleToCell<GasVelT> P2GVelT;
  P2GVelT* p2gvOp_; // particle to gas velocity operator
  
  WasatchCore::UintahPatchContainer* patchContainer_;
};

// ###################################################################
//
//                           Implementation
//
// ###################################################################


//--------------------------------------------------------------------

template< typename GasVelT >
ParticleGasMomentumSrc<GasVelT>::
ParticleGasMomentumSrc( const Expr::Tag& particleDragTag,
                       const Expr::Tag& particleMassTag,
                       const Expr::Tag& particleSizeTag,
                       const Expr::TagList& particlePositionTags)
: Expr::Expression<GasVelT>()
{
  this->set_gpu_runnable(false);  // waiting for GPU-enabled particle interpolants

  pDrag_ = this->template create_field_request<ParticleField>(particleDragTag);
  pSize_ = this->template create_field_request<ParticleField>(particleSizeTag);
  pMass_ = this->template create_field_request<ParticleField>(particleMassTag);
  px_    = this->template create_field_request<ParticleField>(particlePositionTags[0]);
  py_    = this->template create_field_request<ParticleField>(particlePositionTags[1]);
  pz_    = this->template create_field_request<ParticleField>(particlePositionTags[2]);
}

//--------------------------------------------------------------------

template< typename GasVelT >
void
ParticleGasMomentumSrc<GasVelT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  p2gvOp_          = opDB.retrieve_operator<P2GVelT>();
  patchContainer_  = opDB.retrieve_operator<WasatchCore::UintahPatchContainer>();
  vol_ = patchContainer_->get_uintah_patch()->cellVolume();
}

//--------------------------------------------------------------------

template< typename GasVelT >
void
ParticleGasMomentumSrc<GasVelT>::
evaluate()
{
  GasVelT& result = this->value();
  
  using namespace SpatialOps;
  
  const ParticleField& psize = pSize_->field_ref();
  const ParticleField& pdrag = pDrag_->field_ref();
  const ParticleField& pmass = pMass_->field_ref();
  const ParticleField& px = px_->field_ref();
  const ParticleField& py = py_->field_ref();
  const ParticleField& pz = pz_->field_ref();

    SpatFldPtr<GasVelT     >  gasmomsrc = SpatialFieldStore::get<GasVelT     >( result );
  SpatFldPtr<ParticleField> pforcetmp = SpatialFieldStore::get<ParticleField>( px );
  
  *pforcetmp <<= pdrag * pmass; // multiply by mass
  
  p2gvOp_->set_coordinate_information(&px,&py,&pz,&psize);
  p2gvOp_->apply_to_field( *pforcetmp, *gasmomsrc );
  
  result <<= - *gasmomsrc / vol_;
}

//--------------------------------------------------------------------

#endif // ParticleGasMomentumSrc_h
