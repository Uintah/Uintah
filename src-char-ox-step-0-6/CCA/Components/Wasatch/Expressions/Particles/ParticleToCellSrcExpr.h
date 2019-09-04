#ifndef ParticleToCellSrcExpr_h
#define ParticleToCellSrcExpr_h

#include <expression/Expression.h>
#include <spatialops/particles/ParticleFieldTypes.h>
#include <spatialops/particles/ParticleOperators.h>
//====================================================================


/**
 *  @class ParticleToCellSrcExpr
 *  @author Josh McConnell
 *  @date   August 2016
 *
 *  @brief Accumulates extensive source terms from particles and calculates an
 *         extensive source term for a gas phase property.
 */
template< typename FieldT >
class ParticleToCellSrcExpr
: public Expr::Expression<FieldT>
{

public:

  class Builder : public Expr::ExpressionBuilder
  {
    const Expr::Tag     pSizeTag_;
    const Expr::TagList pPosTags_, pSrcTags_;

  public:

    Builder( const Expr::Tag&     resultTag,
             const Expr::TagList& particleSrcTags,
             const Expr::Tag&     particleSizeTag,
             const Expr::TagList& particlePositionTags)
    : ExpressionBuilder( resultTag ),
      pSizeTag_( particleSizeTag      ),
      pPosTags_( particlePositionTags ),
      pSrcTags_( particleSrcTags      )
    {}

    Builder( const Expr::Tag&     resultTag,
             const Expr::Tag&     particleSrcTag,
             const Expr::Tag&     particleSizeTag,
             const Expr::TagList& particlePositionTags)
    : ExpressionBuilder( resultTag ),
      pSizeTag_( particleSizeTag      ),
      pPosTags_( particlePositionTags ),
      pSrcTags_( Expr::tag_list(particleSrcTag) )
    {}

    ~Builder(){}
    Expr::ExpressionBase* build() const{
      return new ParticleToCellSrcExpr<FieldT> ( pSrcTags_, pSizeTag_, pPosTags_ );
    }

  };

  ~ParticleToCellSrcExpr(){}

  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();

private:

  ParticleToCellSrcExpr( const Expr::TagList& particleSrcTags,
                         const Expr::Tag&     particleSizeTag,
                         const Expr::TagList& particlePositionTags );

  DECLARE_FIELDS(ParticleField, px_, py_, pz_, pSize_)
  DECLARE_VECTOR_OF_FIELDS( ParticleField, pSrcs_ )
  double vol_; // cell volume

  typedef typename SpatialOps::Particle::ParticleToCell<FieldT> P2GVelT;
  P2GVelT* p2gvOp_; // particle to gas velocity operator

  WasatchCore::UintahPatchContainer* patchContainer_;
};

// ###################################################################
//
//                           Implementation
//
// ###################################################################


//--------------------------------------------------------------------

template< typename FieldT >
ParticleToCellSrcExpr<FieldT>::
ParticleToCellSrcExpr( const Expr::TagList& particleSrcTags,
                       const Expr::Tag&     particleSizeTag,
                       const Expr::TagList& particlePositionTags )
: Expr::Expression<FieldT>()
{
  this->set_gpu_runnable(false);  // waiting for GPU-enabled particle interpolants

  pSize_ = this->template create_field_request<ParticleField>(particleSizeTag);
  px_    = this->template create_field_request<ParticleField>(particlePositionTags[0]);
  py_    = this->template create_field_request<ParticleField>(particlePositionTags[1]);
  pz_    = this->template create_field_request<ParticleField>(particlePositionTags[2]);

  this->template create_field_vector_request<ParticleField>( particleSrcTags, pSrcs_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
ParticleToCellSrcExpr<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  p2gvOp_          = opDB.retrieve_operator<P2GVelT>();
  patchContainer_  = opDB.retrieve_operator<WasatchCore::UintahPatchContainer>();
  vol_             = patchContainer_->get_uintah_patch()->cellVolume();
}

//--------------------------------------------------------------------

template< typename FieldT >
void
ParticleToCellSrcExpr<FieldT>::
evaluate()
{
  FieldT& result = this->value();

  using namespace SpatialOps;

  const ParticleField& psize = pSize_->field_ref();
  const ParticleField& px = px_->field_ref();
  const ParticleField& py = py_->field_ref();
  const ParticleField& pz = pz_->field_ref();

  SpatFldPtr<FieldT       > p2cExtensiveSrc        = SpatialFieldStore::get<FieldT       >( result );
  SpatFldPtr<ParticleField> cumulativeExtensiveSrc = SpatialFieldStore::get<ParticleField>( px     );
  *cumulativeExtensiveSrc <<= 0.0 ;

  // Accumulate extensive source terms
  for( size_t i=0; i<pSrcs_.size(); ++i ){
    *cumulativeExtensiveSrc <<= *cumulativeExtensiveSrc + pSrcs_[i]->field_ref();
  }

  p2gvOp_->set_coordinate_information(&px,&py,&pz,&psize);
  p2gvOp_->apply_to_field( *cumulativeExtensiveSrc, *p2cExtensiveSrc );

  //calculate intensive source
  result <<= - *p2cExtensiveSrc / vol_;
}

//--------------------------------------------------------------------

#endif // ParticleToCellSrcExpr_h
