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
             const Expr::TagList& particlePositionTags );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  };

  ~ParticleGasMomentumSrc();

  void evaluate();
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );

private:

  ParticleGasMomentumSrc( const Expr::Tag& particleDragTag,
                        const Expr::Tag& particleMassTag,
                        const Expr::Tag& particleSizeTag,
                        const Expr::TagList& particlePositionTags);

  const Expr::Tag     pDragTag_, pMassTag_, pSizeTag_;
  const Expr::TagList pPosTags_;
  double vol_; // cell volume
  
  typedef typename SpatialOps::Particle::ParticleToCell<GasVelT> P2GVelT;
  P2GVelT* p2gvOp_; // particle to gas velocity operator

  Wasatch::UintahPatchContainer* patchContainer_;
  
  const ParticleField *px_, *py_, *pz_, *pDrag_, *pSize_, *pMass_;

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
  : Expr::Expression<GasVelT>(),
    pDragTag_( particleDragTag ),
    pMassTag_  ( particleMassTag        ),
    pSizeTag_  ( particleSizeTag        ),
    pPosTags_  ( particlePositionTags   )
{}

 //------------------------------------------------------------------

  template<typename GasVelT>
  ParticleGasMomentumSrc<GasVelT>::
  ~ParticleGasMomentumSrc()
  {}
//--------------------------------------------------------------------

template< typename GasVelT >
void
ParticleGasMomentumSrc<GasVelT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( pDragTag_     );
  exprDeps.requires_expression( pMassTag_     );
  exprDeps.requires_expression( pSizeTag_     );
  exprDeps.requires_expression( pPosTags_     );
}

//--------------------------------------------------------------------

template< typename GasVelT >
void
ParticleGasMomentumSrc<GasVelT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<ParticleField>::type& pfm = fml.template field_manager<ParticleField>();

  pDrag_ = &pfm.field_ref( pDragTag_ );
  pMass_ =  &pfm.field_ref( pMassTag_ );
  pSize_  = &pfm.field_ref( pSizeTag_  );
  px_  = &pfm.field_ref( pPosTags_[0]  );
  py_  = &pfm.field_ref( pPosTags_[1]  );
  pz_  = &pfm.field_ref( pPosTags_[2]  );
}

//--------------------------------------------------------------------

template< typename GasVelT >
void
ParticleGasMomentumSrc<GasVelT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  p2gvOp_ = opDB.retrieve_operator<P2GVelT>();
  patchContainer_  = opDB.retrieve_operator<Wasatch::UintahPatchContainer>();
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

  SpatFldPtr<GasVelT     >  gasmomsrc = SpatialFieldStore::get<GasVelT     >( result );
  SpatFldPtr<ParticleField> pforcetmp    = SpatialFieldStore::get<ParticleField>( *px_ );

  *pforcetmp <<= *pDrag_ * *pMass_; // multiply by mass

  p2gvOp_->set_coordinate_information( px_,py_,pz_, pSize_ );
  p2gvOp_->apply_to_field( *pforcetmp, *gasmomsrc );

  result <<= - *gasmomsrc / vol_;
}

//--------------------------------------------------------------------

template< typename GasVelT >
ParticleGasMomentumSrc<GasVelT>::Builder::
Builder( const Expr::Tag& resultTag,
        const Expr::Tag& particleDragTag,
        const Expr::Tag& particleMassTag,
        const Expr::Tag& particleSizeTag,
        const Expr::TagList& particlePositionTags)
  : ExpressionBuilder(resultTag),
    pDragTag_( particleDragTag ),
    pMassTag_(particleMassTag),
    pSizeTag_(particleSizeTag),
    pPosTags_(particlePositionTags)
{}

//--------------------------------------------------------------------

template< typename GasVelT >
Expr::ExpressionBase*
ParticleGasMomentumSrc<GasVelT>::Builder::build() const
{
  return new ParticleGasMomentumSrc<GasVelT> (pDragTag_, pMassTag_, pSizeTag_, pPosTags_ );
}


#endif // ParticleGasMomentumSrc_h
