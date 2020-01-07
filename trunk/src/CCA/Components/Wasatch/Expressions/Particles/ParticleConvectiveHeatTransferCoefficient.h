#ifndef ParticleConvectiveHeatTransferCoefficient_Expr_h
#define ParticleConvectiveHeatTransferCoefficient_Expr_h

#include <expression/Expression.h>
#include <spatialops/structured/SpatialFieldStore.h>
#include <spatialops/particles/ParticleOperators.h>
#include <spatialops/OperatorDatabase.h>

  /**
   *  \class SetupCoalModels
   *  \author Babak Goshayeshi, Josh McConnell
   *  \ingroup WasatchParticles
   *  \date August 2016
   *
   *  \brief Evaluates the convective heat transfer coefficient for spherical particles.
   */

  template< typename FieldT >
  class ParticleConvectiveHeatTransferCoefficient
  : public Expr::Expression<ParticleField>
  {

    DECLARE_FIELDS( ParticleField, px_, py_, pz_, pRe_, psize_)
    DECLARE_FIELDS( FieldT, gcp_, gmu_, glambda_, gtemp_ )

    typedef typename SpatialOps::Particle::CellToParticle<FieldT> C2POpT;

    C2POpT* c2pOp_;

    ParticleConvectiveHeatTransferCoefficient( const Expr::TagList& pPosTags,
                                               const Expr::Tag&     pReTag,
                                               const Expr::Tag&     psizeTag,
                                               const Expr::Tag&     gcpTag,
                                               const Expr::Tag&     gmuTag,
                                               const Expr::Tag&     glambdaTag );

  public:
    class Builder : public Expr::ExpressionBuilder
    {
    public:
      Builder( const Expr::Tag&     pConvCoefTag,
               const Expr::TagList& pPosTags,
               const Expr::Tag&     pReTag,
               const Expr::Tag&     psizeTag,
               const Expr::Tag&     gcpTag,
               const Expr::Tag&     gmuTag,
               const Expr::Tag&     glambdaTag );
      ~Builder(){}
      Expr::ExpressionBase* build() const;
    private:
      const Expr::Tag     pReTag_,psizeTag_,gcpTag_,gmuTag_,glambdaTag_;
      const Expr::TagList pPosTags_;
    };

    ParticleConvectiveHeatTransferCoefficient(){}
    void bind_operators( const SpatialOps::OperatorDatabase& opDB );
    void evaluate();

  };

  //---------------------------------------------------------------------
  //
  //                   Implementation
  //
  //---------------------------------------------------------------------
  template<typename FieldT>
  ParticleConvectiveHeatTransferCoefficient<FieldT>::
  ParticleConvectiveHeatTransferCoefficient( const Expr::TagList& pPosTags,
                                             const Expr::Tag&     pReTag,
                                             const Expr::Tag&     psizeTag,
                                             const Expr::Tag&     gcpTag,
                                             const Expr::Tag&     gmuTag,
                                             const Expr::Tag&     glambdaTag )
    : Expr::Expression<ParticleField>()
  {
    this->set_gpu_runnable(false);  // waiting for GPU-enabled particle interpolants

    px_    = this->template create_field_request<ParticleField>( pPosTags[0] );
    py_    = this->template create_field_request<ParticleField>( pPosTags[1] );
    pz_    = this->template create_field_request<ParticleField>( pPosTags[2] );
    pRe_   = this->template create_field_request<ParticleField>( pReTag      );
    psize_ = this->template create_field_request<ParticleField>( psizeTag    );

    gcp_     = this->template create_field_request<FieldT>( gcpTag     );
    gmu_     = this->template create_field_request<FieldT>( gmuTag     );
    glambda_ = this->template create_field_request<FieldT>( glambdaTag );
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  void
  ParticleConvectiveHeatTransferCoefficient<FieldT>::
  bind_operators( const SpatialOps::OperatorDatabase& opDB )
  {
    c2pOp_ = opDB.retrieve_operator<C2POpT>();
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  void
  ParticleConvectiveHeatTransferCoefficient<FieldT>::
  evaluate()
  {
    using namespace SpatialOps;

    ParticleField& coef_h = this->value();

    const ParticleField& px    = px_   ->field_ref();
    const ParticleField& py    = py_   ->field_ref();
    const ParticleField& pz    = pz_   ->field_ref();
    const ParticleField& pRe   = pRe_  ->field_ref();
    const ParticleField& psize = psize_->field_ref();

    const FieldT& gcp     = gcp_    ->field_ref();
    const FieldT& gmu     = gmu_    ->field_ref();
    const FieldT& glambda = glambda_->field_ref();

    //calculating gas Prandtl number
    SpatialOps::SpatFldPtr<FieldT> gPr = SpatialOps::SpatialFieldStore::get<FieldT>( gcp );
    *gPr <<= gcp * gmu / glambda;

    //interpolate the gas Prandtl number to particle locations.....
    SpatialOps::SpatFldPtr<ParticleField> pPr = SpatialOps::SpatialFieldStore::get<ParticleField>( coef_h );
    c2pOp_->set_coordinate_information( &px, &py, &pz, &psize );
    c2pOp_->apply_to_field( *gPr, *pPr );

    //calculating gas Nusselt number from Re_p and interpolated Prandtl number
    SpatialOps::SpatFldPtr<ParticleField> Nu = SpatialOps::SpatialFieldStore::get<ParticleField>( coef_h );
    *Nu <<= 2.0 + 0.6 * pow(pRe, 0.5) * pow(*pPr, (1.0/3.0));

    //interpolate gas phase thermal conductivity to particle locations
    SpatialOps::SpatFldPtr<ParticleField> plambda = SpatialOps::SpatialFieldStore::get<ParticleField>( coef_h );
    c2pOp_->apply_to_field( glambda, *plambda );

    //calculate the convective heat transfer coefficient
    coef_h <<= *Nu * *plambda / psize ;
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  ParticleConvectiveHeatTransferCoefficient<FieldT>::
  Builder::Builder( const Expr::Tag&     pConvCoefTag,
                    const Expr::TagList& pPosTags,
                    const Expr::Tag&     pReTag,
                    const Expr::Tag&     psizeTag,
                    const Expr::Tag&     gcpTag,
                    const Expr::Tag&     gmuTag,
                    const Expr::Tag&     glambdaTag)
  : ExpressionBuilder( pConvCoefTag ),
    pReTag_    ( pReTag    ),
    psizeTag_  ( psizeTag  ),
    gcpTag_    ( gcpTag    ),
    gmuTag_    ( gmuTag    ),
    glambdaTag_( glambdaTag),
    pPosTags_  ( pPosTags  )
  {}

  //------------------------------------------------------------------

  template<typename FieldT>
  Expr::ExpressionBase*
  ParticleConvectiveHeatTransferCoefficient<FieldT>::Builder::build() const
  {
    return new ParticleConvectiveHeatTransferCoefficient<FieldT>( pPosTags_,pReTag_,psizeTag_,gcpTag_,gmuTag_,glambdaTag_ );
  }

#endif // ParticleConvectiveHeatTransferCoefficient_Expr_h
