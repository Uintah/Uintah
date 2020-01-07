#ifndef CellToParticleInterpExpr_h
#define CellToParticleInterpExpr_h

#include <expression/Expression.h>
#include <spatialops/particles/ParticleFieldTypes.h>
#include <spatialops/particles/ParticleOperators.h>
#include <spatialops/OperatorDatabase.h>

//==================================================================

/**
 *  @class CelltoParticleInterpExpr
 *  @author Josh McConnell
 *  @date   August, 2016
 *
 *  @brief Interpolates gas field data to particle locations.
 */
template< typename FieldT >
class CellToParticleInterpExpr : public Expr::Expression<ParticleField>
{

  DECLARE_FIELDS(ParticleField, px_, py_, pz_, psize_)
  DECLARE_FIELD (FieldT, gasField_)

  typedef typename SpatialOps::Particle::CellToParticle<FieldT> C2POpT;
  C2POpT* c2pOp_;

  CellToParticleInterpExpr( const Expr::Tag&     gFieldTag,
                            const Expr::Tag&     particleSizeTag,
                            const Expr::TagList& particlePositionTags );

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  \param resultTag The gas property interpolated to particle locations
     *  \param gasTag A gas-phase property
     *  \param particleSizeTag The particle size
     *  \param particlePositionTags Particle positions: x, y, and z - respectively
     */
    Builder( const Expr::Tag&     resultTag,
             const Expr::Tag&     gFieldTag,
             const Expr::Tag&     particleSizeTag,
             const Expr::TagList& particlePositionTags )
      : ExpressionBuilder(resultTag),
        gFieldTag_    ( gFieldTag            ),
        pSizeTag_     ( particleSizeTag      ),
        pPosTags_     ( particlePositionTags )
    {}

    ~Builder(){}

    Expr::ExpressionBase* build() const{
      return new CellToParticleInterpExpr<FieldT>( gFieldTag_, pSizeTag_, pPosTags_);
    }

  private:
    const Expr::Tag gFieldTag_, pSizeTag_;
    const Expr::TagList pPosTags_;
  };

  ~CellToParticleInterpExpr(){}

  void bind_operators( const SpatialOps::OperatorDatabase& opDB ){
    c2pOp_ = opDB.retrieve_operator<C2POpT>();
  }

  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################



template<typename FieldT>
CellToParticleInterpExpr<FieldT>::
CellToParticleInterpExpr( const Expr::Tag& gFieldTag,
                          const Expr::Tag& particleSizeTag,
                          const Expr::TagList& particlePositionTags )
: Expr::Expression<ParticleField>()
{
  this->set_gpu_runnable(false);  // waiting for GPU-enabled particle interpolants

  px_    = this->template create_field_request<ParticleField>(particlePositionTags[0]);
  py_    = this->template create_field_request<ParticleField>(particlePositionTags[1]);
  pz_    = this->template create_field_request<ParticleField>(particlePositionTags[2]);
  psize_ = this->template create_field_request<ParticleField>(particleSizeTag        );

  gasField_ = this->template create_field_request<FieldT>(gFieldTag);
}

//------------------------------------------------------------------

template<typename FieldT>
void
CellToParticleInterpExpr<FieldT>::
evaluate()
{
  ParticleField& result = this->value();

  using namespace SpatialOps;

  const ParticleField& px      = px_      ->field_ref();
  const ParticleField& py      = py_      ->field_ref();
  const ParticleField& pz      = pz_      ->field_ref();
  const ParticleField& psize   = psize_   ->field_ref();
  const FieldT       & gField  = gasField_->field_ref();

  c2pOp_->set_coordinate_information(&px,&py,&pz,&psize);
  c2pOp_->apply_to_field( gField, result );
}

//------------------------------------------------------------------

#endif // CellToParticleInterpExpr_h
