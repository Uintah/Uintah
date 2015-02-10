#include <CCA/Components/Wasatch/Expressions/DivmomStar.h>
#include <CCA/Components/Wasatch/TagNames.h>

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>


DivmomStar::DivmomStar( const Expr::TagList& velStarTags,
                            const Expr::Tag densStarTag )
: Expr::Expression<SVolField>(),
  doX_      ( velStarTags[0]!=Expr::Tag() ),
  doY_      ( velStarTags[1]!=Expr::Tag() ),
  doZ_      ( velStarTags[2]!=Expr::Tag() ),
  is3d_     ( doX_ && doY_ && doZ_    )
{
  set_gpu_runnable( true );
  
  if(doX_) create_field_request(velStarTags[0], uStar_);
  if(doY_) create_field_request(velStarTags[1], vStar_);
  if(doZ_) create_field_request(velStarTags[2], wStar_);
  create_field_request(densStarTag, densStar_);
}

//------------------------------------------------------------------

DivmomStar::~DivmomStar()
{}

//------------------------------------------------------------------

void DivmomStar::bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
    if( doX_ ){
      gradXOp_       = opDB.retrieve_operator<GradXT>();
      s2XInterpOp_ = opDB.retrieve_operator<S2XInterpOpT>();
    }
    if( doY_ ){
      gradYOp_       = opDB.retrieve_operator<GradYT>();
      s2YInterpOp_ = opDB.retrieve_operator<S2YInterpOpT>();
    }
    if( doZ_ ){
      gradZOp_       = opDB.retrieve_operator<GradZT>();
      s2ZInterpOp_ = opDB.retrieve_operator<S2ZInterpOpT>();
  }
}

//------------------------------------------------------------------

void DivmomStar::evaluate()
{
  using namespace SpatialOps;
  SVolField& divmomstar = this->value();
  
  const SVolField& rho = densStar_->field_ref();
  if (is3d_) {
    const XVolField& u = uStar_->field_ref();
    const YVolField& v = vStar_->field_ref();
    const ZVolField& w = wStar_->field_ref();
    divmomstar <<=   (*gradXOp_) ( (*s2XInterpOp_)(rho) * (u) )
                   + (*gradYOp_) ( (*s2YInterpOp_)(rho) * (v) )
                   + (*gradZOp_) ( (*s2ZInterpOp_)(rho) * (w) );
  } else {
    if(doX_) divmomstar <<=              (*gradXOp_) ( (*s2XInterpOp_)(rho) * (uStar_->field_ref()) );
    else     divmomstar <<= 0.0;
    if(doY_) divmomstar <<= divmomstar + (*gradYOp_) ( (*s2YInterpOp_)(rho) * (vStar_->field_ref()) );
    if(doZ_) divmomstar <<= divmomstar + (*gradZOp_) ( (*s2ZInterpOp_)(rho) * (wStar_->field_ref()) );
  }
}

//------------------------------------------------------------------

DivmomStar::Builder::Builder( const Expr::Tag& result,
                              const Expr::TagList& velStarTags,
                              const Expr::Tag densStarTag )
: ExpressionBuilder(result),
  velStarTs_  ( densStarTag==Expr::Tag() ? Expr::TagList() : velStarTags ),
  densStart_ ( densStarTag  )
{}

//------------------------------------------------------------------

Expr::ExpressionBase*
DivmomStar::Builder::build() const
{
  return new DivmomStar( velStarTs_, densStart_ );
}
//------------------------------------------------------------------

