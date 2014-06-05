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
  is3d_     ( doX_ && doY_ && doZ_    ),
  xVelStart_  ( densStarTag==Expr::Tag() ? Expr::Tag() : velStarTags[0] ),
  yVelStart_  ( densStarTag==Expr::Tag() ? Expr::Tag() : velStarTags[1] ),
  zVelStart_  ( densStarTag==Expr::Tag() ? Expr::Tag() : velStarTags[2] ),
  densStart_  ( densStarTag==Expr::Tag() ? Expr::Tag() : densStarTag    )
{
  set_gpu_runnable( true );
}

//------------------------------------------------------------------

DivmomStar::~DivmomStar()
{}

//------------------------------------------------------------------

void DivmomStar::advertise_dependents( Expr::ExprDeps& exprDeps )
{  
    if( doX_ )
    {
      exprDeps.requires_expression( xVelStart_ );
    }
    if( doY_ )
    {
      exprDeps.requires_expression( yVelStart_ );
    }  
    if( doZ_ )
    {
      exprDeps.requires_expression( zVelStart_ );
    }  
    
    exprDeps.requires_expression( densStart_ );
  
}

//------------------------------------------------------------------

void DivmomStar::bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldMgrSelector<SVolField>::type& scalarFM = fml.field_manager<SVolField>();
  
    const Expr::FieldMgrSelector<XVolField>::type& xVolFM  = fml.field_manager<XVolField>();
    const Expr::FieldMgrSelector<YVolField>::type& yVolFM  = fml.field_manager<YVolField>();
    const Expr::FieldMgrSelector<ZVolField>::type& zVolFM  = fml.field_manager<ZVolField>(); 
    
    if( doX_ ){
      uStar_  = &xVolFM.field_ref( xVelStart_ );
    }
    if( doY_ ){
      vStar_ = &yVolFM.field_ref( yVelStart_ );
    }
    if( doZ_ ){
      wStar_ = &zVolFM.field_ref( zVelStart_ );
    }
    
    densStar_  = &scalarFM.field_ref( densStart_ );
}

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
  
  
    if (is3d_) {
      divmomstar <<=   (*gradXOp_) ( (*s2XInterpOp_)(*densStar_) * (*uStar_) )
                     + (*gradYOp_) ( (*s2YInterpOp_)(*densStar_) * (*vStar_) )
                     + (*gradZOp_) ( (*s2ZInterpOp_)(*densStar_) * (*wStar_) );
    } else {
      if(doX_) divmomstar <<=              (*gradXOp_) ( (*s2XInterpOp_)(*densStar_) * (*uStar_) );
      else     divmomstar <<= 0.0;
      if(doY_) divmomstar <<= divmomstar + (*gradYOp_) ( (*s2YInterpOp_)(*densStar_) * (*vStar_) );
      if(doZ_) divmomstar <<= divmomstar + (*gradZOp_) ( (*s2ZInterpOp_)(*densStar_) * (*wStar_) );
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

