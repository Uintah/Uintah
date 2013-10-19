#include <CCA/Components/Wasatch/Expressions/Drhodt.h>

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

Drhodt::Drhodt( const Expr::TagList& velStarTags,
                      const Expr::Tag densTag,
                      const Expr::Tag densStarTag,
                      const Expr::Tag dens2StarTag,
                      const Expr::Tag timestepTag )
: Expr::Expression<SVolField>(),
  doX_      ( velStarTags[0]!=Expr::Tag() ),
  doY_      ( velStarTags[1]!=Expr::Tag() ),
  doZ_      ( velStarTags[2]!=Expr::Tag() ),
  xVelStart_  ( velStarTags[0] ),
  yVelStart_  ( velStarTags[1] ),
  zVelStart_  ( velStarTags[2] ),
  denst_      ( densTag  ),
  densStart_  ( densStarTag    ),
  dens2Start_ ( dens2StarTag   ),
  timestept_  ( timestepTag )
{
  set_gpu_runnable( true );
}

//------------------------------------------------------------------

Drhodt::~Drhodt()
{}

//------------------------------------------------------------------

void Drhodt::advertise_dependents( Expr::ExprDeps& exprDeps )
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
  exprDeps.requires_expression( dens2Start_ );
  exprDeps.requires_expression( denst_ );
  exprDeps.requires_expression( timestept_ );  
}

//------------------------------------------------------------------

void Drhodt::bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldMgrSelector<SVolField>::type& scalarFM = fml.field_manager<SVolField>();
  const Expr::FieldMgrSelector<TimeField>::type& tsfm     = fml.field_manager<TimeField>();
  
  const Expr::FieldMgrSelector<XVolField>::type& xVolFM  = fml.field_manager<XVolField>();
  const Expr::FieldMgrSelector<YVolField>::type& yVolFM  = fml.field_manager<YVolField>();
  const Expr::FieldMgrSelector<ZVolField>::type& zVolFM  = fml.field_manager<ZVolField>();
  
  if( doX_ )
  {
    uStar_  = &xVolFM.field_ref( xVelStart_ );
  }
  if( doY_ )
  {
    vStar_ = &yVolFM.field_ref( yVelStart_ );
  }
  if( doZ_ )
  {
    wStar_ = &zVolFM.field_ref( zVelStart_ );
  }
  
  densStar_ = &scalarFM.field_ref( densStart_ );
  dens2Star_ = &scalarFM.field_ref( dens2Start_ );
  dens_ = &scalarFM.field_ref( denst_ );
  timestep_ = &tsfm.field_ref( timestept_ );  
    
}

//------------------------------------------------------------------

void Drhodt::bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  if( doX_ ){
    divXOp_       = opDB.retrieve_operator<DivXT>();
    xFInterpOp_   = opDB.retrieve_operator<XFaceInterpT>();
    s2XFInterpOp_ = opDB.retrieve_operator<Scalar2XFInterpT>();
  }
  if( doY_ ){
    divYOp_       = opDB.retrieve_operator<DivYT>();
    yFInterpOp_   = opDB.retrieve_operator<YFaceInterpT>();
    s2YFInterpOp_ = opDB.retrieve_operator<Scalar2YFInterpT>();
  }
  if( doZ_ ){
    divZOp_       = opDB.retrieve_operator<DivZT>();
    zFInterpOp_   = opDB.retrieve_operator<ZFaceInterpT>();
    s2ZFInterpOp_ = opDB.retrieve_operator<Scalar2ZFInterpT>();
  }
}

//------------------------------------------------------------------

void Drhodt::evaluate()
{
  using namespace SpatialOps;
  SVolField& result = this->value();
  
  const double alpha = 0.1;   // the continuity equation weighting factor
  if( doX_ && doY_ && doZ_ ){ // for 3D cases, inline the whole thing
    result <<=
    - (1.0 - alpha) * ( (*divXOp_) ( (*s2XFInterpOp_)(*densStar_) * (*xFInterpOp_)(*uStar_) )
                       +(*divYOp_) ( (*s2YFInterpOp_)(*densStar_) * (*yFInterpOp_)(*vStar_) )
                       +(*divZOp_) ( (*s2ZFInterpOp_)(*densStar_) * (*zFInterpOp_)(*wStar_) ) )
    + alpha * ((*dens2Star_ - *dens_)/(2 * *timestep_));
  }
  else{
    // for 1D and 2D cases, we are not as efficient - add terms as needed...
    if( doX_ ){
      // drhodt^(n+1) = - (1-alpha) * nabla.(rho^(n+1)*u^(n+1))
      result <<= - (1.0 - alpha) * (*divXOp_) ( (*s2XFInterpOp_)(*densStar_) * (*xFInterpOp_)(*uStar_) );
    }
    else{
      result <<= 0.0;
    }
    
    if( doY_ ){
      // drhodt^(n+1) = drhodt^(n+1) - (1-alpha) * nabla.(rho^(n+1)*v^(n+1))
      result <<= result - (1.0 - alpha) * (*divYOp_) ( (*s2YFInterpOp_)(*densStar_) * (*yFInterpOp_)(*vStar_) );
    }
    
    if( doZ_ ){
      // drhodt^(n+1) = drhodt^(n+1) - (1-alpha) * nabla.(rho^(n+1)*w^(n+1))
      result <<= result - (1.0 - alpha) * (*divZOp_) ( (*s2ZFInterpOp_)(*densStar_) * (*zFInterpOp_)(*wStar_) );
    }
    
    result <<= result + alpha * ((*dens2Star_ - *dens_)/(2 * *timestep_));  // drhodt^(n+1) = drhodt^(n+1) + alpha * (drho/dt)^(n+1)
 
  } // 1D, 2D cases

}

//------------------------------------------------------------------

Drhodt::Builder::Builder( const Expr::Tag& result,
                             const Expr::TagList& velStarTags,
                             const Expr::Tag densTag,
                             const Expr::Tag densStarTag,
                             const Expr::Tag dens2StarTag,
                             const Expr::Tag timestepTag )
: ExpressionBuilder(result),
  velStarTs_ ( velStarTags ),
  denst_     ( densTag      ),
  densStart_ ( densStarTag  ),
  dens2Start_( dens2StarTag ),
  tstpt_( timestepTag )
{}

//------------------------------------------------------------------

Expr::ExpressionBase*
Drhodt::Builder::build() const
{
  return new Drhodt( velStarTs_, denst_, densStart_, dens2Start_, tstpt_);
}
//------------------------------------------------------------------

