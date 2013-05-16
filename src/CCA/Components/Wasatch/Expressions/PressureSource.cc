#include "PressureSource.h"

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>


PressureSource::PressureSource( const Expr::TagList& momTags,
                                const Expr::TagList& velStarTags,
                                const bool isConstDensity,
                                const Expr::Tag densTag,
                                const Expr::Tag densStarTag,
                                const Expr::Tag dens2StarTag,
                                const Expr::Tag dilTag,
                                const Expr::Tag timestepTag )
: Expr::Expression<SVolField>(),
xMomt_      ( densStarTag==Expr::Tag() ? Expr::Tag() : momTags[0]     ),
yMomt_      ( densStarTag==Expr::Tag() ? Expr::Tag() : momTags[1]     ),
zMomt_      ( densStarTag==Expr::Tag() ? Expr::Tag() : momTags[2]     ),
haveX_      ( momTags[0]==Expr::Tag() ? false : true                  ),
haveY_      ( momTags[1]==Expr::Tag() ? false : true                  ),
haveZ_      ( momTags[2]==Expr::Tag() ? false : true                  ),
xVelStart_  ( densStarTag==Expr::Tag() ? Expr::Tag() : velStarTags[0] ),
yVelStart_  ( densStarTag==Expr::Tag() ? Expr::Tag() : velStarTags[1] ),
zVelStart_  ( densStarTag==Expr::Tag() ? Expr::Tag() : velStarTags[2] ),
densStart_  ( densStarTag==Expr::Tag() ? Expr::Tag() : densStarTag    ),
dens2Start_ ( densStarTag==Expr::Tag() ? Expr::Tag() : dens2StarTag   ),
timestept_  ( timestepTag ),
denst_         ( densTag  ),
dilt_          ( isConstDensity ? dilTag  : Expr::Tag() ),
isConstDensity_( isConstDensity )
{
}

//------------------------------------------------------------------

PressureSource::~PressureSource()
{}

//------------------------------------------------------------------

void PressureSource::advertise_dependents( Expr::ExprDeps& exprDeps )
{  
  if (!isConstDensity_) {
    if( haveX_ )  
    {
      exprDeps.requires_expression( xMomt_ );
      exprDeps.requires_expression( xVelStart_ );
    }
    if( haveY_ )
    {
      exprDeps.requires_expression( yMomt_ );
      exprDeps.requires_expression( yVelStart_ );
    }  
    if( haveZ_ )
    {
      exprDeps.requires_expression( zMomt_ );
      exprDeps.requires_expression( zVelStart_ );
    }  
    
    exprDeps.requires_expression( densStart_ );
    exprDeps.requires_expression( dens2Start_ );
  }
  else {
    exprDeps.requires_expression( dilt_ );
  }
  exprDeps.requires_expression( denst_ );
  
  exprDeps.requires_expression( timestept_ );
  
}

//------------------------------------------------------------------

void PressureSource::bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldMgrSelector<SVolField>::type& scalarFM = fml.field_manager<SVolField>();
  const Expr::FieldMgrSelector<double   >::type& tsfm     = fml.field_manager<double>();
  
  if (!isConstDensity_) {
    const Expr::FieldMgrSelector<XVolField>::type& xVolFM  = fml.field_manager<XVolField>();
    const Expr::FieldMgrSelector<YVolField>::type& yVolFM  = fml.field_manager<YVolField>();
    const Expr::FieldMgrSelector<ZVolField>::type& zVolFM  = fml.field_manager<ZVolField>(); 
    
    if( haveX_ )  
    {
      xMom_  = &xVolFM.field_ref( xMomt_ );
      xVelStar_  = &xVolFM.field_ref( xVelStart_ );    
    }
    if( haveY_ )    
    {
      yMom_ = &yVolFM.field_ref( yMomt_ );
      yVelStar_ = &yVolFM.field_ref( yVelStart_ );    
    }
    if( haveZ_ )   
    {
      zMom_ = &zVolFM.field_ref( zMomt_ );
      zVelStar_ = &zVolFM.field_ref( zVelStart_ );    
    }
    
    densStar_ = &scalarFM.field_ref( densStart_ );
    dens2Star_ = &scalarFM.field_ref( dens2Start_ );
  }
  else {
    dil_  = &scalarFM.field_ref( dilt_ );
  }
  dens_ = &scalarFM.field_ref( denst_ );
  
  timestep_ = &tsfm.field_ref( timestept_ );  
  
}

//------------------------------------------------------------------

void PressureSource::bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  if (!isConstDensity_) {
    if( haveX_ ) {
      divXOp_         = opDB.retrieve_operator<DivXT>();
      xFaceInterpOp_  = opDB.retrieve_operator<XFaceInterpT>();
      scalar2XFInterpOp_ = opDB.retrieve_operator<Scalar2XFInterpT>();
    }
    if( haveY_ ) {
      divYOp_          = opDB.retrieve_operator<DivYT>();
      yFaceInterpOp_   = opDB.retrieve_operator<YFaceInterpT>();
      scalar2YFInterpOp_ = opDB.retrieve_operator<Scalar2YFInterpT>();
    }
    if( haveZ_ ) {
      divZOp_          = opDB.retrieve_operator<DivZT>();
      zFaceInterpOp_   = opDB.retrieve_operator<ZFaceInterpT>();
      scalar2ZFInterpOp_ = opDB.retrieve_operator<Scalar2ZFInterpT>();
    }
  }
}

//------------------------------------------------------------------

void PressureSource::evaluate()
{
  using namespace SpatialOps;
  SVolField& result = this->value();
  result <<= 0.0;
  
  if (!isConstDensity_) {
    SpatialOps::SpatFldPtr<SVolField> tmp  = SpatialOps::SpatialFieldStore::get<SVolField>( result );
    SpatialOps::SpatFldPtr<SVolField> tmp1 = SpatialOps::SpatialFieldStore::get<SVolField>( result );
    SpatialOps::SpatFldPtr<SVolField> tmp2 = SpatialOps::SpatialFieldStore::get<SVolField>( result );
    SpatialOps::SpatFldPtr<SVolField> tmp3 = SpatialOps::SpatialFieldStore::get<SVolField>( result );
    SpatialOps::SpatFldPtr<SVolField> tmp4 = SpatialOps::SpatialFieldStore::get<SVolField>( result );
    SpatialOps::SpatFldPtr<SVolField> tmp5 = SpatialOps::SpatialFieldStore::get<SVolField>( result );
    SpatialOps::SpatFldPtr<XFace> xftmp  = SpatialOps::SpatialFieldStore::get<XFace>( result );
    SpatialOps::SpatFldPtr<XFace> xftmp1 = SpatialOps::SpatialFieldStore::get<XFace>( result );
    SpatialOps::SpatFldPtr<XFace> xftmp2 = SpatialOps::SpatialFieldStore::get<XFace>( result );
    SpatialOps::SpatFldPtr<YFace> yftmp  = SpatialOps::SpatialFieldStore::get<YFace>( result );
    SpatialOps::SpatFldPtr<YFace> yftmp1 = SpatialOps::SpatialFieldStore::get<YFace>( result );
    SpatialOps::SpatFldPtr<YFace> yftmp2 = SpatialOps::SpatialFieldStore::get<YFace>( result );
    SpatialOps::SpatFldPtr<ZFace> zftmp  = SpatialOps::SpatialFieldStore::get<ZFace>( result );
    SpatialOps::SpatFldPtr<ZFace> zftmp1 = SpatialOps::SpatialFieldStore::get<ZFace>( result );
    SpatialOps::SpatFldPtr<ZFace> zftmp2 = SpatialOps::SpatialFieldStore::get<ZFace>( result );

    double alpha = 0.1;   // the continuity equation weighting factor 

    if( haveX_ ){
      *xftmp <<= 0.0;
      *tmp <<= 0.0;
      *xftmp1 <<= 0.0;
      *xftmp2 <<= 0.0;
      *tmp1 <<=0.0;
      xFaceInterpOp_->apply_to_field( *xMom_, *xftmp );
      divXOp_->apply_to_field( *xftmp, *tmp );
      result <<= result + *tmp;                 // P_src = nabla.(rho*u)^n
      
      scalar2XFInterpOp_->apply_to_field( *densStar_, *xftmp2 );
      xFaceInterpOp_->apply_to_field( *xVelStar_, *xftmp1 );
      *xftmp1 <<= *xftmp1 * *xftmp2;
      divXOp_->apply_to_field( *xftmp1, *tmp1 );
      result <<= result - (1-alpha) * *tmp1;    // P_src = nabla.(rho*u)^n - (1-alpha) * nabla.(rho*u)^(n+1)
    }
    
    if( haveY_ ){
      *yftmp <<= 0.0;
      *tmp2 <<= 0.0;
      *yftmp1 <<= 0.0;
      *yftmp2 <<= 0.0;
      *tmp3 <<= 0.0;
      yFaceInterpOp_->apply_to_field( *yMom_, *yftmp );
      divYOp_->apply_to_field( *yftmp, *tmp2 );
      result <<= result + *tmp2;                // P_src = P_src + nabla.(rho*v)^n
      
      scalar2YFInterpOp_->apply_to_field( *densStar_, *yftmp2 );
      yFaceInterpOp_->apply_to_field( *yVelStar_, *yftmp1 );
      *yftmp1 <<= *yftmp1 * *yftmp2;
      divYOp_->apply_to_field( *yftmp1, *tmp3 );
      result <<= result - (1-alpha) * *tmp3;    // P_src = P_src + nabla.(rho*v)^n - (1-alpha) * nabla.(rho*v)^(n+1)
    }
    
    if( haveZ_ ){
      *zftmp <<= 0.0;
      *tmp4 <<= 0.0;
      *zftmp1 <<= 0.0;
      *zftmp2 <<= 0.0;
      *tmp5 <<= 0.0;
      zFaceInterpOp_->apply_to_field( *zMom_, *zftmp );
      divZOp_->apply_to_field( *zftmp, *tmp4 );
      result <<= result + *tmp4;                // P_src = P_src + nabla.(rho*w)^n
      
      scalar2ZFInterpOp_->apply_to_field( *densStar_, *zftmp2 );
      zFaceInterpOp_->apply_to_field( *zVelStar_, *zftmp1 );
      *zftmp1 <<= *zftmp1 * *zftmp2;
      divZOp_->apply_to_field( *zftmp1, *tmp5 );
      result <<= result - (1-alpha) * *tmp5;    // P_src = P_src + nabla.(rho*w)^n - (1-alpha) * nabla.(rho*w)^(n+1)
    }
    
    result <<= result + alpha * ((*dens2Star_ - *dens_)/(2 * *timestep_));  // P_src = P_src + alpha * (drho/dt)^(n+1)
  }
  else {
    result <<= *dens_ * *dil_;
  }
  result <<= result / *timestep_ ;
  
}

//------------------------------------------------------------------

PressureSource::Builder::Builder( const Expr::Tag& result,
                                  const Expr::TagList& momTags,
                                  const Expr::TagList& velStarTags,
                                  const bool isConstDensity,
                                  const Expr::Tag densTag,
                                  const Expr::Tag densStarTag,
                                  const Expr::Tag dens2StarTag,
                                  const Expr::Tag dilTag,
                                  const Expr::Tag timestepTag )
: ExpressionBuilder(result),
momTs_      ( densStarTag==Expr::Tag() ? Expr::TagList() : momTags     ),
velStarTs_  ( densStarTag==Expr::Tag() ? Expr::TagList() : velStarTags ),
isConstDens_( isConstDensity ),
densStart_  ( densStarTag    ),
dens2Start_ ( dens2StarTag   ),
tstpt_    ( timestepTag ),
denst_    ( densTag     ),
dilt_     ( isConstDensity ? dilTag  : Expr::Tag() )
{}

//------------------------------------------------------------------

Expr::ExpressionBase*
PressureSource::Builder::build() const
{
  return new PressureSource( momTs_, velStarTs_, isConstDens_, denst_, densStart_, dens2Start_, dilt_, tstpt_);
}
//------------------------------------------------------------------

