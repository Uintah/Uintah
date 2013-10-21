#include <CCA/Components/Wasatch/Expressions/PressureSource.h>

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
  isConstDensity_( isConstDensity ),
  doX_      ( momTags[0]!=Expr::Tag() ),
  doY_      ( momTags[1]!=Expr::Tag() ),
  doZ_      ( momTags[2]!=Expr::Tag() ),
  xMomt_      ( densStarTag==Expr::Tag() ? Expr::Tag() : momTags[0]     ),
  yMomt_      ( densStarTag==Expr::Tag() ? Expr::Tag() : momTags[1]     ),
  zMomt_      ( densStarTag==Expr::Tag() ? Expr::Tag() : momTags[2]     ),
  xVelStart_  ( densStarTag==Expr::Tag() ? Expr::Tag() : velStarTags[0] ),
  yVelStart_  ( densStarTag==Expr::Tag() ? Expr::Tag() : velStarTags[1] ),
  zVelStart_  ( densStarTag==Expr::Tag() ? Expr::Tag() : velStarTags[2] ),
  denst_      ( densTag  ),
  densStart_  ( densStarTag==Expr::Tag() ? Expr::Tag() : densStarTag    ),
  dens2Start_ ( densStarTag==Expr::Tag() ? Expr::Tag() : dens2StarTag   ),
  dilt_       ( isConstDensity ? dilTag  : Expr::Tag() ),
  timestept_  ( timestepTag )
{
  set_gpu_runnable( true );
}

//------------------------------------------------------------------

PressureSource::~PressureSource()
{}

//------------------------------------------------------------------

void PressureSource::advertise_dependents( Expr::ExprDeps& exprDeps )
{  
  if (!isConstDensity_) {
    if( doX_ )  
    {
      exprDeps.requires_expression( xMomt_ );
      exprDeps.requires_expression( xVelStart_ );
    }
    if( doY_ )
    {
      exprDeps.requires_expression( yMomt_ );
      exprDeps.requires_expression( yVelStart_ );
    }  
    if( doZ_ )
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
  const Expr::FieldMgrSelector<TimeField>::type& tsfm     = fml.field_manager<TimeField>();
  
  if (!isConstDensity_) {
    const Expr::FieldMgrSelector<XVolField>::type& xVolFM  = fml.field_manager<XVolField>();
    const Expr::FieldMgrSelector<YVolField>::type& yVolFM  = fml.field_manager<YVolField>();
    const Expr::FieldMgrSelector<ZVolField>::type& zVolFM  = fml.field_manager<ZVolField>(); 
    
    if( doX_ ){
      xMom_  = &xVolFM.field_ref( xMomt_ );
      uStar_  = &xVolFM.field_ref( xVelStart_ );
    }
    if( doY_ ){
      yMom_ = &yVolFM.field_ref( yMomt_ );
      vStar_ = &yVolFM.field_ref( yVelStart_ );
    }
    if( doZ_ ){
      zMom_ = &zVolFM.field_ref( zMomt_ );
      wStar_ = &zVolFM.field_ref( zVelStart_ );
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
}

//------------------------------------------------------------------

void PressureSource::evaluate()
{
  using namespace SpatialOps;
  SVolField& result = this->value();
  
  if (isConstDensity_ ){
    result <<= *dens_ * *dil_ / *timestep_;
  }
  else{ // variable density

    const double alpha = 0.1;   // the continuity equation weighting factor

    if( doX_ && doY_ && doZ_ ){ // for 3D cases, inline the whole thing
      result <<=
          ( (*gradXOp_)(*xMom_) - (1.0 - alpha) * (*gradXOp_) ( (*s2XInterpOp_)(*densStar_) * *uStar_ )
          + (*gradYOp_)(*yMom_) - (1.0 - alpha) * (*gradYOp_) ( (*s2YInterpOp_)(*densStar_) * *vStar_ )
          + (*gradZOp_)(*zMom_) - (1.0 - alpha) * (*gradZOp_) ( (*s2ZInterpOp_)(*densStar_) * *wStar_ )
          + alpha * ((*dens2Star_ - *dens_)/(2. * *timestep_))
          ) / *timestep_;
    }
    else{
      // for 1D and 2D cases, we are not as efficient - add terms as needed...
      if( doX_ ){
        // P_src = nabla.(rho*u)^n - (1-alpha) * nabla.(rho*u)^(n+1)
        result <<= (*gradXOp_)(*xMom_) - (1.0 - alpha) * (*gradXOp_) ( (*s2XInterpOp_)(*densStar_) * *uStar_ );
      }
      else{
        result <<= 0.0;
      }

      if( doY_ ){
        // P_src = P_src + nabla.(rho*v)^n - (1-alpha) * nabla.(rho*v)^(n+1)
        result <<= result + (*gradYOp_)(*yMom_) - (1.0 - alpha) * (*gradYOp_) ( (*s2YInterpOp_)(*densStar_) * *vStar_ );
      }

      if( doZ_ ){
        // P_src = P_src + nabla.(rho*w)^n - (1-alpha) * nabla.(rho*w)^(n+1)
        result <<= result + (*gradZOp_)(*zMom_) - (1.0 - alpha) * (*gradZOp_) ( (*s2ZInterpOp_)(*densStar_) * *wStar_ );
      }

      result <<= ( result + alpha * ((*dens2Star_ - *dens_)/(2. * *timestep_))) / *timestep_;  // P_src = P_src + alpha * (drho/dt)^(n+1)
    } // 1D, 2D cases

  }
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
  isConstDens_( isConstDensity ),
  momTs_      ( densStarTag==Expr::Tag() ? Expr::TagList() : momTags     ),
  velStarTs_  ( densStarTag==Expr::Tag() ? Expr::TagList() : velStarTags ),
  denst_     ( densTag      ),
  densStart_ ( densStarTag  ),
  dens2Start_( dens2StarTag ),
  dilt_( isConstDensity ? dilTag  : Expr::Tag() ),
  tstpt_( timestepTag )
{}

//------------------------------------------------------------------

Expr::ExpressionBase*
PressureSource::Builder::build() const
{
  return new PressureSource( momTs_, velStarTs_, isConstDens_, denst_, densStart_, dens2Start_, dilt_, tstpt_);
}
//------------------------------------------------------------------

