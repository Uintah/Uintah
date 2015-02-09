#include <CCA/Components/Wasatch/Expressions/PressureSource.h>
#include <CCA/Components/Wasatch/TimeIntegratorTools.h>
#include <CCA/Components/Wasatch/TagNames.h>

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>


PressureSource::PressureSource( const Expr::TagList& momTags,
                                const Expr::TagList& oldMomTags,
                                const Expr::TagList& velTags,
                                const Expr::TagList& velStarTags,
                                const bool isConstDensity,
                                const Expr::Tag densTag,
                                const Expr::Tag densStarTag,
                                const Expr::Tag dens2StarTag,
                                const Wasatch::VarDenParameters varDenParams,
                                const Expr::Tag divmomstarTag)
: Expr::Expression<SVolField>(),
  isConstDensity_( isConstDensity ),
  doX_      ( momTags[0]!=Expr::Tag() ),
  doY_      ( momTags[1]!=Expr::Tag() ),
  doZ_      ( momTags[2]!=Expr::Tag() ),
  is3d_     ( doX_ && doY_ && doZ_    ),
  useOnePredictor_(varDenParams.onePredictor),
  xMomt_      ( densStarTag==Expr::Tag() ? Expr::Tag() : momTags[0]     ),
  yMomt_      ( densStarTag==Expr::Tag() ? Expr::Tag() : momTags[1]     ),
  zMomt_      ( densStarTag==Expr::Tag() ? Expr::Tag() : momTags[2]     ),
  xMomOldt_   ( oldMomTags[0]     ),
  yMomOldt_   ( oldMomTags[1]     ),
  zMomOldt_   ( oldMomTags[2]     ),
  xVelt_  ( velTags[0] ),
  yVelt_  ( velTags[1] ),
  zVelt_  ( velTags[2] ),
  xVelStart_  ( densStarTag==Expr::Tag() ? Expr::Tag() : velStarTags[0] ),
  yVelStart_  ( densStarTag==Expr::Tag() ? Expr::Tag() : velStarTags[1] ),
  zVelStart_  ( densStarTag==Expr::Tag() ? Expr::Tag() : velStarTags[2] ),
  denst_      ( densTag  ),
  densStart_  ( densStarTag==Expr::Tag() ? Expr::Tag() : densStarTag    ),
  dens2Start_ ( densStarTag==Expr::Tag() ? Expr::Tag() : dens2StarTag   ),
  dilt_       ( Wasatch::TagNames::self().dilatation ),
  timestept_  ( Wasatch::TagNames::self().dt ),
  rkstaget_   ( Wasatch::TagNames::self().rkstage ),
  divmomstart_( divmomstarTag ),
  a0_   (varDenParams.alpha0),
  model_(varDenParams.model )
{
  set_gpu_runnable( true );
}

//------------------------------------------------------------------

PressureSource::~PressureSource()
{}

//------------------------------------------------------------------

void PressureSource::advertise_dependents( Expr::ExprDeps& exprDeps )
{
  if( doX_ ) exprDeps.requires_expression( xMomOldt_ );
  if( doY_ ) exprDeps.requires_expression( yMomOldt_ );
  if( doZ_ ) exprDeps.requires_expression( zMomOldt_ );

  if (!isConstDensity_) {
    if( doX_ )
    {
      exprDeps.requires_expression( xMomt_ );
      exprDeps.requires_expression( xVelt_ );
      exprDeps.requires_expression( xVelStart_ );
    }
    if( doY_ )
    {
      exprDeps.requires_expression( yMomt_ );
      exprDeps.requires_expression( yVelt_ );
      exprDeps.requires_expression( yVelStart_ );
    }  
    if( doZ_ )
    {
      exprDeps.requires_expression( zMomt_ );
      exprDeps.requires_expression( zVelt_ );
      exprDeps.requires_expression( zVelStart_ );
    }  
    
    exprDeps.requires_expression( densStart_ );
    if (!useOnePredictor_) exprDeps.requires_expression( dens2Start_ );
    exprDeps.requires_expression( divmomstart_ );
  }
  else {
    exprDeps.requires_expression( dilt_ );
  }
  exprDeps.requires_expression( denst_ );
  
  exprDeps.requires_expression( timestept_ );
  exprDeps.requires_expression( rkstaget_  );
}

//------------------------------------------------------------------

void PressureSource::bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldMgrSelector<SVolField>::type& scalarFM = fml.field_manager<SVolField>();
  const Expr::FieldMgrSelector<TimeField>::type& tsfm     = fml.field_manager<TimeField>();

  const Expr::FieldMgrSelector<XVolField>::type& xVolFM  = fml.field_manager<XVolField>();
  const Expr::FieldMgrSelector<YVolField>::type& yVolFM  = fml.field_manager<YVolField>();
  const Expr::FieldMgrSelector<ZVolField>::type& zVolFM  = fml.field_manager<ZVolField>();

  if( doX_ ) xMomOld_  = &xVolFM.field_ref( xMomOldt_ );
  if( doY_ ) yMomOld_  = &yVolFM.field_ref( yMomOldt_ );
  if( doZ_ ) zMomOld_  = &zVolFM.field_ref( zMomOldt_ );

  if (!isConstDensity_) {
    
    if( doX_ ){
      xMom_  = &xVolFM.field_ref( xMomt_ );
      xVel_  = &xVolFM.field_ref( xVelt_ );
      uStar_ = &xVolFM.field_ref( xVelStart_ );
    }
    if( doY_ ){
      yMom_  = &yVolFM.field_ref( yMomt_ );
      yVel_  = &yVolFM.field_ref( yVelt_ );
      vStar_ = &yVolFM.field_ref( yVelStart_ );
    }
    if( doZ_ ){
      zMom_  = &zVolFM.field_ref( zMomt_ );
      zVel_  = &zVolFM.field_ref( zVelt_ );
      wStar_ = &zVolFM.field_ref( zVelStart_ );
    }
    
    densStar_ = &scalarFM.field_ref( densStart_ );
    // if we are using more than one predictor then we will need rho**
    if (!useOnePredictor_) dens2Star_ = &scalarFM.field_ref( dens2Start_ );
    divmomstar_ = &scalarFM.field_ref( divmomstart_ );
  }
  else {
    dil_  = &scalarFM.field_ref( dilt_ );
  }
  dens_ = &scalarFM.field_ref( denst_ );
  
  timestep_ = &tsfm.field_ref( timestept_ );
  rkStage_  = &tsfm.field_ref( rkstaget_  );
}

//------------------------------------------------------------------

void PressureSource::bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  if( doX_ ) gradXOp_     = opDB.retrieve_operator<GradXT>();
  if( doY_ ) gradYOp_     = opDB.retrieve_operator<GradYT>();
  if( doZ_ ) gradZOp_     = opDB.retrieve_operator<GradZT>();
  
  timeIntInfo_ = opDB.retrieve_operator<Wasatch::TimeIntegrator>();
  
  if (!isConstDensity_) {
    if( doX_ ){
      s2XInterpOp_ = opDB.retrieve_operator<S2XInterpOpT>();
      gradXSOp_    = opDB.retrieve_operator<GradXST>();
      x2SInterpOp_ = opDB.retrieve_operator<X2SInterpOpT>();
    }
    if( doY_ ){
      s2YInterpOp_ = opDB.retrieve_operator<S2YInterpOpT>();
      gradYSOp_    = opDB.retrieve_operator<GradYST>();
      y2SInterpOp_ = opDB.retrieve_operator<Y2SInterpOpT>();
    }
    if( doZ_ ){
      s2ZInterpOp_ = opDB.retrieve_operator<S2ZInterpOpT>();
      gradZSOp_    = opDB.retrieve_operator<GradZST>();
      z2SInterpOp_ = opDB.retrieve_operator<Z2SInterpOpT>();
    }
  }
}

//------------------------------------------------------------------

void PressureSource::evaluate()
{
  using namespace SpatialOps;
  typedef std::vector<SVolField*> SVolFieldVec;
  SVolFieldVec& results = this->get_value_vec();
  
  const TimeField& dt = *timestep_;
  
  SVolField& psrc = *results[0];

  
  const Wasatch::TimeIntegrator& timeIntInfo = *timeIntInfo_;
  const double a2 = timeIntInfo.alpha[1];
  const double a3 = timeIntInfo.alpha[2];
  
  const double b2 = timeIntInfo.beta[1];
  const double b3 = timeIntInfo.beta[2];

  if( isConstDensity_ ) {

    if( is3d_ ){ // for 3D cases, inline the whole thing
      psrc <<= cond( *rkStage_ > 1.0, (*gradXOp_)(*xMomOld_) + (*gradYOp_)(*yMomOld_) + (*gradZOp_)(*zMomOld_) )
                   ( 0.0 );
    } else {
      // for 1D and 2D cases, we are not as efficient - add terms as needed...
      if( doX_ ) psrc <<=        (*gradXOp_)( (*xMomOld_) );
      else       psrc <<= 0.0;
      if( doY_ ) psrc <<= psrc + (*gradYOp_)( (*yMomOld_) );
      if( doZ_ ) psrc <<= psrc + (*gradZOp_)( (*zMomOld_) );
    } // 1D, 2D cases
    
    psrc <<= cond( *rkStage_ == 1.0,                  *dens_ * *dil_  / dt        )
                 ( *rkStage_ == 2.0, (a2* psrc + b2 * *dens_ * *dil_) / (b2 * dt) )
                 ( *rkStage_ == 3.0, (a3* psrc + b3 * *dens_ * *dil_) / (b3 * dt) )
                 ( 0.0 ); // should never get here.
    
  } else { // variable density
    
    SVolField& drhodt     = *results[1];
    SVolField& alpha      = *results[2];
//    SVolField& beta       = *results[3];
    SVolField& drhodtstar = *results[4];

    if (useOnePredictor_) drhodtstar <<= (*densStar_ - *dens_) / *timestep_;
    else                  drhodtstar <<= (*dens2Star_ - *dens_) / (2.0 * *timestep_);
    
    // beta is the ratio of drhodt to div(mom*)
//    beta  <<= cond (abs(drhodtstar - divmomstar) <= 1e-10, 1.0)
//                   (abs(divmomstar) <= 1e-12, 2.0)
//                   (abs(drhodtstar/divmomstar));
    
//    alpha <<= cond( beta != beta, 0.0 )
//                  ( 1.0/(beta + 1.0)  );
    
//    alpha <<= cond( beta != beta, 0.0 )
//                  ( exp(log(0.2)*(pow(beta,0.2))) ); // Increase the value of the exponent to get closer to a step function
    
    // MAYBE THE FOLLOWING BETA BETA BEHAVES A BIT BETTER?
//    beta  <<= abs( drhodtstar/ (divmomstar + 1) );

    // INSTEAD OF SCALING WITH BETA, TRY DRHODT
//    alpha <<=  0.9/(abs(drhodtstar) + 1.0) + 0.1 ;

    switch (model_) {
      case Wasatch::VarDenParameters::CONSTANT:
        alpha <<= a0_;
        break;
      case Wasatch::VarDenParameters::IMPULSE:
        alpha <<= cond(drhodtstar == 0.0, 1.0)(a0_);
        break;
      case Wasatch::VarDenParameters::DYNAMIC:
      {
        SpatialOps::SpatFldPtr<SVolField> velDotDensGrad = SpatialOps::SpatialFieldStore::get<SVolField>( alpha );
        if( is3d_ ){ // for 3D cases, inline the whole thing
          *velDotDensGrad <<= (*x2SInterpOp_)(*xVel_) * (*gradXSOp_)(*dens_) + (*y2SInterpOp_)(*yVel_) * (*gradYSOp_)(*dens_) + (*z2SInterpOp_)(*zVel_) * (*gradZSOp_)(*dens_);
        } else {
          // for 1D and 2D cases, we are not as efficient - add terms as needed...
          if( doX_ ) *velDotDensGrad <<= (*x2SInterpOp_)(*xVel_) * (*gradXSOp_)(*dens_);
          else       *velDotDensGrad <<= 0.0;
          if( doY_ ) *velDotDensGrad <<= *velDotDensGrad + (*y2SInterpOp_)(*yVel_) * (*gradYSOp_)(*dens_);
          if( doZ_ ) *velDotDensGrad <<= *velDotDensGrad + (*z2SInterpOp_)(*zVel_) * (*gradZSOp_)(*dens_);
        } // 1D, 2D cases
        *velDotDensGrad <<= abs(*velDotDensGrad);
        alpha <<= cond(drhodtstar == 0.0, 1.0)( (1.0 - a0_) * ((0.1 * *velDotDensGrad) / ( 0.1 * *velDotDensGrad + 1)) + a0_ );
      }
        //    case Wasatch::VarDenParameters::DYNAMIC:
        //    {
        //      SpatialOps::SpatFldPtr<SVolField> densGrad = SpatialOps::SpatialFieldStore::get<SVolField>( alpha );
        //      *densGrad <<= sqrt( (*gradXOp_)(*dens_) * (*gradXOp_)(*dens_) + (*gradYOp_)(*dens_) * (*gradYOp_)(*dens_) + (*gradZOp_)(*dens_) * (*gradZOp_)(*dens_));
        //
        //      alpha <<= 0.9*((0.1 * *densGrad) / ( 0.1 * *densGrad + 1))+0.1;
        //    }
        break;
      default:
        alpha <<= 0.1;
        break;
    }

    drhodt <<= alpha * drhodtstar - (1.0 - alpha) * *divmomstar_;
    
    SpatFldPtr<SVolField> divmomlatest_ = SpatialFieldStore::get<SVolField>( psrc );
    SVolField& divmomlatest = *divmomlatest_;
    
    if( is3d_ ){ // for 3D cases, inline the whole thing
      // always add the divergence of momentum from the old timestep div(r u)^n
      psrc <<= (*gradXOp_)(*xMomOld_) + (*gradYOp_)(*yMomOld_) + (*gradZOp_)(*zMomOld_);
      // if we are at an rkstage > 1, then add the divergence of the most recent momentum
      psrc <<= cond( *rkStage_ == 1.0,                  psrc                                                             )
                   ( *rkStage_ == 2.0, a2* psrc + b2 * ((*gradXOp_)(*xMom_) + (*gradYOp_)(*yMom_) + (*gradZOp_)(*zMom_)) )
                   ( *rkStage_ == 3.0, a3* psrc + b3 * ((*gradXOp_)(*xMom_) + (*gradYOp_)(*yMom_) + (*gradZOp_)(*zMom_)) )
                   ( 0.0 ); // should never get here.

    } else {
      // for 1D and 2D cases, we are not as efficient - add terms as needed...
      if( doX_ ) {
        // always add the divergence of momentum from the old timestep div(r u)^n
        psrc <<= (*gradXOp_)( (*xMomOld_) );
        if (timeIntInfo_->nStages > 1) divmomlatest <<= (*gradXOp_)(*xMom_);
      }
      else       psrc <<= 0.0;
      if( doY_ )
      {
        psrc <<= psrc + (*gradYOp_)( (*yMomOld_) );
        if (timeIntInfo_->nStages > 1) divmomlatest <<= divmomlatest + (*gradYOp_)(*yMom_);
      }
      if( doZ_ )
      {
        psrc <<= psrc + (*gradZOp_)( (*zMomOld_) );
        if (timeIntInfo_->nStages > 1) divmomlatest <<= divmomlatest + (*gradZOp_)(*zMom_);
      }
      
      psrc <<= cond( *rkStage_ == 1.0,                  psrc         )
                   ( *rkStage_ == 2.0, a2* psrc + b2 * divmomlatest  )
                   ( *rkStage_ == 3.0, a3* psrc + b3 * divmomlatest  )
                   ( 0.0 ); // should never get here.
      
    } // 1D, 2D cases

    //
    psrc <<= cond( *rkStage_ == 1.0, (psrc + drhodt)  / dt       )
                 ( *rkStage_ == 2.0, (psrc + drhodt) / (b2 * dt) )
                 ( *rkStage_ == 3.0, (psrc + drhodt) / (b3 * dt) )
                 ( 0.0 ); // should never get here.

  } // Variable density
}

//------------------------------------------------------------------

PressureSource::Builder::Builder( const Expr::TagList& results,
                                  const Expr::TagList& momTags,
                                  const Expr::TagList& oldMomTags,
                                  const Expr::TagList& velTags,
                                  const Expr::TagList& velStarTags,
                                  const bool isConstDensity,
                                  const Expr::Tag densTag,
                                  const Expr::Tag densStarTag,
                                  const Expr::Tag dens2StarTag,
                                  const Wasatch::VarDenParameters varDenParams,
                                  const Expr::Tag divmomstarTag)
: ExpressionBuilder(results),
  isConstDens_( isConstDensity ),
  momTs_      ( densStarTag==Expr::Tag() ? Expr::TagList() : momTags     ),
  oldMomTags_ ( oldMomTags     ),
  velTs_      ( densStarTag==Expr::Tag() ? Expr::TagList() : velTags     ),
  velStarTs_  ( densStarTag==Expr::Tag() ? Expr::TagList() : velStarTags ),
  denst_      ( densTag       ),
  densStart_  ( densStarTag   ),
  dens2Start_ ( dens2StarTag  ),
  divmomstart_( divmomstarTag ),
  varDenParams_(varDenParams)
{}

//------------------------------------------------------------------

Expr::ExpressionBase*
PressureSource::Builder::build() const
{
  return new PressureSource( momTs_, oldMomTags_, velTs_, velStarTs_, isConstDens_, denst_, densStart_, dens2Start_, varDenParams_, divmomstart_ );
}
//------------------------------------------------------------------

