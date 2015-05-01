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
                                const Expr::Tag& densTag,
                                const Expr::Tag& densStarTag,
                                const Expr::Tag& dens2StarTag,
                                const Wasatch::VarDenParameters varDenParams,
                                const Expr::Tag& divmomstarTag)
: Expr::Expression<SVolField>(),
  isConstDensity_( isConstDensity ),
  doX_      ( momTags[0]!=Expr::Tag() ),
  doY_      ( momTags[1]!=Expr::Tag() ),
  doZ_      ( momTags[2]!=Expr::Tag() ),
  is3d_     ( doX_ && doY_ && doZ_    ),
  useOnePredictor_(varDenParams.onePredictor),
  a0_   (varDenParams.alpha0),
  model_(varDenParams.model )
{
  set_gpu_runnable( true );
  
  if( doX_ )  xMomOld_ = create_field_request<XVolField>(oldMomTags[0]);
  if( doY_ )  yMomOld_ = create_field_request<YVolField>(oldMomTags[1]);
  if( doZ_ )  zMomOld_ = create_field_request<ZVolField>(oldMomTags[2]);
  
  if (!isConstDensity_) {
    
    if( doX_ ){
       xMom_ = create_field_request<XVolField>(momTags[0]);
       xVel_ = create_field_request<XVolField>(velTags[0]);
       uStar_ = create_field_request<XVolField>(velStarTags[0]);
    }
    if( doY_ ){
       yMom_ = create_field_request<YVolField>(momTags[1]);
       yVel_ = create_field_request<YVolField>(velTags[1]);
       vStar_ = create_field_request<YVolField>(velStarTags[1]);
    }
    if( doZ_ ){
       zMom_ = create_field_request<ZVolField>(momTags[2]);
       zVel_ = create_field_request<ZVolField>(velTags[2]);
       wStar_ = create_field_request<ZVolField>(velStarTags[2]);
    }
     densStar_ = create_field_request<SVolField>(densStarTag);

    // if we are using more than one predictor then we will need rho**
    if (!useOnePredictor_)  dens2Star_ = create_field_request<SVolField>(dens2StarTag);
     divmomstar_ = create_field_request<SVolField>(divmomstarTag);
  }
  else {
    const Expr::Tag dilt = Wasatch::TagNames::self().dilatation;
     dil_ = create_field_request<SVolField>(dilt);
  }
   dens_ = create_field_request<SVolField>(densTag);
  const Expr::Tag dtt = Wasatch::TagNames::self().dt;
   dt_ = create_field_request<TimeField>(dtt);
  const Expr::Tag rkst = Wasatch::TagNames::self().rkstage;
   rkStage_ = create_field_request<TimeField>(rkst);
}

//------------------------------------------------------------------

PressureSource::~PressureSource()
{}

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
  
  const TimeField& dt = dt_->field_ref();
  const TimeField& rkStage = rkStage_->field_ref();
  const SVolField& rho = dens_->field_ref();
  
  
  SVolField& psrc = *results[0];

  
  const Wasatch::TimeIntegrator& timeIntInfo = *timeIntInfo_;
  const double a2 = timeIntInfo.alpha[1];
  const double a3 = timeIntInfo.alpha[2];
  
  const double b2 = timeIntInfo.beta[1];
  const double b3 = timeIntInfo.beta[2];

  if( isConstDensity_ ) {
    const SVolField& dil = dil_->field_ref();

    if( is3d_ ){ // for 3D cases, inline the whole thing
      const XVolField& rUOld = xMomOld_->field_ref();
      const YVolField& rVOld = yMomOld_->field_ref();
      const ZVolField& rWOld = zMomOld_->field_ref();
      psrc <<= cond( rkStage > 1.0, (*gradXOp_)(rUOld) + (*gradYOp_)(rVOld) + (*gradZOp_)(rWOld) )
                   ( 0.0 );
    } else {
      // for 1D and 2D cases, we are not as efficient - add terms as needed...
      if( doX_ ) psrc <<=        (*gradXOp_)( xMomOld_->field_ref() );
      else       psrc <<= 0.0;
      if( doY_ ) psrc <<= psrc + (*gradYOp_)( yMomOld_->field_ref() );
      if( doZ_ ) psrc <<= psrc + (*gradZOp_)( zMomOld_->field_ref() );
    } // 1D, 2D cases

    psrc <<= cond( rkStage == 1.0,                  rho * dil  / dt        )
                 ( rkStage == 2.0, (a2* psrc + b2 * rho * dil) / (b2 * dt) )
                 ( rkStage == 3.0, (a3* psrc + b3 * rho * dil) / (b3 * dt) )
                 ( 0.0 ); // should never get here.
    
  } else { // variable density
    
    SVolField& drhodt     = *results[1];
    SVolField& alpha      = *results[2];
//    SVolField& beta       = *results[3];
    SVolField& drhodtstar = *results[4];

    const SVolField& rhoStar = densStar_->field_ref();
    const SVolField& divMomStar = divmomstar_->field_ref();

    if (useOnePredictor_) drhodtstar <<= (rhoStar - rho) / dt;
    else                  drhodtstar <<= (dens2Star_->field_ref() - rho) / (2.0 * dt);
    
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
          const XVolField& u = xVel_->field_ref();
          const YVolField& v = yVel_->field_ref();
          const ZVolField& w = zVel_->field_ref();
          *velDotDensGrad <<= (*x2SInterpOp_)(u) * (*gradXSOp_)(rho) + (*y2SInterpOp_)(v) * (*gradYSOp_)(rho) + (*z2SInterpOp_)(w) * (*gradZSOp_)(rho);
        } else {
          // for 1D and 2D cases, we are not as efficient - add terms as needed...
          if( doX_ ) *velDotDensGrad <<= (*x2SInterpOp_)(xVel_->field_ref()) * (*gradXSOp_)(rho);
          else       *velDotDensGrad <<= 0.0;
          if( doY_ ) *velDotDensGrad <<= *velDotDensGrad + (*y2SInterpOp_)(yVel_->field_ref()) * (*gradYSOp_)(rho);
          if( doZ_ ) *velDotDensGrad <<= *velDotDensGrad + (*z2SInterpOp_)(zVel_->field_ref()) * (*gradZSOp_)(rho);
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

    drhodt <<= alpha * drhodtstar - (1.0 - alpha) * divMomStar;
    
    SpatFldPtr<SVolField> divmomlatest_ = SpatialFieldStore::get<SVolField>( psrc );
    SVolField& divmomlatest = *divmomlatest_;
    
    if( is3d_ ){ // for 3D cases, inline the whole thing
      // always add the divergence of momentum from the old timestep div(r u)^n
      const XVolField& rUOld = xMomOld_->field_ref();
      const YVolField& rVOld = yMomOld_->field_ref();
      const ZVolField& rWOld = zMomOld_->field_ref();
      const XVolField& rU = xMom_->field_ref();
      const YVolField& rV = yMom_->field_ref();
      const ZVolField& rW = zMom_->field_ref();

      psrc <<= (*gradXOp_)(rUOld) + (*gradYOp_)(rVOld) + (*gradZOp_)(rWOld);
      // if we are at an rkstage > 1, then add the divergence of the most recent momentum
      psrc <<= cond( rkStage == 1.0,                  psrc                                                             )
                   ( rkStage == 2.0, a2* psrc + b2 * ((*gradXOp_)(rU) + (*gradYOp_)(rV) + (*gradZOp_)(rW)) )
                   ( rkStage == 3.0, a3* psrc + b3 * ((*gradXOp_)(rU) + (*gradYOp_)(rV) + (*gradZOp_)(rW)) )
                   ( 0.0 ); // should never get here.

    } else {
      // for 1D and 2D cases, we are not as efficient - add terms as needed...
      if( doX_ ) {
        // always add the divergence of momentum from the old timestep div(r u)^n
        psrc <<= (*gradXOp_)( xMomOld_->field_ref() );
        if (timeIntInfo_->nStages > 1) divmomlatest <<= (*gradXOp_)(xMom_->field_ref());
      }
      else       psrc <<= 0.0;
      if( doY_ )
      {
        psrc <<= psrc + (*gradYOp_)( yMomOld_->field_ref() );
        if (timeIntInfo_->nStages > 1) divmomlatest <<= divmomlatest + (*gradYOp_)(yMom_->field_ref());
      }
      if( doZ_ )
      {
        psrc <<= psrc + (*gradZOp_)( zMomOld_->field_ref() );
        if (timeIntInfo_->nStages > 1) divmomlatest <<= divmomlatest + (*gradZOp_)(zMom_->field_ref());
      }
      
      psrc <<= cond( rkStage == 1.0,                  psrc         )
                   ( rkStage == 2.0, a2* psrc + b2 * divmomlatest  )
                   ( rkStage == 3.0, a3* psrc + b3 * divmomlatest  )
                   ( 0.0 ); // should never get here.
      
    } // 1D, 2D cases

    //
    psrc <<= cond( rkStage == 1.0, (psrc + drhodt)  / dt       )
                 ( rkStage == 2.0, (psrc + drhodt) / (b2 * dt) )
                 ( rkStage == 3.0, (psrc + drhodt) / (b3 * dt) )
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
                                  const Expr::Tag& densTag,
                                  const Expr::Tag& densStarTag,
                                  const Expr::Tag& dens2StarTag,
                                  const Wasatch::VarDenParameters varDenParams,
                                  const Expr::Tag& divmomstarTag)
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

