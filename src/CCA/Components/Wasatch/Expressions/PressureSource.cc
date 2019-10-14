 #include <CCA/Components/Wasatch/Expressions/PressureSource.h>
#include <CCA/Components/Wasatch/TimeIntegratorTools.h>
#include <CCA/Components/Wasatch/TagNames.h>

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>


PressureSource::PressureSource( const Expr::TagList& momTags,
                                const Expr::TagList& oldMomTags,
                                const Expr::TagList& velTags,
                                const Expr::Tag& divuTag,
                                const bool isConstDensity,
                                const Expr::Tag& rhoTag,
                                const Expr::Tag& rhoStarTag)
: Expr::Expression<SVolField>(),
  isConstDensity_( isConstDensity ),
  doX_      ( momTags[0]!=Expr::Tag() ),
  doY_      ( momTags[1]!=Expr::Tag() ),
  doZ_      ( momTags[2]!=Expr::Tag() ),
  is3d_     ( doX_ && doY_ && doZ_    )
{
  set_gpu_runnable( true );
  
  if( doX_ )  xMomOld_ = create_field_request<XVolField>(oldMomTags[0]);
  if( doY_ )  yMomOld_ = create_field_request<YVolField>(oldMomTags[1]);
  if( doZ_ )  zMomOld_ = create_field_request<ZVolField>(oldMomTags[2]);
  
  if (!isConstDensity_) {
    
    if( doX_ ) xMom_ = create_field_request<XVolField>(momTags[0]);
    if( doY_ ) yMom_ = create_field_request<YVolField>(momTags[1]);
    if( doZ_ ) zMom_ = create_field_request<ZVolField>(momTags[2]);
    
    rhoStar_ = create_field_request<SVolField>(rhoStarTag);
    divu_ = create_field_request<SVolField>(divuTag);
  } else {
    const Expr::Tag dilt = WasatchCore::TagNames::self().dilatation;
    dil_ = create_field_request<SVolField>(dilt);
  }
  
  rho_ = create_field_request<SVolField>(rhoTag);
  
  const Expr::Tag dtt = WasatchCore::TagNames::self().dt;
  dt_ = create_field_request<TimeField>(dtt);
  
  const Expr::Tag rkst = WasatchCore::TagNames::self().rkstage;
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
  
  timeIntInfo_ = opDB.retrieve_operator<WasatchCore::TimeIntegrator>();
  
  if (!isConstDensity_) {
    if( doX_ ){
      s2XInterpOp_ = opDB.retrieve_operator<S2XInterpOpT>();
      gradXOp_    = opDB.retrieve_operator<GradXT>();
      x2SInterpOp_ = opDB.retrieve_operator<X2SInterpOpT>();
    }
    if( doY_ ){
      s2YInterpOp_ = opDB.retrieve_operator<S2YInterpOpT>();
      gradYOp_    = opDB.retrieve_operator<GradYT>();
      y2SInterpOp_ = opDB.retrieve_operator<Y2SInterpOpT>();
    }
    if( doZ_ ){
      s2ZInterpOp_ = opDB.retrieve_operator<S2ZInterpOpT>();
      gradZOp_    = opDB.retrieve_operator<GradZT>();
      z2SInterpOp_ = opDB.retrieve_operator<Z2SInterpOpT>();
    }
  }
}

//------------------------------------------------------------------

void PressureSource::evaluate()
{
  using namespace SpatialOps;
  typedef typename Expr::Expression<SVolField>::ValVec SVolFieldVec;
  SVolFieldVec& results = this->get_value_vec();
  
  const TimeField& dt = dt_->field_ref();
  const TimeField& rkStage = rkStage_->field_ref();
  const SVolField& rho = rho_->field_ref();
  
  SVolField& psrc = *results[0];
  
  const WasatchCore::TimeIntegrator& timeIntInfo = *timeIntInfo_;
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
    
    SVolField& drhodtstar = *results[1];
    const SVolField& divu       = divu_->field_ref();    
    const SVolField& rhoStar = rhoStar_->field_ref();
    
    drhodtstar <<= (rhoStar - rho)/dt;
    
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
        psrc <<= (*gradXOp_)( xMomOld_->field_ref()/ (*s2XInterpOp_)(rhoStar) );
        if (timeIntInfo_->nStages > 1) divmomlatest <<= (*gradXOp_)(xMom_->field_ref());
      }
      else       psrc <<= 0.0;
      if( doY_ )
      {
        psrc <<= psrc + (*gradYOp_)( yMomOld_->field_ref()/ (*s2YInterpOp_)(rhoStar) );
        if (timeIntInfo_->nStages > 1) divmomlatest <<= divmomlatest + (*gradYOp_)(yMom_->field_ref());
      }
      if( doZ_ )
      {
        psrc <<= psrc + (*gradZOp_)( zMomOld_->field_ref() / (*s2ZInterpOp_)(rhoStar) );
        if (timeIntInfo_->nStages > 1) divmomlatest <<= divmomlatest + (*gradZOp_)(zMom_->field_ref());
      }
      
      psrc <<= cond( rkStage == 1.0,                  psrc         )
                   ( rkStage == 2.0, a2* psrc + b2 * divmomlatest  )
                   ( rkStage == 3.0, a3* psrc + b3 * divmomlatest  )
                   ( 0.0 ); // should never get here.
      
    } // 1D, 2D cases

    //
    psrc <<= cond( rkStage == 1.0, (psrc - divu)  / dt       )
                 ( rkStage == 2.0, (psrc - divu) / (b2 * dt) )
                 ( rkStage == 3.0, (psrc - divu) / (b3 * dt) )
                 ( 0.0 ); // should never get here.

  } // Variable density
}

//------------------------------------------------------------------

PressureSource::Builder::Builder( const Expr::TagList& results,
                                  const Expr::TagList& momTags,
                                  const Expr::TagList& oldMomTags,
                                  const Expr::TagList& velTags,
                                  const Expr::Tag& divuTag,
                                  const bool isConstDensity,
                                  const Expr::Tag& rhoTag,
                                  const Expr::Tag& rhoStarTag)
: ExpressionBuilder(results),
  isConstDens_( isConstDensity ),
  momTs_      ( rhoStarTag==Expr::Tag() ? Expr::TagList() : momTags ),
  oldMomTags_ ( oldMomTags ),
  velTs_      ( rhoStarTag==Expr::Tag() ? Expr::TagList() : velTags ),
  rhot_       ( rhoTag     ),
  rhoStart_   ( rhoStarTag ),
  divuTag_    ( divuTag    )
{}

//------------------------------------------------------------------

Expr::ExpressionBase*
PressureSource::Builder::build() const
{
  return new PressureSource( momTs_, oldMomTags_, velTs_, divuTag_, isConstDens_, rhot_, rhoStart_ );
}
//------------------------------------------------------------------

