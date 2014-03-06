#include <CCA/Components/Wasatch/Expressions/PressureSource.h>
#include <CCA/Components/Wasatch/TagNames.h>

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>


PressureSource::PressureSource( const Expr::TagList& momTags,
                                const Expr::TagList& velStarTags,
                                const bool isConstDensity,
                                const Expr::Tag densTag,
                                const Expr::Tag densStarTag,
                                const Expr::Tag dens2StarTag )
: Expr::Expression<SVolField>(),
  isConstDensity_( isConstDensity ),
  doX_      ( momTags[0]!=Expr::Tag() ),
  doY_      ( momTags[1]!=Expr::Tag() ),
  doZ_      ( momTags[2]!=Expr::Tag() ),
  is3d_     ( doX_ && doY_ && doZ_    ),
  xMomt_      ( densStarTag==Expr::Tag() ? Expr::Tag() : momTags[0]     ),
  yMomt_      ( densStarTag==Expr::Tag() ? Expr::Tag() : momTags[1]     ),
  zMomt_      ( densStarTag==Expr::Tag() ? Expr::Tag() : momTags[2]     ),
  xVelStart_  ( densStarTag==Expr::Tag() ? Expr::Tag() : velStarTags[0] ),
  yVelStart_  ( densStarTag==Expr::Tag() ? Expr::Tag() : velStarTags[1] ),
  zVelStart_  ( densStarTag==Expr::Tag() ? Expr::Tag() : velStarTags[2] ),
  denst_      ( densTag  ),
  densStart_  ( densStarTag==Expr::Tag() ? Expr::Tag() : densStarTag    ),
  dens2Start_ ( densStarTag==Expr::Tag() ? Expr::Tag() : dens2StarTag   ),
  dilt_       ( Wasatch::TagNames::self().dilatation ),
  timestept_  ( Wasatch::TagNames::self().dt )
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
  typedef std::vector<SVolField*> SVolFieldVec;
  SVolFieldVec& results = this->get_value_vec();
  
  SVolField& psrc = *results[0];

  if( isConstDensity_ ){
    
    psrc <<= *dens_ * *dil_ / *timestep_;
    
  } else { // variable density
    
    SVolField& drhodt     = *results[1];
    SVolField& alpha      = *results[2];
//    SVolField& beta       = *results[3];
    SVolField& divmomstar = *results[4];
    SVolField& drhodtstar = *results[5];

    drhodtstar <<= (*dens2Star_ - *dens_)/(2. * *timestep_);
    
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

    alpha <<= 0.1; // use this for the moment until we figure out the proper model for alpha

    drhodt <<= alpha * drhodtstar - (1.0 - alpha) * divmomstar;
    
    if( is3d_ ){ // for 3D cases, inline the whole thing
      psrc <<=    (*gradXOp_)(*xMom_) + (*gradYOp_)(*yMom_) + (*gradZOp_)(*zMom_) ;
    } else {
      // for 1D and 2D cases, we are not as efficient - add terms as needed...
      if( doX_ ) psrc <<=        (*gradXOp_)( (*xMom_) );
      else       psrc <<= 0.0;
      if( doY_ ) psrc <<= psrc + (*gradYOp_)( (*yMom_) );
      if( doZ_ ) psrc <<= psrc + (*gradZOp_)( (*zMom_) );
    } // 1D, 2D cases
    
    psrc <<= (psrc + drhodt)/ *timestep_;  // P_src = ( div(mom) + drhodt ) / dt
  } // Variable density
//  int count=1;
//  for( SVolField::interior_iterator iStrTsr= psrc.begin(); iStrTsr!=psrc.end(); ++iStrTsr, ++count)
//  std::cout <<count << " : Psource: " << *iStrTsr << std::endl;
  
}

//------------------------------------------------------------------

PressureSource::Builder::Builder( const Expr::TagList& results,
                                  const Expr::TagList& momTags,
                                  const Expr::TagList& velStarTags,
                                  const bool isConstDensity,
                                  const Expr::Tag densTag,
                                  const Expr::Tag densStarTag,
                                  const Expr::Tag dens2StarTag )
: ExpressionBuilder(results),
  isConstDens_( isConstDensity ),
  momTs_      ( densStarTag==Expr::Tag() ? Expr::TagList() : momTags     ),
  velStarTs_  ( densStarTag==Expr::Tag() ? Expr::TagList() : velStarTags ),
  denst_     ( densTag      ),
  densStart_ ( densStarTag  ),
  dens2Start_( dens2StarTag )
{}

//------------------------------------------------------------------

Expr::ExpressionBase*
PressureSource::Builder::build() const
{
  return new PressureSource( momTs_, velStarTs_, isConstDens_, denst_, densStart_, dens2Start_ );
}
//------------------------------------------------------------------

