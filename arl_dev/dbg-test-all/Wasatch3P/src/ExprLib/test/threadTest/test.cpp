#include "test.h"
#include <spatialops/Nebo.h>

RHSExpr::RHSExpr( const Expr::Tag& fluxTag,
                  const Expr::Tag& srcTag )
  : Expr::Expression<VolT>(),
    fluxTag_( fluxTag ),
    srcTag_ ( srcTag  ),
    doFlux_( fluxTag != Expr::Tag() ),
    doSrc_ ( srcTag  != Expr::Tag() )
{
#ifdef ENABLE_CUDA
  this->set_gpu_runnable( true );
#endif
  flux_ = NULL;
  src_  = NULL;
  div_  = NULL;
}

//--------------------------------------------------------------------

void
RHSExpr::advertise_dependents( Expr::ExprDeps& exprDeps )
{
  if( doFlux_ ) exprDeps.requires_expression( fluxTag_ );
  if( doSrc_  ) exprDeps.requires_expression( srcTag_  );
}

//--------------------------------------------------------------------

void
RHSExpr::bind_fields( const Expr::FieldManagerList& fml )
{
  if( doFlux_ )  flux_ = &fml.field_manager<XFluxT>().field_ref( fluxTag_ );
  if( doSrc_  )  src_  = &fml.field_manager<VolT  >().field_ref( srcTag_  );
}

//--------------------------------------------------------------------

void
RHSExpr::bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  if( doFlux_ )
    div_ = opDB.retrieve_operator<XDivT>();
}

//--------------------------------------------------------------------

void
RHSExpr::evaluate()
{
  using namespace SpatialOps;
  VolT& rhs = this->value();
  if( doFlux_ ){
    rhs <<= -(*div_)(*flux_);
  }
  else{
    rhs <<= 0.0;
  }
  if( doSrc_ ){
    rhs <<= rhs + *src_;
  }
}

//--------------------------------------------------------------------

Expr::ExpressionBase*
RHSExpr::Builder::
build() const
{
  return new RHSExpr( fluxTag_, srcTag_ );
}

//--------------------------------------------------------------------

RHSExpr::Builder::
Builder( const Expr::Tag& rhsTag,
         const Expr::Tag& fluxTag,
         const Expr::Tag& srcTag )
  : ExpressionBuilder(rhsTag),
    fluxTag_( fluxTag ),
    srcTag_ ( srcTag  )
{}

//--------------------------------------------------------------------


//====================================================================


//--------------------------------------------------------------------

FluxExpr::FluxExpr( const Expr::Tag& varTag,
                    const double diffCoef )
  : Expr::Expression<XFluxT>(),
    phiTag_( varTag ),
    diffCoef_( diffCoef )
{
#ifdef ENABLE_CUDA
  this->set_gpu_runnable( true );
#endif
  phi_  = NULL;
  grad_ = NULL;
}

//--------------------------------------------------------------------

void
FluxExpr::advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( phiTag_ );
}

//--------------------------------------------------------------------

void
FluxExpr::bind_fields( const Expr::FieldManagerList& fml )
{
  phi_ = &fml.field_manager<VolT  >().field_ref( phiTag_ );
}

//--------------------------------------------------------------------

void
FluxExpr::bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  grad_ = opDB.retrieve_operator<XGradT>();
}

//--------------------------------------------------------------------

void
FluxExpr::evaluate()
{
  using namespace SpatialOps;
  XFluxT& flux = value();
  flux <<= -diffCoef_ * (*grad_)(*phi_);
}

//--------------------------------------------------------------------

Expr::ExpressionBase*
FluxExpr::Builder::build() const
{
  return new FluxExpr( tag_, coef_ );
}

//--------------------------------------------------------------------

FluxExpr::Builder::
Builder( const Expr::Tag& fluxTag,
         const Expr::Tag& phiTag,
         const double diffCoef )
  : ExpressionBuilder(fluxTag),
    tag_ ( phiTag   ),
    coef_( diffCoef )
{}

//--------------------------------------------------------------------

//====================================================================

//--------------------------------------------------------------------

BusyWork::BusyWork( const Expr::Tag& varTag, const int nvar )
  : Expr::Expression<VolT>(),
    phiTag_( varTag ),
    nvar_( nvar )
{
  this->set_gpu_runnable( true );
  phi_  = NULL;
}

//--------------------------------------------------------------------

void
BusyWork::advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( phiTag_ );
}

//--------------------------------------------------------------------

void
BusyWork::bind_fields( const Expr::FieldManagerList& fml )
{
  phi_ = &fml.field_manager<VolT>().field_ref( phiTag_ );
}

//--------------------------------------------------------------------

void
BusyWork::evaluate()
{
  using namespace SpatialOps;
  VolT& val = value();

  for( int i=0; i<nvar_; ++i ){
    val <<= *phi_ / exp(double(i));
  }
}

//--------------------------------------------------------------------

Expr::ExpressionBase*
BusyWork::Builder::build() const
{
  return new BusyWork( tag_, nvar_ );
}

//--------------------------------------------------------------------

BusyWork::Builder::
Builder( const Expr::Tag& result, const Expr::Tag& phiTag, const int nvar )
: ExpressionBuilder(result),
  tag_( phiTag ),
  nvar_( nvar )
{}

//--------------------------------------------------------------------

//====================================================================

//--------------------------------------------------------------------

CoupledBusyWork::CoupledBusyWork( const Expr::Tag& varTag,
                                  const int nvar )
  : Expr::Expression<VolT>(),
    phiTag_( varTag ),
    nvar_( nvar )
{
#ifdef ENABLE_CUDA
  this->set_gpu_runnable( true );
#endif
  tmpVec_.resize( nvar, 0.0 );
}

//--------------------------------------------------------------------

void
CoupledBusyWork::advertise_dependents( Expr::ExprDeps& exprDeps )
{
  for( int i=0; i!=nvar_; ++i ){
    std::ostringstream nam;
    nam << "var_" << i;
    const std::string varnam = nam.str();
    exprDeps.requires_expression( Expr::Tag( varnam, Expr::STATE_NONE ) );
  }
}

//--------------------------------------------------------------------

void
CoupledBusyWork::bind_fields( const Expr::FieldManagerList& fml )
{
  phi_.clear();
  for( int i=0; i!=nvar_; ++i ){
    std::ostringstream nam;
    nam << "var_" << i;
    const std::string varnam = nam.str();
    phi_.push_back( &fml.field_manager<VolT>().field_ref( Expr::Tag(varnam,Expr::STATE_NONE) ) );
  }
}

//--------------------------------------------------------------------

void
CoupledBusyWork::evaluate()
{
  using namespace SpatialOps;
  VolT& val = value();
  val <<= 0.0;

  for( FieldVecT::const_iterator ifld=phi_.begin(); ifld!=phi_.end(); ++ifld ){
    val <<= val + exp(**ifld);
  }

  // NOTE: the following commented code is a mockup for point-wise calls to a
  //       third-party library.  For now, we aren't going to use this since it
  //       won't allow GPU execution via Nebo.

  //  // pack iterators into a vector
//  iterVec_.clear();
//  for( FieldVecT::const_iterator ifld=phi_.begin(); ifld!=phi_.end(); ++ifld ){
//    iterVec_.push_back( (*ifld)->begin() );
//  }
//
//  for( VolT::iterator ival=val.begin(); ival!=val.end(); ++ival ){
//    // unpack into temporary
//    tmpVec_.clear();
//    for( IterVec::const_iterator ii=iterVec_.begin(); ii!=iterVec_.end(); ++ii ){
//      tmpVec_.push_back( **ii );
//    }
//
//    double src=1.0;
//    for( std::vector<double>::const_iterator isrc=tmpVec_.begin(); isrc!=tmpVec_.end(); ++isrc ){
//      src += exp(*isrc);
//    }
//
//    *ival = src;
//
//    // advance iterators to next point
//    for( IterVec::iterator ii=iterVec_.begin(); ii!=iterVec_.end(); ++ii ){
//      ++(*ii);
//    }
//  }
}

//--------------------------------------------------------------------

Expr::ExpressionBase*
CoupledBusyWork::Builder::
build() const
{
  return new CoupledBusyWork( tag_, nvar_ );
}

//--------------------------------------------------------------------

CoupledBusyWork::Builder::
Builder( const Expr::Tag& result,
         const Expr::Tag& phiTag,
         const int nvar )
  : ExpressionBuilder(result),
    tag_( phiTag ),
    nvar_( nvar )
{}

//--------------------------------------------------------------------
