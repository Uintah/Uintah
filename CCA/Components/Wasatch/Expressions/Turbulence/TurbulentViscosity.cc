#include "TurbulentViscosity.h"

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

#include <math.h>

//====================================================================

TurbulentViscosity::TurbulentViscosity( const Expr::Tag rhoTag,
                    const Expr::Tag strTsrMagTag,
                    const Expr::Tag sqStrTsrMagTag,                                       
                    const Wasatch::TurbulenceParameters turbParams )
: Expr::Expression<SVolField>(),
  isConstSmag_(turbulenceParameters_.turbulenceModelName != Wasatch::DYNAMIC),
  turbulenceParameters_ ( turbParams ),
  rhoTag_        ( rhoTag            ),
  strTsrMagTag_  ( strTsrMagTag      ),
  sqStrTsrMagTag_( sqStrTsrMagTag    ),
  smagTag_       ( Expr::Tag() )
{}

//--------------------------------------------------------------------

TurbulentViscosity::
~TurbulentViscosity()
{}

//--------------------------------------------------------------------

void
TurbulentViscosity::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( rhoTag_ );
  exprDeps.requires_expression( strTsrMagTag_ );
  if (turbulenceParameters_.turbulenceModelName == Wasatch::WALE) 
    exprDeps.requires_expression( sqStrTsrMagTag_ );
  if( turbulenceParameters_.turbulenceModelName == Wasatch::DYNAMIC ) 
    exprDeps.requires_expression( smagTag_ );
}

//--------------------------------------------------------------------

void
TurbulentViscosity::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<SVolField>& scalarfm = fml.field_manager<SVolField>();

  rho_       = &scalarfm.field_ref( rhoTag_       );
  strTsrMag_ = &scalarfm.field_ref( strTsrMagTag_ );
  if ( turbulenceParameters_.turbulenceModelName == Wasatch::WALE )   
    sqStrTsrMag_ = &scalarfm.field_ref( sqStrTsrMagTag_ );
  if( turbulenceParameters_.turbulenceModelName == Wasatch::DYNAMIC ) 
    smag_ = &scalarfm.field_ref ( smagTag_ );
}

//--------------------------------------------------------------------

void
TurbulentViscosity::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  gradXOp_ = opDB.retrieve_operator< GradXT >();
  gradYOp_ = opDB.retrieve_operator< GradYT >();
  gradZOp_ = opDB.retrieve_operator< GradZT >();
}

//--------------------------------------------------------------------

void
TurbulentViscosity::
evaluate()
{
  using namespace SpatialOps;
  SVolField& result = this->value();
  result <<= 0.0;
  
  const double dx = 1.0 / std::abs( gradXOp_->get_plus_coef() );
  const double dy = 1.0 / std::abs( gradYOp_->get_plus_coef() );
  const double dz = 1.0 / std::abs( gradZOp_->get_plus_coef() );
  const double avgVol = pow(dx*dy*dz, 1.0/3.0);
  double mixingLengthSq = turbulenceParameters_.eddyViscosityConstant * avgVol * (1.0 - avgVol/turbulenceParameters_.kolmogorovScale);
  mixingLengthSq = mixingLengthSq * mixingLengthSq;
  //const double deltaSquared  = pow(dx * dy * dz, 2.0/3.0);
  //const double eddyViscConstSq = turbulenceParameters_.eddyViscosityConstant * turbulenceParameters_.eddyViscosityConstant;
  
  switch ( turbulenceParameters_.turbulenceModelName ) {
    case Wasatch::SMAGORINSKY:
      result <<= *rho_ * mixingLengthSq  * sqrt(2.0 * *strTsrMag_) ; // rho * delta^2 * |S|
      break;
    case Wasatch::DYNAMIC:
      std::cout << "WARNING: Dynamic smagorinsky model not implemented yet.\n";
      std::cout << "returning 0.0 for turbulent viscosity.\n";
      result <<= 0.0;
      break;
    case Wasatch::WALE:
      result <<= *rho_ * mixingLengthSq * pow(*sqStrTsrMag_, 1.5) / ( pow(*strTsrMag_, 2.5) + pow(*sqStrTsrMag_, 1.25) + 1e-15);
      break;
    default:
      break;
  }
}