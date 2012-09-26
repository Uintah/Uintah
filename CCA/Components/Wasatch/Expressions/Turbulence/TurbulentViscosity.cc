/*
 * Copyright (c) 2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "TurbulentViscosity.h"

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

#include <cmath>

//====================================================================

Expr::Tag turbulent_viscosity_tag()
{
  return Expr::Tag( "TurbulentViscosity", Expr::STATE_NONE );
}

//====================================================================

TurbulentViscosity::
TurbulentViscosity( const Expr::Tag rhoTag,
                    const Expr::Tag strTsrMagTag,
                    const Expr::Tag sqStrTsrMagTag,
                    const Expr::Tag vremanTsrMagTag,                   
                    const Wasatch::TurbulenceParameters turbParams )
: Expr::Expression<SVolField>(),
  isConstSmag_(turbulenceParameters_.turbulenceModelName != Wasatch::DYNAMIC),
  turbulenceParameters_ ( turbParams ),
  strTsrMagTag_  ( strTsrMagTag      ),
  sqStrTsrMagTag_( sqStrTsrMagTag    ),
  vremanTsrMagTag_( vremanTsrMagTag  ),
  smagTag_       ( Expr::Tag()       ),
  rhoTag_        ( rhoTag            )
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
  if (turbulenceParameters_.turbulenceModelName == Wasatch::SMAGORINSKY)
    exprDeps.requires_expression( strTsrMagTag_ );  
  
  else if (turbulenceParameters_.turbulenceModelName == Wasatch::VREMAN)
    exprDeps.requires_expression( vremanTsrMagTag_ );  
  
  else if (turbulenceParameters_.turbulenceModelName == Wasatch::WALE) {
    exprDeps.requires_expression( strTsrMagTag_ );    
    exprDeps.requires_expression( sqStrTsrMagTag_ );
  }
  
  else if( turbulenceParameters_.turbulenceModelName == Wasatch::DYNAMIC )
    exprDeps.requires_expression( smagTag_ );
}

//--------------------------------------------------------------------

void
TurbulentViscosity::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldMgrSelector<SVolField>::type& scalarfm = fml.field_manager<SVolField>();

  rho_       = &scalarfm.field_ref( rhoTag_       );
  if ( turbulenceParameters_.turbulenceModelName == Wasatch::SMAGORINSKY )
      strTsrMag_ = &scalarfm.field_ref( strTsrMagTag_ );
  
  else if ( turbulenceParameters_.turbulenceModelName == Wasatch::WALE ) {
    strTsrMag_ = &scalarfm.field_ref( strTsrMagTag_ );
    sqStrTsrMag_ = &scalarfm.field_ref( sqStrTsrMagTag_ );    
  }
  
  else if( turbulenceParameters_.turbulenceModelName == Wasatch::DYNAMIC )
    smag_ = &scalarfm.field_ref ( smagTag_ );
  
  else if( turbulenceParameters_.turbulenceModelName == Wasatch::VREMAN )
    vremanTsrMag_ = &scalarfm.field_ref( vremanTsrMagTag_ );    
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
  const double avgVol = std::pow(dx*dy*dz, 1.0/3.0);
  double mixingLengthSq = turbulenceParameters_.eddyViscosityConstant * avgVol * (1.0 - avgVol/turbulenceParameters_.kolmogorovScale);
  mixingLengthSq = mixingLengthSq * mixingLengthSq;
  //const double deltaSquared  = pow(dx * dy * dz, 2.0/3.0);
  //const double eddyViscConstSq = turbulenceParameters_.eddyViscosityConstant * turbulenceParameters_.eddyViscosityConstant;

  switch ( turbulenceParameters_.turbulenceModelName ) {

    case Wasatch::SMAGORINSKY:
      result <<= *rho_ * mixingLengthSq  * sqrt(2.0 * *strTsrMag_) ; // rho * (Cs * delta)^2 * |S|
      break;

    case Wasatch::DYNAMIC:
      std::cout << "WARNING: Dynamic smagorinsky model not implemented yet.\n";
      std::cout << "returning 0.0 for turbulent viscosity.\n";
      result <<= 0.0;
      break;

    case Wasatch::WALE:
      result <<= mixingLengthSq * pow(*sqStrTsrMag_, 1.5) / ( pow(*strTsrMag_, 2.5) + pow(*sqStrTsrMag_, 1.25) + 1e-15);
      break;

    case Wasatch::VREMAN:
      result <<= *rho_ * 2.5 * mixingLengthSq  * *vremanTsrMag_ ; // rho * 2.5 * (Cs * delta)^2 * |S|
      break;
      
    default:
      break;
      
  }
}
