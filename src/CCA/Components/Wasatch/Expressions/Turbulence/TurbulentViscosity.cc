/*
 * The MIT License
 *
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

//-- Wasatch Includes --//
#include <CCA/Components/Wasatch/StringNames.h>

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

#include <cmath>

//====================================================================

Expr::Tag turbulent_viscosity_tag()
{
  const Wasatch::StringNames& sName = Wasatch::StringNames::self();
  return Expr::Tag( sName.turbulentviscosity, Expr::STATE_NONE );
}

//====================================================================

TurbulentViscosity::
TurbulentViscosity( const Expr::Tag rhoTag,
                    const Expr::Tag strTsrSqTag,
                    const Expr::Tag waleTsrMagTag,
                    const Expr::Tag vremanTsrMagTag,
                    const Expr::Tag dynamicSmagCoefTag,
                    const Wasatch::TurbulenceParameters turbParams )
: Expr::Expression<SVolField>(),
  isConstSmag_(turbulenceParameters_.turbulenceModelName != Wasatch::DYNAMIC),
  turbulenceParameters_ ( turbParams          ),
  strTsrSqTag_          ( strTsrSqTag        ),
  waleTsrMagTag_        ( waleTsrMagTag      ),
  vremanTsrMagTag_      ( vremanTsrMagTag     ),
  dynCoefTag_           ( dynamicSmagCoefTag ),
  rhoTag_               ( rhoTag              )
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
    exprDeps.requires_expression( strTsrSqTag_ );  
  
  else if (turbulenceParameters_.turbulenceModelName == Wasatch::VREMAN)
    exprDeps.requires_expression( vremanTsrMagTag_ );  
  
  else if (turbulenceParameters_.turbulenceModelName == Wasatch::WALE) {
    exprDeps.requires_expression( strTsrSqTag_ );
    exprDeps.requires_expression( waleTsrMagTag_ );
  }
  
  else if( turbulenceParameters_.turbulenceModelName == Wasatch::DYNAMIC ) {
    exprDeps.requires_expression( strTsrSqTag_ );
    exprDeps.requires_expression( dynCoefTag_ );
  }
  
}

//--------------------------------------------------------------------

void
TurbulentViscosity::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldMgrSelector<SVolField>::type& scalarfm = fml.field_manager<SVolField>();

  rho_       = &scalarfm.field_ref( rhoTag_       );
  
  if ( turbulenceParameters_.turbulenceModelName == Wasatch::SMAGORINSKY )
      strTsrSq_ = &scalarfm.field_ref( strTsrSqTag_ );
  
  else if ( turbulenceParameters_.turbulenceModelName == Wasatch::WALE ) {
    strTsrSq_ = &scalarfm.field_ref( strTsrSqTag_ );
    waleTsrMag_ = &scalarfm.field_ref( waleTsrMagTag_ );    
  }
  
  else if( turbulenceParameters_.turbulenceModelName == Wasatch::DYNAMIC ) {
    strTsrSq_ = &scalarfm.field_ref( strTsrSqTag_ );
    dynCoef_ = &scalarfm.field_ref ( dynCoefTag_ );
  }
  
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
      result <<= *rho_ * mixingLengthSq  * sqrt(2.0 * *strTsrSq_) ; // rho * (Cs * delta)^2 * |S|, Cs is the Smagorinsky constant
      break;

    case Wasatch::DYNAMIC:
//      result <<= 0.0;
      // tsaad.Note: When the dynamic model is used, the DynamicSmagorinskyCoefficient expression calculates both the coefficient and the StrainTensorMagnitude. Unlike the StrainTensorMagnitude.cc Expression, which calculates SijSij instead. That's why for the constant smagorinsky case, we have take the sqrt() of that quanitity. In the Dynamic model case, we don't.
      result <<= *rho_ * *dynCoef_ * *strTsrSq_;//*rho_ * *dynCoef_ * sqrt(2.0 * *strTsrSq_);
      break;

    case Wasatch::WALE:
    {
      SpatFldPtr<SVolField> denom = SpatialFieldStore::get<SVolField>( result );
      *denom <<= 0.0;
      *denom <<= pow(*strTsrSq_, 2.5) + pow(*waleTsrMag_, 1.25);
      result <<= cond( *denom == 0.0, 0.0 )
                     ( *rho_ * mixingLengthSq * pow(*waleTsrMag_, 1.5) / *denom );
    }
      break;

    case Wasatch::VREMAN:
      // NOTE: the constant used in the Vreman model input corresponds to the
      // best Smagorinsky constant when using the constant Smagorinsky model
      // for the problem being simulated. The Vreman constant is estimated at Cv ~ 2.5 Cs
      result <<= *rho_ * 2.5 * mixingLengthSq  * *vremanTsrMag_ ; // rho * 2.5 * (Cs * delta)^2 * |V|, Cs is the Smagorinsky constant
      break;
      
    default:
      break;
      
  }
}
