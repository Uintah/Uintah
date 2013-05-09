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

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

#include <cmath>

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
  exOp_    = opDB.retrieve_operator< ExOpT  >();
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
  double mixingLengthSq = turbulenceParameters_.eddyViscosityConstant * avgVol; //* (1.0 - avgVol/turbulenceParameters_.kolmogorovScale);
  mixingLengthSq = mixingLengthSq * mixingLengthSq; // (C_s * Delta)^2
  const double eps = 2.0*std::numeric_limits<double>::epsilon();
  //const double deltaSquared  = pow(dx * dy * dz, 2.0/3.0);
  //const double eddyViscConstSq = turbulenceParameters_.eddyViscosityConstant * turbulenceParameters_.eddyViscosityConstant;

  switch ( turbulenceParameters_.turbulenceModelName ) {

    case Wasatch::SMAGORINSKY:
      result <<= *rho_ * mixingLengthSq  * sqrt(2.0 * *strTsrSq_) ; // rho * (Cs * delta)^2 * |S|, Cs is the Smagorinsky constant
      break;

    case Wasatch::DYNAMIC:
      // tsaad.Note: When the dynamic model is used, the DynamicSmagorinskyCoefficient expression calculates both the coefficient and the StrainTensorMagnitude = sqrt(2*Sij*Sij). Unlike the StrainTensorMagnitude.cc Expression, which calculates SijSij instead. That's why for the constant smagorinsky case, we have take the sqrt() of that quanitity. In the Dynamic model case, we don't.
      result <<= *rho_ * *dynCoef_ * *strTsrSq_;//*rho_ * *dynCoef_ * sqrt(2.0 * *strTsrSq_);
      break;

    case Wasatch::WALE:
    {
      SpatFldPtr<SVolField> denom = SpatialFieldStore::get<SVolField>( result );
      *denom <<= 0.0;
      *denom <<= pow(*strTsrSq_, 2.5) + pow(*waleTsrMag_, 1.25);
      result <<= cond( *denom <= eps, 0.0 )
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

  // extrapolate from interior cells to ptach boundaries (both process and physical boundaries. the latter is optional):
  // this is necessary to avoid problems when calculating the stress tensor where
  // a viscosity interpolant is needed. If problems arise due to this extrapolation,
  // you should consider performing an MPI communication on the turbulent viscosity
  // (i.e. cleave the turbulent viscosity from its parents). This can be done
  // in transport/MomentumTransportEquation.cc
  // Based on data that I collected, an MPI communication costs about twice as
  // much as the extrapolation in terms of speedup.
  // You may also need to skip extrapolation at physical boundaries. This is the
  // case when using Warches. With regular extrapolation, you may end up with
  // a negative value for the turbulent viscosity in the extra cell if the
  // first interior cell value is zero. To avoid this, you can turn on the "skipBCs" flag
  // when using apply_to_field, or specify a min value for the extraplated cells.
  exOp_->apply_to_field(result);
//  exOp_->apply_to_field(result,0.0);
}
