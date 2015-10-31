//
//  VardenMMS.cc
//  uintah-xcode-local
//
//  Created by Tony Saad on 11/1/13.
//
//

#include "VardenMMS.h"

//-- SpatialOps Includes --//
#include <spatialops/structured/FVStaggered.h>

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif

//--------------------------------------------------------------------

template<typename FieldT>
VarDen1DMMSMixFracSrc<FieldT>::
VarDen1DMMSMixFracSrc( const Expr::Tag& xTag,
                      const Expr::Tag& tTag,
                      const Expr::Tag& dtTag,
                      const double D,
                      const double rho0,
                      const double rho1,
                      const bool atNPlus1)
: Expr::Expression<FieldT>(),
d_   ( D    ),
rho0_( rho0 ),
rho1_( rho1 ),
atNPlus1_( atNPlus1 )
{
  this->set_gpu_runnable( true );
  x_ = this->template create_field_request<FieldT>(xTag);
  t_ = this->template create_field_request<TimeField>(tTag);
  dt_ = this->template create_field_request<TimeField>(dtTag);
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDen1DMMSMixFracSrc<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  const FieldT& x = x_->field_ref();
  const TimeField& t = t_->field_ref();
  
  SpatialOps::SpatFldPtr<TimeField> tNew_ = SpatialOps::SpatialFieldStore::get<TimeField>( result );
  if (atNPlus1_) {
    const TimeField& dt = dt_->field_ref();
    *tNew_ <<= t + dt;
  } else {
    *tNew_ <<= t;
  }
  
  const TimeField& tNew = *tNew_;
  
  result <<=
  (
   ( 5 * rho0_ * (rho1_ * rho1_) * exp( (5 * (x * x))/(tNew + 10.))
    * ( 1500. * d_ - 120. * tNew + 75. * (tNew * tNew) * (x * x)
       + 30. * (tNew * tNew * tNew) * (x * x) + 750 * d_ * tNew
       + 1560. * d_ * (tNew * tNew) + 750 * d_ * (tNew * tNew * tNew)
       + 60. * d_ * (tNew * tNew * tNew * tNew) - 1500 * d_ * (x * x)
       + 30. * tNew * (x * x) - 606. * (tNew * tNew) - 120. * (tNew * tNew * tNew)
       - 6. * (tNew * tNew * tNew * tNew) + 75 * (x * x)
       + 7500. * tNew * x * sin((2 * PI * x)/(3. * (tNew + 10.)))
       - 250. * PI * (tNew * tNew) * cos((2 * PI * x)/(3.*(tNew + 10.)))
       - 20. * PI * (tNew * tNew * tNew)*cos((2 * PI * x)/(3.*(tNew + 10.)))
       - 1500. * d_ * (tNew * tNew) * (x * x)
       - 600. * d_ * (tNew * tNew * tNew) * (x * x)
       + 3750. * (tNew * tNew) * x * sin((2 * PI * x)/(3. * (tNew + 10.)))
       + 300. * (tNew * tNew * tNew) * x * sin((2 * PI * x)/(3.*(tNew + 10.)))
       - 500. * PI * tNew * cos((2 * PI * x)/(3. * (tNew + 10.)))
       - 600. * d_ * tNew * (x * x) - 600
       )
    )/3
   +
   ( 250 * rho0_ * rho1_ * (rho0_ - rho1_) * (tNew + 10.) * (3 * d_ + 3 * d_ * (tNew * tNew)
                                                          - PI * tNew * cos(
                                                                         (2 * PI * x)/(3. * (tNew + 10.))))
    )/ 3
   )
  /
  (
   ( (tNew * tNew) + 1.)*((tNew + 10.) * (tNew + 10.))
   *
   (
    (5 * rho0_ - 5 * rho1_ + 5 * rho1_ * exp((5 * (x * x))/(tNew + 10.)) + 2 * rho1_ * tNew * exp((5 * (x * x))/(tNew + 10.)))
    *(5 * rho0_ - 5 * rho1_ + 5 * rho1_ * exp((5 * (x * x))/(tNew + 10.)) + 2 * rho1_ * tNew * exp((5 * (x * x))/(tNew + 10.)))
    )
   );
}

//--------------------------------------------------------------------

template< typename FieldT >
VarDen1DMMSMixFracSrc<FieldT>::Builder::
Builder( const Expr::Tag& result,
        const Expr::Tag& xTag,
        const Expr::Tag& tTag,
        const Expr::Tag& dtTag,
        const double D,
        const double rho0,
        const double rho1,
        const bool atNPlus1)
: ExpressionBuilder(result),
d_   ( D    ),
rho0_( rho0 ),
rho1_( rho1 ),
atNPlus1_(atNPlus1),
xTag_( xTag ),
tTag_( tTag ),
dtTag_(dtTag)
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
VarDen1DMMSMixFracSrc<FieldT>::Builder::
build() const
{
  return new VarDen1DMMSMixFracSrc<FieldT>( xTag_, tTag_, dtTag_, d_, rho0_, rho1_, atNPlus1_ );
}

//--------------------------------------------------------------------

template<typename FieldT>
VarDen1DMMSContinuitySrc<FieldT>::
VarDen1DMMSContinuitySrc( const double rho0,
                         const double rho1,
                         const Expr::Tag& xTag,
                         const Expr::Tag& tTag,
                         const Expr::Tag& dtTag )
: Expr::Expression<FieldT>(),
rho0_( rho0 ),
rho1_( rho1 )
{
  this->set_gpu_runnable( true );
  x_ = this->template create_field_request<FieldT>(xTag);
  t_ = this->template create_field_request<TimeField>(tTag);
  dt_ = this->template create_field_request<TimeField>(dtTag);
}

//--------------------------------------------------------------------
template< typename FieldT >
void
VarDen1DMMSContinuitySrc<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDen1DMMSContinuitySrc<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  const TimeField& time = t_->field_ref();
  const TimeField& dt = dt_->field_ref();
  
  const FieldT& x = x_->field_ref();
  
  SpatFldPtr<TimeField> t = SpatialFieldStore::get<TimeField>( time );
  *t <<= time + dt;
  
  result <<=
  - (
   (
    ( 10/( exp((5 * (x * x))/( *t + 10)) * ((2 * *t + 5) * (2 * *t + 5)) )
     - (25 * (x * x))/(exp(( 5 * (x * x))/(*t + 10)) * (2 * *t + 5) * ((*t + 10) * (*t + 10)) )
     ) / rho0_
    - 10/( rho1_ * exp((5 * (x * x))/(*t + 10)) * ((2 * *t + 5) * (2 * *t + 5)) )
    + (25 * (x * x)) / (rho1_ * exp((5 * (x * x))/(*t + 10)) * (2 * *t + 5) * ((*t + 10) * (*t + 10)) )
    )
   /
   (
    ( (5 / (exp((5 * (x * x))/(*t + 10)) * (2 * *t + 5)) - 1)/rho0_
     - 5/(rho1_ * exp((5 * (x * x))/(*t + 10)) * (2 * *t + 5))
     )
    *
    ((5 / (exp((5 * (x * x))/(*t + 10)) * (2 * *t + 5)) - 1)/rho0_ - 5/(rho1_ * exp((5 * (x * x))/(*t + 10)) * (2 * *t + 5)))
    )
   -
   ( 5 * *t * sin((2 * PI * x)/(3 * *t + 30)) *
    ((50 * x) / (rho0_ * exp((5 * (x * x))/(*t + 10)) * (2 * *t + 5) * (*t + 10)) - (50 * x)/(rho1_ * exp((5 * (x * x))/(*t + 10)) * (2 * *t + 5) * (*t + 10)))
    )
   /
   ( ( (*t * *t) + 1.)
    * ( ((5/(exp((5 * (x * x))/(*t + 10)) * (2 * *t + 5)) - 1)/rho0_ - 5/(rho1_ * exp((5 * (x * x))/(*t + 10)) * (2 * *t + 5)))
       * ((5/(exp((5 * (x * x))/(*t + 10)) * (2 * *t + 5)) - 1)/rho0_ - 5/(rho1_ * exp((5 * (x * x))/(*t + 10)) * (2 * *t + 5)))
       )
    )
   -
   ( 10 * PI * *t * cos((2 * PI * x)/(3 * *t + 30)))
   /
   ( (3 * *t + 30) * ((*t * *t) + 1.)
    * ( (5/(exp((5 * (x * x))/(*t + 10)) * (2 * *t + 5)) - 1)/rho0_
       - 5/(rho1_ * exp((5 * (x * x))/(*t + 10)) * (2 * *t + 5))
       )
    )
   );
}

//--------------------------------------------------------------------

template< typename FieldT >
VarDen1DMMSContinuitySrc<FieldT>::Builder::
Builder( const Expr::Tag& result,
        const double rho0,
        const double rho1,
        const Expr::Tag& xTag,
        const Expr::Tag& tTag,
        const Expr::Tag& dtTag )
: ExpressionBuilder(result),
rho0_( rho0 ),
rho1_( rho1 ),
xTag_( xTag ),
tTag_( tTag ),
dtTag_( dtTag )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
VarDen1DMMSContinuitySrc<FieldT>::Builder::
build() const
{
  return new VarDen1DMMSContinuitySrc<FieldT>( rho0_, rho1_, xTag_, tTag_, dtTag_ );
}

//--------------------------------------------------------------------

template<typename FieldT>
VarDen1DMMSPressureContSrc<FieldT>::
VarDen1DMMSPressureContSrc( const Expr::Tag continutySrcTag,
                           const Expr::Tag rhoStarTag,
                           const Expr::Tag fStarTag,
                           const Expr::Tag dRhoDfStarTag,
                           const Expr::Tag& dtTag)
: Expr::Expression<FieldT>()
{
  this->set_gpu_runnable( true );
  continutySrc_ = this->template create_field_request<FieldT>(continutySrcTag);
  rhoStar_ = this->template create_field_request<FieldT>(rhoStarTag);
  fStar_ = this->template create_field_request<FieldT>(fStarTag);
  dRhoDfStar_ = this->template create_field_request<FieldT>(dRhoDfStarTag);
  dt_ = this->template create_field_request<TimeField>(dtTag);
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDen1DMMSPressureContSrc<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  const SVolField& f = fStar_->field_ref();
  const SVolField& rho = rhoStar_->field_ref();
  const SVolField& dRhoDf = dRhoDfStar_->field_ref();
  const SVolField& Srho = continutySrc_->field_ref();
  result <<= 1.0/rho/rho*dRhoDf*f*Srho + 1.0/rho*Srho;
}

//--------------------------------------------------------------------

template< typename FieldT >
VarDen1DMMSPressureContSrc<FieldT>::Builder::
Builder( const Expr::Tag& result,
        const Expr::Tag continutySrcTag,
        const Expr::Tag rhoStarTag,
        const Expr::Tag fStarTag,
        const Expr::Tag dRhoDfStarTag,
        const Expr::Tag& timestepTag )
: ExpressionBuilder(result),
continutySrcTag_( continutySrcTag ),
rhoStarTag_(rhoStarTag),
fStarTag_(fStarTag),
dRhoDfStarTag_(dRhoDfStarTag),
dtTag_    ( timestepTag     )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
VarDen1DMMSPressureContSrc<FieldT>::Builder::
build() const
{
  return new VarDen1DMMSPressureContSrc<FieldT>( continutySrcTag_, rhoStarTag_, fStarTag_, dRhoDfStarTag_, dtTag_ );
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------

template<typename FieldT>
VarDenEOSCouplingMixFracSrc<FieldT>::
VarDenEOSCouplingMixFracSrc( const Expr::Tag mixFracSrcTag,
                           const Expr::Tag rhoStarTag,
                           const Expr::Tag dRhoDfStarTag)
: Expr::Expression<FieldT>()
{
  this->set_gpu_runnable( true );
  mixFracSrc_ = this->template create_field_request<FieldT>(mixFracSrcTag);
  rhoStar_ = this->template create_field_request<FieldT>(rhoStarTag);
  dRhoDfStar_ = this->template create_field_request<FieldT>(dRhoDfStarTag);
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDenEOSCouplingMixFracSrc<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  const SVolField& rho = rhoStar_->field_ref();
  const SVolField& dRhoDf = dRhoDfStar_->field_ref();
  const SVolField& Sf = mixFracSrc_->field_ref();
  result <<= -1.0/rho/rho*dRhoDf*Sf;
}

//--------------------------------------------------------------------

template< typename FieldT >
VarDenEOSCouplingMixFracSrc<FieldT>::Builder::
Builder( const Expr::Tag& result,
        const Expr::Tag mixFracSrcTag,
        const Expr::Tag rhoStarTag,
        const Expr::Tag dRhoDfStarTag )
: ExpressionBuilder(result),
mixFracSrcTag_( mixFracSrcTag ),
rhoStarTag_(rhoStarTag),
dRhoDfStarTag_(dRhoDfStarTag)
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
VarDenEOSCouplingMixFracSrc<FieldT>::Builder::
build() const
{
  return new VarDenEOSCouplingMixFracSrc<FieldT>( mixFracSrcTag_, rhoStarTag_, dRhoDfStarTag_ );
}

//--------------------------------------------------------------------

// EXPLICIT INSTANTIATION
#include <CCA/Components/Wasatch/FieldTypes.h>
template class VarDen1DMMSMixFracSrc<SVolField>;
template class VarDen1DMMSContinuitySrc<SVolField>;
template class VarDen1DMMSPressureContSrc<SVolField>;
template class VarDenEOSCouplingMixFracSrc<SVolField>;
