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
                         const Expr::Tag densTag,
                         const Expr::Tag densStarTag,
                         const Expr::Tag dens2StarTag,
                         const Expr::TagList& velTags,
                         const Expr::Tag& xTag,
                         const Expr::Tag& tTag,
                         const Expr::Tag& dtTag,
                         const Wasatch::VarDenParameters varDenParams)
: Expr::Expression<FieldT>(),
rho0_( rho0 ),
rho1_( rho1 ),
doX_( velTags[0]!=Expr::Tag() ),
doY_( velTags[1]!=Expr::Tag() ),
doZ_( velTags[2]!=Expr::Tag() ),
is3d_( doX_ && doY_ && doZ_ ),
a0_( varDenParams.alpha0 ),
model_( varDenParams.model ),
useOnePredictor_(varDenParams.onePredictor),
varDenParams_(varDenParams)
{
  this->set_gpu_runnable( true );
  x_ = this->template create_field_request<FieldT>(xTag);
  t_ = this->template create_field_request<TimeField>(tTag);
  dt_ = this->template create_field_request<TimeField>(dtTag);
  if (model_ != Wasatch::VarDenParameters::CONSTANT) {
    dens_ = this->template create_field_request<FieldT>(densTag);
    if (useOnePredictor_)  densStar_ = this->template create_field_request<SVolField>(densStarTag);
    else                   dens2Star_ = this->template create_field_request<SVolField>(dens2StarTag);
    if (doX_)  u_ = this->template create_field_request<XVolField>(velTags[0]);
    if (doY_)  v_ = this->template create_field_request<YVolField>(velTags[0]);
    if (doZ_)  w_ = this->template create_field_request<ZVolField>(velTags[0]);
  }
}

//--------------------------------------------------------------------
template< typename FieldT >
void
VarDen1DMMSContinuitySrc<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  if( doX_ ){
    gradXOp_       = opDB.retrieve_operator<GradXT>();
    x2SInterpOp_ = opDB.retrieve_operator<X2SInterpOpT>();
  }
  if( doY_ ){
    gradYOp_       = opDB.retrieve_operator<GradYT>();
    y2SInterpOp_ = opDB.retrieve_operator<Y2SInterpOpT>();
  }
  if( doZ_ ){
    gradZOp_       = opDB.retrieve_operator<GradZT>();
    z2SInterpOp_ = opDB.retrieve_operator<Z2SInterpOpT>();
  }
}

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
  
  
  SpatialOps::SpatFldPtr<SVolField> drhodtstar = SpatialOps::SpatialFieldStore::get<SVolField>( result );
  switch (model_) {
    case Wasatch::VarDenParameters::IMPULSE:
    case Wasatch::VarDenParameters::SMOOTHIMPULSE:
    case Wasatch::VarDenParameters::DYNAMIC:
    {
      const SVolField& dens = dens_->field_ref();
      if (useOnePredictor_)  *drhodtstar <<= (densStar_->field_ref()  - dens) / dt;
      else                   *drhodtstar <<= (dens2Star_->field_ref() - dens) / (2. * dt);
    }
      break;
    default:
      break;
  }
  
  SpatialOps::SpatFldPtr<SVolField> alpha = SpatialOps::SpatialFieldStore::get<SVolField>( result );
  
  switch (model_) {
    case Wasatch::VarDenParameters::CONSTANT:
      *alpha <<= a0_;
      break;
    case Wasatch::VarDenParameters::IMPULSE:
    {
      *alpha <<= cond(*drhodtstar == 0.0, 1.0)(a0_);
    }
      break;
    case Wasatch::VarDenParameters::SMOOTHIMPULSE:
    {
      const double c = varDenParams_.gaussWidth;
      *alpha <<= a0_ + (1.0 - a0_)*exp(- *drhodtstar * *drhodtstar/(2.0*c*c));
    }
      break;
    case Wasatch::VarDenParameters::DYNAMIC:
    {
      const SVolField& dens = dens_->field_ref();
      SpatialOps::SpatFldPtr<SVolField> velDotDensGrad = SpatialOps::SpatialFieldStore::get<SVolField>( result );
      
      if( is3d_ ){ // for 3D cases, inline the whole thing
        *velDotDensGrad <<= (*x2SInterpOp_)(u_->field_ref()) * (*gradXOp_)(dens) + (*y2SInterpOp_)(v_->field_ref()) * (*gradYOp_)(dens) + (*z2SInterpOp_)(w_->field_ref()) * (*gradZOp_)(dens);
      } else {
        // for 1D and 2D cases, we are not as efficient - add terms as needed...
        if( doX_ ) *velDotDensGrad <<= (*x2SInterpOp_)(u_->field_ref()) * (*gradXOp_)(dens);
        else       *velDotDensGrad <<= 0.0;
        if( doY_ ) *velDotDensGrad <<= *velDotDensGrad + (*y2SInterpOp_)(v_->field_ref()) * (*gradYOp_)(dens);
        if( doZ_ ) *velDotDensGrad <<= *velDotDensGrad + (*z2SInterpOp_)(w_->field_ref()) * (*gradZOp_)(dens);
      } // 1D, 2D cases
      *velDotDensGrad <<= abs(*velDotDensGrad);
      *alpha <<= cond(*drhodtstar == 0.0, 1.0)( (1.0 - a0_) * ((0.1 * *velDotDensGrad) / ( 0.1 * *velDotDensGrad + 1)) + a0_ );
    }
      //    case Wasatch::VarDenParameters::DYNAMIC:
      //    {
      //      SpatialOps::SpatFldPtr<SVolField> densGrad = SpatialOps::SpatialFieldStore::get<SVolField>( result );
      //      *densGrad <<= sqrt( (*gradXOp_)(*dens_) * (*gradXOp_)(*dens_) + (*gradYOp_)(*dens_) * (*gradYOp_)(*dens_) + (*gradZOp_)(*dens_) * (*gradZOp_)(*dens_));
      //
      //      //      alpha <<= 1.0 / ( 1.0 + exp(10- *densGrad));
      //      *alpha <<= 0.9*((0.1 * *densGrad) / ( 0.1 * *densGrad + 1))+0.1;
      //    }
      break;
    default:
      *alpha <<= 0.1;
      break;
  }
  
  result <<= *alpha *
  (
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
        const Expr::Tag densTag,
        const Expr::Tag densStarTag,
        const Expr::Tag dens2StarTag,
        const Expr::TagList& velTags,
        const Expr::Tag& xTag,
        const Expr::Tag& tTag,
        const Expr::Tag& timestepTag,
        const Wasatch::VarDenParameters varDenParams)
: ExpressionBuilder(result),
rho0_( rho0 ),
rho1_( rho1 ),
velTs_( velTags ),
densTag_(densTag),
densStarTag_(densStarTag),
dens2StarTag_(dens2StarTag),
xTag_( xTag ),
tTag_( tTag ),
timestepTag_( timestepTag ),
varDenParams_(varDenParams)
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
VarDen1DMMSContinuitySrc<FieldT>::Builder::
build() const
{
  return new VarDen1DMMSContinuitySrc<FieldT>( rho0_, rho1_, densTag_, densStarTag_, dens2StarTag_, velTs_, xTag_, tTag_, timestepTag_, varDenParams_ );
}

//--------------------------------------------------------------------

template<typename FieldT>
VarDen1DMMSPressureContSrc<FieldT>::
VarDen1DMMSPressureContSrc( const Expr::Tag continutySrcTag,
                           const Expr::Tag& dtTag)
: Expr::Expression<FieldT>()
{
  this->set_gpu_runnable( true );
  continutySrc_ = this->template create_field_request<FieldT>(continutySrcTag);
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
  
  result <<= continutySrc_->field_ref() / dt_->field_ref();
}

//--------------------------------------------------------------------

template< typename FieldT >
VarDen1DMMSPressureContSrc<FieldT>::Builder::
Builder( const Expr::Tag& result,
        const Expr::Tag continutySrcTag,
        const Expr::Tag& timestepTag )
: ExpressionBuilder(result),
continutySrcTag_( continutySrcTag ),
timestepTag_    ( timestepTag     )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
VarDen1DMMSPressureContSrc<FieldT>::Builder::
build() const
{
  return new VarDen1DMMSPressureContSrc<FieldT>( continutySrcTag_, timestepTag_ );
}

//--------------------------------------------------------------------

// EXPLICIT INSTANTIATION
#include <CCA/Components/Wasatch/FieldTypes.h>
template class VarDen1DMMSMixFracSrc<SVolField>;
template class VarDen1DMMSContinuitySrc<SVolField>;
template class VarDen1DMMSPressureContSrc<SVolField>;
