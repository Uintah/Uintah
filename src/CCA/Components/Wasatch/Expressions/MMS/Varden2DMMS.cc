#include "Varden2DMMS.h"

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>
#include <CCA/Components/Wasatch/TagNames.h>
#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif

//**********************************************************************
// OSCILLATING MMS: MIXTURE FRACTION SOURCE TERM
//**********************************************************************

template<typename FieldT>
VarDenMMSOscillatingMixFracSrc<FieldT>::
VarDenMMSOscillatingMixFracSrc( const Expr::Tag& xTag,
                                const Expr::Tag& yTag,
                               const Expr::Tag& tTag,
                               const double r0,
                               const double r1,
                               const double d,
                               const double w,
                               const double k,
                               const double uf,
                               const double vf,
                               const bool atNP1)
: Expr::Expression<FieldT>(),
r0_( r0 ),
r1_( r1 ),
d_ ( d ),
w_ ( w ),
k_ ( k ),
uf_ ( uf ),
vf_ ( vf ),
atNP1_(atNP1)
{
  this->set_gpu_runnable( true );
  x_ = this->template create_field_request<FieldT>(xTag);
  y_ = this->template create_field_request<FieldT>(yTag);
  t_ = this->template create_field_request<TimeField>(tTag);
  dt_ = this->template create_field_request<TimeField>(WasatchCore::TagNames::self().dt);
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDenMMSOscillatingMixFracSrc<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  
  const FieldT& x = x_->field_ref();
  const FieldT& y = y_->field_ref();
  
  
  SpatFldPtr<TimeField> ta = SpatialFieldStore::get<TimeField>( result );
  *ta <<= t_->field_ref();
  if (atNP1_) {
    *ta <<= t_->field_ref() + dt_->field_ref();
  }
  
  const TimeField& t = *ta;
  
  SpatFldPtr<FieldT> xh = SpatialFieldStore::get<FieldT>( x );
  SpatFldPtr<FieldT> yh = SpatialFieldStore::get<FieldT>( y );
  *xh <<= x - uf_ * t;
  *yh <<= y - vf_ * t;

  SpatFldPtr<FieldT> s0_ = SpatialFieldStore::get<FieldT>( result );
  SpatFldPtr<FieldT> s1_ = SpatialFieldStore::get<FieldT>( result );
  SpatFldPtr<FieldT> s2_ = SpatialFieldStore::get<FieldT>( result );
  SpatFldPtr<FieldT> s3_ = SpatialFieldStore::get<FieldT>( result );
  SpatFldPtr<FieldT> s4_ = SpatialFieldStore::get<FieldT>( result );
  SpatFldPtr<FieldT> s5_ = SpatialFieldStore::get<FieldT>( result );
  SpatFldPtr<FieldT> s6_ = SpatialFieldStore::get<FieldT>( result );
  SpatFldPtr<FieldT> s7_ = SpatialFieldStore::get<FieldT>( result );
  SpatFldPtr<FieldT> s10_ = SpatialFieldStore::get<FieldT>( result );
  SpatFldPtr<FieldT> s11_ = SpatialFieldStore::get<FieldT>( result );
  SpatFldPtr<FieldT> s12_ = SpatialFieldStore::get<FieldT>( result );
  
  FieldT& s0  = *s0_;
  FieldT& s1  = *s1_;
  FieldT& s2  = *s2_;
  FieldT& s3  = *s3_;
  FieldT& s4  = *s4_;
  FieldT& s5  = *s5_;
  FieldT& s6  = *s6_;
  FieldT& s7  = *s7_;
  FieldT& s10  = *s10_;
  FieldT& s11  = *s11_;
  FieldT& s12  = *s12_;


  s0 <<= cos(PI * w_ * t);
  s1 <<= sin(PI * w_ * t);
  s2 <<= sin(k_ * PI * (x - uf_ * t) );
  s3 <<= sin(k_ * PI * (y - vf_ * t) );

  s4 <<= cos(k_ * PI * (x - uf_ * t) );
  s5 <<= cos(k_ * PI * (y - vf_ * t) );
  
  s6 <<= sin(2.0*PI*w_*t);
  s7 <<= cos(2.0 * k_ * PI * (x - uf_ * t) );
  const double s8 = r0_ - r1_;
  const double s9 = r0_ + r1_;
  s10 <<= 8.0*d_*k_*k_*PI + s1 * s2 * s3 * s8 * w_;
  s11 <<= 4.0*d_*k_*k_*PI + s1*s2*s3*s8*w_;
  s12 <<= -5.0 + 3.0 * s7;
  
  result <<= -(PI*r0_*r1_*(pow(s0,2)*(pow(s3,2)*(2*s11*pow(s2,2) + s10*pow(s4,2))
             + s10*pow(s2,2)*pow(s5,2))*s8 - 8*d_*pow(k_,2)*PI*s0*s2*s3*s9 -
              (s9*(2*pow(s2,2)*pow(s5,2)*s6*s8 - s3*(s12*s3*s6*s8 + 8*s1*s2*s9))*w_)/4.))/
              (2.*pow(-(s0*s2*s3*s8) + s9,3));
}

//--------------------------------------------------------------------

template< typename FieldT >
VarDenMMSOscillatingMixFracSrc<FieldT>::Builder::
Builder( const Expr::Tag& result,
         const Expr::Tag& xTag,
         const Expr::Tag& yTag,
         const Expr::Tag& tTag,
        const double r0,
        const double r1,
        const double d,
        const double w,
        const double k,
        const double uf,
        const double vf,
        const bool atNP1)
: ExpressionBuilder(result),
r0_( r0 ),
r1_( r1 ),
d_ ( d ),
w_ ( w ),
k_ ( k ),
uf_ ( uf ),
vf_ ( vf ),
xTag_( xTag ),
yTag_( yTag ),
tTag_( tTag ),
atNP1_(atNP1)
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
VarDenMMSOscillatingMixFracSrc<FieldT>::Builder::
build() const
{
  return new VarDenMMSOscillatingMixFracSrc<FieldT>( xTag_, yTag_, tTag_, r0_, r1_, d_, w_, k_, uf_, vf_, atNP1_ );
}

//--------------------------------------------------------------------

template<typename FieldT>
VarDenOscillatingMMSMixFrac<FieldT>::
VarDenOscillatingMMSMixFrac( const Expr::Tag& xTag,
                 const Expr::Tag& yTag,
                 const Expr::Tag& tTag,
                 const double r0,
                 const double r1,
                 const double w,
                 const double k,
                 const double uf,
                 const double vf )
: Expr::Expression<FieldT>(),
r0_( r0 ),
r1_( r1 ),
w_ ( w ),
k_ ( k ),
uf_ ( uf ),
vf_ ( vf )
{
  this->set_gpu_runnable( true );
   x_ = this->template create_field_request<FieldT>(xTag);
   y_ = this->template create_field_request<FieldT>(yTag);
   t_ = this->template create_field_request<TimeField>(tTag);
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDenOscillatingMMSMixFrac<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  
  const FieldT& x = x_->field_ref();
  const FieldT& y = y_->field_ref();
  const TimeField& t = t_->field_ref();
  
  SpatFldPtr<FieldT> xh = SpatialFieldStore::get<FieldT>( x );
  SpatFldPtr<FieldT> yh = SpatialFieldStore::get<FieldT>( y );
  *xh <<= x - uf_ * t;
  *yh <<= y - vf_ * t;
  
  SpatFldPtr<FieldT> xb = SpatialFieldStore::get<FieldT>( x );
  SpatFldPtr<FieldT> yb = SpatialFieldStore::get<FieldT>( y );
  SpatFldPtr<TimeField> tb = SpatialFieldStore::get<TimeField>( t );
  *xb <<= PI * k_ * *xh;
  *yb <<= PI * k_ * *yh;
  *tb <<= PI * w_ * t;
  
  result <<= (sin(*xb) * sin(*yb) * cos(*tb) + 1.0) / ( (1.0-(r0_/r1_)) * sin(*xb) * sin(*yb) * cos(*tb) + (1.0+(r0_/r1_)));
}

//--------------------------------------------------------------------

template< typename FieldT >
VarDenOscillatingMMSMixFrac<FieldT>::Builder::
Builder( const Expr::Tag& result,
         const Expr::Tag& xTag,
         const Expr::Tag& yTag,
         const Expr::Tag& tTag,
        const double r0,
        const double r1,
        const double w,
        const double k,
        const double uf,
        const double vf )
: ExpressionBuilder(result),
r0_( r0 ),
r1_( r1 ),
w_ ( w ),
k_ ( k ),
uf_ ( uf ),
vf_ ( vf ),
xTag_( xTag ),
yTag_( yTag ),
tTag_( tTag )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
VarDenOscillatingMMSMixFrac<FieldT>::Builder::
build() const
{
  return new VarDenOscillatingMMSMixFrac<FieldT>( xTag_, yTag_, tTag_, r0_, r1_, w_, k_, uf_, vf_ );
}


//--------------------------------------------------------------------
//--------------------------------------------------------------------


template<typename FieldT>
DiffusiveConstant<FieldT>::
DiffusiveConstant( const Expr::Tag& rhoTag,
                   const double d)
: Expr::Expression<FieldT>(),
  d_(d)
{
  this->set_gpu_runnable( true );
  rho_ = this->template create_field_request<FieldT>(rhoTag);
}

//--------------------------------------------------------------------

template< typename FieldT >
void
DiffusiveConstant<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  
  result <<= d_ / rho_->field_ref();
}

//--------------------------------------------------------------------

template< typename FieldT >
DiffusiveConstant<FieldT>::Builder::
Builder( const Expr::Tag& result,
        const Expr::Tag& rhoTag,
        const double d)
: ExpressionBuilder(result),
  rhoTag_( rhoTag ),
  d_ (d)
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
DiffusiveConstant<FieldT>::Builder::
build() const
{
  return new DiffusiveConstant<FieldT>( rhoTag_, d_ );
}

//--------------------------------------------------------------------

// EXPLICIT INSTANTIATION
#include <CCA/Components/Wasatch/FieldTypes.h>
template class VarDenMMSOscillatingMixFracSrc<SVolField>;
template class VarDenOscillatingMMSMixFrac<SVolField>;
template class DiffusiveConstant<SVolField>;
