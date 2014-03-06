#include "Varden2DMMS.h"

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif

//**********************************************************************
// CORRUGATED MMS: BASE CLASS
//**********************************************************************

template<typename FieldT>
VarDenCorrugatedMMSBase<FieldT>::
VarDenCorrugatedMMSBase( const Expr::Tag& xTag,
                           const Expr::Tag& yTag,
                           const Expr::Tag& tTag,
                           const double r0,
                           const double r1,
                           const double d,
                           const double w,
                           const double k,
                           const double a,
                           const double b,
                           const double uf,
                           const double vf)
: Expr::Expression<FieldT>(),
r0_( r0 ),
r1_( r1 ),
d_ ( d ),
w_ ( w ),
k_ ( k ),
a_ ( a ),
b_ ( b ),
uf_ ( uf ),
vf_ ( vf ),
xTag_( xTag ),
yTag_( yTag ),
tTag_( tTag )
{
  this->set_gpu_runnable( true );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDenCorrugatedMMSBase<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( xTag_ );
  exprDeps.requires_expression( yTag_ );
  exprDeps.requires_expression( tTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDenCorrugatedMMSBase<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  x_ = &fml.template field_manager<FieldT   >().field_ref( xTag_ );
  y_ = &fml.template field_manager<FieldT   >().field_ref( yTag_ );
  t_ = &fml.template field_manager<TimeField>().field_ref( tTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDenCorrugatedMMSBase<FieldT>::
evaluate()
{}

//**********************************************************************
// CORRUGATED MMS: MIXTURE FRACTION (initialization, etc...)
//**********************************************************************

template<typename FieldT>
VarDenCorrugatedMMSMixFrac<FieldT>::
VarDenCorrugatedMMSMixFrac( const Expr::Tag& xTag,
                              const Expr::Tag& yTag,
                              const Expr::Tag& tTag,
                              const double r0,
                              const double r1,
                              const double d,
                              const double w,
                              const double k,
                              const double a,
                              const double b,
                              const double uf,
                              const double vf)
: VarDenCorrugatedMMSBase<FieldT>(xTag, yTag, tTag, r0, r1, d, w, k, a, b, uf, vf)
{
  this->set_gpu_runnable( true );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDenCorrugatedMMSMixFrac<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  
  const double k = this->k_;
  const double uf = this->uf_;
  const double vf = this->vf_;
  const double w = this->w_;
  const double a = this->a_;
  const double b = this->b_;
  const double r0 = this->r0_;
  const double r1 = this->r1_;
  
  const double dn1 = 1.0 + r0/r1;
  const double dn2 = 1.0 - r0/r1;
  
  const FieldT& x = *(this->x_);
  const FieldT& y = *(this->y_);
  
  typedef SpatialOps::structured::SingleValueField TimeField;
  const TimeField& t = *(this->t_);
  
  SpatFldPtr<FieldT> xh_ = SpatialFieldStore::get<FieldT>( result );
  FieldT& xh = *xh_;
  xh <<= uf * t - x + a * cos(k*(vf * t - y));
  
  SpatFldPtr<FieldT> s0_ = SpatialFieldStore::get<FieldT>( result );
  FieldT& s0  = *s0_;
  
  s0 <<= tanh(b*xh*exp(w * t));
  
  result <<= (1.0 + s0)/(dn1 + dn2 * s0);
}

//--------------------------------------------------------------------

template< typename FieldT >
VarDenCorrugatedMMSMixFrac<FieldT>::Builder::
Builder( const Expr::Tag& result,
        const Expr::Tag& xTag,
        const Expr::Tag& yTag,
        const Expr::Tag& tTag,
        const double r0,
        const double r1,
        const double d,
        const double w,
        const double k,
        const double a,
        const double b,
        const double uf,
        const double vf)
: ExpressionBuilder(result),
r0_( r0 ),
r1_( r1 ),
d_ ( d ),
w_ ( w ),
k_ ( k ),
a_ ( a ),
b_ ( b ),
uf_ ( uf ),
vf_ ( vf ),
xTag_( xTag ),
yTag_( yTag ),
tTag_( tTag )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
VarDenCorrugatedMMSMixFrac<FieldT>::Builder::
build() const
{
  return new VarDenCorrugatedMMSMixFrac<FieldT>( xTag_, yTag_, tTag_, r0_, r1_, d_, w_, k_, a_, b_, uf_, vf_ );
}


//**********************************************************************
// CORRUGATED MMS: MIXTURE FRACTION SOURCE TERM
//**********************************************************************

template<typename FieldT>
VarDenCorrugatedMMSMixFracSrc<FieldT>::
VarDenCorrugatedMMSMixFracSrc( const Expr::Tag& xTag,
                               const Expr::Tag& yTag,
                               const Expr::Tag& tTag,
                               const double r0,
                               const double r1,
                               const double d,
                               const double w,
                               const double k,
                               const double a,
                               const double b,
                               const double uf,
                               const double vf)
: VarDenCorrugatedMMSBase<FieldT>(xTag, yTag, tTag, r0, r1, d, w, k, a, b, uf, vf)
{
  this->set_gpu_runnable( true );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDenCorrugatedMMSMixFracSrc<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();

  const double k = this->k_;
  const double uf = this->uf_;
  const double vf = this->vf_;
  const double w = this->w_;
  const double a = this->a_;
  const double b = this->b_;
    const double d = this->d_;
  const double r0 = this->r0_;
  const double r1 = this->r1_;

  const double k2 = k * k;
  const double b2 = b * b;
  const double a2 = a * a;

  const FieldT& x = *(this->x_);
  const FieldT& y = *(this->y_);
  typedef SpatialOps::structured::SingleValueField TimeField;
  const TimeField& t = *(this->t_);
  
  SpatFldPtr<FieldT> s0_ = SpatialFieldStore::get<FieldT>( result );
  SpatFldPtr<FieldT> s1_ = SpatialFieldStore::get<FieldT>( result );
  SpatFldPtr<FieldT> s2_ = SpatialFieldStore::get<FieldT>( result );
  
  FieldT& s0  = *s0_;
  FieldT& s1  = *s1_;
  FieldT& s2  = *s2_;
  
  s0 <<= exp(w * t);
  s1 <<= uf * t - x + a * cos(k * (vf * t - y));
  s2 <<= exp(2*b*s1/s0);

  result <<= (r0*r1*s2)/(s0*s0*pow(r0 + r1*s2,3))*(-4*d*b2*r0 -
                        2*a2*d*b2*k2*r0 +
                        4*d*b2*r1*s2 +
                        2*a2*d*b2*k2*r1*s2 +
                        2*b*r0*r1*s0*uf + 2*b*r1*r1*s0*s2*uf -
                        2*b*r0*r0*s0*t*uf*w - 2*b*r0*r1*s0*s2*t*uf*w +
                        2*b*r0*r0*s0*w*x + 2*b*r0*r1*s0*s2*w*x +
                        2*a*b*s0*(r0 + r1*s2)*(d*k2 - r0*w)*
                        cos(k*(t*vf - y)) +
                        2*a2*d*b2*k2*(r0 - r1*s2)*
                        cos(2*k*(t*vf - y)) +
                        r0*r0*s0*s0*w*log(1 + s2) -
                        r0*r1*s0*s0*w*log(1 + s2) +
                        r0*r1*s0*s0*s2*w*log(1 + s2) -
                        r1*r1*s0*s0*s2*w*log(1 + s2));
}

//--------------------------------------------------------------------

template< typename FieldT >
VarDenCorrugatedMMSMixFracSrc<FieldT>::Builder::
Builder( const Expr::Tag& result,
        const Expr::Tag& xTag,
        const Expr::Tag& yTag,
        const Expr::Tag& tTag,
        const double r0,
        const double r1,
        const double d,
        const double w,
        const double k,
        const double a,
        const double b,
        const double uf,
        const double vf)
: ExpressionBuilder(result),
r0_( r0 ),
r1_( r1 ),
d_ ( d ),
w_ ( w ),
k_ ( k ),
a_ ( a ),
b_ ( b ),
uf_ ( uf ),
vf_ ( vf ),
xTag_( xTag ),
yTag_( yTag ),
tTag_( tTag )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
VarDenCorrugatedMMSMixFracSrc<FieldT>::Builder::
build() const
{
  return new VarDenCorrugatedMMSMixFracSrc<FieldT>( xTag_, yTag_, tTag_, r0_, r1_, d_, w_, k_, a_, b_, uf_, vf_ );
}

//**********************************************************************
// CORRUGATED MMS: AXIAL VELOCITY (initialization, etc...)
//**********************************************************************

template<typename FieldT>
VarDenCorrugatedMMSVelocity<FieldT>::
VarDenCorrugatedMMSVelocity( const Expr::Tag& xTag,
                           const Expr::Tag& yTag,
                           const Expr::Tag& tTag,
                           const double r0,
                           const double r1,
                           const double d,
                           const double w,
                           const double k,
                           const double a,
                           const double b,
                           const double uf,
                           const double vf)
: VarDenCorrugatedMMSBase<FieldT>(xTag, yTag, tTag, r0, r1, d, w, k, a, b, uf, vf)
{
  this->set_gpu_runnable( true );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDenCorrugatedMMSVelocity<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  
  const double k = this->k_;
  const double uf = this->uf_;
  const double vf = this->vf_;
  const double w = this->w_;
  const double a = this->a_;
  const double b = this->b_;
  const double r0 = this->r0_;
  const double r1 = this->r1_;
  
  const FieldT& x = *(this->x_);
  const FieldT& y = *(this->y_);
  
  typedef SpatialOps::structured::SingleValueField TimeField;
  const TimeField& t = *(this->t_);
  
  SpatFldPtr<FieldT> xh_ = SpatialFieldStore::get<FieldT>( result );
  FieldT& xh = *xh_;
  xh <<= uf * t - x + a * cos(k*(vf * t - y));
  
  SpatFldPtr<FieldT> s0_ = SpatialFieldStore::get<FieldT>( result );
  FieldT& s0  = *s0_;
  s0 <<= exp(2.0*b*xh*exp(w * t)) + 1;
  
  SpatFldPtr<FieldT> rho_ = SpatialFieldStore::get<FieldT>( result );
  FieldT& rho = *rho_;
  rho <<= 0.5*(r0 + r1 + (r1-r0)*tanh( b*exp(-w*t)*xh ));
  result <<= (r1 - r0) / rho * ( - w * xh + (w*xh - uf)/s0 + w*log(s0)/(2.0*b*exp(-w*t)) );
}

//--------------------------------------------------------------------

template< typename FieldT >
VarDenCorrugatedMMSVelocity<FieldT>::Builder::
Builder(const Expr::Tag& result,
        const Expr::Tag& xTag,
        const Expr::Tag& yTag,
        const Expr::Tag& tTag,
        const double r0,
        const double r1,
        const double d,
        const double w,
        const double k,
        const double a,
        const double b,
        const double uf,
        const double vf)
: ExpressionBuilder(result),
r0_( r0 ),
r1_( r1 ),
d_ ( d ),
w_ ( w ),
k_ ( k ),
a_ ( a ),
b_ ( b ),
uf_ ( uf ),
vf_ ( vf ),
xTag_( xTag ),
yTag_( yTag ),
tTag_( tTag )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
VarDenCorrugatedMMSVelocity<FieldT>::Builder::
build() const
{
  return new VarDenCorrugatedMMSVelocity<FieldT>(xTag_, yTag_, tTag_, r0_, r1_, d_, w_, k_, a_, b_, uf_, vf_ );
}


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
                               const double vf)
: Expr::Expression<FieldT>(),
r0_( r0 ),
r1_( r1 ),
d_ ( d ),
w_ ( w ),
k_ ( k ),
uf_ ( uf ),
vf_ ( vf ),
xTag_( xTag ),
yTag_( yTag ),
tTag_( tTag )
{
  this->set_gpu_runnable( true );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDenMMSOscillatingMixFracSrc<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( xTag_ );
  exprDeps.requires_expression( yTag_ );
  exprDeps.requires_expression( tTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDenMMSOscillatingMixFracSrc<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  x_ = &fml.template field_manager<FieldT   >().field_ref( xTag_ );
  y_ = &fml.template field_manager<FieldT   >().field_ref( yTag_ );
  t_ = &fml.template field_manager<TimeField>().field_ref( tTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDenMMSOscillatingMixFracSrc<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  
//  const double A = 0.001, w=2.0, k=2.0, uf=0.5, vf=0.5; // Parameter values for the manufactured solutions
  const double k2 = k_*k_;

  SpatFldPtr<FieldT> xh = SpatialFieldStore::get<FieldT>( *x_ );
  SpatFldPtr<FieldT> yh = SpatialFieldStore::get<FieldT>( *y_ );
  *xh <<= *x_ - uf_ * *t_;
  *yh <<= *y_ - vf_ * *t_;
  
  SpatFldPtr<FieldT> s0_ = SpatialFieldStore::get<FieldT>( result );
  SpatFldPtr<FieldT> s1_ = SpatialFieldStore::get<FieldT>( result );
  SpatFldPtr<FieldT> s2_ = SpatialFieldStore::get<FieldT>( result );
  SpatFldPtr<FieldT> s3_ = SpatialFieldStore::get<FieldT>( result );
  SpatFldPtr<FieldT> s4_ = SpatialFieldStore::get<FieldT>( result );
  SpatFldPtr<FieldT> s5_ = SpatialFieldStore::get<FieldT>( result );
  SpatFldPtr<FieldT> s6_ = SpatialFieldStore::get<FieldT>( result );
  SpatFldPtr<FieldT> s7_ = SpatialFieldStore::get<FieldT>( result );
  SpatFldPtr<FieldT> s8_ = SpatialFieldStore::get<FieldT>( result );
  SpatFldPtr<FieldT> s9_ = SpatialFieldStore::get<FieldT>( result );
  
  FieldT& s0  = *s0_;
  FieldT& s1  = *s1_;
  FieldT& s2  = *s2_;
  FieldT& s3  = *s3_;
  FieldT& s4  = *s4_;
  FieldT& s5  = *s5_;
  FieldT& s6  = *s6_;
  FieldT& s7  = *s7_;
  FieldT& s8  = *s8_;
  FieldT& s9  = *s9_;
  
  const FieldT& x = *x_;
  const FieldT& y = *y_;
  const TimeField& t = *t_;
  

  s0 <<= cos(PI * w_ * *t_);
  s1 <<= sin(PI * w_ * *t_);
  s2 <<= cos(k_ * PI * *x_);
  s3 <<= sin(k_ * PI * *x_);
  s4 <<= cos(k_ * PI * *y_);
  s5 <<= sin(k_ * PI * *y_);

  s6 <<= sin(k_ * PI * (x - uf_ * t) );
  s7 <<= sin(k_ * PI * (y - vf_ * t) );
  
  s8 <<= cos(k_ * PI * (x - uf_ * t) );
  s9 <<= cos(k_ * PI * (y - vf_ * t) );

  const double s10 = r0_ + r1_;
  const double s11 = r0_ - r1_;

  result <<= -0.25*PI*r1_/pow(s10 - s0*s11*s6*s7,3)*
            ( 16*d_*k2*PI*r0_*s0*s0*s9*s9*s11*s6*s6
             - 16*d_*k2*PI*r0_*s0*s10*s6*s7
             + 16*d_*k2*PI*r0_*s0*s0*s8*s8*s11*s7*s7
             + 16*d_*k2*PI*r0_*s0*s0*s11*s6*s6*s7*s7
             + 2*k_*s0*s8*s10*s10*s10*s7*uf_
             - 6*k_*r0_*r0_*s0*s0*s8*s11*s6*s7*s7*uf_
             - 12*k_*r0_*r1_*s0*s0*s8*s11*s6*s7*s7*uf_
             - 6*k_*r1_*r1_*s0*s0*s8*s11*s6*s7*s7*uf_
             + 6*k_*s0*s0*s0*s8*s10*s11*s11*s6*s6*s7*s7*s7*uf_
             - 2*k_*s0*s0*s0*s0*s8*s11*s11*s11*s6*s6*s6*s7*s7*s7*s7*uf_
             + 2*k_*s0*s9*s10*s10*s10*s6*vf_
             - 6*k_*r0_*r0_*s0*s0*s9*s11*s6*s6*s7*vf_
             - 12*k_*r0_*r1_*s0*s0*s9*s11*s6*s6*s7*vf_
             - 6*k_*r1_*r1_*s0*s0*s9*s11*s6*s6*s7*vf_
             + 6*k_*s0*s0*s0*s9*s10*s11*s11*s6*s6*s6*s7*s7*vf_
             - 2*k_*s0*s0*s0*s0*s9*s11*s11*s11*s6*s6*s6*s6*s7*s7*s7*vf_
             - 2*r0_*s0*s1*s9*s9*s10*s11*s6*s6*w_
             + 4*r0_*s1*s10*s10*s6*s7*w_
             + 2*r0_*s0*s0*s1*s9*s9*s11*s11*s6*s6*s6*s7*w_
             - 5*r0_*s0*s1*s10*s11*s7*s7*w_
             + 3*r0_*s0*s1*s8*s8*s10*s11*s7*s7*w_
             - 3*r0_*s0*s1*s10*s11*s6*s6*s7*s7*w_
             + 2*r0_*s0*s0*s1*s8*s8*s11*s11*s6*s7*s7*s7*w_
             + 4*r0_*r0_*s0*s0*s1*s11*s6*s6*s6*s7*s7*s7*w_
             - 4*r0_*r1_*s0*s0*s1*s11*s6*s6*s6*s7*s7*s7*w_ );
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
        const double vf)
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
tTag_( tTag )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
VarDenMMSOscillatingMixFracSrc<FieldT>::Builder::
build() const
{
  return new VarDenMMSOscillatingMixFracSrc<FieldT>( xTag_, yTag_, tTag_, r0_, r1_, d_, w_, k_, uf_, vf_ );
}

//**********************************************************************
// OSCILLATING MMS: CONTINUITY SOURCE TERM
//**********************************************************************

template<typename FieldT>
VarDenMMSOscillatingContinuitySrc<FieldT>::
VarDenMMSOscillatingContinuitySrc( const Expr::Tag densTag,
                                   const Expr::Tag densStarTag,
                                   const Expr::Tag dens2StarTag,
                                   const Expr::TagList& velStarTags,
                                   const double r0,
                                   const double r1,
                                   const double w,
                                   const double k,
                                   const double uf,
                                   const double vf,
                                   const Expr::Tag& xTag,
                                   const Expr::Tag& yTag,
                                   const Expr::Tag& tTag,
                                   const Expr::Tag& timestepTag)
: Expr::Expression<FieldT>(),
  xVelStart_( velStarTags[0] ),
  yVelStart_( velStarTags[1] ),
  zVelStart_( velStarTags[2] ),
  denst_     ( densTag  ),
  densStart_ ( densStarTag ),
  dens2Start_( dens2StarTag ),
  doX_( velStarTags[0]!=Expr::Tag() ),
  doY_( velStarTags[1]!=Expr::Tag() ),
  doZ_( velStarTags[2]!=Expr::Tag() ),
  r0_( r0 ),
  r1_( r1 ),
  w_ ( w ),
  k_ ( k ),
  uf_ ( uf ),
  vf_ ( vf ),
  xTag_( xTag ),
  yTag_( yTag ),
  tTag_( tTag ),
  timestepTag_( timestepTag )
{
  this->set_gpu_runnable( true );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDenMMSOscillatingContinuitySrc<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  if( doX_ ) exprDeps.requires_expression( xVelStart_ );
  if( doY_ ) exprDeps.requires_expression( yVelStart_ );
  if( doZ_ ) exprDeps.requires_expression( zVelStart_ );

  exprDeps.requires_expression( denst_ );
  exprDeps.requires_expression( densStart_ );
  exprDeps.requires_expression( dens2Start_ );

  exprDeps.requires_expression( xTag_ );
  exprDeps.requires_expression( yTag_ );
  exprDeps.requires_expression( tTag_ );
  exprDeps.requires_expression( timestepTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDenMMSOscillatingContinuitySrc<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  if( doX_ ) uStar_ = &fml.field_ref<XVolField>( xVelStart_ );
  if( doY_ ) vStar_ = &fml.field_ref<YVolField>( yVelStart_ );
  if( doZ_ ) wStar_ = &fml.field_ref<ZVolField>( zVelStart_ );

  const typename Expr::FieldMgrSelector<SVolField>::type& scalarFM = fml.field_manager<SVolField>();
  dens_      = &scalarFM.field_ref( denst_ );
  densStar_  = &scalarFM.field_ref( densStart_ );
  dens2Star_ = &scalarFM.field_ref( dens2Start_ );

  const typename Expr::FieldMgrSelector<FieldT>::type& fm = fml.field_manager<FieldT>();
  x_ = &fm.field_ref( xTag_ );
  y_ = &fm.field_ref( yTag_ );
  
  const typename Expr::FieldMgrSelector<TimeField>::type& tfm = fml.field_manager<TimeField>();
  t_        = &tfm.field_ref( tTag_        );
  timestep_ = &tfm.field_ref( timestepTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDenMMSOscillatingContinuitySrc<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
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

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDenMMSOscillatingContinuitySrc<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
    
  SpatialOps::SpatFldPtr<SVolField> drhodtstar = SpatialOps::SpatialFieldStore::get<SVolField>( result );
  *drhodtstar <<= (*dens2Star_ - *dens_)/(2. * *timestep_);
  
  SpatialOps::SpatFldPtr<SVolField> divmomstar = SpatialOps::SpatialFieldStore::get<SVolField>( result );
  const bool is3d = doX_ && doY_ && doZ_;
  if (is3d) {
    *divmomstar <<=   (*gradXOp_) ( (*s2XInterpOp_)(*densStar_) * (*uStar_) )
    + (*gradYOp_) ( (*s2YInterpOp_)(*densStar_) * (*vStar_) )
    + (*gradZOp_) ( (*s2ZInterpOp_)(*densStar_) * (*wStar_) );
  } else {
    if(doX_) *divmomstar <<=               (*gradXOp_) ( (*s2XInterpOp_)(*densStar_) * (*uStar_) );
    else     *divmomstar <<= 0.0;
    if(doY_) *divmomstar <<= *divmomstar + (*gradYOp_) ( (*s2YInterpOp_)(*densStar_) * (*vStar_) );
    if(doZ_) *divmomstar <<= *divmomstar + (*gradZOp_) ( (*s2ZInterpOp_)(*densStar_) * (*wStar_) );
  }
  
  SpatialOps::SpatFldPtr<SVolField> beta = SpatialOps::SpatialFieldStore::get<SVolField>( result );
  
  // beta is the ratio of drhodt to div(mom*)
  *beta  <<= //cond (abs(abs(*drhodtstar) - abs(*divmomstar)) <= 1e-16, 1.0)
  cond (abs(*drhodtstar) <= 1e-10, 0.0)
  (abs(*divmomstar) <= 1e-10, abs(*drhodtstar) / 1e-10)
  (abs(*drhodtstar / *divmomstar));
  
  SpatialOps::SpatFldPtr<SVolField> alpha = SpatialOps::SpatialFieldStore::get<SVolField>( result );
//      *alpha <<= cond( *beta != *beta, 0.0 )
//                    ( 1.0/(*beta + 1.0)  );
  
//  const double r = ( r1_/r0_ >= r0_/r1_ ? r0_/r1_ : r1_/r0_ );
  *alpha <<= cond( *beta != *beta, 0.0 )
                 ( exp(log(0.2)*(pow(*beta, 0.2))) ); // Increase the value of the exponent to get closer to a step function
  
  *alpha <<= 0.1; // use this for the moment until we figure out the proper model for alpha
  
  SpatFldPtr<TimeField> t = SpatialFieldStore::get<TimeField>( *t_ );
  *t <<= *t_ + *timestep_;
  
  SpatFldPtr<FieldT> xh = SpatialFieldStore::get<FieldT>( *x_ );
  SpatFldPtr<FieldT> yh = SpatialFieldStore::get<FieldT>( *y_ );
  *xh <<= *x_ - uf_ * *t;
  *yh <<= *y_ - vf_ * *t;
  
  SpatFldPtr<FieldT> xb = SpatialFieldStore::get<FieldT>( *x_ );
  SpatFldPtr<FieldT> yb = SpatialFieldStore::get<FieldT>( *y_ );
  SpatFldPtr<TimeField> tb = SpatialFieldStore::get<TimeField>( *t_ );
  *xb <<= PI * k_ * *xh;
  *yb <<= PI * k_ * *yh;
  *tb <<= PI * w_ * *t;
  
  result <<= *alpha *
  (
   -0.5 * (r0_ - r1_) * PI*k_*cos(*tb) * ( uf_ * cos(*xb)*sin(*yb) + vf_ * cos(*yb)*sin(*xb) )
   );
}

//--------------------------------------------------------------------

template< typename FieldT >
VarDenMMSOscillatingContinuitySrc<FieldT>::Builder::
Builder( const Expr::Tag& result,
         const Expr::Tag densTag,
         const Expr::Tag densStarTag,
         const Expr::Tag dens2StarTag,
         const Expr::TagList& velStarTags,
         const double r0,
         const double r1,
         const double w,
         const double k,
         const double uf,
         const double vf,
         const Expr::Tag& xTag,
         const Expr::Tag& yTag,
         const Expr::Tag& tTag,
         const Expr::Tag& timestepTag )
: ExpressionBuilder(result),
  r0_( r0 ),
  r1_( r1 ),
  w_ ( w ),
  k_ ( k ),
  uf_ ( uf ),
  vf_ ( vf ),
  xTag_( xTag ),
  yTag_( yTag ),
  tTag_( tTag ),
  timestepTag_( timestepTag ),
  denst_     ( densTag      ),
  densStart_ ( densStarTag  ),
  dens2Start_( dens2StarTag ),
  velStarTs_ ( velStarTags  )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
VarDenMMSOscillatingContinuitySrc<FieldT>::Builder::
build() const
{
  return new VarDenMMSOscillatingContinuitySrc<FieldT>( denst_, densStart_, dens2Start_, velStarTs_, r0_, r1_, w_, k_, uf_, vf_, xTag_, yTag_, tTag_, timestepTag_ );
}


//--------------------------------------------------------------------
//--------------------------------------------------------------------


template<typename FieldT>
VarDenOscillatingMMSxVel<FieldT>::
VarDenOscillatingMMSxVel( const Expr::Tag& rhoTag,
                          const Expr::Tag& xTag,
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
  vf_ ( vf ),
  xTag_  ( xTag   ),
  yTag_  ( yTag   ),
  tTag_  ( tTag   ),
  rhoTag_( rhoTag )
{
  this->set_gpu_runnable( true );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDenOscillatingMMSxVel<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( rhoTag_ );
  exprDeps.requires_expression( xTag_ );
  exprDeps.requires_expression( yTag_ );
  exprDeps.requires_expression( tTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDenOscillatingMMSxVel<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  x_ = &fml.template field_manager<FieldT>().field_ref( xTag_ );
  y_ = &fml.template field_manager<FieldT>().field_ref( yTag_ );

  const typename Expr::FieldMgrSelector<SVolField>::type& scalarfm = fml.field_manager<SVolField>();
  rho_  = &scalarfm.field_ref( rhoTag_ );
  
  const typename Expr::FieldMgrSelector<TimeField>::type& tfm = fml.field_manager<TimeField>();
  t_ = &tfm.field_ref( tTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDenOscillatingMMSxVel<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  s2FInterpOp_ = opDB.retrieve_operator<S2FInterpOpT>();
}
//--------------------------------------------------------------------

template< typename FieldT >
void
VarDenOscillatingMMSxVel<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  
  const double w=2.0, k=2.0, uf=0.5, vf=0.5; // Parameter values for the manufactured solutions
    
  SpatFldPtr<FieldT> xh_ = SpatialFieldStore::get<FieldT>( *x_ ); FieldT& xh = *xh_;
  SpatFldPtr<FieldT> yh_ = SpatialFieldStore::get<FieldT>( *y_ ); FieldT& yh = *yh_;
  const TimeField& t = *t_;
  xh <<= *x_ - uf * t;
  yh <<= *y_ - vf * t;
  
  SpatFldPtr<FieldT> xb_ = SpatialFieldStore::get<FieldT>( *x_ ); FieldT& xb = *xb_;
  SpatFldPtr<FieldT> yb_ = SpatialFieldStore::get<FieldT>( *y_ ); FieldT& yb = *yb_;
  SpatFldPtr<TimeField> tb_ = SpatialFieldStore::get<TimeField>( *t_ ); TimeField& tb = *tb_;
  xb <<= PI * k * xh;
  yb <<= PI * k * yh;
  tb <<= PI * w * t;
  
  SpatFldPtr<FieldT> rho_interp = SpatialFieldStore::get<FieldT>( result );
  s2FInterpOp_->apply_to_field(*rho_,*rho_interp);
  const double q =  (-w/(4*k)) * (r1_-r0_);
  result <<= q * cos(xb) * sin(yb) * sin(tb) / *rho_interp;
}

//--------------------------------------------------------------------

template< typename FieldT >
VarDenOscillatingMMSxVel<FieldT>::Builder::
Builder( const Expr::Tag& result,
         const Expr::Tag& rhoTag,
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
  xTag_  ( xTag   ),
  yTag_  ( yTag   ),
  tTag_  ( tTag   ),
  rhoTag_( rhoTag )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
VarDenOscillatingMMSxVel<FieldT>::Builder::
build() const
{
  return new VarDenOscillatingMMSxVel<FieldT>( rhoTag_, xTag_, yTag_, tTag_, r0_, r1_, w_, k_, uf_, vf_ );
}

//--------------------------------------------------------------------
//--------------------------------------------------------------------


template<typename FieldT>
VarDenOscillatingMMSyVel<FieldT>::
VarDenOscillatingMMSyVel( const Expr::Tag& rhoTag,
                          const Expr::Tag& xTag,
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
  vf_ ( vf ),
  xTag_  ( xTag   ),
  yTag_  ( yTag   ),
  tTag_  ( tTag   ),
  rhoTag_( rhoTag )
{
  this->set_gpu_runnable( true );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDenOscillatingMMSyVel<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( rhoTag_ );
  exprDeps.requires_expression( xTag_ );
  exprDeps.requires_expression( yTag_ );
  exprDeps.requires_expression( tTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDenOscillatingMMSyVel<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  x_ = &fml.template field_manager<FieldT>().field_ref( xTag_ );
  y_ = &fml.template field_manager<FieldT>().field_ref( yTag_ );
  
  const typename Expr::FieldMgrSelector<SVolField>::type& scalarfm = fml.field_manager<SVolField>();
  rho_  = &scalarfm.field_ref( rhoTag_ );

  const typename Expr::FieldMgrSelector<TimeField>::type& tfm = fml.field_manager<TimeField>();
  t_        = &tfm.field_ref( tTag_        );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDenOscillatingMMSyVel<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  s2FInterpOp_ = opDB.retrieve_operator<S2FInterpOpT>();
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDenOscillatingMMSyVel<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  
  const double om=2.0, k=2.0, uf=0.5, vf=0.5; // Parameter values for the manufactured solutions
  
  SpatFldPtr<FieldT> xh = SpatialFieldStore::get<FieldT>( *x_ );
  SpatFldPtr<FieldT> yh = SpatialFieldStore::get<FieldT>( *y_ );
  *xh <<= *x_ - uf * *t_;
  *yh <<= *y_ - vf * *t_;
  
  SpatFldPtr<FieldT> xb = SpatialFieldStore::get<FieldT>( *x_ );
  SpatFldPtr<FieldT> yb = SpatialFieldStore::get<FieldT>( *y_ );
  SpatFldPtr<TimeField> tb = SpatialFieldStore::get<TimeField>( *t_ );
  *xb <<= PI * k * *xh;
  *yb <<= PI * k * *yh;
  *tb <<= PI * om * *t_;
  
  SpatFldPtr<FieldT> rho_interp = SpatialFieldStore::get<FieldT>( result );
  s2FInterpOp_->apply_to_field(*rho_,*rho_interp);
  result <<= (-om/(4*k)) * (r1_-r0_) * sin(*xb) * cos(*yb) * sin(*tb) / *rho_interp;
}

//--------------------------------------------------------------------

template< typename FieldT >
VarDenOscillatingMMSyVel<FieldT>::Builder::
Builder( const Expr::Tag& result,
         const Expr::Tag& rhoTag,
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
  xTag_  ( xTag   ),
  yTag_  ( yTag   ),
  tTag_  ( tTag   ),
  rhoTag_( rhoTag )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
VarDenOscillatingMMSyVel<FieldT>::Builder::
build() const
{
  return new VarDenOscillatingMMSyVel<FieldT>( rhoTag_, xTag_, yTag_, tTag_, r0_, r1_, w_, k_, uf_, vf_ );
}


//--------------------------------------------------------------------
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
vf_ ( vf ),
xTag_( xTag ),
yTag_( yTag ),
tTag_( tTag )
{
  this->set_gpu_runnable( true );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDenOscillatingMMSMixFrac<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( xTag_ );
  exprDeps.requires_expression( yTag_ );
  exprDeps.requires_expression( tTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDenOscillatingMMSMixFrac<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  x_ = &fml.template field_manager<FieldT>().field_ref( xTag_ );
  y_ = &fml.template field_manager<FieldT>().field_ref( yTag_ );
  
  const typename Expr::FieldMgrSelector<TimeField>::type& tfm = fml.field_manager<TimeField>();
  t_        = &tfm.field_ref( tTag_        );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDenOscillatingMMSMixFrac<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  
  SpatFldPtr<FieldT> xh = SpatialFieldStore::get<FieldT>( *x_ );
  SpatFldPtr<FieldT> yh = SpatialFieldStore::get<FieldT>( *y_ );
  *xh <<= *x_ - uf_ * *t_;
  *yh <<= *y_ - vf_ * *t_;
  
  SpatFldPtr<FieldT> xb = SpatialFieldStore::get<FieldT>( *x_ );
  SpatFldPtr<FieldT> yb = SpatialFieldStore::get<FieldT>( *y_ );
  SpatFldPtr<TimeField> tb = SpatialFieldStore::get<TimeField>( *t_ );
  *xb <<= PI * k_ * *xh;
  *yb <<= PI * k_ * *yh;
  *tb <<= PI * w_ * *t_;
  
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
  rhoTag_( rhoTag ),
  d_(d)
{
  this->set_gpu_runnable( true );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
DiffusiveConstant<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( rhoTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
DiffusiveConstant<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  rho_ = &fml.template field_manager<FieldT>().field_ref( rhoTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
DiffusiveConstant<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  
  result <<= d_ / *rho_;
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

template class VarDenCorrugatedMMSBase<SVolField>;
template class VarDenCorrugatedMMSMixFrac<SVolField>;
template class VarDenCorrugatedMMSMixFracSrc<SVolField>;

template class VarDenCorrugatedMMSBase<XVolField>;
template class VarDenCorrugatedMMSVelocity<XVolField>;


template class VarDenMMSOscillatingMixFracSrc<XVolField>;
template class VarDenMMSOscillatingMixFracSrc<YVolField>;
template class VarDenMMSOscillatingMixFracSrc<ZVolField>;

template class VarDenMMSOscillatingContinuitySrc<SVolField>;

template class VarDenOscillatingMMSxVel<XVolField>;
template class VarDenOscillatingMMSxVel<YVolField>;
template class VarDenOscillatingMMSxVel<ZVolField>;
template class VarDenOscillatingMMSxVel<SVolField>;
template class VarDenOscillatingMMSyVel<YVolField>;
template class VarDenOscillatingMMSyVel<XVolField>;
template class VarDenOscillatingMMSyVel<ZVolField>;
template class VarDenOscillatingMMSyVel<SVolField>;
template class VarDenOscillatingMMSMixFrac<SVolField>;
template class VarDenOscillatingMMSMixFrac<XVolField>;
template class VarDenOscillatingMMSMixFrac<YVolField>;
template class VarDenOscillatingMMSMixFrac<ZVolField>;
template class DiffusiveConstant<SVolField>;
template class DiffusiveConstant<XVolField>;
template class DiffusiveConstant<YVolField>;
template class DiffusiveConstant<ZVolField>;
