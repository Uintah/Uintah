//
//  VardenMMS.cc
//  uintah-xcode-local
//
//  Created by Tony Saad on 11/1/13.
//
//

#include "VardenMMS.h"

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif

//--------------------------------------------------------------------

template<typename FieldT>
VarDen1DMMSMixFracSrc<FieldT>::
VarDen1DMMSMixFracSrc( const Expr::Tag& xTag,
                     const Expr::Tag& tTag,
                     const double D,
                     const double rho0,
                     const double rho1 )
: Expr::Expression<FieldT>(),
d_   ( D    ),
rho0_( rho0 ),
rho1_( rho1 ),
xTag_( xTag ),
tTag_( tTag )
{
  this->set_gpu_runnable( true );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDen1DMMSMixFracSrc<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( xTag_ );
  exprDeps.requires_expression( tTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDen1DMMSMixFracSrc<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  x_ = &fml.template field_manager<FieldT   >().field_ref( xTag_ );
  t_ = &fml.template field_manager<TimeField>().field_ref( tTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDen1DMMSMixFracSrc<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  
  result <<=
  (
   ( 5 * rho0_ * (rho1_ * rho1_) * exp( (5 * (*x_ * *x_))/(*t_ + 10.))
    * ( 1500. * d_ - 120. * *t_ + 75. * (*t_ * *t_) * (*x_ * *x_)
       + 30. * (*t_ * *t_ * *t_) * (*x_ * *x_) + 750 * d_ * *t_
       + 1560. * d_ * (*t_ * *t_) + 750 * d_ * (*t_ * *t_ * *t_)
       + 60. * d_ * (*t_ * *t_ * *t_ * *t_) - 1500 * d_ * (*x_ * *x_)
       + 30. * *t_ * (*x_ * *x_) - 606. * (*t_ * *t_) - 120. * (*t_ * *t_ * *t_)
       - 6. * (*t_ * *t_ * *t_ * *t_) + 75 * (*x_ * *x_)
       + 7500. * *t_ * *x_ * sin((2 * PI * *x_)/(3. * (*t_ + 10.)))
       - 250. * PI * (*t_ * *t_) * cos((2 * PI * *x_)/(3.*(*t_ + 10.)))
       - 20. * PI * (*t_ * *t_ * *t_)*cos((2 * PI * *x_)/(3.*(*t_ + 10.)))
       - 1500. * d_ * (*t_ * *t_) * (*x_ * *x_)
       - 600. * d_ * (*t_ * *t_ * *t_) * (*x_ * *x_)
       + 3750. * (*t_ * *t_) * *x_ * sin((2 * PI * *x_)/(3. * (*t_ + 10.)))
       + 300. * (*t_ * *t_ * *t_) * *x_ * sin((2 * PI * *x_)/(3.*(*t_ + 10.)))
       - 500. * PI * *t_ * cos((2 * PI * *x_)/(3. * (*t_ + 10.)))
       - 600. * d_ * *t_ * (*x_ * *x_) - 600
       )
    )/3
   +
   ( 250 * rho0_ * rho1_ * (rho0_ - rho1_) * (*t_ + 10.) * (3 * d_ + 3 * d_ * (*t_ * *t_)
                                                            - PI * *t_ * cos(
                                                                             (2 * PI * *x_)/(3. * (*t_ + 10.))))
    )/ 3
   )
  /
  (
   ( (*t_ * *t_) + 1.)*((*t_ + 10.) * (*t_ + 10.))
   *
   (
    (5 * rho0_ - 5 * rho1_ + 5 * rho1_ * exp((5 * (*x_ * *x_))/(*t_ + 10.)) + 2 * rho1_ * *t_ * exp((5 * (*x_ * *x_))/(*t_ + 10.)))
    *(5 * rho0_ - 5 * rho1_ + 5 * rho1_ * exp((5 * (*x_ * *x_))/(*t_ + 10.)) + 2 * rho1_ * *t_ * exp((5 * (*x_ * *x_))/(*t_ + 10.)))
    )
   );
}

//--------------------------------------------------------------------

template< typename FieldT >
VarDen1DMMSMixFracSrc<FieldT>::Builder::
Builder( const Expr::Tag& result,
        const Expr::Tag& xTag,
        const Expr::Tag& tTag,
        const double D,
        const double rho0,
        const double rho1  )
: ExpressionBuilder(result),
d_   ( D    ),
rho0_( rho0 ),
rho1_( rho1 ),
xTag_( xTag ),
tTag_( tTag )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
VarDen1DMMSMixFracSrc<FieldT>::Builder::
build() const
{
  return new VarDen1DMMSMixFracSrc<FieldT>( xTag_, tTag_, d_, rho0_, rho1_ );
}

//--------------------------------------------------------------------

template<typename FieldT>
VarDen1DMMSContinuitySrc<FieldT>::
VarDen1DMMSContinuitySrc( const double rho0,
                        const double rho1,
                        const Expr::Tag& xTag,
                        const Expr::Tag& tTag,
                        const Expr::Tag& timestepTag)
: Expr::Expression<FieldT>(),
rho0_( rho0 ),
rho1_( rho1 ),
xTag_( xTag ),
tTag_( tTag ),
timestepTag_( timestepTag )
{
  this->set_gpu_runnable( true );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDen1DMMSContinuitySrc<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( xTag_ );
  exprDeps.requires_expression( tTag_ );
  exprDeps.requires_expression( timestepTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDen1DMMSContinuitySrc<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  x_ = &fml.template field_manager<FieldT>().field_ref( xTag_ );
  
  const typename Expr::FieldMgrSelector<TimeField>::type& tfm = fml.field_manager<TimeField>();
  t_        = &tfm.field_ref( tTag_        );
  timestep_ = &tfm.field_ref( timestepTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDen1DMMSContinuitySrc<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  
  const double alpha = 0.1;   // the continuity equation weighting factor
  
  SpatFldPtr<TimeField> t = SpatialFieldStore::get<TimeField>( *t_ );
  *t <<= *t_ + *timestep_;
  
  result <<= alpha *
  (
   (
    ( 10/( exp((5 * (*x_ * *x_))/( *t + 10)) * ((2 * *t + 5) * (2 * *t + 5)) )
     - (25 * (*x_ * *x_))/(exp(( 5 * (*x_ * *x_))/(*t + 10)) * (2 * *t + 5) * ((*t + 10) * (*t + 10)) )
     ) / rho0_
    - 10/( rho1_ * exp((5 * (*x_ * *x_))/(*t + 10)) * ((2 * *t + 5) * (2 * *t + 5)) )
    + (25 * (*x_ * *x_)) / (rho1_ * exp((5 * (*x_ * *x_))/(*t + 10)) * (2 * *t + 5) * ((*t + 10) * (*t + 10)) )
    )
   /
   (
    ( (5 / (exp((5 * (*x_ * *x_))/(*t + 10)) * (2 * *t + 5)) - 1)/rho0_
     - 5/(rho1_ * exp((5 * (*x_ * *x_))/(*t + 10)) * (2 * *t + 5))
     )
    *
    ((5 / (exp((5 * (*x_ * *x_))/(*t + 10)) * (2 * *t + 5)) - 1)/rho0_ - 5/(rho1_ * exp((5 * (*x_ * *x_))/(*t + 10)) * (2 * *t + 5)))
    )
   -
   ( 5 * *t * sin((2 * PI * *x_)/(3 * *t + 30)) *
    ((50 * *x_) / (rho0_ * exp((5 * (*x_ * *x_))/(*t + 10)) * (2 * *t + 5) * (*t + 10)) - (50 * *x_)/(rho1_ * exp((5 * (*x_ * *x_))/(*t + 10)) * (2 * *t + 5) * (*t + 10)))
    )
   /
   ( ( (*t * *t) + 1.)
    * ( ((5/(exp((5 * (*x_ * *x_))/(*t + 10)) * (2 * *t + 5)) - 1)/rho0_ - 5/(rho1_ * exp((5 * (*x_ * *x_))/(*t + 10)) * (2 * *t + 5)))
       * ((5/(exp((5 * (*x_ * *x_))/(*t + 10)) * (2 * *t + 5)) - 1)/rho0_ - 5/(rho1_ * exp((5 * (*x_ * *x_))/(*t + 10)) * (2 * *t + 5)))
       )
    )
   -
   ( 10 * PI * *t * cos((2 * PI * *x_)/(3 * *t + 30)))
   /
   ( (3 * *t + 30) * ((*t * *t) + 1.)
    * ( (5/(exp((5 * (*x_ * *x_))/(*t + 10)) * (2 * *t + 5)) - 1)/rho0_
       - 5/(rho1_ * exp((5 * (*x_ * *x_))/(*t + 10)) * (2 * *t + 5))
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
        const Expr::Tag& timestepTag )
: ExpressionBuilder(result),
rho0_( rho0 ),
rho1_( rho1 ),
xTag_( xTag ),
tTag_( tTag ),
timestepTag_( timestepTag )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
VarDen1DMMSContinuitySrc<FieldT>::Builder::
build() const
{
  return new VarDen1DMMSContinuitySrc<FieldT>( rho0_, rho1_, xTag_, tTag_, timestepTag_ );
}

//--------------------------------------------------------------------

template<typename FieldT>
VarDen1DMMSPressureContSrc<FieldT>::
VarDen1DMMSPressureContSrc( const Expr::Tag continutySrcTag,
                          const Expr::Tag& timestepTag)
: Expr::Expression<FieldT>(),
continutySrcTag_( continutySrcTag ),
timestepTag_    ( timestepTag     )
{
  this->set_gpu_runnable( true );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDen1DMMSPressureContSrc<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( continutySrcTag_ );
  exprDeps.requires_expression( timestepTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDen1DMMSPressureContSrc<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  continutySrc_ = &fml.template field_manager<FieldT>().field_ref( continutySrcTag_ );
  
  const typename Expr::FieldMgrSelector<TimeField>::type& dfm = fml.field_manager<TimeField>();
  timestep_ = &dfm.field_ref( timestepTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
VarDen1DMMSPressureContSrc<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& result = this->value();
  
  result <<= *continutySrc_ / *timestep_;
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
template class VarDen1DMMSMixFracSrc<XVolField>;
template class VarDen1DMMSMixFracSrc<YVolField>;
template class VarDen1DMMSMixFracSrc<ZVolField>;

template class VarDen1DMMSContinuitySrc<SVolField>;

template class VarDen1DMMSPressureContSrc<SVolField>;