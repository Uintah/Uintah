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
                          const Expr::Tag densTag,
                          const Expr::Tag densStarTag,
                          const Expr::Tag dens2StarTag,
                          const Expr::TagList& velTags,
                          const Expr::Tag& xTag,
                          const Expr::Tag& tTag,
                          const Expr::Tag& timestepTag,
                          const Wasatch::VarDenParameters varDenParams)
: Expr::Expression<FieldT>(),
  rho0_( rho0 ),
  rho1_( rho1 ),
  densTag_(densTag),
  densStarTag_(densStarTag),
  dens2StarTag_(dens2StarTag),
  xTag_( xTag ),
  tTag_( tTag ),
  timestepTag_( timestepTag ),
  xVelt_( velTags[0] ),
  yVelt_( velTags[1] ),
  zVelt_( velTags[2] ),
  doX_( velTags[0]!=Expr::Tag() ),
  doY_( velTags[1]!=Expr::Tag() ),
  doZ_( velTags[2]!=Expr::Tag() ),
  is3d_( doX_ && doY_ && doZ_ ),
  a0_( varDenParams.alpha0 ),
  model_( varDenParams.model ),
  useOnePredictor_(varDenParams.onePredictor)
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
  if (model_ != Wasatch::VarDenParameters::CONSTANT) {
    exprDeps.requires_expression( densTag_ );
    exprDeps.requires_expression( useOnePredictor_ ? densStarTag_ : dens2StarTag_ );
    if( doX_ ) exprDeps.requires_expression( xVelt_ );
    if( doY_ ) exprDeps.requires_expression( yVelt_ );
    if( doZ_ ) exprDeps.requires_expression( zVelt_ );
  }
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
  
  if (model_ != Wasatch::VarDenParameters::CONSTANT) {
    const typename Expr::FieldMgrSelector<SVolField>::type& sfm = fml.field_manager<SVolField>();
    dens_ = &sfm.field_ref( densTag_ );
    
    if   (useOnePredictor_) densStar_  = &sfm.field_ref( densStarTag_ );
    else                    dens2Star_ = &sfm.field_ref( dens2StarTag_ );
    
    if( doX_ ) xVel_  = &fml.field_ref<XVolField>( xVelt_ );
    if( doY_ ) yVel_  = &fml.field_ref<YVolField>( yVelt_ );
    if( doZ_ ) zVel_  = &fml.field_ref<ZVolField>( zVelt_ );
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
  
  SpatFldPtr<TimeField> t = SpatialFieldStore::get<TimeField>( *t_ );
  *t <<= *t_ + *timestep_;
  

  SpatialOps::SpatFldPtr<SVolField> alpha = SpatialOps::SpatialFieldStore::get<SVolField>( result );

  switch (model_) {
    case Wasatch::VarDenParameters::CONSTANT:
      *alpha <<= a0_;
      break;
    case Wasatch::VarDenParameters::IMPULSE:
    {
      SpatialOps::SpatFldPtr<SVolField> drhodtstar = SpatialOps::SpatialFieldStore::get<SVolField>( result );
      if (useOnePredictor_)  *drhodtstar <<= (*densStar_  - *dens_) / *timestep_;
      else                   *drhodtstar <<= (*dens2Star_ - *dens_) / (2. * *timestep_);
      *alpha <<= cond(*drhodtstar == 0.0, 1.0)(a0_);
    }
      break;
    case Wasatch::VarDenParameters::DYNAMIC:
    {
      SpatialOps::SpatFldPtr<SVolField> velDotDensGrad = SpatialOps::SpatialFieldStore::get<SVolField>( result );
      SpatialOps::SpatFldPtr<SVolField> drhodtstar = SpatialOps::SpatialFieldStore::get<SVolField>( result );

      if (useOnePredictor_)  *drhodtstar <<= (*densStar_  - *dens_) / *timestep_;
      else                   *drhodtstar <<= (*dens2Star_ - *dens_) / (2. * *timestep_);
      
      if( is3d_ ){ // for 3D cases, inline the whole thing
        *velDotDensGrad <<= (*x2SInterpOp_)(*xVel_) * (*gradXOp_)(*dens_) + (*y2SInterpOp_)(*yVel_) * (*gradYOp_)(*dens_) + (*z2SInterpOp_)(*zVel_) * (*gradZOp_)(*dens_);
      } else {
        // for 1D and 2D cases, we are not as efficient - add terms as needed...
        if( doX_ ) *velDotDensGrad <<= (*x2SInterpOp_)(*xVel_) * (*gradXOp_)(*dens_);
        else       *velDotDensGrad <<= 0.0;
        if( doY_ ) *velDotDensGrad <<= *velDotDensGrad + (*y2SInterpOp_)(*yVel_) * (*gradYOp_)(*dens_);
        if( doZ_ ) *velDotDensGrad <<= *velDotDensGrad + (*z2SInterpOp_)(*zVel_) * (*gradZOp_)(*dens_);
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
