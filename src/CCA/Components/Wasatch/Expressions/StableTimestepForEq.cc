//
//  StableTimestepForEq.cc
//
//  Created by Tony Saad on 03/01/2022.
//
//

#include "StableTimestepForEq.h"
//-- ExprLib includes --//
#include <expression/ExprLib.h>

//-- SpatialOps includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

// ###################################################################
//
//                          Implementation
//
// ###################################################################

template< typename Vel1T, typename Vel2T, typename Vel3T >
StableTimestepForEq<Vel1T,Vel2T,Vel3T>::
StableTimestepForEq( const Expr::Tag& rhoTag,
            const Expr::Tag& viscTag,
            const Expr::Tag& uTag,
            const Expr::Tag& vTag,
            const Expr::Tag& wTag,
            const Expr::Tag& csoundTag,
            const std::string timeIntegratorName)
: Expr::Expression<SpatialOps::SingleValueField>(),
  dx_(1.0),
  dy_(1.0),
  dz_(1.0),
  doX_( uTag != Expr::Tag() ),
  doY_( vTag != Expr::Tag() ),
  doZ_( wTag != Expr::Tag() ),
  isCompressible_(csoundTag != Expr::Tag()),
  is3dconvdiff_( doX_ && doY_ && doZ_),
  timeIntegratorName_(timeIntegratorName)
{
  rho_ = create_field_request<SVolField>(rhoTag);
  visc_ = create_field_request<SVolField>(viscTag);
  if (doX_)  u_ = create_field_request<Vel1T>(uTag);
  if (doY_)  v_ = create_field_request<Vel2T>(vTag);
  if (doZ_)  w_ = create_field_request<Vel3T>(wTag);
  if(isCompressible_) csound_ = create_field_request<SVolField>(csoundTag);
}

//--------------------------------------------------------------------
template< typename Vel1T, typename Vel2T, typename Vel3T >
StableTimestepForEq<Vel1T,Vel2T,Vel3T>::
~StableTimestepForEq()
{}

//--------------------------------------------------------------------
template< typename Vel1T, typename Vel2T, typename Vel3T >
void
StableTimestepForEq<Vel1T,Vel2T,Vel3T>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  // bind operators as follows:
  // op_ = opDB.retrieve_operator<OpT>();
  if (doX_) {
    x2SInterp_ = opDB.retrieve_operator<X2SOpT>();
    gradXOp_   = opDB.retrieve_operator<GradXT>();
    dx_ = 1.0 / std::abs( gradXOp_->coefs().get_coef(1) ); //high coefficient
  }
  if (doY_) {
    y2SInterp_ = opDB.retrieve_operator<Y2SOpT>();
    gradYOp_   = opDB.retrieve_operator<GradYT>();
    dy_ = 1.0 / std::abs( gradYOp_->coefs().get_coef(1) ); //high coefficient
  }
  if (doZ_) {
    z2SInterp_ = opDB.retrieve_operator<Z2SOpT>();
    gradZOp_   = opDB.retrieve_operator<GradZT>();
    dz_ = 1.0 / std::abs( gradZOp_->coefs().get_coef(1) ); //high coefficient
  }
}

//--------------------------------------------------------------------
template< typename Vel1T, typename Vel2T, typename Vel3T >
void
StableTimestepForEq<Vel1T,Vel2T,Vel3T>::
evaluate()
{
  using namespace SpatialOps;
  SpatialOps::SingleValueField& result = this->value();
  
  // jcs we need to switch to this once we have field_min_interior fixed.  This expression would then be ready for gpu.
//  if( is3dconvdiff_ ){
//    result <<= field_min_interior( 1.0 / (
//        (*x2SInterp_)(abs(*u_)) * invDx_ + *visc_/ *rho_ * invDx_ * invDx_ +
//        (*y2SInterp_)(abs(*v_)) * invDy_ + *visc_/ *rho_ * invDy_ * invDy_ +
//        (*z2SInterp_)(abs(*w_)) * invDz_ + *visc_/ *rho_ * invDz_ * invDz_ )
//        );
//  }
//  else{
  const SVolField& rho = rho_->field_ref();
  SpatialOps::SpatFldPtr<SVolField> kinVisc_ = SpatialOps::SpatialFieldStore::get<SVolField>( rho );
  *kinVisc_ <<= visc_->field_ref() / rho;
  
  SpatialOps::SpatFldPtr<SVolField> c_ = SpatialOps::SpatialFieldStore::get<SVolField>( rho );
  SVolField& c = *c_;
  if(isCompressible_) c <<= abs(csound_->field_ref());
  else c <<= 0.0;
  
  SpatialOps::SpatFldPtr<SVolField> innerdt = SpatialOps::SpatialFieldStore::get<SVolField>( rho );
  SpatialOps::SpatFldPtr<SVolField> outerdt = SpatialOps::SpatialFieldStore::get<SVolField>( rho );
  
  // inner rule:
  if (!doX_) *innerdt <<= 0.0;
  if (doX_)  *innerdt <<=             *kinVisc_ / dx_ / dx_;
  if (doY_)  *innerdt <<= *innerdt +  *kinVisc_ / dy_ / dy_;
  if (doZ_)  *innerdt <<= *innerdt +  *kinVisc_ / dz_ / dz_;

  if (timeIntegratorName_ == "FE") {
    // Inner Rule
    *innerdt <<= 0.5 / *innerdt;

    // Outer Rule
    if (!doX_) *outerdt <<= 0.0;
    if (doX_)  *outerdt <<=        ( (*x2SInterp_)( abs(u_->field_ref()) ) + c ) * ( (*x2SInterp_)( abs(u_->field_ref()) ) + c );
    if (doY_)  *outerdt <<= *outerdt + ( (*y2SInterp_)( abs(v_->field_ref()) ) + c ) * ( (*y2SInterp_)( abs(v_->field_ref()) ) + c );
    if (doZ_)  *outerdt <<= *outerdt + ( (*z2SInterp_)( abs(w_->field_ref()) ) + c ) * ( (*z2SInterp_)( abs(w_->field_ref()) ) + c );
    *outerdt <<= 2.0 * *kinVisc_ / *outerdt;
    
  } else if (timeIntegratorName_ == "RK2SSP") {
    // Inner Rule
    *innerdt <<= 0.5 / *innerdt;

    // Outer Rule
    if (!doX_) *outerdt <<= 0.0;
    if (doX_)  *outerdt <<=       0.420168*( (*x2SInterp_)( abs(u_->field_ref()) ) + c ) / dx_ * pow( ((*x2SInterp_)( abs(u_->field_ref())) + c ) * dx_ / *kinVisc_, 1.0/3.0);
    if (doY_)  *outerdt <<= *outerdt + 0.420168*( (*y2SInterp_)( abs(v_->field_ref()) ) + c ) /dy_ * pow( ((*y2SInterp_)( abs(v_->field_ref())) + c ) * dy_ / *kinVisc_, 1.0/3.0);
    if (doZ_)  *outerdt <<= *outerdt + 0.420168*( (*z2SInterp_)( abs(w_->field_ref()) ) + c ) /dz_ * pow( ((*z2SInterp_)( abs(w_->field_ref())) + c ) * dz_ / *kinVisc_, 1.0/3.0);
    *outerdt <<= 1.0 / *outerdt;
    
  }else if (timeIntegratorName_ == "RK3SSP") {
    // Inner Rule
    *innerdt <<= 0.628931 / *innerdt;

    // Outer Rule
    if (!doX_) *outerdt <<= 0.0;
    if (doX_)  *outerdt <<=            ( (*x2SInterp_)( abs(u_->field_ref()) ) + c ) / dx_;
    if (doY_)  *outerdt <<= *outerdt + ( (*y2SInterp_)( abs(v_->field_ref()) ) + c ) / dy_;
    if (doZ_)  *outerdt <<= *outerdt + ( (*z2SInterp_)( abs(w_->field_ref()) ) + c ) / dz_;
    *outerdt <<= 1.732 / *outerdt;
  }

  SpatialOps::SpatFldPtr<SingleValueField> innerdtMin = SpatialOps::SpatialFieldStore::get<SingleValueField>( result );
  *innerdtMin <<= 999999999.0;
  *innerdtMin <<= field_min_interior(*innerdt);
  result <<= min( *innerdtMin, field_min_interior(*outerdt) );
}

//--------------------------------------------------------------------
template< typename Vel1T, typename Vel2T, typename Vel3T >
StableTimestepForEq<Vel1T,Vel2T,Vel3T>::
Builder::Builder( const Expr::Tag& resultTag,
                 const Expr::Tag& rhoTag,
                 const Expr::Tag& viscTag,
                 const Expr::Tag& uTag,
                 const Expr::Tag& vTag,
                 const Expr::Tag& wTag,
                 const Expr::Tag& csoundTag,
                 const std::string timeIntegratorName)
: ExpressionBuilder( resultTag ),
rhoTag_( rhoTag ),
viscTag_( viscTag ),
uTag_( uTag ),
vTag_( vTag ),
wTag_( wTag ),
csoundTag_(csoundTag),
timeIntegratorName_(timeIntegratorName)
{}

//--------------------------------------------------------------------
template< typename Vel1T, typename Vel2T, typename Vel3T >
Expr::ExpressionBase*
StableTimestepForEq<Vel1T,Vel2T,Vel3T>::
Builder::build() const
{
  return new StableTimestepForEq( rhoTag_,viscTag_,uTag_,vTag_,wTag_, csoundTag_, timeIntegratorName_ );
}

//==========================================================================
// Explicit template instantiation for supported versions of this expression
#include <spatialops/structured/FVStaggered.h>
template class StableTimestepForEq<SpatialOps::XVolField, SpatialOps::YVolField, SpatialOps::ZVolField>;
template class StableTimestepForEq<SpatialOps::SVolField, SpatialOps::SVolField, SpatialOps::SVolField>;
