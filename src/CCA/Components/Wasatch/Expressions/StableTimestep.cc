//
//  StableTimestep.cc
//  uintah-xcode-aurora
//
//  Created by Tony Saad on 6/10/13.
//
//

#include "StableTimestep.h"
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
StableTimestep<Vel1T,Vel2T,Vel3T>::
StableTimestep( const Expr::Tag& rhoTag,
            const Expr::Tag& viscTag,
            const Expr::Tag& uTag,
            const Expr::Tag& vTag,
            const Expr::Tag& wTag,
            const Expr::Tag& puTag,
            const Expr::Tag& pvTag,
            const Expr::Tag& pwTag,
            const Expr::Tag& csoundTag)
: Expr::Expression<SpatialOps::SingleValueField>(),
  invDx_(1.0),
  invDy_(1.0),
  invDz_(1.0),
  doX_( uTag != Expr::Tag() ),
  doY_( vTag != Expr::Tag() ),
  doZ_( wTag != Expr::Tag() ),
  isViscous_( viscTag != Expr::Tag() ),
  doParticles_(puTag != Expr::Tag() && pvTag != Expr::Tag() && pwTag != Expr::Tag() ),
  isCompressible_(csoundTag != Expr::Tag()),
  is3dconvdiff_( doX_ && doY_ && doZ_ && isViscous_ )
{
  rho_ = create_field_request<SVolField>(rhoTag);
  if (isViscous_)  visc_ = create_field_request<SVolField>(viscTag);
  if (doX_)  u_ = create_field_request<Vel1T>(uTag);
  if (doY_)  v_ = create_field_request<Vel2T>(vTag);
  if (doZ_)  w_ = create_field_request<Vel3T>(wTag);
  if (doParticles_) {
    pu_ = create_field_request<ParticleField>(puTag);
    pv_ = create_field_request<ParticleField>(pvTag);
    pw_ = create_field_request<ParticleField>(pwTag);
  }
  if(isCompressible_) csound_ = create_field_request<SVolField>(csoundTag);
}

//--------------------------------------------------------------------
template< typename Vel1T, typename Vel2T, typename Vel3T >
StableTimestep<Vel1T,Vel2T,Vel3T>::
~StableTimestep()
{}

//--------------------------------------------------------------------
template< typename Vel1T, typename Vel2T, typename Vel3T >
void
StableTimestep<Vel1T,Vel2T,Vel3T>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  // bind operators as follows:
  // op_ = opDB.retrieve_operator<OpT>();
  if (doX_) {
    x2SInterp_ = opDB.retrieve_operator<X2SOpT>();
    gradXOp_   = opDB.retrieve_operator<GradXT>();
    invDx_ = std::abs( gradXOp_->coefs().get_coef(1) ); //high coefficient
  }
  if (doY_) {
    y2SInterp_ = opDB.retrieve_operator<Y2SOpT>();
    gradYOp_   = opDB.retrieve_operator<GradYT>();
    invDy_ = std::abs( gradYOp_->coefs().get_coef(1) ); //high coefficient
  }
  if (doZ_) {
    z2SInterp_ = opDB.retrieve_operator<Z2SOpT>();
    gradZOp_   = opDB.retrieve_operator<GradZT>();
    invDz_ = std::abs( gradZOp_->coefs().get_coef(1) ); //high coefficient
  }
}

//--------------------------------------------------------------------
template< typename Vel1T, typename Vel2T, typename Vel3T >
void
StableTimestep<Vel1T,Vel2T,Vel3T>::
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
  if (isViscous_) *kinVisc_ <<= visc_->field_ref()/ rho;
  else            *kinVisc_ <<= 0.0;
  
  SpatialOps::SpatFldPtr<SVolField> c_ = SpatialOps::SpatialFieldStore::get<SVolField>( rho );
  SVolField& c = *c_;
  if(isCompressible_) c <<= abs(csound_->field_ref());
  else c <<= 0.0;
  
  SpatialOps::SpatFldPtr<SVolField> tmp = SpatialOps::SpatialFieldStore::get<SVolField>( rho );
  if (!doX_) *tmp <<= 0.0;
  if (doX_)  *tmp <<=        ( (*x2SInterp_)( abs(u_->field_ref()) ) + c ) * invDx_ + *kinVisc_ * invDx_ * invDx_; // u/dx + nu/dx2
  if (doY_)  *tmp <<= *tmp + ( (*y2SInterp_)( abs(v_->field_ref()) ) + c ) * invDy_ + *kinVisc_ * invDy_ * invDy_; // v/dy + nu/dy2
  if (doZ_)  *tmp <<= *tmp + ( (*z2SInterp_)( abs(w_->field_ref()) ) + c ) * invDz_ + *kinVisc_ * invDz_ * invDz_; // w/dz + nu/dz2
  *tmp <<= 1.0 / *tmp;
  if (doParticles_) {
    SpatialOps::SpatFldPtr<SingleValueField> minPDt_ = SpatialOps::SpatialFieldStore::get<SingleValueField>( result ); // particle dt
    SingleValueField& minPDt = *minPDt_;
  
    SpatialOps::SpatFldPtr<ParticleField> tmpP = SpatialOps::SpatialFieldStore::get<ParticleField>( pu_->field_ref() );
    if (!doX_) *tmpP <<= 0.0;
    if (doX_)  *tmpP <<=         abs(pu_->field_ref()) * invDx_; // u_particle/dx
    if (doY_)  *tmpP <<= *tmpP + abs(pv_->field_ref()) * invDy_; // v_particle/dy
    if (doZ_)  *tmpP <<= *tmpP + abs(pw_->field_ref()) * invDz_; // w_particle/dz
    
    *tmpP <<= 1.0 / *tmpP;
    
    minPDt <<= field_min(*tmpP);
    result <<= min( minPDt, field_min_interior(*tmp) );
  } else {
    result <<= field_min_interior(*tmp);
  }
//  }
}

//--------------------------------------------------------------------
template< typename Vel1T, typename Vel2T, typename Vel3T >
StableTimestep<Vel1T,Vel2T,Vel3T>::
Builder::Builder( const Expr::Tag& resultTag,
                 const Expr::Tag& rhoTag,
                 const Expr::Tag& viscTag,
                 const Expr::Tag& uTag,
                 const Expr::Tag& vTag,
                 const Expr::Tag& wTag,
                 const Expr::Tag& puTag,
                 const Expr::Tag& pvTag,
                 const Expr::Tag& pwTag,
                 const Expr::Tag& csoundTag)
: ExpressionBuilder( resultTag ),
rhoTag_( rhoTag ),
viscTag_( viscTag ),
uTag_( uTag ),
vTag_( vTag ),
wTag_( wTag ),
puTag_(puTag),
pvTag_(pvTag),
pwTag_(pwTag),
csoundTag_(csoundTag)
{}

//--------------------------------------------------------------------
template< typename Vel1T, typename Vel2T, typename Vel3T >
Expr::ExpressionBase*
StableTimestep<Vel1T,Vel2T,Vel3T>::
Builder::build() const
{
  return new StableTimestep( rhoTag_,viscTag_,uTag_,vTag_,wTag_, puTag_, pvTag_, pwTag_, csoundTag_ );
}

//==========================================================================
// Explicit template instantiation for supported versions of this expression
#include <spatialops/structured/FVStaggered.h>
template class StableTimestep<SpatialOps::XVolField, SpatialOps::YVolField, SpatialOps::ZVolField>;
template class StableTimestep<SpatialOps::SVolField, SpatialOps::SVolField, SpatialOps::SVolField>;
