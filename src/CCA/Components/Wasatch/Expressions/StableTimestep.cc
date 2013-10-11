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



StableTimestep::
StableTimestep( const Expr::Tag& rhoTag,
            const Expr::Tag& viscTag,
            const Expr::Tag& uTag,
            const Expr::Tag& vTag,
            const Expr::Tag& wTag )
: Expr::Expression<SpatialOps::structured::SingleValueField>(),
  rhoTag_( rhoTag ),
  viscTag_( viscTag ),
  uTag_( uTag ),
  vTag_( vTag ),
  wTag_( wTag ),
  invDx_(1.0),
  invDy_(1.0),
  invDz_(1.0),
  doX_( uTag_ != Expr::Tag() ),
  doY_( vTag_ != Expr::Tag() ),
  doZ_( wTag_ != Expr::Tag() ),
  isViscous_( viscTag_ != Expr::Tag() ),
  is3dconvdiff_( doX_ && doY_ && doZ_ && isViscous_ )
{}

//--------------------------------------------------------------------

StableTimestep::
~StableTimestep()
{}

//--------------------------------------------------------------------

void
StableTimestep::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( rhoTag_ );
  exprDeps.requires_expression( viscTag_ );
  if (doX_) exprDeps.requires_expression( uTag_ );
  if (doY_) exprDeps.requires_expression( vTag_ );
  if (doZ_) exprDeps.requires_expression( wTag_ );
}

//--------------------------------------------------------------------

void
StableTimestep::
bind_fields( const Expr::FieldManagerList& fml )
{
  rho_ = &fml.field_ref< SVolField >( rhoTag_ );
  if (isViscous_) visc_ = &fml.field_ref< SVolField >( viscTag_ );
  if (doX_)       u_ = &fml.field_ref< XVolField >( uTag_ );
  if (doY_)       v_ = &fml.field_ref< YVolField >( vTag_ );
  if (doZ_)       w_ = &fml.field_ref< ZVolField >( wTag_ );
}

//--------------------------------------------------------------------

void
StableTimestep::
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

void
StableTimestep::
evaluate()
{
  using namespace SpatialOps;
  SpatialOps::structured::SingleValueField& result = this->value();
  
  // jcs we need to switch to this once we have field_min_interior fixed.  This expression would then be ready for gpu.
//  if( is3dconvdiff_ ){
//    result <<= field_min_interior( 1.0 / (
//        (*x2SInterp_)(abs(*u_)) * invDx_ + *visc_/ *rho_ * invDx_ * invDx_ +
//        (*y2SInterp_)(abs(*v_)) * invDy_ + *visc_/ *rho_ * invDy_ * invDy_ +
//        (*z2SInterp_)(abs(*w_)) * invDz_ + *visc_/ *rho_ * invDz_ * invDz_ )
//        );
//  }
//  else{
  SpatialOps::SpatFldPtr<SVolField> kinVisc_ = SpatialOps::SpatialFieldStore::get<SVolField>( *rho_ );
  if (isViscous_) *kinVisc_ <<= *visc_/ *rho_;
  else            *kinVisc_ <<= 0.0;
  
  SpatialOps::SpatFldPtr<SVolField> tmp = SpatialOps::SpatialFieldStore::get<SVolField>( *rho_ );
  if (!doX_) *tmp <<= 0.0;
  if (doX_)  *tmp <<=        (*x2SInterp_)(abs(*u_)) * invDx_ + *kinVisc_ * invDx_ * invDx_; // u/dx + nu/dx2
  if (doY_)  *tmp <<= *tmp + (*y2SInterp_)(abs(*v_)) * invDy_ + *kinVisc_ * invDy_ * invDy_; // v/dy + nu/dy2
  if (doZ_)  *tmp <<= *tmp + (*z2SInterp_)(abs(*w_)) * invDz_ + *kinVisc_ * invDz_ * invDz_; // w/dz + nu/dz2
  *tmp <<= 1.0 / *tmp;
  result <<= field_min_interior(*tmp);
//  }
}

//--------------------------------------------------------------------

StableTimestep::
Builder::Builder( const Expr::Tag& resultTag,
                 const Expr::Tag& rhoTag,
                 const Expr::Tag& viscTag,
                 const Expr::Tag& uTag,
                 const Expr::Tag& vTag,
                 const Expr::Tag& wTag )
: ExpressionBuilder( resultTag ),
rhoTag_( rhoTag ),
viscTag_( viscTag ),
uTag_( uTag ),
vTag_( vTag ),
wTag_( wTag )
{}

//--------------------------------------------------------------------

Expr::ExpressionBase*
StableTimestep::
Builder::build() const
{
  return new StableTimestep( rhoTag_,viscTag_,uTag_,vTag_,wTag_ );
}
