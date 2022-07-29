//
//  CourantNumber.cc
//
//
//

#include "CourantNumber.h"
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


template< typename VelT>
CourantNumber<VelT>::
CourantNumber(const Expr::Tag& rhoTag,const Expr::Tag& rhovelTag, const Expr::Tag& dtTag, const std::string& direction)
: Expr::Expression<SpatialOps::SVolField>(),
  h_(1.0),
  direction_(direction)
{
  rho_ = create_field_request<SVolField>(rhoTag);
  rhovel_ = create_field_request<VelT>(rhovelTag);
  dt_ = create_field_request<TimeField>(dtTag);
}

//--------------------------------------------------------------------
template< typename VelT>
CourantNumber<VelT>::
~CourantNumber()
{}

//--------------------------------------------------------------------
template< typename VelT>
void
CourantNumber<VelT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  // bind operators as follows:
  // op_ = opDB.retrieve_operator<OpT>();

  interpVelT2SVolOp_ = opDB.retrieve_operator<interpVelT2SVolT>();
  
  gradXOp_   = opDB.retrieve_operator<GradXT>();
  const double dx = 1.0/ std::abs( gradXOp_->coefs().get_coef(1) ); //high coefficient

  gradYOp_   = opDB.retrieve_operator<GradYT>();
  const double dy = 1.0/std::abs( gradYOp_->coefs().get_coef(1) ); //high coefficient
  
 
  gradZOp_   = opDB.retrieve_operator<GradZT>();
  const double dz = 1.0/ std::abs( gradZOp_->coefs().get_coef(1) ); //high coefficient
  
  h_ = (direction_=="xdir") ? dx : ((direction_ == "ydir") ? dy : dz);
}

//--------------------------------------------------------------------
template< typename VelT>
void
CourantNumber<VelT>::
evaluate()
{
  using namespace SpatialOps;
  SpatialOps::SVolField& result = this->value();
  const SVolField& rho = rho_->field_ref();

  SpatialOps::SpatFldPtr<SVolField> vel_ = SpatialOps::SpatialFieldStore::get<SVolField>( rho );
  *vel_ <<= (*interpVelT2SVolOp_)( abs(rhovel_->field_ref()) )/ rho;
  
  result <<=  abs(*vel_) * dt_->field_ref() / h_;
}

//--------------------------------------------------------------------
template< typename VelT>
CourantNumber<VelT>::
Builder::Builder( const Expr::Tag& resultTag,
                 const Expr::Tag& rhoTag, 
                 const Expr::Tag& rhovelTag,
                 const Expr::Tag& dtTag,
                 const std::string direction)
: ExpressionBuilder( resultTag ),
rhoTag_ (rhoTag),
rhovelTag_( rhovelTag ),
dtTag_( dtTag ),
direction_(direction)
{}

//--------------------------------------------------------------------
template< typename VelT>
Expr::ExpressionBase*
CourantNumber<VelT>::
Builder::build() const
{
  return new CourantNumber(rhoTag_, rhovelTag_, dtTag_, direction_ );
}

//==========================================================================
// Explicit template instantiation for supported versions of this expression
#include <spatialops/structured/FVStaggered.h>
template class CourantNumber<SpatialOps::XVolField>;
template class CourantNumber<SpatialOps::YVolField>;
template class CourantNumber<SpatialOps::ZVolField>;
