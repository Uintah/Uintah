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
CourantNumber(const Expr::Tag& velTag, const Expr::Tag& dtTag, const std::string& direction)
: Expr::Expression<SpatialOps::SVolField>(),
  h_(1.0),
  direction_(direction)
{
  vel_ = create_field_request<VelT>(velTag);
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
  
  result <<=(*interpVelT2SVolOp_)( abs(vel_->field_ref()) ) * dt_->field_ref() / h_;
}

//--------------------------------------------------------------------
template< typename VelT>
CourantNumber<VelT>::
Builder::Builder( const Expr::Tag& resultTag,
                 const Expr::Tag& velTag,
                 const Expr::Tag& dtTag,
                 const std::string direction)
: ExpressionBuilder( resultTag ),
velTag_( velTag ),
dtTag_( dtTag ),
direction_(direction)
{}

//--------------------------------------------------------------------
template< typename VelT>
Expr::ExpressionBase*
CourantNumber<VelT>::
Builder::build() const
{
  return new CourantNumber( velTag_, dtTag_, direction_ );
}

//==========================================================================
// Explicit template instantiation for supported versions of this expression
#include <spatialops/structured/FVStaggered.h>
template class CourantNumber<SpatialOps::XVolField>;
template class CourantNumber<SpatialOps::YVolField>;
template class CourantNumber<SpatialOps::ZVolField>;
