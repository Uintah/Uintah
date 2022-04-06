//
//  CellReynoldsNumber.cc
//
//

#include "CellReynoldsNumber.h"
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
CellReynoldsNumber<VelT>::
CellReynoldsNumber(const Expr::Tag& velTag, const Expr::Tag& viscosityTag, const std::string& direction)
: Expr::Expression<SpatialOps::SVolField>(),
  h_(1.0),
  direction_(direction)
{
  vel_ = create_field_request<VelT>(velTag);
  visc_ = create_field_request<SVolField>(viscosityTag);
}

//--------------------------------------------------------------------
template< typename VelT>
CellReynoldsNumber<VelT>::
~CellReynoldsNumber()
{}

//--------------------------------------------------------------------
template< typename VelT>
void
CellReynoldsNumber<VelT>::
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
CellReynoldsNumber<VelT>::
evaluate()
{
  using namespace SpatialOps;
  
  typedef typename Expr::Expression<SVolField>::ValVec SVolFieldVec;
  SVolFieldVec& results = this->get_value_vec();

  // jcs: can we do the linear solve in place? We probably can. If so,
  // we would only need one field, not two...
  SVolField& cellReynolds = *results[0];
  SVolField& cellReynoldsSquared = *results[1];
  
  cellReynolds <<=(*interpVelT2SVolOp_)( abs(vel_->field_ref()) ) * h_ / visc_->field_ref();
  
  cellReynoldsSquared <<= pow(cellReynolds,2);
}

//--------------------------------------------------------------------
template< typename VelT>
CellReynoldsNumber<VelT>::
Builder::Builder( const Expr::TagList& resultTags,
                 const Expr::Tag& velTag,
                 const Expr::Tag& viscosityTag,
                 const std::string direction)
: ExpressionBuilder( resultTags ),
velTag_( velTag ),
viscosityTag_( viscosityTag ),
direction_(direction)
{}

//--------------------------------------------------------------------
template< typename VelT>
Expr::ExpressionBase*
CellReynoldsNumber<VelT>::
Builder::build() const
{
  return new CellReynoldsNumber( velTag_, viscosityTag_, direction_ );
}

//==========================================================================
// Explicit template instantiation for supported versions of this expression
#include <spatialops/structured/FVStaggered.h>
template class CellReynoldsNumber<SpatialOps::XVolField>;
template class CellReynoldsNumber<SpatialOps::YVolField>;
template class CellReynoldsNumber<SpatialOps::ZVolField>;
