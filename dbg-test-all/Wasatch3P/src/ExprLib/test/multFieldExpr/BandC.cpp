#include "BandC.h"


BandC::BandC()
  : Expr::Expression<SpatialOps::SingleValueField>()
{}

//--------------------------------------------------------------------

BandC::~BandC()
{}

//--------------------------------------------------------------------

void
BandC::advertise_dependents( Expr::ExprDeps& exprDeps )
{}

//--------------------------------------------------------------------

void
BandC::bind_fields( const Expr::FieldManagerList& fml )
{}

//--------------------------------------------------------------------

void
BandC::bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//--------------------------------------------------------------------

void
BandC::evaluate()
{
  using namespace SpatialOps;
  std::vector<SpatialOps::SingleValueField*>& values = this->get_value_vec();
  SpatialOps::SingleValueField& b = *values[0];
  SpatialOps::SingleValueField& c = *values[1];
  b <<= 1.0;
  c <<= 2.0;
}

//--------------------------------------------------------------------
