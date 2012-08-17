#include "TurbulentDiffusivity.h"

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

#include <cmath>

//====================================================================

TurbulentDiffusivity::
TurbulentDiffusivity( const Expr::Tag rhoTag,
                      const double tSchmidt,
                      const Expr::Tag tViscTag )
: Expr::Expression<SVolField>(),
  tViscTag_ ( tViscTag ),
  rhoTag_   ( rhoTag   ),
  tSchmidt_ ( tSchmidt )
{}

//--------------------------------------------------------------------

TurbulentDiffusivity::
~TurbulentDiffusivity()
{}

//--------------------------------------------------------------------

void
TurbulentDiffusivity::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( rhoTag_   );
  exprDeps.requires_expression( tViscTag_ );
}

//--------------------------------------------------------------------

void
TurbulentDiffusivity::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldMgrSelector<SVolField>::type& scalarfm = fml.field_manager<SVolField>();

  rho_   = &scalarfm.field_ref( rhoTag_   );
  tVisc_ = &scalarfm.field_ref( tViscTag_ );
}

//--------------------------------------------------------------------

void
TurbulentDiffusivity::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//--------------------------------------------------------------------

void
TurbulentDiffusivity::
evaluate()
{
  using namespace SpatialOps;
  SVolField& result = this->value();
  result <<= 0.0;

  result <<= *tVisc_ / (tSchmidt_ * *rho_);
}
