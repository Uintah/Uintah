/*
 * Copyright (c) 2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

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
