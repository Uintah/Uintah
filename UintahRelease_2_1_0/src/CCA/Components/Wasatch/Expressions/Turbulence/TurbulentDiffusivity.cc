/*
 * The MIT License
 *
 * Copyright (c) 2012-2017 The University of Utah
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

Expr::Tag turbulent_diffusivity_tag()
{
  return Expr::Tag( "TurbulentDiffusivity", Expr::STATE_NONE );
}

//====================================================================

TurbulentDiffusivity::
TurbulentDiffusivity( const Expr::Tag rhoTag,
                      const double tSchmidt,
                      const Expr::Tag tViscTag )
: Expr::Expression<SVolField>(),
  tSchmidt_ ( tSchmidt )
{
  this->set_gpu_runnable(true);
  rho_ = create_field_request<SVolField>(rhoTag);
  tVisc_ = create_field_request<SVolField>(tViscTag);
}

//--------------------------------------------------------------------

void
TurbulentDiffusivity::
evaluate()
{
  using namespace SpatialOps;
  SVolField& result = this->value();
  result <<= 0.0;
  const SVolField& mut = tVisc_->field_ref();
  const SVolField& rho = rho_->field_ref();
  result <<= mut / (tSchmidt_ * rho);
}
