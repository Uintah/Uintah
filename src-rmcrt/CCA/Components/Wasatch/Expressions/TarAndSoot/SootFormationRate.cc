/*
 * The MIT License
 *
 * Copyright (c) 2012-2018 The University of Utah
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

#include <CCA/Components/Wasatch/Expressions/TarAndSoot/SootFormationRate.h>

#ifndef GAS_CONSTANT
#  define GAS_CONSTANT 8.3144621 // J/(mol*K)
#endif

//--------------------------------------------------------------------

template< typename ScalarT >
SootFormationRate<ScalarT>::
SootFormationRate( const Expr::Tag& yTarTag,
		           const Expr::Tag& densityTag,
                   const Expr::Tag& tempTag )
  : Expr::Expression<ScalarT>(),
    A_( 5.02E+8  ),
    E_( 198.9E+3 )
    {
  this->set_gpu_runnable( true );

  yTar_    = this->template create_field_request<ScalarT>( yTarTag    );
  density_ = this->template create_field_request<ScalarT>( densityTag );
  temp_    = this->template create_field_request<ScalarT>( tempTag    );
}

//--------------------------------------------------------------------

template< typename ScalarT >
void
SootFormationRate<ScalarT>::
evaluate()
{
  using namespace SpatialOps;

  ScalarT& result = this->value();

  const ScalarT& yTar    = yTar_   ->field_ref();
  const ScalarT& density = density_->field_ref();
  const ScalarT& temp    = temp_   ->field_ref();

  result <<= density * yTar * A_ *exp(-E_ / (GAS_CONSTANT * temp) );
}

//--------------------------------------------------------------------

// Explicit template instantiation
#include <spatialops/structured/FVStaggeredFieldTypes.h>
template class SootFormationRate<SpatialOps::SVolField>;
