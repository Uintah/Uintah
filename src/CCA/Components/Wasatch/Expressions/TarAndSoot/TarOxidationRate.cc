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

#include <CCA/Components/Wasatch/Expressions/TarAndSoot/TarOxidationRate.h>

#ifndef GAS_CONSTANT
#  define GAS_CONSTANT 8.3144621 // J/(mol*K)
#endif

//--------------------------------------------------------------------

template< typename ScalarT >
TarOxidationRate<ScalarT>::
TarOxidationRate( const Expr::Tag& densityTag,
                  const Expr::Tag& yO2Tag,
                  const Expr::Tag& yTarTag,
                  const Expr::Tag& tempTag )
: Expr::Expression<ScalarT>(),
  A_( 6.77e5 ),
  E_( 52.3e3 )
{
  this->set_gpu_runnable( true );

  density_ = this->template create_field_request<ScalarT>( densityTag );
  yO2_     = this->template create_field_request<ScalarT>( yO2Tag     );
  yTar_    = this->template create_field_request<ScalarT>( yTarTag    );
  temp_    = this->template create_field_request<ScalarT>( tempTag    );
}

//--------------------------------------------------------------------

template< typename ScalarT >
void
TarOxidationRate<ScalarT>::
evaluate()
{
  using namespace SpatialOps;

  ScalarT& result = this->value();

  const ScalarT& density = density_->field_ref();
  const ScalarT& yO2     = yO2_    ->field_ref();
  const ScalarT& yTar    = yTar_   ->field_ref();
  const ScalarT& temp    = temp_   ->field_ref();

  /*
   * result = [tar][O2]A*exp(-E/RT)*tarMW  ( in kg/m^3-s ),
   *
   * [tar] = yTar*density/tarMW,
   * [O2]  = yo2*density/o2MW,
   *
   * Thus,
   * result = density^2 * yTar * yO2/o2MW * A*exp(-E/RT)*tarMW
   */

  result <<= density * yTar
           * density * yO2
           * A_ * exp( -E_ / ( GAS_CONSTANT * temp ) );
}

//--------------------------------------------------------------------

// Explicit template instantiation
#include <spatialops/structured/FVStaggeredFieldTypes.h>
template class TarOxidationRate<SpatialOps::SVolField>;
