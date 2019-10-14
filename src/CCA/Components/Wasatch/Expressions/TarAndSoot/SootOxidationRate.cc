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

#include <CCA/Components/Wasatch/Expressions/TarAndSoot/SootOxidationRate.h>

#ifndef PI
#  define PI 3.1415926535897932384626433832795
#endif

#ifndef GAS_CONSTANT
#  define GAS_CONSTANT 8.3144621 // J/(mol*K)
#endif

//--------------------------------------------------------------------

template< typename ScalarT >
SootOxidationRate<ScalarT>::
SootOxidationRate( const Expr::Tag& yO2Tag,
                   const Expr::Tag& ySootTag,
                   const Expr::Tag& nSootParticleTag,
                   const Expr::Tag& mixMWTag,
                   const Expr::Tag& pressTag,
                   const Expr::Tag& densityTag,
                   const Expr::Tag& tempTag,
                   const double     sootDensity )
  : Expr::Expression<ScalarT>(),
    A_          ( 1.09E+4/101325.),
    E_          ( 164.5E+3       ),
    o2MW_       ( 32.0           ),
    sootDensity_( sootDensity    )
{
   this->set_gpu_runnable( true );

   yO2_           = this->template create_field_request<ScalarT>( yO2Tag           );
   ySoot_         = this->template create_field_request<ScalarT>( ySootTag         );
   nSootParticle_ = this->template create_field_request<ScalarT>( nSootParticleTag );
   mixMW_         = this->template create_field_request<ScalarT>( mixMWTag         );
   press_         = this->template create_field_request<ScalarT>( pressTag         );
   density_       = this->template create_field_request<ScalarT>( densityTag       );
   temp_          = this->template create_field_request<ScalarT>( tempTag          );
}

//--------------------------------------------------------------------

template< typename ScalarT >
void
SootOxidationRate<ScalarT>::
evaluate()
{
  using namespace SpatialOps;

  ScalarT& result = this->value();

  const ScalarT& yO2           = yO2_          ->field_ref();
  const ScalarT& ySoot         = ySoot_        ->field_ref();
  const ScalarT& nSootParticle = nSootParticle_->field_ref();
  const ScalarT& mixMW         = mixMW_        ->field_ref();
  const ScalarT& press         = press_        ->field_ref();
  const ScalarT& density       = density_      ->field_ref();
  const ScalarT& temp          = temp_         ->field_ref();

  result <<= density
             * pow( 36.0*PI * nSootParticle * square(ySoot / sootDensity_), 1.0/3.0 ) // (total surface area of soot particles)/density
             * mixMW/o2MW_ * yO2                                                   // converts O2 from mass to mole fraction
             * press/sqrt(temp) * A_ * exp(-E_ / (GAS_CONSTANT * temp));
}

//--------------------------------------------------------------------

// Explicit template instantiation
#include <spatialops/structured/FVStaggeredFieldTypes.h>
template class SootOxidationRate<SpatialOps::SVolField>;
