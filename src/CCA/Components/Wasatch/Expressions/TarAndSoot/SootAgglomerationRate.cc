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

#include "SootAgglomerationRate.h"

#ifndef PI
#  define PI 3.1415926535897932384626433832795
#endif

//--------------------------------------------------------------------

template< typename ScalarT >
SootAgglomerationRate<ScalarT>::
SootAgglomerationRate( const Expr::Tag& densityTag,
                 const Expr::Tag& tempTag,
                 const Expr::Tag& ySootTag,
                 const Expr::Tag& nSootParticleTag,
                 const double     sootDensity )
  : Expr::Expression<ScalarT>(),
    boltzConst_ ( 1.3806488*10E-23 ),
    ca_         ( 3.0              ), // collision frequency [1]
    sootDensity_( sootDensity      )
{
  this->set_gpu_runnable( true );

  density_       = this->template create_field_request<ScalarT>( densityTag       );
  temp_          = this->template create_field_request<ScalarT>( tempTag          );
  ySoot_         = this->template create_field_request<ScalarT>( ySootTag         );
  nSootParticle_ = this->template create_field_request<ScalarT>( nSootParticleTag );
}

//--------------------------------------------------------------------

template< typename ScalarT >
void
SootAgglomerationRate<ScalarT>::

evaluate()
{
  using namespace SpatialOps;
  ScalarT& result = this->value();

  const ScalarT& density       = density_      ->field_ref();
  const ScalarT& temp          = temp_         ->field_ref();
  const ScalarT& ySoot         = ySoot_        ->field_ref();
  const ScalarT& nSootParticle = nSootParticle_->field_ref();


  result <<= 2 * density * ca_
               * pow(6.0 / (PI * sootDensity_),               1.0/6.0  )
               * pow(6.0 * boltzConst_ * temp / sootDensity_,    0.5   )
               * pow(density * ySoot,                         1.0/6.0  )
               * pow(density * nSootParticle,                 11.0/6.0 );

}

//--------------------------------------------------------------------

// Explicit template instantiation
#include <spatialops/structured/FVStaggeredFieldTypes.h>
template class SootAgglomerationRate<SpatialOps::SVolField>;
