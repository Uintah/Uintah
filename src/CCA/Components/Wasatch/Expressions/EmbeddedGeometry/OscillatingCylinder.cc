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

#include "OscillatingCylinder.h"
#include <CCA/Components/Wasatch/TagNames.h>
//--------------------------------------------------------------------

OscillatingCylinder::
OscillatingCylinder( const std::string& axis,
                     const std::vector<double>& origin,
                     const std::vector<double>& oscillatingdir,
                     const double insideValue,
                     const double outsideValue,
                     const double radius,
                     const double frequency,
                     const double amplitude )
: Expr::Expression<SVolField>(),
  origin_(origin),
  oscillatingdir_(oscillatingdir),
  insidevalue_(insideValue),
  outsidevalue_(outsideValue),
  radius_(radius),
  frequency_(frequency),
  amplitude_(amplitude)
{
  Expr::Tag tag1, tag2;
  const WasatchCore::TagNames& tagNames = WasatchCore::TagNames::self();
  if ( axis == "X" ) {
    tag1 = tagNames.ysvolcoord;
    tag2 = tagNames.zsvolcoord;
  } else if ( axis == "Y" ) {
    tag1 = tagNames.xsvolcoord;
    tag2 = tagNames.zsvolcoord;
  } else if ( axis == "Z" ) {
    tag1 = tagNames.xsvolcoord;
    tag2 = tagNames.ysvolcoord;
  }

  this->set_gpu_runnable( false );
  
   field1_ = create_field_request<SVolField>(tag1);
   field2_ = create_field_request<SVolField>(tag2);
   t_ = create_field_request<TimeField>(tagNames.time);
}

//--------------------------------------------------------------------

void
OscillatingCylinder::
evaluate()
{
  using namespace SpatialOps;
  // jcs need to fold all of this into cond before it is GPU runnable.
  const TimeField& t = t_->field_ref();
  const double orig0 = origin_[0] + oscillatingdir_[0]*amplitude_*sin(frequency_ * t[0] );
  const double orig1 = origin_[1] + oscillatingdir_[1]*amplitude_*sin(frequency_ * t[0] );
  
  const SVolField& field1 = field1_->field_ref();
  const SVolField& field2 = field2_->field_ref();

  SVolField& result = this->value();
  result <<= cond( (field1 - orig0) * (field1 - orig0) + (field2 - orig1)*(field2 - orig1) - radius_*radius_ <= 0,
                   insidevalue_)
                 ( outsidevalue_ );
}

//--------------------------------------------------------------------

OscillatingCylinder::Builder::
Builder( const Expr::Tag& result,
         const std::string& axis,
         const std::vector<double>& origin,
         const std::vector<double>& oscillatingdir,
         const double insideValue,
         const double outsideValue,
         const double radius,
         const double frequency,
         const double amplitude )
: ExpressionBuilder(result),
  axis_(axis),
  origin_(origin),
  oscillatingdir_(oscillatingdir),
  insidevalue_ (insideValue ),
  outsidevalue_(outsideValue),
  radius_(radius),
  frequency_(frequency),
  amplitude_(amplitude)
{}

//--------------------------------------------------------------------

Expr::ExpressionBase*
OscillatingCylinder::Builder::
build() const
{
  return new OscillatingCylinder( axis_, origin_, oscillatingdir_,
                                  insidevalue_, outsidevalue_, radius_,
                                  frequency_, amplitude_ );
}

//==========================================================================
