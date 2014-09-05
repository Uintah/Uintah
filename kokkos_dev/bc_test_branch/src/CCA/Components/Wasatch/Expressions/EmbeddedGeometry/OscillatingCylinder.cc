/*
 * The MIT License
 *
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

#include "OscillatingCylinder.h"
//--------------------------------------------------------------------

OscillatingCylinder::
OscillatingCylinder(const std::string axis,
                    const std::vector<double> origin,
                    const std::vector<double> oscillatingdir,                    
                    const double insideValue,
                    const double outsideValue,
                    const double radius,
                    const double frequency,
                    const double amplitude)
: Expr::Expression<SVolField>(),
origin_(origin),
oscillatingdir_(oscillatingdir),
insidevalue_(insideValue),
outsidevalue_(outsideValue),
radius_(radius),
frequency_(frequency),
amplitude_(amplitude)
{
  if ( axis == "X" ) {
    tag1_ = Expr::Tag("YSVOL", Expr::STATE_NONE);
    tag2_ = Expr::Tag("ZSVOL", Expr::STATE_NONE);
  } else if ( axis == "Y" ) {
    tag1_ = Expr::Tag("XSVOL", Expr::STATE_NONE);
    tag2_ = Expr::Tag("ZSVOL", Expr::STATE_NONE);    
  } else if ( axis == "Z" ) {
    tag1_ = Expr::Tag("XSVOL", Expr::STATE_NONE);
    tag2_ = Expr::Tag("YSVOL", Expr::STATE_NONE);
  }
  timet_ = Expr::Tag("time", Expr::STATE_NONE );

}

//--------------------------------------------------------------------

void
OscillatingCylinder::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( tag1_ );
  exprDeps.requires_expression( tag2_ );
  const Expr::Tag timeVarTag( "time", Expr::STATE_NONE );
  exprDeps.requires_expression(timeVarTag);
}

//--------------------------------------------------------------------

void
OscillatingCylinder::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldMgrSelector<SVolField>::type& fm = fml.field_manager<SVolField>();
  field1_ = &fm.field_ref( tag1_ );
  field2_ = &fm.field_ref( tag2_ );
  
  // get the time tag
  t_ = &fml.field_manager<double>().field_ref( timet_ );
}

//--------------------------------------------------------------------

void
OscillatingCylinder::
evaluate()
{
  using namespace SpatialOps;
  const double orig0 = origin_[0] + oscillatingdir_[0]*amplitude_*sin(frequency_ * *t_);
  const double orig1 = origin_[1] + oscillatingdir_[1]*amplitude_*sin(frequency_ * *t_);
  SVolField& result = this->value();
  result <<= cond( (*field1_ - orig0) * (*field1_ - orig0) + (*field2_ - orig1)*(*field2_ - orig1) - radius_*radius_ <= 0,
                  insidevalue_)
                 ( outsidevalue_ );
}

//--------------------------------------------------------------------

OscillatingCylinder::Builder::
Builder( const Expr::Tag& result,
        const std::string axis,
        const std::vector<double> origin,
        const std::vector<double> oscillatingdir,        
        const double insideValue,
        const double outsideValue,
        const double radius,
        const double frequency,
        const double amplitude)
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
