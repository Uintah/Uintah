/*
 * The MIT License
 *
 * Copyright (c) 1997-2024 The University of Utah
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

#include<CCA/Components/Wasatch/Expressions/PostProcessing/InstantaneousPressure.h>

#include <CCA/Components/Wasatch/OldVariable.h>
#include<CCA/Components/Wasatch/TagNames.h>

InstantaneousPressure::InstantaneousPressure(const int order)
:Expr::Expression<SpatialOps::SVolField>(),
order_(order)
{
  this->set_gpu_runnable( true );

  const Expr::Tag timestept = WasatchCore::TagNames::self().timestep;
  const Expr::Tag pressuret = WasatchCore::TagNames::self().pressure;

  timestep_ = create_field_request<TimeField>(timestept);
  
  oldPressureTags_= WasatchCore::get_old_var_taglist(pressuret, order);

  create_field_vector_request<PFieldT>( oldPressureTags_, old_pressure_Fields_ );
}

//--------------------------------------------------------------------
InstantaneousPressure::~InstantaneousPressure()
{}
//--------------------------------------------------------------------

void
InstantaneousPressure::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{}

//--------------------------------------------------------------------
void
InstantaneousPressure::evaluate()
{
  SpatialOps::SVolField& instantenousPressureValue = this->value();
  const TimeField& timestepValue = timestep_->field_ref();
  if (*timestepValue.begin() > order_)
    {
      switch (order_)
      {
      case 2:
        instantenousPressureValue <<= (5.0 * old_pressure_Fields_[0]->field_ref() - 3.0 * old_pressure_Fields_[1]->field_ref())/2.0 ;
        break;
      
      case 3:
        instantenousPressureValue <<= 13.0 * old_pressure_Fields_[0]->field_ref() /3.0  - 31.0 * old_pressure_Fields_[1]->field_ref()/6.0  + 11.0 * old_pressure_Fields_[2]->field_ref()/6.0 ;
        break;
      default:
        instantenousPressureValue <<= 0.0;
        break;
      }
    }
  else
    instantenousPressureValue <<= 0.0;
}

//--------------------------------------------------------------------
InstantaneousPressure::Builder::Builder( const Expr::Tag& result,
                         const int order)
: ExpressionBuilder(result),
order_( order )
{}

//--------------------------------------------------------------------

Expr::ExpressionBase*
InstantaneousPressure::Builder::build() const
{
  return new InstantaneousPressure( order_ );
}
