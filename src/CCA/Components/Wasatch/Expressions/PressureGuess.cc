/*
 * The MIT License
 *
 * Copyright (c) 2012-2026 The University of Utah
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

#include <CCA/Components/Wasatch/Expressions/PressureGuess.h>
#include <CCA/Components/Wasatch/TagNames.h>
#include <CCA/Components/Wasatch/Wasatch.h>
//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>
#include <spatialops/structured/FVStaggered.h>

PressureGuess::PressureGuess(const int order)
:Expr::Expression<SpatialOps::SVolField>(),
numtimesteps_(order)
{
  this->set_gpu_runnable( true );
  const Expr::Tag rkst = WasatchCore::TagNames::self().rkstage;
  const Expr::Tag dtt = WasatchCore::TagNames::self().dt; 
  const Expr::Tag pressuret = WasatchCore::TagNames::self().pressure;
  oldPressureTags_= WasatchCore::get_old_var_taglist(pressuret, order);
  oldDtTags_= WasatchCore::get_old_var_taglist(WasatchCore::TagNames::self().dt, WasatchCore::Wasatch::get_old_dt_num());

  rkStage_ = create_field_request<TimeField>(rkst);
  dt_= create_field_request<TimeField>(dtt);
  create_field_vector_request<PFieldT>( oldPressureTags_, old_pressure_Fields_ );
  create_field_vector_request<TimeField>( oldDtTags_, old_dts_ );

  std::string integName = WasatchCore::Wasatch::get_timeIntegratorName();
  if (integName == "RK2SSP")
        pressure_approximation_helper_ = new PressureApproximationsHelper(new RK2Approx);
  else if (integName == "RK3SSP")
        pressure_approximation_helper_ = new PressureApproximationsHelper(new RK3Approx);
}

//--------------------------------------------------------------------
PressureGuess::~PressureGuess()
{
  delete pressure_approximation_helper_;
}
//--------------------------------------------------------------------

void
PressureGuess::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  timeIntInfo_ = opDB.retrieve_operator<WasatchCore::TimeIntegrator>();
}

//--------------------------------------------------------------------
void
PressureGuess::evaluate()
{
  SpatialOps::SVolField& pressureGuessValue = this->value();
  if (WasatchCore::Wasatch::low_cost_integ_recompiled())
    pressure_approximation_helper_->compute_approximation(rkStage_, dt_, pressureGuessValue, old_pressure_Fields_, old_dts_, timeIntInfo_); 
  else 
    pressureGuessValue <<= 0.0;
}

//--------------------------------------------------------------------
PressureGuess::Builder::Builder( const Expr::Tag& result,
                         const int order)
: ExpressionBuilder(result),
order_( order )
{}

//--------------------------------------------------------------------

Expr::ExpressionBase*
PressureGuess::Builder::build() const
{
  return new PressureGuess( order_ );
}
