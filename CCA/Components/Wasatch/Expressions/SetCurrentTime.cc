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

#include <expression/ExprLib.h>

#include <CCA/Components/Wasatch/Expressions/SetCurrentTime.h>

namespace Wasatch{

  SetCurrentTime::SetCurrentTime( const Uintah::SimulationStateP sharedState,
                                  const int RKStage )
    : Expr::Expression<double>(),
      state_( sharedState ),
      RKStage_( RKStage )
  {}

  //--------------------------------------------------------------------

  SetCurrentTime::~SetCurrentTime()
  {}

  //--------------------------------------------------------------------

  void
  SetCurrentTime::evaluate()
  {
    double& result = this->value();
    result = state_->getElapsedTime();
    if (RKStage_ == 2) result += deltat_;
    if (RKStage_ == 3) result += deltat_*0.5;
    //std::cout<<"Current Simulation Time: " << result << " STAGE = "<< RKStage_	 << std::endl;
  }

  //--------------------------------------------------------------------

  SetCurrentTime::Builder::Builder( const Expr::Tag& result,
                                    const Uintah::SimulationStateP sharedState,
                                    const int RKStage )
    : ExpressionBuilder(result),
      state_( sharedState ),
      RKStage_(RKStage)
  {}

  //--------------------------------------------------------------------

  Expr::ExpressionBase*
  SetCurrentTime::Builder::build() const
  {
    return new SetCurrentTime( state_, RKStage_ );
  }

} // namespace Wasatch
