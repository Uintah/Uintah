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

#include <expression/ExprLib.h>

#include <CCA/Components/Wasatch/Expressions/SetCurrentTime.h>

namespace Wasatch{

  SetCurrentTime::SetCurrentTime( const Uintah::SimulationStateP sharedState,
                                  const int RKStage )
    : Expr::Expression<double>(),
      state_( sharedState ),
      RKStage_( RKStage ),
      deltat_ ( 0.0 )
  {}

  //--------------------------------------------------------------------

  SetCurrentTime::~SetCurrentTime()
  {}

  //--------------------------------------------------------------------

  void
  SetCurrentTime::evaluate()
  {
    //std::cout << "set current time deltat: " << deltat_ << std::endl;
    typedef std::vector<double*>& doubleVec;
    doubleVec results = this->get_value_vec();
    const double delt = state_->d_current_delt;
    const double elapsedTime = state_->getElapsedTime();
    *results[0] = elapsedTime;
    *results[1] = delt;
    if (RKStage_ == 2) *results[0] += delt;
    if (RKStage_ == 3) {
        *results[0] += delt*0.5;
        *results[1]  = 0.5*delt;
    }
  }

  //--------------------------------------------------------------------

  SetCurrentTime::Builder::Builder( const Expr::TagList& resultsTags,
                                    const Uintah::SimulationStateP sharedState,
                                    const int RKStage )
    : ExpressionBuilder(resultsTags),
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
