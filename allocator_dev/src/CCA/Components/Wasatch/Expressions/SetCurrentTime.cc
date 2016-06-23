/*
 * The MIT License
 *
 * Copyright (c) 2012-2016 The University of Utah
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

namespace WasatchCore{

  SetCurrentTime::SetCurrentTime()
    : Expr::Expression<SpatialOps::SingleValueField>(),
      rkStage_ ( 1   ),
      deltat_  ( 0.0 ),
      simTime_ ( 0.0 ),
      timeStep_( 0.0 )
  {
    this->set_gpu_runnable(true);
    timeCor_[0] = 0.0; // for the first rk stage, the time is t0
    timeCor_[1] = 1.0; // for the second rk stage, the time is t0 + dt
    timeCor_[2] = 0.5; // for the third rk stage, the time is t0 + 0.5*dt
  }

  //--------------------------------------------------------------------

  SetCurrentTime::~SetCurrentTime()
  {}

  //--------------------------------------------------------------------

  void
  SetCurrentTime::evaluate()
  {
    using namespace SpatialOps;
    assert( deltat_  >= 0.0 );
    assert( simTime_ >= 0.0 );

    typedef std::vector<SpatialOps::SingleValueField*> Vec;
    Vec& results = this->get_value_vec();
    *results[0] <<= simTime_ + timeCor_[rkStage_ - 1]*deltat_;
    *results[1] <<= deltat_;
    *results[2] <<= timeStep_;
    *results[3] <<= (double) rkStage_;
  }

  //--------------------------------------------------------------------

  SetCurrentTime::Builder::Builder( const Expr::TagList& resultsTags )
    : ExpressionBuilder(resultsTags)
  {}

  //--------------------------------------------------------------------

  Expr::ExpressionBase*
  SetCurrentTime::Builder::build() const
  {
    return new SetCurrentTime();
  }

} // namespace WasatchCore
