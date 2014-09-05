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

#include "SetCurrentTime.h"

namespace Wasatch{

  SetCurrentTime::SetCurrentTime()
    : Expr::Expression<double>(),
      rkStage_( 0 ),
      deltat_ ( 0.0 ),
      simTime_( 0.0 )
  {}

  //--------------------------------------------------------------------

  SetCurrentTime::~SetCurrentTime()
  {}

  //--------------------------------------------------------------------

  void
  SetCurrentTime::evaluate()
  {
    assert( deltat_  >= 0.0 );
    assert( simTime_ >= 0.0 );

    typedef std::vector<double*>& DoubleVec;
    DoubleVec results = this->get_value_vec();
    *results[0] = simTime_;
    *results[1] = deltat_;
    if( rkStage_ == 2 ) *results[0] += deltat_;
    if( rkStage_ == 3 ) {
      *results[0] += 0.5*deltat_;
      *results[1]  = 0.5*deltat_;
    }
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

} // namespace Wasatch
