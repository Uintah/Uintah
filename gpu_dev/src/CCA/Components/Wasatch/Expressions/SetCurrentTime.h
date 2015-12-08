/*
 * The MIT License
 *
 * Copyright (c) 2012-2015 The University of Utah
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

#ifndef SetCurrentTime_Expr_h
#define SetCurrentTime_Expr_h

#include <expression/Expression.h>

namespace WasatchCore{

/**
 *  \class 	SetCurrentTime
 *  \ingroup 	Expressions
 *  \author 	James C. Sutherland
 *
 *  \brief Provides a simple expression to set the current simulation
 *         time.  May be needed for time-varying BCs, etc.
 */
class SetCurrentTime
 : public Expr::Expression<SpatialOps::SingleValueField>
{
  int rkStage_;
  double deltat_, simTime_, timeStep_;
  double timeCor_[3]; // time correction in rkstages
  SetCurrentTime();

public:

  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::TagList& resultsTags );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  };

  ~SetCurrentTime();

  void evaluate();
  void set_integrator_stage( const int rkStage ){rkStage_ = rkStage;}
  void set_deltat( const double deltat ) { deltat_ = deltat; }
  void set_timestep( const int ts ){ timeStep_ = (double) ts; }
  void set_time  ( const double t ){ simTime_ = t; }
};

} // namespace WasatchCore

#endif // SetCurrentTime_Expr_h
