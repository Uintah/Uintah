/*
 * TransportEquation.cpp
 *
 *  Created on: Aug 8, 2012
 *      Author: "James C. Sutherland"
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

#include "TransportEquation.h"

#include <stdexcept>
#include <sstream>

namespace Expr{

  TransportEquation::TransportEquation( const std::string solutionVarName,
                                        const ExpressionID rhsExprID )
  : solnVarName_( solutionVarName ),
    rhsExprID_( rhsExprID )
  {}

  TransportEquation::TransportEquation( ExpressionFactory& exprFactory,
                                        const std::string solnVarName,
                                        const ExpressionBuilder* const solnVarRHSBuilder )
  : solnVarName_( solnVarName ),
    rhsExprID_( exprFactory.register_expression(solnVarRHSBuilder) )
  {}

  TransportEquation::~TransportEquation()
  {}

  ExpressionID
  TransportEquation::initial_condition( ExpressionFactory& exprFactory )
  {
    using namespace std;
    ExpressionID id;
    try{
      id = exprFactory.get_id( Tag(solnVarName_,STATE_N) );
    }
    catch( exception& err ){
      ostringstream msg;
      msg << __FILE__ << " : " << __LINE__ << endl
          << "No initial condition expression was found for " << solnVarName_ << endl
          << err.what();
      throw std::runtime_error( msg.str() );
    }
    return id;
  }
}
