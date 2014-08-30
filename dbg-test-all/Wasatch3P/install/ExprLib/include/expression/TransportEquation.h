/**
 *  \file   TransportEquation.h
 *  \author James C. Sutherland
 *
 * Copyright (c) 2011 The University of Utah
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

#ifndef Expr_TransportEquation_h
#define Expr_TransportEquation_h

#include <string>

#include <expression/ExpressionFactory.h>
#include <expression/ExprFwd.h>
#include <expression/ExpressionID.h>
#include <expression/Context.h>

namespace Expr{

  class ExprDeps;  // forward declaration.

  /**
   *  @class  TransportEquation
   *  @author James C. Sutherland
   *  @date   September, 2008
   *  @brief  Base class for defining a transport equation.
   */
  class TransportEquation
  {
  public:


    /**
     *  @brief Construct a TransportEquation
     *
     *  @param solutionVarName The name of the solution variable that this
     *         TransportEquation describes
     *
     *  @param rhsExprID The ExpressionID for the RHS expression.
     */
    TransportEquation( const std::string solutionVarName,
                       const ExpressionID rhsExprID );

    /**
     *  @brief The base class constructor registers the expression
     *         associated with this transport equation
     *
     *  @param exprFactory The ExpressionFactory that manages creation
     *         of expressions.
     *
     *  @param solnVarName The name of the solution variable that this
     *         TransportEquation describes
     *
     *  @param solnVarRHSBuilder The ExpressionBuilder for the
     *         expression that will calculate the RHS for this
     *         TransportEquation.
     */
    TransportEquation( ExpressionFactory& exprFactory,
                       const std::string solnVarName,
                       const ExpressionBuilder* const solnVarRHSBuilder );

    virtual ~TransportEquation();

    /**
     *  @brief Obtain the name of the solution variable for this transport equation.
     */
    inline const std::string& solution_variable_name() const{ return solnVarName_; }

    inline ExpressionID get_rhs_id() const{ return rhsExprID_; }

    /**
     *  Set up the boundary condition evaluators for this
     *  TransportEquation. Each derived class must implement this
     *  method.  Boundary conditions are imposed by adding additional
     *  tasks to modify values in an Expression after it is evaluated.
     *  This is done by attaching a functor to the applicable expression
     *  via the <code>Expression::process_after_evaluate</code> method.
     */
    virtual void setup_boundary_conditions( ExpressionFactory& factory ) = 0;

    /**
     *  Return the ExpressionID that identifies an expression that will
     *  set the initial condition for this transport equation.
     *
     *  NOTE: the ExpressionFactory object provided here is distinctly
     *  different than the one provided at construction of the
     *  TransportEquation object.  Therefore, you should not assume that
     *  any expressions registered at construction of this
     *  TransportEquation are registered for purposes of calculating the
     *  initial conditions.
     */
    virtual ExpressionID initial_condition( ExpressionFactory& exprFactory );

  protected:
    const std::string  solnVarName_;  ///< Name of the solution variable for this TransportEquation.
    const ExpressionID rhsExprID_;    ///< The label for the rhs expression for this TransportEquation.
  };

} // namespace Expr

#endif // Expr_TransportEquation_h
