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

#ifndef Wasatch_TransportEquation_h
#define Wasatch_TransportEquation_h

#include <string>

//-- ExprLib includes --//
#include <expression/Tag.h>
#include <expression/ExpressionFactory.h>
#include <expression/ExpressionID.h>
#include <expression/Context.h>

//-- Uintah framework includes --//
#include <Core/Grid/Patch.h>

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/PatchInfo.h>
#include <CCA/Components/Wasatch/ParseTools.h>

namespace Wasatch{

  class ExprDeps;  // forward declaration.
  class GraphHelper;
  /**
   *  \ingroup WasatchCore
   *  \class  TransportEquation
   *  \author Tony Saad
   *  \date   April, 2011
   *  \brief  Base class for defining a transport equation.
   */
  class TransportEquation
  {
  public:


    /**
     *  \brief Construct a TransportEquation
     *
     *  \param solutionVarName The name of the solution variable that
     *         this TransportEquation describes
     *
     *  \param rhsExprID The ExpressionID for the RHS expression.
     */
    TransportEquation( const std::string solutionVarName,
                       const Expr::ExpressionID rhsExprID )
      : solnVarName_( solutionVarName ),
        rhsExprID_( rhsExprID ),
        stagLoc_( NODIR )
    {}

    /**
     *  \brief Construct a TransportEquation
     *
     *  \param solutionVarName The name of the solution variable that this
     *         TransportEquation describes
     *
     *  \param rhsExprID The ExpressionID for the RHS expression.
     *
     *  \param stagLoc the staggered location.
     */
    TransportEquation( const std::string solutionVarName,
                       const Expr::ExpressionID rhsExprID,
                       const Direction stagLoc,
                       Uintah::ProblemSpecP params=NULL)
      : solnVarName_( solutionVarName ),
        rhsExprID_( rhsExprID ),
        stagLoc_( stagLoc ),
        volFracTag_( Expr::Tag() ),
        hasVolFrac_( false )
    {
      if (params && params->findBlock("VolumeFractionExpression")) {
        volFracTag_ = parse_nametag( params->findBlock("VolumeFractionExpression")->findBlock("NameTag") );
        hasVolFrac_ = true;
      }
    }

    virtual ~TransportEquation(){}

    /**
     *  \brief Obtain the name of the solution variable for thisa transport equation.
     */
    const std::string& solution_variable_name() const{ return solnVarName_; }

    /**
     *  \brief Obtain the staggered location of the solution variable that is
     *  governed by this transport equation.
     */
    const Direction staggered_location() const{ return stagLoc_; }

    /**
     *  \brief Obtain the name (i.e. string) staggered location of the solution
     *  variable that is governed by this transport equation.
     */
    const std::string dir_name() const {
      switch (stagLoc_) {
      case XDIR:
        return "x";
        break;
      case YDIR:
        return "y";
      case ZDIR:
        return "z";
      case NODIR:
      default:
        return "";
      }
    }

    Expr::ExpressionID get_rhs_id() const{ return rhsExprID_; }

    /**
     *  Set up the boundary condition on initial conditions evaluators for this
     *  TransportEquation. Each derived class must implement this
     *  method.  Boundary conditions are imposed by adding additional
     *  tasks to modify values in an Expression after it is evaluated.
     *  This is done by attaching a functor to the applicable expression
     *  via the <code>Expression::process_after_evaluate</code> method.
     */
    virtual void setup_initial_boundary_conditions( const GraphHelper& graphHelper,
                                                    const Uintah::PatchSet* const localPatches,
                                                    const PatchInfoMap& patchInfoMap,
                                                    const Uintah::MaterialSubset* const materials ) = 0;


    /**
     *  Set up the boundary condition evaluators for this
     *  TransportEquation. Each derived class must implement this
     *  method.  Boundary conditions are imposed by adding additional
     *  tasks to modify values in an Expression after it is evaluated.
     *  This is done by attaching a functor to the applicable expression
     *  via the <code>Expression::process_after_evaluate</code> method.
     */
    virtual void setup_boundary_conditions( const GraphHelper& graphHelper,
                                            const Uintah::PatchSet* const localPatches,
                                            const PatchInfoMap& patchInfoMap,
                                            const Uintah::MaterialSubset* const materials ) = 0;

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
    virtual Expr::ExpressionID initial_condition( Expr::ExpressionFactory& exprFactory ) = 0;

  protected:
    const std::string  solnVarName_;      ///< Name of the solution variable for this TransportEquation.
    const Expr::ExpressionID rhsExprID_;  ///< The label for the rhs expression for this TransportEquation.
    const Direction stagLoc_;             ///< staggered direction for this equation
    Expr::Tag volFracTag_;    
    bool hasVolFrac_;
  };

} // namespace Wasatch

#endif // Wasatch_TransportEquation_h
