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
#include <Core/ProblemSpec/ProblemSpecP.h>

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/PatchInfo.h>
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/Expressions/RHSTerms.h>

namespace Wasatch{

  class ExprDeps;  // forward declaration.
  class GraphHelper;
  class BCHelper;
  /**
   *  \ingroup WasatchCore
   *  \class  TransportEquation
   *  \author Tony Saad
   *  \date   April, 2011
   *  \brief  Base class for defining a transport equation.
   */
  class TransportEquation{

  protected:

    void setup();

    /**
     * Setup the expression(s) to calculate the diffusive fluxes as applicable,
     * populating the FieldTagInfo object supplied with the appropriate
     * diffusive flux tag(s).
     */
    virtual void setup_diffusive_flux( FieldTagInfo& ) = 0;

    /**
     * Setup the expression(s) to calculate the convective fluxes as applicable,
     * populating the FieldTagInfo object supplied with the appropriate
     * convective flux tag(s).
     */
    virtual void setup_convective_flux( FieldTagInfo& ) = 0;

    /**
     * Setup the expression to calculate the source term as applicable,
     * populating the FieldTagInfo object supplied with the appropriate source
     * term tag.
     *
     * Tags for additional source terms may be populated on the supplied TagList.
     */
    virtual void setup_source_terms( FieldTagInfo&, Expr::TagList& ) = 0;

    /**
     * Setup the RHS expression for this transport equation, returning the
     * ExpressionID associated with it.
     *
     * @param info the FieldTagInfo describing terms that are active for the RHS
     * @param srcTags additional source terms for the RHS
     */
    virtual Expr::ExpressionID setup_rhs( FieldTagInfo& info,
                                          const Expr::TagList& srcTags ) = 0;

  public:

    /**
     * @brief Construct a TransportEquation
     * @param gc the GraphCategories object from Wasatch
     * @param solnVarName the name of the solution variable for this equation
     * @param params the parser information for this TransportEquation
     * @param stagLoc the direction that this equation is staggered
     * @param isConstDensity flag for constant density
     */
    TransportEquation( GraphCategories& gc,
                       const std::string solnVarName,
                       Uintah::ProblemSpecP params,
                       const Direction stagLoc,
                       const bool isConstDensity );

    virtual ~TransportEquation(){}

    /**
     *  \brief Obtain the name of the solution variable for this transport equation.
     */
    inline const std::string& solution_variable_name() const{ return solnVarName_; }

    /**
     *  \brief Obtain the tag of the solution variable for this transport equation.
     */
    inline const Expr::Tag& solution_variable_tag() const{ return solnVarTag_; }

    /**
     *  \brief Obtain the rhs tag of the solution variable for this transport equation.
     */
    inline const Expr::Tag& rhs_tag() const { return rhsTag_; }

    /**
     *  \brief Obtain the rhs name of the solution variable for this transport equation.
     */
    inline std::string rhs_name() const{ return rhsTag_.name(); }
    
    /**
     *  \brief Obtain the staggered location of the solution variable that is
     *  governed by this transport equation.
     */
    inline Direction staggered_location() const{ return stagLoc_; }

    /**
     *  \brief Obtain the name (i.e. string) staggered location of the solution
     *  variable that is governed by this transport equation.
     */
    std::string dir_name() const;

    /**
     * @return the ExpressionID for the RHS expression associated with this TransportEquation
     */
    Expr::ExpressionID get_rhs_id() const;

    /**
     *  \brief Used to check the validity of the boundary conditions specified
     *   by the user at a given boundary and also to infer/add new BCs on the
     *   type of boundary.  Example: at a stationary impermeable wall, we can
     *   immediately infer zero-velocity boundary conditions and check whether
     *   the user has specified any velocity BCs at that boundary. See examples
     *   in the momentum transport equation.
     */
    virtual void verify_boundary_conditions( BCHelper& bcHelper,
                                             GraphCategories& graphCat)=0;
    
    /**
     *  Set up the boundary condition on initial conditions evaluators for this
     *  TransportEquation. Each derived class must implement this
     *  method.  Boundary conditions are imposed by adding additional
     *  tasks to modify values in an Expression after it is evaluated.
     *  This is done by attaching a functor to the applicable expression
     *  via the <code>Expression::process_after_evaluate</code> method.
     */
    virtual void setup_initial_boundary_conditions( const GraphHelper& graphHelper,
                                                   BCHelper& bcHelper ) = 0;

    /**
     *  Set up the boundary condition evaluators for this
     *  TransportEquation. Each derived class must implement this
     *  method.  Boundary conditions are imposed by adding additional
     *  tasks to modify values in an Expression after it is evaluated.
     *  This is done by attaching a functor to the applicable expression
     *  via the <code>Expression::process_after_evaluate</code> method.
     */
    virtual void setup_boundary_conditions( const GraphHelper& graphHelper,
                                            BCHelper& bcHelper ) = 0;

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

    inline bool is_constant_density()   const { return isConstDensity_; }
    
  protected:
    Uintah::ProblemSpecP params_;
    GraphCategories& gc_;
    const std::string  solnVarName_;      ///< Name of the solution variable for this TransportEquation.
    const Expr::Tag solnVarTag_;          ///< Tag for the solution variable (at STATE_N)
    const Expr::Tag rhsTag_;              ///< Tag for the rhs
    const Direction stagLoc_;             ///< staggered direction for this equation
    const bool isConstDensity_;
    Expr::ExpressionID rhsExprID_;  ///< The label for the rhs expression for this TransportEquation.
  };

} // namespace Wasatch

#endif // Wasatch_TransportEquation_h
