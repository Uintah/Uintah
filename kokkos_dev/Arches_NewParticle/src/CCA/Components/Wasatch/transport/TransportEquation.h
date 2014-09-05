#ifndef Wasatch_TransportEquation_h
#define Wasatch_TransportEquation_h

#include <string>

//-- ExprLib includes --//
#include <expression/Tag.h>
#include <expression/ExpressionFactory.h>
#include <expression/Expr_ExpressionID.h>
#include <expression/Expr_Context.h>

//-- Uintah framework includes --//
#include <Core/Grid/Patch.h>

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/PatchInfo.h>

namespace Wasatch{
  
  class ExprDeps;  // forward declaration.
  class GraphHelper;
  /**
   *  \ingroup WasatchCore
   *  @class  TransportEquation
   *  @author Tony Saad
   *  @date   April, 2011
   *  @brief  Base class for defining a transport equation.
   */
  class TransportEquation
  {
  public:
    
    
    /**
     *  @brief Construct a TransportEquation
     *
     *  @param solnVarName The name of the solution variable that this
     *         TransportEquation describes
     *
     *  @param rhsExprID The ExpressionID for the RHS expression.
     */
    TransportEquation( const std::string solutionVarName,
                      const Expr::ExpressionID rhsExprID )
    : solnVarName_( solutionVarName ),
    rhsExprID_( rhsExprID ),
    stagLoc_( NODIR )
    {}
    
    /**
     *  @brief Construct a TransportEquation
     *
     *  @param solnVarName The name of the solution variable that this
     *         TransportEquation describes
     *
     *  @param rhsExprID The ExpressionID for the RHS expression.
     */
    TransportEquation( const std::string solutionVarName,
                      const Expr::ExpressionID rhsExprID,
                      const Direction stagLoc)
    : solnVarName_( solutionVarName ),
    rhsExprID_( rhsExprID ),
    stagLoc_( stagLoc )
    {}
    
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
     *
     *  @param rhsExprTag The Expr::Tag for the RHS expression
     *         (corresponding to the builder supplied in the previous
     *         argument)
     */
    TransportEquation( Expr::ExpressionFactory& exprFactory,
                      const std::string solnVarName,
                      const Expr::ExpressionBuilder* const solnVarRHSBuilder,
                      const Expr::Tag rhsExprTag )
    : solnVarName_( solnVarName ),
    rhsExprID_( exprFactory.register_expression(rhsExprTag,solnVarRHSBuilder) ),
    stagLoc_( NODIR )
    {}
    
    virtual ~TransportEquation(){}
    
    /**
     *  @brief Obtain the name of the solution variable for thisa transport equation.
     */
    const std::string& solution_variable_name() const{ return solnVarName_; }
    
    /**
     *  @brief Obtain the staggered location of the solution variable that is
     *  governed by this transport equation.
     */    
    const Direction staggered_location() const{ return stagLoc_; }
    
    /**
     *  @brief Obtain the name (i.e. string) staggered location of the solution 
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
          return "";
        default:
          break;
      }
    }
    
    Expr::ExpressionID get_rhs_id() const{ return rhsExprID_; }
    
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
                                           const Uintah::MaterialSubset* const materials) = 0;
    
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
    const std::string  solnVarName_;  ///< Name of the solution variable for this TransportEquation.
    const Expr::ExpressionID rhsExprID_;    ///< The label for the rhs expression for this TransportEquation.
    const Direction stagLoc_;
  };
  
} // namespace Wasatch

#endif // Wasatch_TransportEquation_h