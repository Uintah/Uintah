/*
 * The MIT License
 *
 * Copyright (c) 2012-2017 The University of Utah
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

#ifndef Wasatch_CoalEquation_h
#define Wasatch_CoalEquation_h

#include <string>

//-- ExprLib includes --//
#include <expression/Tag.h>
#include <expression/ExpressionFactory.h>
#include <expression/ExpressionID.h>
#include <expression/ExprLib.h>
#include <expression/Context.h>

//-- Uintah framework includes --//
#include <Core/Grid/Patch.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/PatchInfo.h>
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/Expressions/RHSTerms.h>
#include <CCA/Components/Wasatch/Transport/EquationBase.h>

namespace WasatchCore{
  class ExprDeps;  // forward declaration.
  class GraphHelper;
  class BCHelper;
}

namespace Coal{

  using WasatchCore::ExprDeps;
  using WasatchCore::GraphHelper;
  using WasatchCore::BCHelper;
  using WasatchCore::WasatchBCHelper;
  using WasatchCore::GraphCategories;
  using WasatchCore::EquationBase;

  class CoalEquation : public EquationBase {

  public:

    /**
     * @brief Construct a CoalEquation (used for coal mass-balances)
     * \param solnVarName The name of the solution variable for this equation
     *
     * \param particleMassTag Particle mass tag.
     *
     * \param initialmassFraction The initial mass fraction for the component being solved for.
     *
     * \param gc The GraphCategories object from Wasatch
     */
    CoalEquation( const std::string& solnVarName,
                  const Expr::Tag&   particleMassTag,
                  const double       initialMassFraction,
                  GraphCategories&   gc );

    /**
     * @brief Construct a CoalEquation (used for a generic ODE)
     * \param solnVarName The name of the solution variable for this equation
     *
     * \param initialValue The initial value for the variable being solved for.
     *
     * \param gc The GraphCategories object from Wasatch
     */
    CoalEquation( const std::string& solnVarName,
                  const double       initialvalue,
                  GraphCategories&   gc );

    /**
     *  \brief setup the initial conditions for this transport equation.
     */
    Expr::ExpressionID initial_condition( Expr::ExpressionFactory& );


    /*
     * \brief ODEs related to the coal do not require boundary conditions so
     *        BC-related functions do nothing.
     */
    void setup_boundary_conditions( WasatchBCHelper& bcHelper,
                                    GraphCategories& graphCat)
    {}

    /**
     *  \brief Set up the boundary condition on initial conditions equation.
     */
    void apply_initial_boundary_conditions( const GraphHelper& graphHelper,
                                            WasatchBCHelper& bcHelper )
    {}

    /**
     *  \brief Set up the boundary condition evaluators for this
     *  Equation.
     */
    void apply_boundary_conditions( const GraphHelper& graphHelper,
                                    WasatchBCHelper& bcHelper )
    {}

    ~CoalEquation(){}
    
  private:
    const Expr::Tag pMassTag_;
    const double    initialValue_;
    void setup();
//    Expr::ExpressionID setup_rhs();
  };

  typedef std::vector<CoalEquation*> CoalEqVec;

} // namespace coal

#endif // Wasatch_CoalEquation_h
