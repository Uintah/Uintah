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
#include <CCA/Components/Wasatch/Transport/EquationBase.h>

namespace WasatchCore{

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
  class TransportEquation : public EquationBase {

  protected:

    void setup();

    /**
     * \brief Setup the expression(s) to calculate the diffusive fluxes as applicable,
     * populating the FieldTagInfo object supplied with the appropriate
     * diffusive flux tag(s).
     */
    virtual void setup_diffusive_flux( FieldTagInfo& ) = 0;

    /**
     * \brief Setup the expression(s) to calculate the convective fluxes as applicable,
     * populating the FieldTagInfo object supplied with the appropriate
     * convective flux tag(s).
     */
    virtual void setup_convective_flux( FieldTagInfo& ) = 0;

    /**
     * \brief Setup the expression to calculate the source term as applicable,
     * populating the FieldTagInfo object supplied with the appropriate source
     * term tag.
     *
     * Tags for additional source terms may be populated on the supplied TagList.
     */
    virtual void setup_source_terms( FieldTagInfo&, Expr::TagList& ) = 0;

    /**
     * \brief Setup the RHS expression for this transport equation, returning the
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
     * @param stagLoc the direction that this equation is staggered
     * @param isConstDensity flag for constant density
     */
    TransportEquation( GraphCategories& gc,
                       const std::string& solnVarName,
                       const Direction stagLoc,
                       const bool isConstDensity );

    virtual ~TransportEquation(){}

    inline bool is_constant_density() const { return isConstDensity_; }

  protected:
    const bool isConstDensity_;
  };

} // namespace WasatchCore

#endif // Wasatch_TransportEquation_h
