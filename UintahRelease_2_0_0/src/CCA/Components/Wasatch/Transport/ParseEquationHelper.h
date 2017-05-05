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

#ifndef Wasatch_ParseEquationHelper_h
#define Wasatch_ParseEquationHelper_h

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/TimeStepper.h>
#include <CCA/Components/Wasatch/Transport/EquationBase.h>
#include <CCA/Components/Wasatch/ConvectiveInterpolationMethods.h>

namespace WasatchCore{

  /**
   * \brief Register diffusive flux calculation, \f$J_\phi = -\rho \Gamma_\phi \nabla \phi\f$,
   *        for the scalar quantity \f$ \phi \f$.
   * \param diffFluxParams Parser block "DiffusiveFlux"
   * \param densityTag the mixture mass density
   * \param primVarTag The primitive variable, \f$\phi\f$.
   * \param turbDiffTag The scalar turbulent diffusivity
   * \param factory the factory to register the resulting expression on
   * \param info the FieldTagInfo object that will be populated with the appropriate convective flux entry.
   */
  template< typename FieldT>
  void setup_diffusive_flux_expression( Uintah::ProblemSpecP diffFluxParams,
                                        const Expr::Tag densityTag,
                                        const Expr::Tag primVarTag,
                                        const Expr::Tag turbDiffTag,
                                        Expr::ExpressionFactory& factory,
                                        FieldTagInfo& info );
  template< typename FieldT>
  void setup_diffusive_velocity_expression( Uintah::ProblemSpecP diffVelParams,
                                            const Expr::Tag primVarTag,
                                            const Expr::Tag turbDiffTag,
                                            Expr::ExpressionFactory& factory,
                                            FieldTagInfo& info );

  /**
   * \brief Build the convective flux expression
   * \param dir the direction that this flux is associated with
   * \param solnVarTag the solution variable tag
   * \param convFluxTag the convective flux tag - leave empty to assemble a
   *        flux, populate it to use a flux expression that already exists.
   * \param convMethod the upwind method to use
   * \param advVelocityTag the advecting velocity, which lives at staggered cell centers
   * \param factory the factory to associate the convective flux expression with
   * \param info this will be populated for use in the ScalarRHS expression if needed.
   */
  template< typename FieldT >
  void setup_convective_flux_expression( const std::string& dir,
                                         const Expr::Tag& solnVarTag,
                                         Expr::Tag convFluxTag,
                                         const ConvInterpMethods convMethod,
                                         const Expr::Tag& advVelocityTag,
                                         Expr::ExpressionFactory& factory,
                                         FieldTagInfo& info );
  
  /**
   * \brief Register convective flux calculation for the given scalar quantity
   * \param convFluxParams Parser block "ConvectiveFlux"
   * \param solnVarTag the solution variable to be advected
   * \param factory the factory to register the resulting expression on
   * \param info the FieldTagInfo object that will be populated with the appropriate convective flux entry.
   */
  template< typename FieldT >
  void setup_convective_flux_expression( Uintah::ProblemSpecP convFluxParams,
                                         const Expr::Tag& solnVarTag,
                                         Expr::ExpressionFactory& factory,
                                         FieldTagInfo& info );

}// namespace WasatchCore

#endif // Wasatch_ParseEquations_h
