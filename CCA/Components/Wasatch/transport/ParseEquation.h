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

#ifndef Wasatch_ParseEquations_h
#define Wasatch_ParseEquations_h

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Ports/SolverInterface.h>

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/GraphHelperTools.h>
#include <CCA/Components/Wasatch/transport/TransportEquation.h>
#include <CCA/Components/Wasatch/ConvectiveInterpolationMethods.h>
#include <CCA/Components/Wasatch/Expressions/Turbulence/TurbulenceParameters.h>

#include <CCA/Components/Wasatch/Expressions/RHSTerms.h>

/**
 *  \file ParseEquation.h
 *  \brief Parser tools for transport equations.
 */

namespace Wasatch{

  class TimeStepper;
  class TransportEquation;

  /** \addtogroup WasatchParser
   *  @{
   */

  /**
   *  \class EqnTimestepAdaptorBase
   *  \author James C. Sutherland, Tony Saad, Amir Biglari
   *  \date June, 2010
   *
   *  This serves as a means to have a container of adaptors.  These
   *  adaptors will plug a strongly typed transport equation into a
   *  time integrator, preserving the type information which is
   *  required for use by the integrator.
   */
  class EqnTimestepAdaptorBase
  {
  protected:
    EqnTimestepAdaptorBase( TransportEquation* eqn);
    TransportEquation* const eqn_;

  public:
    virtual ~EqnTimestepAdaptorBase();
    virtual void hook( TimeStepper& ts ) const = 0;
    TransportEquation* equation(){ return eqn_; }
    const TransportEquation* equation() const{ return eqn_; }
  };


  /**
   *  \brief Build the transport equation specified by "params"
   *
   *  \param params the tag from the input file specifying the
   *         transport equation.
   *
   *  \param densityTag a tag for the density to be passed to
   *         the scalar transport equations if it is needed.
   *         othwise it will be an empty tag.
   *
   *  \param gc the GraphCategories.
   *
   *  \return an EqnTimestepAdaptorBase object that can be used to
   *          plug this transport equation into a TimeStepper.
   */
  EqnTimestepAdaptorBase*
  parse_equation( Uintah::ProblemSpecP params,
                  TurbulenceParameters turbParams,
                  const Expr::Tag densityTag,
                  const bool isConstDensity,
                  GraphCategories& gc );

  /**
   *  \brief Build the momentum equation specified by "params"
   *
   *  \param params The XML block from the input file specifying the
   *         momentum equation. This will be \verbatim <MomentumEquations>\endverbatim.
   *
   *  \param gc The GraphCategories.
   *
   *  \return an EqnTimestepAdaptorBase object that can be used to
   *          plug this transport equation into a TimeStepper.
   */
  std::vector<EqnTimestepAdaptorBase*> parse_scalability_test( Uintah::ProblemSpecP params,
                                                               GraphCategories& gc );

  /**
   *  \brief Build the momentum equation specified by "params"
   *
   *  \param params The XML block from the input file specifying the
   *         momentum equation. This will be \verbatim <MomentumEquations>\endverbatim.
   *
   *  \param gc The GraphCategories.
   *
   *  \return an EqnTimestepAdaptorBase object that can be used to
   *          plug this transport equation into a TimeStepper.
   */
  std::vector<EqnTimestepAdaptorBase*> parse_momentum_equations( Uintah::ProblemSpecP params,
                                                                 TurbulenceParameters turbParams,
                                                                 const bool hasEmbeddedGeometry,
                                                                 const bool hasMovingGeometry,
                                                                 const Expr::Tag densityTag,
                                                                 GraphCategories& gc,
                                                                 Uintah::SolverInterface& linSolver,
                                                                 Uintah::SimulationStateP& sharedState);
  void parse_poisson_equation( Uintah::ProblemSpecP params,
                               GraphCategories& gc,
                               Uintah::SolverInterface& linSolver,
                               Uintah::SimulationStateP& sharedState);
  
  /**
   *  \brief Build moment transport equations specified by "params"
   *
   *  \param params The XML block from the input file specifying the
   *         momentum equation. This will be <MomentumEquations>.
   *
   *  \param densityTag a tag for the density to be passed to
   *         the momentum equations if it is needed. othwise
   *         it will be an empty tag.
   *
   *  \param gc The GraphCategories.
   *
   *  \return a vector of EqnTimestepAdaptorBase objects that can be used to
   *          plug this transport equation into a TimeStepper.
   */
  std::vector<EqnTimestepAdaptorBase*> parse_moment_transport_equations( Uintah::ProblemSpecP params,
                                                                         Uintah::ProblemSpecP wasatchParams,
                                                                         const bool hasEmbeddedGeometry,
                                                                         GraphCategories& gc);


  /**
   * \brief Register diffusive flux calculation, \f$J_\phi = -\rho \Gamma_\phi \nabla \phi\f$,
   *        for the scalar quantity \f$ \phi \f$.
   * \param convFluxParams Parser block "DiffusiveFluxExpression"
   * \param solnVarTag the solution variable to be advected (\f$ \phi \f$).
   * \param volFracTag volume fraction tag - okay if empty for no volume fraction specification
   * \param factory the factory to register the resulting expression on
   * \param info the FieldTagInfo object that will be populated with the appropriate convective flux entry.
   */
  template< typename FieldT>
  void setup_diffusive_flux_expression( Uintah::ProblemSpecP diffFluxParams,
                                        const Expr::Tag densityTag,
                                        const Expr::Tag primVarTag,
                                        const bool isStrong,
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
   * \param volFracTag the volume fraction (optional, can leave empty)
   * \param convMethod the upwind method to use
   * \param advVelocityTag the advecting velocity, which lives at staggered cell centers
   * \param factory the factory to associate the convective flux expression with
   * \param info this will be populated for use in the ScalarRHS expression if needed.
   */
  template< typename FieldT >
  void setup_convective_flux_expression( const std::string dir,
                                         const Expr::Tag solnVarTag,
                                         Expr::Tag convFluxTag,
                                         const Expr::Tag volFracTag,
                                         const ConvInterpMethods convMethod,
                                         const Expr::Tag advVelocityTag,
                                         Expr::ExpressionFactory& factory,
                                         FieldTagInfo& info );

  /**
   * \brief Register convective flux calculation for the given scalar quantity
   * \param convFluxParams Parser block "ConvectiveFluxExpression"
   * \param solnVarTag the solution variable to be advected
   * \param volFracTag volume fraction tag - okay if empty for no volume fraction specification
   * \param factory the factory to register the resulting expression on
   * \param info the FieldTagInfo object that will be populated with the appropriate convective flux entry.
   */
  template< typename FieldT >
  void setup_convective_flux_expression( Uintah::ProblemSpecP convFluxParams,
                                         const Expr::Tag solnVarTag,
                                         const Expr::Tag volFracTag,
                                         Expr::ExpressionFactory& factory,
                                         FieldTagInfo& info );

  /** @} */

}// namespace Wasatch

#endif // Wasatch_ParseEquations_h
