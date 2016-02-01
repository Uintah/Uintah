/*
 * The MIT License
 *
 * Copyright (c) 2012-2016 The University of Utah
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
#include <CCA/Components/Wasatch/Transport/TransportEquation.h>
#include <CCA/Components/Wasatch/ConvectiveInterpolationMethods.h>
#include <CCA/Components/Wasatch/Expressions/Turbulence/TurbulenceParameters.h>

#include <CCA/Components/Wasatch/Expressions/RHSTerms.h>

/**
 *  \file ParseEquation.h
 *  \brief Parser tools for transport equations.
 */

namespace WasatchCore{

  class TimeStepper;
  class EquationBase;

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
    EqnTimestepAdaptorBase( EquationBase* eqn);
    EquationBase* const eqn_;

  public:
    virtual ~EqnTimestepAdaptorBase();
    virtual void hook( TimeStepper& ts ) const = 0;
    EquationBase* equation(){ return eqn_; }
    const EquationBase* equation() const{ return eqn_; }
  };


  /**
   *  \brief Build the transport equation specified by "params"
   *
   *  \param params the tag from the input file specifying the
   *         transport equation.
   *  \param turbParams
   *  \param densityTag a tag for the density to be passed to
   *         the scalar transport equations if it is needed.
   *         otherwise it will be an empty tag.
   *  \param isConstDensity true if density is constant
   *  \param gc the GraphCategories.
   *
   *  \return an EqnTimestepAdaptorBase object that can be used to
   *          plug this transport equation into a TimeStepper.
   */
  EqnTimestepAdaptorBase*
  parse_scalar_equation( Uintah::ProblemSpecP params,
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
   *  \param momentumSpec The XML block from the input file specifying the
   *         momentum equation. This will be \verbatim <MomentumEquations>\endverbatim.
   *  \param turbParams
   *  \param useAdaptiveDt true for variable dt
   *  \param isConstDensity true if density is constant
   *  \param densityTag the tag for the mixture mass density
   *  \param gc The GraphCategories.
   *  \param linSolver
   *  \param sharedState
   *
   *  \return an EqnTimestepAdaptorBase object that can be used to
   *          plug this transport equation into a TimeStepper.
   */
  std::vector<EqnTimestepAdaptorBase*>
  parse_momentum_equations( Uintah::ProblemSpecP wasatchSpec,
                            const TurbulenceParameters turbParams,
                            const bool useAdaptiveDt,
                            const bool doParticles,
                            const bool isConstDensity,
                            const Expr::Tag densityTag,
                            GraphCategories& gc,
                            Uintah::SolverInterface& linSolver,
                            Uintah::SimulationStateP& sharedState );

  void parse_poisson_equation( Uintah::ProblemSpecP params,
                               GraphCategories& gc,
                               Uintah::SolverInterface& linSolver,
                               Uintah::SimulationStateP& sharedState );

  void parse_radiation_solver( Uintah::ProblemSpecP params,
                               GraphHelper& gh,
                               Uintah::SolverInterface& linSolver,
                               Uintah::SimulationStateP& sharedState,
                               std::set<std::string>& persistentFields );


  /**
   *  \brief Build mms source terms and parse them properly to the RHS's
   *
   *  \param wasatchParams The XML block from the input file specifying the
   *         wasatch block.
   *
   *  \param varDensMMSParams The XML block from the input file specifying the
   *         parameters of the MMS. This will be \verbatim<VariableDensityMMS>\endverbatim.
   *
   *  \param computeContinuityResidual a boolean showing whether the continuity
   *         residual is being calculated or not
   *
   *  \param gc The GraphCategories.
   */
  void parse_var_den_mms( Uintah::ProblemSpecP wasatchParams,
                          Uintah::ProblemSpecP varDensMMSParams,
                          const bool computeContinuityResidual,
                          GraphCategories& gc);
  
  
  /**
   *  \brief Build 2D mms source terms and parse them properly to the RHS's
   *
   *  \param wasatchParams The XML block from the input file specifying the
   *         wasatch block.
   *
   *  \param varDensMMSParams The XML block from the input file specifying the
   *         parameters of the MMS. This will be \verbatim<VariableDensityMMS>\endverbatim.
   *
   *  \param computeContinuityResidual a boolean showing whether the continuity
   *         residual is being calculated or not
   *
   *  \param gc The GraphCategories.
   */
  void parse_var_den_oscillating_mms( Uintah::ProblemSpecP wasatchParams,
                                      Uintah::ProblemSpecP varDensMMSParams,
                                      const bool computeContinuityResidual,
                                      GraphCategories& gc);

  /**
   *  \brief Build moment transport equations specified by "params"
   *
   *  \param params The XML block from the input file specifying the
   *         momentum equation. This will be \verbatim <MomentumEquations> \endverbatim.
   *
   *  \param wasatchParams The XML block from the input file specifying the
   *         wasatch block.
   *
   *  \param isConstDensity true for constant density problems
   *
   *  \param gc The GraphCategories.
   *
   *  \return a vector of EqnTimestepAdaptorBase objects that can be used to
   *          plug this transport equation into a TimeStepper.
   */
  std::vector<EqnTimestepAdaptorBase*> parse_moment_transport_equations( Uintah::ProblemSpecP params,
                                                                         Uintah::ProblemSpecP wasatchParams,
                                                                         const bool isConstDensity,
                                                                         GraphCategories& gc);

  template<typename GasVel1T, typename GasVel2T, typename GasVel3T>
  std::vector<EqnTimestepAdaptorBase*>
  parse_particle_transport_equations( Uintah::ProblemSpecP particleSpec,
                                      Uintah::ProblemSpecP wasatchSpec,
                                      const bool useAdaptiveDt,
                                      GraphCategories& gc);


  /**
   * \brief Register diffusive flux calculation, \f$J_\phi = -\rho \Gamma_\phi \nabla \phi\f$,
   *        for the scalar quantity \f$ \phi \f$.
   * \param diffFluxParams Parser block "DiffusiveFlux"
   * \param densityTag the mixture mass density
   * \param primVarTag The primitive variable, \f$\phi\f$.
   * \param turbDiffTag The scalar turbulent diffusivity
   * \param suffix a string containing the "_*" suffix or not, according to whether we
   *        want to calculate the convection term at time step "n+1" or the current time step 
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
   * \param suffix a string containing the "_*" suffix or not, according to whether we
   *        want to calculate the convection term at time step "n+1" or the current time step 
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
   * \param suffix a string containing the "_*" suffix or not, according to whether we
   *        want to calculate the convection term at time step "n+1" or the current time step 
   * \param factory the factory to register the resulting expression on
   * \param info the FieldTagInfo object that will be populated with the appropriate convective flux entry.
   */
  template< typename FieldT >
  void setup_convective_flux_expression( Uintah::ProblemSpecP convFluxParams,
                                         const Expr::Tag& solnVarTag,
                                         Expr::ExpressionFactory& factory,
                                         FieldTagInfo& info );

  /** @} */

}// namespace WasatchCore

#endif // Wasatch_ParseEquations_h
