/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
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

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Ports/SolverInterface.h>

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/GraphHelperTools.h>
#include <CCA/Components/Wasatch/Expressions/Turbulence/TurbulenceParameters.h>

/**
 *  \file ParseEquation.h
 *  \brief Parser tools for transport equations.
 */

namespace WasatchCore{
  
  class DualTimeMatrixInfo;  // forward declaration

  class EquationBase;

  /**
   *  \brief Build the transport equation specified by "params"
   *
   *  \param scalarEqnParams the tag from the input file specifying the
   *         transport equation.
   *  \param wasatchParams the tag from the input file specifying the wasatch
   *         component of uintah
   *  \param turbParams
   *  \param densityTag a tag for the density to be passed to
   *         the scalar transport equations if it is needed.
   *         otherwise it will be an empty tag.
   *  \param gc the GraphCategories.
   *  \param dualTimeMatrixInfo stores dual time matrix information (tags).
   *  \param persistentFields set of names that will be persistent in memory
   *
   *  \return an EqnTimestepAdaptorBase object that can be used to
   *          plug this transport equation into a TimeStepper.
   */
  EqnTimestepAdaptorBase*
  parse_scalar_equation( Uintah::ProblemSpecP scalarEqnParams,
                         Uintah::ProblemSpecP wasatchParams,
                         TurbulenceParameters turbParams,
                         const Expr::Tag densityTag,
                         GraphCategories& gc,
                         WasatchCore::DualTimeMatrixInfo& dualTimeMatrixInfo,
                         std::set<std::string>& persistentFields );

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
   *  \param wasatchSpec The XML block from the input file specifying the
   *         wasatch block.
   *  \param turbParams information on turbulence models
   *  \param turbParams
   *  \param useAdaptiveDt true for variable dt
   *  \param doParticles true if particle transport is active
   *  \param densityTag the tag for the mixture mass density
   *  \param gc The GraphCategories.
   *  \param linSolver
   *  \param materialManager
   *  \param dualTimeMatrixInfo stores dual time matrix information (tags).
   *
   *  \return an EqnTimestepAdaptorBase object that can be used to
   *          plug this transport equation into a TimeStepper.
   */
  std::vector<EqnTimestepAdaptorBase*>
  parse_momentum_equations( Uintah::ProblemSpecP wasatchSpec,
                            const TurbulenceParameters turbParams,
                            const bool useAdaptiveDt,
                            const bool doParticles,
                            const Expr::Tag densityTag,
                            GraphCategories& gc,
                            Uintah::SolverInterface& linSolver,
                            Uintah::MaterialManagerP& materialManager,
                            WasatchCore::DualTimeMatrixInfo& dualTimeMatrixInfo,
                            std::set<std::string>& persistentFields );

  void parse_poisson_equation( Uintah::ProblemSpecP params,
                               GraphCategories& gc,
                               Uintah::SolverInterface& linSolver,
                               Uintah::MaterialManagerP& materialManager );

  void parse_radiation_solver( Uintah::ProblemSpecP params,
                               GraphHelper& gh,
                               Uintah::SolverInterface& linSolver,
                               Uintah::MaterialManagerP& materialManager,
                               std::set<std::string>& persistentFields );

  std::vector<EqnTimestepAdaptorBase*>
  parse_species_equations( Uintah::ProblemSpecP params,
                           Uintah::ProblemSpecP wasatchSpec,
                           Uintah::ProblemSpecP momentumParams,
                           const TurbulenceParameters& turbParams,
                           const Expr::Tag& densityTag,
                           GraphCategories& gc,
                           std::set<std::string>& persistentFields,
                           WasatchCore::DualTimeMatrixInfo& dualTimeMatrixInfo,
                           bool computeKineticsJacobian );

  EqnTimestepAdaptorBase*
  parse_thermodynamic_pressure_equation( Uintah::ProblemSpecP wasatchSpec,
                                         GraphCategories& gc,
                                         std::set<std::string>& persistentFields );

  std::vector<EqnTimestepAdaptorBase*>
  parse_tar_and_soot_equations( Uintah::ProblemSpecP params,
                                const TurbulenceParameters& turbParams,
                                const Expr::Tag& densityTag,
                                GraphCategories& gc );

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
   *  \param gc The GraphCategories.
   *
   *  \return a vector of EqnTimestepAdaptorBase objects that can be used to
   *          plug this transport equation into a TimeStepper.
   */
  std::vector<EqnTimestepAdaptorBase*> parse_moment_transport_equations( Uintah::ProblemSpecP params,
                                                                         Uintah::ProblemSpecP wasatchParams,
                                                                         GraphCategories& gc);


}// namespace WasatchCore

#endif // Wasatch_ParseEquations_h
