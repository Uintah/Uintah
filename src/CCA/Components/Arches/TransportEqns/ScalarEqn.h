/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#ifndef Uintah_Component_Arches_ScalarEqn_h
#define Uintah_Component_Arches_ScalarEqn_h

#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/Directives.h>

//==========================================================================

/**
* @class ScalarEqn
* @author Jeremy Thornock
* @date Oct 16, 2008
*
* @brief Transport equation class for a CCVariable scalar
*
*
*/

namespace Uintah{

//---------------------------------------------------------------------------
// Builder
class ScalarEqn;
class CCScalarEqnBuilder: public EqnBuilder
{
public:
  CCScalarEqnBuilder( ArchesLabel* fieldLabels,
                      ExplicitTimeInt* timeIntegrator,
                      std::string eqnName );
  ~CCScalarEqnBuilder();

  EqnBase* build();
private:

};
// End Builder
//---------------------------------------------------------------------------

class ArchesLabel;
class ExplicitTimeInt;
class SourceTerm;
class ScalarEqn:
public EqnBase{

public:

  ScalarEqn( ArchesLabel* fieldLabels, ExplicitTimeInt* timeIntegrator, std::string eqnName );

  ~ScalarEqn();

  /** @brief Set any parameters from input file, initialize any constants, etc.. */
  void problemSetup(const ProblemSpecP& inputdb);

  /** @brief Assign the stage to the sources as dictated by the eqn **/
  void assign_stage_to_sources();

  /** @brief Schedule a transport equation to be built and solved */
  void sched_evalTransportEqn( const LevelP&,
                               SchedulerP& sched, int timeSubStep, 
                               EQN_BUILD_PHASE phase );

  /** @brief Schedule the build for the terms needed in the transport equation */
  void sched_buildTransportEqn( const LevelP& level,
                                SchedulerP& sched, int timeSubStep );
  /** @brief Actually build the transport equation */
  void buildTransportEqn( const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset*,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw,
                          int timeSubStep );

  /** @brief Schedule the solution the transport equation */
  void sched_solveTransportEqn(const LevelP& level,
                                SchedulerP& sched, int timeSubStep );
  /** @brief Solve the transport equation */
  void solveTransportEqn(const ProcessorGroup*,
                         const PatchSubset* patches,
                         const MaterialSubset*,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw,
                         int timeSubStep);
  /** @brief Schedule the initialization of the variables */
  void sched_initializeVariables( const LevelP& level, SchedulerP& sched );

  /** @brief Actually initialize the variables at the begining of a time step */
  void initializeVariables( const ProcessorGroup* pc,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw );

  /** @brief Compute all source terms for this scalar eqn */
  void sched_computeSources( const LevelP& level, SchedulerP& sched, int timeSubStep);

  /** @brief Add the source terms to the RHS **/
  void sched_addSourceTerms( const LevelP& level, SchedulerP& sched, int timeSubStep );
  void addSourceTerms( const ProcessorGroup* pc,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse* old_dw,
                       DataWarehouse* new_dw,
                       const int timeSubStep );

  /** @brief Apply boundary conditions */
  template <class phiType> void computeBCs( const Patch* patch, std::string varName, phiType& phi );

  /** @brief Actually clean up after the equation. This just reinitializes
             source term booleans so that the code can determine if the source
             term label should be allocated or just retrieved from the data
             warehouse. */
  void cleanUp( const ProcessorGroup* pc,
                const PatchSubset* patches,
                const MaterialSubset* matls,
                DataWarehouse* old_dw,
                DataWarehouse* new_dw  );

  void sched_timeAve( const LevelP& level, SchedulerP& sched, int timeSubStep );
  void timeAve( const ProcessorGroup* pc,
                const PatchSubset* patches,
                const MaterialSubset* matls,
                DataWarehouse* old_dw,
                DataWarehouse* new_dw,
                int timeSubStep );

  // ---------------------------------
  // Access functions:

  /** @brief Sets the time integrator. */
  inline void setTimeInt( ExplicitTimeInt* timeIntegrator ) {
    d_timeIntegrator = timeIntegrator;
  }

  void sched_advClipping( const LevelP& level, SchedulerP& sched, int timeSubStep );
  void advClipping( const ProcessorGroup* pc,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse* old_dw,
                    DataWarehouse* new_dw,
                    int timeSubStep );

  /** @brief Clip values of phi that are too high or too low (after RK time averaging). */
  template<class phiType>
  void clipPhi( const Patch* p,
                phiType& phi );

private:

  struct constCCVarWrapper {
    constCCVariable<double> data;
    double sign;
  };

  bool d_laminar_pr;
  std::string d_pr_label;

  const VarLabel* d_prNo_label;           ///< Label for the prandlt number

  //For advanced clipping
  bool _reinitialize_from_other_var;
  std::string _reinit_var_name;
  const VarLabel* _reinit_var_label;

  int nExtraSources;
  std::vector<const VarLabel *> extraSourceLabels;

}; // class ScalarEqn
} // namespace Uintah

#endif
