#ifndef Uintah_Component_Arches_ScalarEqn_h
#define Uintah_Component_Arches_ScalarEqn_h

#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/Directives.h>

namespace Uintah{

//==========================================================================

/**
* @class ScalarEqn
* @author Jeremy Thornock
* @date Oct 16, 2008
*
* @brief  Scalar transport equation class 
*
* @details
* This class is currently only implemented for cell-centered (CC) variables.
* Eventually it could be extended to face-centered variables.
*
*/

//---------------------------------------------------------------------------
// Builder 
class ScalarEqn; 
class CCScalarEqnBuilder: public EqnBuilder
{
public:
  CCScalarEqnBuilder( ArchesLabel* fieldLabels, 
                      ExplicitTimeInt* timeIntegrator, 
                      string eqnName );

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

  ScalarEqn( ArchesLabel* fieldLabels, ExplicitTimeInt* timeIntegrator, string eqnName );

  ~ScalarEqn();

  /** @brief Set any parameters from input file, initialize any constants, etc.. */
  void problemSetup(const ProblemSpecP& inputdb);
  

  ////////////////////////////////////////////////
  // Calculation methods
  
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
                                SchedulerP& sched, 
                                int timeSubStep,
                                bool copyOldIntoNew );

  /** @brief  Solve the transport equation */ 
  void solveTransportEqn(const ProcessorGroup*, 
                         const PatchSubset* patches, 
                         const MaterialSubset*, 
                         DataWarehouse* old_dw, 
                         DataWarehouse* new_dw,
                         int timeSubStep, 
                         bool copyOldIntoNew );

  /** @brief Schedule the initialization of the variables */ 
  void sched_initializeVariables( const LevelP& level, SchedulerP& sched );

  /** @brief Actually initialize the variables at the begining of a time step */ 
  void initializeVariables( const ProcessorGroup* pc, 
                            const PatchSubset* patches, 
                            const MaterialSubset* matls, 
                            DataWarehouse* old_dw, 
                            DataWarehouse* new_dw );

  /** @brief Compute all source terms for this scalar eqn */
  void sched_computeSources( const LevelP& level, 
                             SchedulerP& sched, 
                             int timeSubStep);

  /** @brief Return a list of all sources associated with this transport equation */ 
  inline const vector<string> getSourcesList(){
    return d_sources; };

  /** @brief Apply boundary conditions */
  template <class phiType> void computeBCs( const Patch* patch, string varName, phiType& phi );

  /** @brief Schedule the cleanup after this equation. */ 
  void sched_cleanUp( const LevelP&, SchedulerP& sched ); 

  /** @brief Actually clean up after the equation. This just reinitializes 
             source term booleans so that the code can determine if the source
             term label should be allocated or just retrieved from the data 
             warehouse. */ 
  void cleanUp( const ProcessorGroup* pc, 
                const PatchSubset* patches, 
                const MaterialSubset* matls, 
                DataWarehouse* old_dw, 
                DataWarehouse* new_dw  ); 

  void sched_timeAveraging( const LevelP& level, SchedulerP& sched, int timeSubStep );

  void timeAveraging( const ProcessorGroup* pc, 
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

  /** @brief Schedule dummy initialization for MPMArches */
  void sched_dummyInit( const LevelP& level, SchedulerP& sched );
  
  /** @brief Do dummy initialization for MPMArches */
  void dummyInit( const ProcessorGroup* pc, 
                     const PatchSubset* patches, 
                     const MaterialSubset* matls, 
                     DataWarehouse* old_dw, 
                     DataWarehouse* new_dw );

  /** @brief Clip values of phi that are too high or too low (after RK time averaging). */
  void sched_clipPhi( const LevelP& level, SchedulerP& sched );

  void clipPhi( const ProcessorGroup* pc, 
                const PatchSubset* patches, 
                const MaterialSubset* matls, 
                DataWarehouse* old_dw, 
                DataWarehouse* new_dw );

private:

  double d_timestepMultiplier;

  vector<std::string> d_sources;

  const VarLabel* d_prNo_label;           ///< Label for the prandlt number 

}; // class ScalarEqn
} // namespace Uintah

#endif


