#ifndef UT_EqnFactory_h
#define UT_EqnFactory_h
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Math/MinMax.h>

namespace Uintah {
//---------------------------------------------------------------------------
// Builder 

/**
  * @class EqnBuilder
  * @author Jeremy Thornock, James Sutherland
  * @date November 19, 2008
  *
  * @brief Abstract base class to support scalar equation additions.  Meant to
  * be used with the EqnFactory. 
  *
  */
class ExplicitTimeInt;
class ArchesLabel;
class EqnBase; 
class ArchesLabel;
class ExplicitTimeInt;
class EqnBuilder
{
public:
  EqnBuilder( ArchesLabel* fieldLabels, 
              ExplicitTimeInt* timeIntegrator,
              string eqnName ) : 
              d_fieldLabels(fieldLabels), 
              d_timeIntegrator(timeIntegrator),
              d_eqnName(eqnName) {};

  virtual ~EqnBuilder(){};

  virtual EqnBase* build() = 0;  

protected: 
  ArchesLabel* d_fieldLabels; 
  ExplicitTimeInt* d_timeIntegrator;
  string d_eqnName; 
}; // class EqnBuilder

// End builder 
//---------------------------------------------------------------------------

/**
  * @class  EqnFactory
  * @author Jeremy Thornock, Adapted from James Sutherland's code
  * @date   November 19, 2008
  * 
  * @brief  A factory to manage non-DQMOM scalar transport equations.
  *
  * @details
  * This class is implemented as a singleton.
  * 
  * @todo Implement a minimum timestep thing, similar to DQMOMEqnFactory
  * 
  */
class EqnFactory
{
public:

  typedef std::map< std::string, EqnBase* > EqnMap; 

  /** @brief Return an instance of the factory.  */
  static EqnFactory& self(); 

  //////////////////////////////////////////////
  // Initialization/setup methods

  /** @brief  Grab input parameters from the ups file */
  void problemSetup( const ProblemSpecP & params );

  /** @brief  Schedule initialization of scalar equations */
  void sched_scalarInit( const LevelP& level, 
                         SchedulerP& sched );

  /** @brief  Actually initialize scalar equations */
  void scalarInit( const ProcessorGroup* ,
                   const PatchSubset* patches,
                   const MaterialSubset*,
                   DataWarehouse* old_dw,
                   DataWarehouse* new_dw );

  /** @brief  Schedule dummy initialization for MPM nosolve (calls dummySolve of all objects owned/managed by the factory) */
  void sched_dummyInit( const LevelP& level,
                        SchedulerP& sched );

  /** @brief  Schedule initializing value of the minimum timestep var label */
  void sched_initializeMinTimestepLabel( const LevelP& level,
                                         SchedulerP& sched,
                                         int timeSubStep );

  /** @brief  Initialize the value of the minimum timestep label for DQMOM scalar equations in the data warehouse, as well as initializing the value of the private member d_MinTimestepVar. */
  void initializeMinTimestepLabel( const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset*,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw );

  /** @brief  Schedule setting the value of the minimum timestep var label */
  void sched_setMinTimestepLabel( const LevelP& level,
                                  SchedulerP& sched );

  /** @brief  Set the value of the minimum timestep label for DQMOM scalar equations in the data warehouse
              (Contrast this with setMinTimestepVar() below, which merely sets the value of the private member d_MinTimestepVar. */
  void setMinTimestepLabel( const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset*,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw );

  //////////////////////////////////////////////
  // Evaluate the transport equations

  /** @brief  Schedule the evaluation of the scalar equations and their source terms 
      @param  evalDensityGuessEqns    (Boolean) If  true, only equations with a density guess are evaluated;
                                      if false, only equations without a density guess are evaluated
  */
  void sched_evalTransportEqns( const LevelP& level,
                                SchedulerP&,
                                int timeSubStep,
                                bool evalDensityGuessEqns,
                                bool lastTimeSubstep );

  void sched_timeAveraging( const LevelP& level,
                            SchedulerP&,
                            int timeSubStep );

  void cleanUp( const LevelP& level,
                SchedulerP& );

  ///////////////////////////////////////////////
  // Equation retrieval methods

  /** @brief Register a scalar eqn with the builder.    */
  void register_scalar_eqn( const std::string name, 
                            EqnBuilder* builder);

  /** @brief Retrieve a given scalar eqn.    */
  EqnBase& retrieve_scalar_eqn( const std::string name ); 

  /** @brief Determine if a given scalar eqn is contained in the factory. */
  bool find_scalar_eqn( const std::string name );

  /////////////////////////////////////////////
  // Get/set methods

  /** @brief Get access to the eqn map */ 
  EqnMap& retrieve_all_eqns(){
    return eqns_; };

  /** @brief  Set the label object associated with the EqnFactory */ 
  inline void setArchesLabel( ArchesLabel* fieldLabels ) {
    d_fieldLabels = fieldLabels;
    d_labelSet = true;
  };

  /** @brief  Set the time integrator object associated with the EqnFactory */ 
  inline void setTimeIntegrator( ExplicitTimeInt* timeIntegrator ){
    d_timeIntegrator = timeIntegrator;
    d_timeIntegratorSet = true;
  };

  /** @brief  Method to determine whether any scalar equations are used/registered with the EqnFactory */
  inline bool useScalarEqns() {
    return d_useScalarEqns; };

  /** @brief  Set the value of the private member d_MinTimestepVar, which is the minimum timestep required for stability by the DQMOM scalar equations */
  void setMinTimestepVar( string eqnName, double new_min );

private:

  typedef std::map< std::string, EqnBuilder* >  BuildMap; 

  ArchesLabel* d_fieldLabels;
  ExplicitTimeInt* d_timeIntegrator;

  double d_MinTimestepVar;  ///< Since we can't modify a variable multiple times (the memory usage spikes after you modify a variable ~10 or more times), 
                            ///  we have to modify a private member, then put that private member in a data warehouse variable ONCE

  bool d_useScalarEqns;     ///< Boolean: are any scalar equations being tracked?
  bool d_labelSet;          ///< Boolean: has the ArchesLabel been set using setArchesLabel()?
  bool d_timeIntegratorSet; ///< Boolean: has the time integrator been set using setTimeIntegrator()?

  BuildMap builders_;       ///< Storage container for equation builder objects
  EqnMap eqns_;             ///< Storage container for equation objects

  EqnFactory(); 
  ~EqnFactory(); 
}; // class EqnFactory 



} // end namespace Uintah

#endif
