#ifndef UT_DQMOMEqnFactory_h
#define UT_DQMOMEqnFactory_h

#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ExplicitTimeInt.h> // should this be here?  -Jeremy
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Math/MinMax.h>

namespace Uintah {

//==========================================================================

/**
* @class  DQMOMEqnFactory
* @author Jeremy Thornock
* @date   October 2008 : Initial version 
*
* @brief  Factory to manage DQMOM scalar transport equations (weights and weighted abscissas).
*
* @details
* This class is implemented as a singleton.
* 
*/

class DQMOM;
class EqnBase;
class DQMOMEqnBuilder;
class DQMOMEqnFactory
{
public:

  typedef std::map< std::string, EqnBase* >     EqnMap; 

  /** @brief Return an instance of the factory.  */
  static DQMOMEqnFactory& self(); 

  /////////////////////////////////////////////////
  // Initialization/setup methods

  /** @brief  Grab input parameters from the ups file */
  void problemSetup( const ProblemSpecP & params );

  /** @brief  Schedule/perform initialization of weight equations */
  void sched_weightInit( const LevelP& level, 
                         SchedulerP& ); 

  void weightInit( const ProcessorGroup*,
                   const PatchSubset* patches,
                   const MaterialSubset*,
                   DataWarehouse* old_dw,
                   DataWarehouse* new_dw);

  /** @brief  Schedule initialization of weighted abscissa equations */
  void sched_weightedAbscissaInit( const LevelP& level, 
                                   SchedulerP& ); 

  /** @brief  Actually initialize weighted abscissa equations */
  void weightedAbscissaInit( const ProcessorGroup*,
                             const PatchSubset* patches,
                             const MaterialSubset*,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw);

  /** @brief  Schedule dummy initialization for MPM nosolve (calls dummySolve of all objects owned/managed by the factory) */
  void sched_dummyInit( const LevelP& level, SchedulerP& );

  /** @brief  Initialize the value of the minimum timestep label for DQMOM scalar equations IN THE DATA WAREHOUSE, as well as initializing the value of the private member d_MinTimestepVar. */
  void initializeMinTimestepLabel( const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset*,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw );

  /** @brief  Set the value of the minimum timestep label for DQMOM scalar equations IN THE DATA WAREHOUSE
              (Contrast this with setMinTimestepVar() below, which merely sets the value of the private member d_MinTimestepVar. */
  void setMinTimestepLabel( const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset*,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw );

  //////////////////////////////////////////////
  // Evaluate the transport equations

  /** @brief  Schedule the evaluation of the DQMOMEqns and their source terms */
  void sched_evalTransportEqns( const LevelP& level,
                                SchedulerP&,
                                int timeSubStep );

  /** @brief  Schedule the evaluation of the DQMOMEqns and their source terms, and clean up after the evaluation is done */
  void sched_evalTransportEqnsWithCleanup( const LevelP& level,
                                           SchedulerP&,
                                           int timeSubStep );

  void sched_evalTransportEqns( const LevelP& level,
                                SchedulerP&,
                                int timeSubStep,
                                bool cleanup );

  ////////////////////////////////////////////
  // Equation retrieval

  /** @brief  Register a scalar eqn with the builder.    */
  void register_scalar_eqn( const std::string name, 
                            DQMOMEqnBuilder* builder);

  /** @brief  Retrieve a given scalar eqn.    */
  EqnBase& retrieve_scalar_eqn( const std::string name ); 

  /** @brief  Determine if a given scalar eqn is contained in the factory. */
  bool find_scalar_eqn( const std::string name );


  /////////////////////////////////////////////////////////
  // Set/Get methods

  /** @brief  Get access to the eqn map */ 
  EqnMap& retrieve_all_eqns(){
    return eqns_; };

  /** @brief  Set the field labels for the DQMOM equation factory */
  inline void setArchesLabel( ArchesLabel* fieldLabels ) {
    d_fieldLabels = fieldLabels;
    d_labelSet = true;
  }

  /** @brief  Set the time integrator object for the DQMOM equation factory */
  inline void setTimeIntegrator( ExplicitTimeInt* timeIntegrator ){
    d_timeIntegrator = timeIntegrator;
    d_timeIntegratorSet = true;
  };

  /** @brief  Get the number of quadrature nodes */ 
  inline const int get_quad_nodes( ) {
    return d_quadNodes; };

  /** @brief  Set number quadrature nodes */ 
  inline void set_quad_nodes( int qn ) {
    d_quadNodes = qn; };

  /** @brief  Get a boolean: is DQMOM used? (Is there a <DQMOM> block?) */
  inline bool getDoDQMOM() {
    return d_doDQMOM; };

  /** @brief  Set the DQMOM solver object (this is managed by the DQMOM equation factory becuase it generates source terms for the DQMOM equations) */
  inline void setDQMOMSolver( DQMOM* solver ) {
    d_dqmomSolver = solver; }; 

  /** @brief  Set the value of the private member d_MinTimestepVar, which is the minimum timestep required for stability by the DQMOM scalar equations */
  void setMinTimestepVar( string eqnName, double new_min );

//cmr
//  inline void setDQMOMSolvers( vector<DQMOM*> solvers ) {
//    d_dqmomSolvers = solvers; };
    

private:
  typedef std::map< std::string, DQMOMEqnBuilder* >  BuildMap; 

  ArchesLabel* d_fieldLabels;
  ExplicitTimeInt* d_timeIntegrator;
  DQMOM* d_dqmomSolver;

  double d_MinTimestepVar;  ///< Since we can't modify a variable multiple times (the memory usage spikes after you modify a variable ~10 or more times), 
                            ///  we have to modify a private member, then put that private member in a data warehouse variable ONCE

  bool d_timeIntegratorSet; ///< Boolean: has the time integrator been set?
  bool d_labelSet;          ///< Boolean: has the ArchesLabel been set using setArchesLabel()?
  bool d_doDQMOM;           ///< Boolean: is DQMOM being used?

  BuildMap builders_;       ///< Structure to hold the equation builder objects
  EqnMap eqns_;             ///< Structure to hold the equation objects

  DQMOMEqnFactory(); 
  ~DQMOMEqnFactory(); 

  int d_quadNodes;

}; // class DQMOMEqnFactory 
} // end namespace Uintah

#endif
