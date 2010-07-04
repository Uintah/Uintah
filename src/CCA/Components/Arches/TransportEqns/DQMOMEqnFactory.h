#ifndef UT_DQMOMEqnFactory_h
#define UT_DQMOMEqnFactory_h

#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ExplicitTimeInt.h> // should this be here?  -Jeremy
#include <Core/Grid/Variables/VarLabel.h>

//---------------------------------------------------------------------------
// Builder 

/**
  * @class DQMOMEqnBuilder
  * @author Jeremy Thornock, Adapted from James Sutherland's code
  * @date November 19, 2008
  *
  * @brief Abstract base class to support scalar equation additions.  Meant to
  * be used with the DQMOMEqnFactory. 
  *
  */
namespace Uintah {
class EqnBase; 
class DQMOMEqnBuilderBase
{
public:
  DQMOMEqnBuilderBase( ArchesLabel* fieldLabels, 
                       ExplicitTimeInt* timeIntegrator,
                       string eqnName ) : 
                       d_fieldLabels(fieldLabels), 
                       d_eqnName(eqnName), 
                       d_timeIntegrator(timeIntegrator) {};

  virtual ~DQMOMEqnBuilderBase(){};

  virtual EqnBase* build() = 0;  

protected: 
  ArchesLabel* d_fieldLabels; 
  string d_eqnName; 
  ExplicitTimeInt* d_timeIntegrator; 
}; // class DQMOMEqnBuilder

// End builder 
//---------------------------------------------------------------------------

/**
  * @class  DQMOMEqnFactory
  * @author Jeremy Thornock, Adapted from James Sutherland's code
  * @date   November 19, 2008
  * 
  * @brief  A Factory for building eqns. 
  * 
  */
class DQMOM;
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

  void weightedAbscissaInit( const ProcessorGroup*,
                             const PatchSubset* patches,
                             const MaterialSubset*,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw);

  void sched_dummyInit( const LevelP& level, SchedulerP& );


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

  ////////////////////////////////////////////
  // Equation retrieval

  /** @brief  Register a scalar eqn with the builder.    */
  void register_scalar_eqn( const std::string name, 
                            DQMOMEqnBuilderBase* builder);

  /** @brief  Retrieve a given scalar eqn.    */
  EqnBase& retrieve_scalar_eqn( const std::string name ); 

  /** @brief  Determine if a given scalar eqn is contained in the factory. */
  bool find_scalar_eqn( const std::string name );


  /////////////////////////////////////////////////////////
  // Set/Get methods

  /** @brief  Get access to the eqn map */ 
  EqnMap& retrieve_all_eqns(){
    return eqns_; };

  /** @brief  Get the number of quadrature nodes */ 
  inline const int get_quad_nodes( ) {
    return d_quadNodes; };

  /** @brief  Set number quadrature nodes */ 
  inline void set_quad_nodes( int qn ) {
    d_quadNodes = qn; };

  /** @brief  Set the field labels for the DQMOM equation factory */
  inline void setArchesLabel( ArchesLabel* fieldLabels ) {
    d_fieldLabels = fieldLabels;
    d_labelSet = true;
  }

  inline void setTimeIntegrator( ExplicitTimeInt* timeIntegrator ){
    d_timeIntegrator = timeIntegrator;
    d_timeIntegratorSet = true;
  };

  inline bool getDoDQMOM() {
    return d_doDQMOM; };

  inline void setDQMOMSolver( DQMOM* solver ) {
    d_dqmomSolver = solver; }; 

//  inline void setDQMOMSolvers( vector<DQMOM*> solvers ) {
//    d_dqmomSolvers = solvers; };
    

private:
  typedef std::map< std::string, DQMOMEqnBuilderBase* >  BuildMap; 

  ArchesLabel* d_fieldLabels;
  ExplicitTimeInt* d_timeIntegrator;
  DQMOM* d_dqmomSolver;
  
  bool d_timeIntegratorSet; ///< Boolean: has the time integrator been set?
  bool d_labelSet; ///< Boolean: has the ArchesLabel been set using setArchesLabel()?
  bool d_doDQMOM;  ///< Boolean: is DQMOM being used?

  BuildMap builders_; 
  EqnMap eqns_; 

  DQMOMEqnFactory(); 
  ~DQMOMEqnFactory(); 

  int d_quadNodes;

}; // class DQMOMEqnFactory 
} // end namespace Uintah

#endif
