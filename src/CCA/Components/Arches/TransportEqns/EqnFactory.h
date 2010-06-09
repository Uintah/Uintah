#ifndef UT_EqnFactory_h
#define UT_EqnFactory_h
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Ports/DataWarehouseP.h>
//#include <Core/Parallel/Parallel.h>

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
namespace Uintah {
class ExplicitTimeInt;
class ArchesLabel;
class EqnBase; 
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
  * @brief  A Factory for building eqns. 
  * 
  */
class ArchesLabel;
class ExplicitTimeInt;

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

  /** @brief  Schedule/do initialization of scalar equations */
  void sched_scalarInit( const LevelP& level, 
                         SchedulerP& sched );

  void scalarInit( const ProcessorGroup* ,
                   const PatchSubset* patches,
                   const MaterialSubset*,
                   DataWarehouse* old_dw,
                   DataWarehouse* new_dw );

  ///////////////////////////////////////////////
  // Equation retrieval methods

  /** @brief Register a scalar eqn with the builder.    */
  void register_scalar_eqn( const std::string name, 
                            EqnBuilder* builder);

  /** @brief Retrieve a given scalar eqn.    */
  EqnBase& retrieve_scalar_eqn( const std::string name ); 

  /** @brief Determine if a given scalar eqn is contained in the factory. */
  bool find_scalar_eqn( const std::string name );

  /** @brief Get access to the eqn map */ 
  EqnMap& retrieve_all_eqns(){
    return eqns_; };

  /////////////////////////////////////////////
  // Get/set methods

  inline void setArchesLabel( ArchesLabel* fieldLabels ) {
    d_fieldLabels = fieldLabels;
    d_labelSet = true;
  };

  inline void setTimeIntegrator( ExplicitTimeInt* timeIntegrator ){
    d_timeIntegrator = timeIntegrator;
    d_timeIntegratorSet = true;
  };

private:

  typedef std::map< std::string, EqnBuilder* >  BuildMap; 

  ArchesLabel* d_fieldLabels;

  ExplicitTimeInt* d_timeIntegrator;

  bool d_labelSet; ///< Boolean: has the ArchesLabel been set using setArchesLabel()?
  bool d_timeIntegratorSet; ///< Boolean: has the time integrator been set using setTimeIntegrator()?

  BuildMap builders_; 
  EqnMap eqns_; 

  EqnFactory(); 
  ~EqnFactory(); 
}; // class EqnFactory 
} // end namespace Uintah

#endif
