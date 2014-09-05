#ifndef Uintah_Component_SpatialOps_SpatialOps_h
#define Uintah_Component_SpatialOps_SpatialOps_h

#include <sci_defs/petsc_defs.h>
#include <CCA/Components/OnTheFlyAnalysis/AnalysisModule.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <CCA/Components/OnTheFlyAnalysis/AnalysisModuleFactory.h>
#include <CCA/Ports/SimulationInterface.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Util/Handle.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

#define YDIM
//#define ZDIM

//===========================================================================

/**
*   @class SpatialOps
*   @author Jeremy Thornock
*   
*   @brief The main component class for Spatial Ops.   
*
*/ 

namespace Uintah {
class Fields;
class CalcManual;
class ExplicitTimeInt;
class ScalarEqn; 
class BoundaryCond;
class EqnBase;
class EqnFactory;
class DQMOM; 
class ArchesLabel; 
class SpatialOps : public UintahParallelComponent, public SimulationInterface {

public:

  SpatialOps(const ProcessorGroup* myworld);

  virtual ~SpatialOps();
  /** @brief Interface to the input file and intializes objects/constants */
  virtual void problemSetup(const ProblemSpecP& params, 
                            const ProblemSpecP& materials_ps, 
                            GridP& grid, SimulationStateP&);
  /** @brief t=0 intialization for all variables that need values in DW */ 
  virtual void scheduleInitialize(const LevelP& level, SchedulerP&);
  /** @brief Computes a stable timestep based on some stability criteria */  
  virtual void scheduleComputeStableTimestep(const LevelP& level, SchedulerP&);
  /** @brief Schedules the advancement of the transport equations */ 
  virtual void scheduleTimeAdvance( const LevelP& level, SchedulerP&);

  virtual bool needRecompile(double time, double dt, const GridP& grid);
  /** @brief Actual implementation of the t=0 initialization */ 
  virtual void actuallyInitialize(const ProcessorGroup* ,
                                  const PatchSubset* patches,
                                  const MaterialSubset*,
                                  DataWarehouse* ,
                                  DataWarehouse* new_dw);

  // This is probably in bad from here  
  SimulationStateP d_sharedState;

private: 

  /** @brief Actual implementation of the calculation of a stable time step */ 
  void computeStableTimestep(const ProcessorGroup* ,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse* ,
                             DataWarehouse* new_dw);

  /** @brief Registers all possible source terms by instantiating a builder in the factory */     
  //void registerSources(ProblemSpecP& db);

  /** @brief Registers all possible models for DQMOM */ 
  //void registerModels( ProblemSpecP& db ); 

  /** @brief Registers all possible equations by instantiating a builder in the factory */     
  //void registerTransportEqns(ProblemSpecP& db);

  //Fields* d_fieldLabels; 
  ArchesLabel* d_fieldLabels; 
  CalcManual* d_manualSolver;
  ExplicitTimeInt* d_timeIntegrator;
  ScalarEqn* d_temperatureEqnMan; 
  BoundaryCond* d_boundaryCond;
  DQMOM* d_dqmomSolver; 

  bool d_recompile;
  int nofTimeSteps;
  double d_pi;
  double d_initlambda;
  double d_initTemperature;
  int d_tOrder;
  vector<string> d_scalarEqnNames; 

  typedef map< string, EqnBase* > EqnList; //a map of equations and their names
  EqnList d_myEqnList; 
  EqnBase* d_transportEqn; 
  EqnFactory* d_scalarFactory; 

  bool d_doDQMOM; 

}; // end class SpatialOps 
} // end namespace Uintah

#endif
