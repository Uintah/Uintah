/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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

#ifndef Uintah_Component_Arches_Arches_h
#define Uintah_Component_Arches_Arches_h

/**
 * \file
 *
 * \mainpage
 *
 * \author Various Generations of CRSim
 * \date   September 1997 (?)
 *
 * Arches is a variable density, low-mach (pressure projection) CFD formulation, with radiation, chemistry and
 * turbulence subgrid closure.
 *
 */

/* ----- Arches.h -----------------------------------------------

CLASS
   Arches

   Short description...

GENERAL INFORMATION

   Arches.h

   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   Department of Chemical Engineering
   University of Utah

   All major modifications since 01.01.2004 done by:
   Stanislav Borodai(borodai@crsim.utah.edu)

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)


KEYWORDS
   Arches

DESCRIPTION
   Long description...

WARNING

****************************************/

#include <sci_defs/petsc_defs.h>
#include <sci_defs/uintah_defs.h>

#include <CCA/Components/OnTheFlyAnalysis/AnalysisModule.h>
#include <CCA/Components/OnTheFlyAnalysis/AnalysisModuleFactory.h>
#include <CCA/Ports/SimulationInterface.h>

#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Util/Handle.h>

#include <string>

#include <boost/shared_ptr.hpp>

// Divergence constraint instead of drhodt in pressure equation
//#define divergenceconstraint

// Exact Initialization for first time step in
// MPMArches problem, to eliminate problem of
// sudden appearance of mass in second step
//
// #define ExactMPMArchesInitialize

namespace Uintah {

  class TaskFactoryBase; 
  class VarLabel;
  class PhysicalConstants;
  class NonlinearSolver;
  class Properties;
  class TurbulenceModel;
  class ScaleSimilarityModel;
  class BoundaryCondition;
  class MPMArchesLabel;
  class ArchesLabel;
  class TimeIntegratorLabel;
  class ExplicitTimeInt;
  class PartVel;
  class DQMOM;
  class CQMOM;
  class CQMOM_Convection;
  class CQMOMSourceWrapper;
  class RadPropertyCalculator; 
  class ArchesParticlesHelper; 

class Arches : public UintahParallelComponent, public SimulationInterface {

public:

  static const int NDIM;

  enum d_eqnType { PRESSURE, MOMENTUM, SCALAR };
  enum d_dirName { NODIR, XDIR, YDIR, ZDIR };
  enum d_stencilName { AP, AE, AW, AN, AS, AT, AB };
  enum d_numGhostCells {ZEROGHOSTCELLS , ONEGHOSTCELL, TWOGHOSTCELLS,
                        THREEGHOSTCELLS, FOURGHOSTCELLS, FIVEGHOSTCELLS };
  int nofTimeSteps;

  Arches(const ProcessorGroup* myworld, const bool doAMR);

  virtual ~Arches();

  virtual void problemSetup(const ProblemSpecP& params,
                            const ProblemSpecP& materials_ps,
                            GridP& grid, SimulationStateP&);

  virtual void scheduleInitialize(const LevelP& level,
                                  SchedulerP&);
                                  
  virtual void scheduleRestartInitialize(const LevelP& level,
                                         SchedulerP& sched);

  virtual void restartInitialize();

  virtual void sched_paramInit(const LevelP& level,
                               SchedulerP&);

  virtual void scheduleComputeStableTimestep(const LevelP& level,
                                             SchedulerP&);
  void 
  MPMArchesIntrusionSetupForResart( const LevelP& level, SchedulerP& sched, 
                                    bool& recompile, bool doing_restart );

  virtual void scheduleTimeAdvance( const LevelP& level,
                                    SchedulerP&);

  virtual bool needRecompile(double time, double dt,
                             const GridP& grid);

  virtual void sched_getCCVelocities(const LevelP& level,
                                     SchedulerP&);
  virtual void sched_weightInit( const LevelP& level,
                                SchedulerP& );
  virtual void sched_weightedAbsInit( const LevelP& level,
                                      SchedulerP& );
                                      
  virtual void sched_scalarInit( const LevelP& level,
                                 SchedulerP& sched );
  virtual void sched_momentInit( const LevelP& level,
                                 SchedulerP& sched );
                                 
  //__________________________________
  //  Multi-level/AMR 
  // stub functions.  Needed for multi-level RMCRT and
  // if you want to change the coarse level patch configuration
  virtual void scheduleCoarsen(const LevelP& ,
                                SchedulerP& ){ /* do Nothing */};

  virtual void scheduleRefineInterface(const LevelP&,
                                       SchedulerP&,
                                       bool, bool){ /* do Nothing */};

  virtual void scheduleInitialErrorEstimate( const LevelP& , 
                                             SchedulerP&  ){/* do Nothing */};
  virtual void scheduleErrorEstimate( const LevelP& , 
                                      SchedulerP&  ){/* do Nothing */};

  // for multimaterial
  void setMPMArchesLabel(const MPMArchesLabel* MAlb){
    d_MAlab = MAlb;
  }

  const ArchesLabel* getArchesLabel(){
    return d_lab;
  }
  NonlinearSolver* getNonlinearSolver(){
    return d_nlSolver;
  }

  BoundaryCondition* getBoundaryCondition(){
    return d_boundaryCondition;
  }

  TurbulenceModel* getTurbulenceModel(){
    return d_turbModel;
  }
  virtual double recomputeTimestep(double current_dt);

  virtual bool restartableTimesteps();

  void setWithMPMARCHES() {
    d_with_mpmarches = true;
  };

  void sched_create_patch_operators( const LevelP& level, SchedulerP& sched ); 
  void create_patch_operators( const ProcessorGroup* pg,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw);


protected:

private:

  Arches();

  Arches(const Arches&);

  Arches& operator=(const Arches&);


  void paramInit(const ProcessorGroup*,
                 const PatchSubset* patches,
                 const MaterialSubset*,
                 DataWarehouse* ,
                 DataWarehouse* new_dw );

  void computeStableTimeStep(const ProcessorGroup* ,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse* ,
                             DataWarehouse* new_dw);

  void readUVWInitialCondition(const ProcessorGroup*,
                              const PatchSubset* patches,
                              const MaterialSubset*,
                              DataWarehouse* ,
                              DataWarehouse* new_dw);
  
  void getCCVelocities(const ProcessorGroup*,
                       const PatchSubset* patches,
                       const MaterialSubset*,
                       DataWarehouse* ,
                       DataWarehouse* new_dw);

  void weightInit( const ProcessorGroup*,
                  const PatchSubset* patches,
                  const MaterialSubset*,
                  DataWarehouse* old_dw,
                  DataWarehouse* new_dw);
  void weightedAbsInit( const ProcessorGroup*,
                  const PatchSubset* patches,
                  const MaterialSubset*,
                  DataWarehouse* old_dw,
                  DataWarehouse* new_dw);

  void scalarInit( const ProcessorGroup* ,
                   const PatchSubset* patches,
                   const MaterialSubset*,
                   DataWarehouse* old_dw,
                   DataWarehouse* new_dw );
  
  void momentInit( const ProcessorGroup* ,
                   const PatchSubset* patches,
                   const MaterialSubset*,
                   DataWarehouse* old_dw,
                   DataWarehouse* new_dw );


  const Uintah::ProblemSpecP get_arches_spec(){ return _arches_spec; }

  /** @brief Registers all possible user defined source terms by instantiating a builder in the factory */
  void registerUDSources(ProblemSpecP& db);

  /** @brief Registers developer specific sources that may not have an input file specification */
  void registerSources();

  /** @brief Registers all possible models for DQMOM */
  void registerModels( ProblemSpecP& db );

  /** @brief Registers all possible equations by instantiating a builder in the factory */
  void registerTransportEqns(ProblemSpecP& db);

  /** @brief Registers all possible CQMOM equations by instantiating a builder in the factory */
  void registerCQMOMEqns(ProblemSpecP& db);

  /** @brief Registers all possible Property Models by instantiating a builder in the factory */
  void registerPropertyModels( ProblemSpecP& db );

  std::string d_whichTurbModel;
  bool d_mixedModel;
  bool d_with_mpmarches;
  bool d_extraProjection;
  double d_initial_dt; 

  ScaleSimilarityModel* d_scaleSimilarityModel;
  PhysicalConstants* d_physicalConsts;
  NonlinearSolver* d_nlSolver;
  // properties...solves density, temperature and species concentrations
  Properties* d_props;
  // Turbulence Model
  TurbulenceModel* d_turbModel;
  // Boundary conditions
  BoundaryCondition* d_boundaryCondition;
  SimulationStateP d_sharedState;
  // Variable labels that are used by the simulation controller
  ArchesLabel* d_lab;

  //Radiation properties
  RadPropertyCalculator* d_rad_prop_calc;

  const MPMArchesLabel* d_MAlab;

  std::string d_timeIntegratorType;

  std::vector<AnalysisModule*> d_analysisModules;

  std::string d_init_inputfile;

  TimeIntegratorLabel* init_timelabel;
  bool init_timelabel_allocated;
  bool d_dynScalarModel;
  bool d_underflow;

  // Variables----
  std::vector<std::string> d_scalarEqnNames;
  bool d_doDQMOM; // do we need this as a private member?
  std::string d_which_dqmom;
  int d_tOrder;
  ExplicitTimeInt* d_timeIntegrator;
  PartVel* d_partVel;
  DQMOM* d_dqmomSolver;
  
  bool d_doCQMOM;
  bool d_usePartVel;
  CQMOM* d_cqmomSolver;
  CQMOM_Convection* d_cqmomConvect;
  CQMOMSourceWrapper* d_cqmomSource;

  bool d_doingRestart;
  bool d_newBC_on_Restart;
  bool d_do_dummy_solve; 
  
  //__________________________________
  //  Multi-level related
  int d_archesLevelIndex;
  bool d_doAMR;

  //NEW TASK INTERFACE STUFF: 
  std::map<std::string, TaskFactoryBase*> _factory_map; 

  Uintah::ProblemSpecP _arches_spec; 

  ArchesParticlesHelper* _particlesHelper; 
  bool _doLagrangianParticles; 

  std::map<std::string, boost::shared_ptr<TaskFactoryBase> > _boost_factory_map; 

}; // end class Arches

} // End namespace Uintah

#endif

