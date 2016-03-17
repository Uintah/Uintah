/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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
#include <CCA/Components/Wasatch/BCHelper.h>
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
  class MPMArchesLabel;
  class ArchesLabel;
  class ArchesParticlesHelper;
  class ArchesBCHelper;

class Arches : public UintahParallelComponent, public SimulationInterface {

public:

  typedef std::map< int, ArchesBCHelper* > BCHelperMapT;

  enum DIRNAME { NODIR, XDIR, YDIR, ZDIR };
  enum STENCILNAME { AP, AE, AW, AN, AS, AT, AB };
  enum NUMGHOSTS {ZEROGHOSTCELLS , ONEGHOSTCELL, TWOGHOSTCELLS,
                  THREEGHOSTCELLS, FOURGHOSTCELLS, FIVEGHOSTCELLS };

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

  virtual void scheduleComputeStableTimestep(const LevelP& level,
                                             SchedulerP&);
  void
  MPMArchesIntrusionSetupForResart( const LevelP& level, SchedulerP& sched,
                                    bool& recompile, bool doing_restart );

  virtual void scheduleTimeAdvance( const LevelP& level,
                                    SchedulerP&);

  virtual bool needRecompile(double time, double dt,
                             const GridP& grid);

  void setMPMArchesLabel(const MPMArchesLabel* MAlb){
    d_MAlab = MAlb;
  }

  virtual double recomputeTimestep(double current_dt);

  virtual bool restartableTimesteps();

  void setWithMPMARCHES() {
    d_with_mpmarches = true;
  };

  void sched_create_patch_operators( const LevelP& level, SchedulerP& sched );

  int nofTimeSteps;
  static const int NDIM;

  //________________________________________________________________________________________________
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
  //________ end stub functions ____________________________________________________________________

private:

  Arches();

  Arches(const Arches&);

  Arches& operator=(const Arches&);

  void computeStableTimeStep(const ProcessorGroup* ,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse* ,
                             DataWarehouse* new_dw);

  void create_patch_operators( const ProcessorGroup* pg,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw);


  const Uintah::ProblemSpecP get_arches_spec(){ return _arches_spec; }

  /** @brief Assign a unique name to each BC where name is not specified in the UPS.
             Note that this functionality was taken from Wasatch. (credit: Tony Saad) **/
  void assign_unique_boundary_names( Uintah::ProblemSpecP bcProbSpec );

  /** @brief convert a number to a string **/
  template <typename T>
  std::string number_to_string ( T n )
  {
    std::stringstream ss;
    ss << n;
    return ss.str();
  }



  PhysicalConstants* d_physicalConsts;
  NonlinearSolver* d_nlSolver;
  SimulationStateP d_sharedState;
  const MPMArchesLabel* d_MAlab;
  BCHelperMapT _bcHelperMap;
  std::vector<AnalysisModule*> d_analysisModules;
  //NEW TASK INTERFACE STUFF:
  std::map<std::string, TaskFactoryBase*> _factory_map;
  Uintah::ProblemSpecP _arches_spec;
  ArchesParticlesHelper* _particlesHelper;
  std::map<std::string, boost::shared_ptr<TaskFactoryBase> > _task_factory_map;

  bool d_doingRestart;
  bool d_newBC_on_Restart;
  bool d_with_mpmarches;
  bool d_doAMR;             //<<< Multilevel related
  bool _doLagrangianParticles;
  bool d_recompile_taskgraph;

  int d_archesLevelIndex;   //<<< Multilevel related

  double d_initial_dt;

  const VarLabel* d_x_vel_label;
  const VarLabel* d_y_vel_label;
  const VarLabel* d_z_vel_label;
  const VarLabel* d_viscos_label;
  const VarLabel* d_rho_label;
  const VarLabel* d_celltype_label;

}; // end class Arches

} // End namespace Uintah

#endif
