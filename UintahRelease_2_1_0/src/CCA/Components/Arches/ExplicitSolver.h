/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

//----- ExplicitSolver.h -----------------------------------------------

#ifndef Uintah_Component_Arches_ExplicitSolver_h
#define Uintah_Component_Arches_ExplicitSolver_h

/**************************************
CLASS
   NonlinearSolver

   Class ExplicitSolver is a subclass of NonlinearSolver
   which implements the Forward Euler/RK2/ RK3 methods

GENERAL INFORMATION
   ExplicitSolver.h - declaration of the class

   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   Author of RK2/RK3 implementation: Stanislav Borodai (borodai@crsim.utah.edu)

   Creation Date:   Mar 1, 2000

   C-SAFE



KEYWORDS


DESCRIPTION
   Class ExplicitSolver implements ...

WARNING
   none
****************************************/

#include <CCA/Components/Arches/Arches.h>
#include <CCA/Components/Arches/NonlinearSolver.h>
#include <CCA/Components/Arches/MomentumSolver.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <CCA/Ports/SolverInterface.h>
#include <Core/Grid/Variables/VarTypes.h>

namespace Uintah {
class TaskFactoryBase;
class PressureSolver;
class MomentumSolver;
class ScalarSolver;
class ExtraScalarSolver;
class TurbulenceModel;
class ScaleSimilarityModel;
class Properties;
class TableLookup;
class BoundaryCondition;
class PhysicalConstants;
class PartVel;
class DQMOM;
class CQMOM;
class CQMOM_Convection;
class CQMOMSourceWrapper;
class EfficiencyCalculator;
class WallModelDriver;
class RadPropertyCalculator;
class ExplicitTimeInt;
class WBCHelper;
class ArchesParticlesHelper;
class ExplicitSolver: public NonlinearSolver {

public:

  //Builder::
  class Builder : public  NonlinearSolver::NLSolverBuilder {

  public:

    Builder( SimulationStateP& sharedState,
             const MPMArchesLabel* MAlb,
             PhysicalConstants* physConst,
             const ProcessorGroup* myworld,
             ArchesParticlesHelper* particle_helper,
             SolverInterface* hypreSolver ) :
             _sharedState(sharedState),
             _MAlb(MAlb),
             _physConst(physConst),
             _myworld(myworld),
             _particle_helper(particle_helper),
             _hypreSolver(hypreSolver)
    { }

    ~Builder(){}

    ExplicitSolver* build(){
      return scinew ExplicitSolver( _sharedState,
                                    _MAlb,
                                    _physConst,
                                    _myworld,
                                    _particle_helper,
                                    _hypreSolver);
    }

  private:

    SimulationStateP& _sharedState;
    const MPMArchesLabel* _MAlb;
    PhysicalConstants* _physConst;
    const ProcessorGroup* _myworld;
    ArchesParticlesHelper* _particle_helper;
    SolverInterface* _hypreSolver;

  };

  ExplicitSolver( SimulationStateP& sharedState,
                  const MPMArchesLabel* MAlb,
                  PhysicalConstants* physConst,
                  const ProcessorGroup* myworld,
                  ArchesParticlesHelper* particle_helper,
                  SolverInterface* hypreSolver );

  virtual ~ExplicitSolver();

  void sched_initializeVariables( const LevelP& level, SchedulerP& sched );

  void sched_restartInitialize( const LevelP& level, SchedulerP& sched );

  void sched_restartInitializeTimeAdvance( const LevelP& level, SchedulerP& sched );

  /** @brief Input file interface. **/
  virtual void problemSetup( const ProblemSpecP& input_db,
                             SimulationStateP& state,
                             GridP& grid );

  /** @brief Solve the nonlinear system. (also does some actual computations) **/
  virtual int nonlinearSolve( const LevelP& level,
                              SchedulerP& sched );

  /** @brief Sets the initial guess for several variables **/
  void sched_setInitialGuess(SchedulerP&,
                             const PatchSet* patches,
                             const MaterialSet* matls);

  /** @brief Schedule compute of a stable timestep **/
  void computeTimestep(const LevelP& level, SchedulerP& sched);

  void computeStableTimeStep( const ProcessorGroup*,
                              const PatchSubset* patches,
                              const MaterialSubset*,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw );

  /** @brief Check the momentum boundary conditions for errors (and do some other stuff too)**/
  void checkMomBCs( SchedulerP&,
                    const LevelP& level,
                    const MaterialSet* matls);

  /** @brief Interpolates face centered vars to cell centered **/
  void sched_interpolateFromFCToCC(SchedulerP&,
                                   const PatchSet* patches,
                                   const MaterialSet* matls,
                                   const TimeIntegratorLabel* timelabels,
                                   const int curr_level);

  /** @brief Compute the kinetic energy of the system **/
  void sched_computeKE( SchedulerP& sched,
                        const PatchSet* patches,
                        const MaterialSet* matls );

  void initialize( const LevelP& level, SchedulerP& sched, const bool doing_restart );

  /** @brief Print the reduced kinetic energy values to the screen output **/
  void sched_printTotalKE( SchedulerP& sched,
                           const PatchSet* patches,
                           const MaterialSet* matls );


  virtual void sched_weightInit( const LevelP& level,
                                SchedulerP& );

  virtual void sched_scalarInit( const LevelP& level,
                                 SchedulerP& sched );

  virtual void sched_weightedAbsInit( const LevelP& level,
                                      SchedulerP& );

  virtual void sched_momentInit( const LevelP& level,
                                 SchedulerP& sched );

  void weightInit( const ProcessorGroup*,
                  const PatchSubset* patches,
                  const MaterialSubset*,
                  DataWarehouse* old_dw,
                  DataWarehouse* new_dw);

  void scalarInit( const ProcessorGroup* ,
                   const PatchSubset* patches,
                   const MaterialSubset*,
                   DataWarehouse* old_dw,
                   DataWarehouse* new_dw );

  void weightedAbsInit( const ProcessorGroup*,
                  const PatchSubset* patches,
                  const MaterialSubset*,
                  DataWarehouse* old_dw,
                  DataWarehouse* new_dw);

  void momentInit( const ProcessorGroup* ,
                   const PatchSubset* patches,
                   const MaterialSubset*,
                   DataWarehouse* old_dw,
                   DataWarehouse* new_dw );

  void sched_updatePressure(SchedulerP& sched,
                          const PatchSet* patches,
                          const MaterialSet* matls,
                          const TimeIntegratorLabel* timelabels);

  void sched_saveTempCopies(SchedulerP&,
                            const PatchSet* patches,
                            const MaterialSet* matls,
                            const TimeIntegratorLabel* timelabels);

  void sched_computeDensityLag(SchedulerP&,
                               const PatchSet* patches,
                               const MaterialSet* matls,
                               const TimeIntegratorLabel* timelabels,
                               bool after_average);

  void sched_getDensityGuess(SchedulerP&,
                             const PatchSet* patches,
                             const MaterialSet* matls,
                             const TimeIntegratorLabel* timelabels);

  void sched_checkDensityGuess(SchedulerP&,
                               const PatchSet* patches,
                               const MaterialSet* matls,
                               const TimeIntegratorLabel* timelabels);

  void sched_checkDensityLag(SchedulerP&,
                             const PatchSet* patches,
                             const MaterialSet* matls,
                             const TimeIntegratorLabel* timelabels,
                             bool after_average);

  void sched_updateDensityGuess(SchedulerP&,
                                const PatchSet* patches,
                                const MaterialSet* matls,
                                const TimeIntegratorLabel* timelabels);

  /** @brief Set the initial condition on velocity on the interior **/
  void sched_setInitVelCond( const LevelP& level,
                             SchedulerP& sched,
                             const MaterialSet* matls );

  virtual void sched_getCCVelocities(const LevelP& level,
                                     SchedulerP&);

  inline double recomputeTimestep(double current_dt) {
    return current_dt/2;
  }

  inline bool restartableTimesteps() {
    return true;
  }

  inline void setNumSourceBoundaries(int numSourceBoundaries){
    d_numSourceBoundaries = numSourceBoundaries;
  }

  void initializeVariables( const ProcessorGroup* pc,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw );

  void setInitialGuess(const ProcessorGroup* pc,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse* old_dw,
                       DataWarehouse* new_dw);

  /** @brief Set the initial condition on velocity **/
  void setInitVelCond( const ProcessorGroup* pc,
                       const PatchSubset* patches,
                       const MaterialSubset*,
                       DataWarehouse* old_dw,
                       DataWarehouse* new_dw );


  /** @brief Interpolate from face-center to cell-center **/
  void interpolateFromFCToCC(const ProcessorGroup* pc,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw,
                             const TimeIntegratorLabel* timelabels,
                             const int curr_level);

  void getCCVelocities(const ProcessorGroup*,
                       const PatchSubset* patches,
                       const MaterialSubset*,
                       DataWarehouse* ,
                       DataWarehouse* new_dw);

  void computeVorticity(const ProcessorGroup* pc,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* old_dw,
                        DataWarehouse* new_dw,
                        const TimeIntegratorLabel* timelabels);

  void computeKE( const ProcessorGroup* ,
                  const PatchSubset* patches,
                  const MaterialSubset*,
                  DataWarehouse*,
                  DataWarehouse* new_dw );


  void printTotalKE(const ProcessorGroup* ,
                    const PatchSubset* patches,
                    const MaterialSubset*,
                    DataWarehouse*,
                    DataWarehouse* new_dw );

  void updatePressure(const ProcessorGroup* ,
                      const PatchSubset* patches,
                      const MaterialSubset*,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw,
                      const TimeIntegratorLabel* timelabels);

  void saveTempCopies(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* matls,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw,
                      const TimeIntegratorLabel* timelabels);

  void computeDensityLag(const ProcessorGroup*,
                         const PatchSubset* patches,
                         const MaterialSubset* matls,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw,
                         const TimeIntegratorLabel* timelabels,
                         bool after_average);

  void getDensityGuess(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* matls,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw,
                      const TimeIntegratorLabel* timelabels);

  void checkDensityGuess(const ProcessorGroup*,
                         const PatchSubset* patches,
                         const MaterialSubset* matls,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw,
                         const TimeIntegratorLabel* timelabels);

  void checkDensityLag(const ProcessorGroup*,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse* old_dw,
                       DataWarehouse* new_dw,
                       const TimeIntegratorLabel* timelabels,
                       bool after_average);

  void updateDensityGuess(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw,
                          const TimeIntegratorLabel* timelabels);

  void allocateAndInitializeToC( const VarLabel* label,
                                 DataWarehouse* dw,
                                 const int index,
                                 const Patch* patch, const double C );

  void setupBoundaryConditions( const LevelP& level,
                                SchedulerP& sched,
                                const bool doing_restart );

  int getTaskGraphIndex(const int time_step ) {
    if (d_num_taskgraphs==1){  
      return 0;
    }else{
      return ((time_step % d_rad_calc_frequency == 0));
    }
  }

  int taskGraphsRequested() {
  return d_num_taskgraphs;
  }

  void registerModels( ProblemSpecP& db );
  void registerTransportEqns( ProblemSpecP& db );
  void registerPropertyModels( ProblemSpecP& db );
  void registerCQMOMEqns( ProblemSpecP& db );

  // const VarLabel*
  ArchesLabel* d_lab;

  // problemspec for Arches
  Uintah::ProblemSpecP _arches_spec;

  // Total number of nonlinear iterates
  int d_nonlinear_its;
  // for probing data for debuging or plotting
  // properties...solves density, temperature and specie concentrations
  Properties* d_props;
  TableLookup* d_tabulated_properties; 
  // Boundary conditions
  BoundaryCondition* d_boundaryCondition;
  // Turbulence Model
  TurbulenceModel* d_turbModel;
  ScaleSimilarityModel* d_scaleSimilarityModel;
  TimeIntegratorLabel* d_init_timelabel;
  bool d_mixedModel;

  MomentumSolver* d_momSolver;             ///< Momentum solver
  WallModelDriver* d_wall_ht_models;       ///< Heat transfer models for walls
  SimulationStateP& d_sharedState;
  const MPMArchesLabel* d_MAlab;
  PhysicalConstants* d_physicalConsts;     ///< Physical constants

  //NEW TASK INTERFACE STUFF:
  std::map<std::string, std::shared_ptr<TaskFactoryBase> > _task_factory_map;
  bool _doLagrangianParticles;

  std::vector<TimeIntegratorLabel* > d_timeIntegratorLabels;
  TimeIntegratorLabel* nosolve_timelabels;
  int numTimeIntegratorLevels;
  bool nosolve_timelabels_allocated;

  bool d_3d_periodic;
  int d_turbModelCalcFreq;
  bool d_turbModelRKsteps;
  int d_turbCounter;
  bool d_restart_on_negative_density_guess;
  bool d_noisyDensityGuess;
  std::string d_mms;
  std::string d_mmsErrorType;
  double d_airDensity, d_heDensity;
  Vector d_gravity;
  double d_viscosity;

  bool d_KE_fromFC;
  double d_maxDensityLag;
  std::vector<std::string> d_mass_sources;
  bool d_solvability;

  bool d_extra_table_lookup;

  //linear mms
  double cu, cv, cw, cp, phi0;
  // sine mms
  double amp;

  int d_numSourceBoundaries;

  ExplicitTimeInt* d_timeIntegrator;
  int d_tOrder;

  //DQMOM
  bool d_doDQMOM;
  PartVel* d_partVel;
  DQMOM* d_dqmomSolver;
  std::string d_which_dqmom;

  //CQMOM
  bool d_doCQMOM;
  CQMOM* d_cqmomSolver;
  CQMOM_Convection* d_cqmomConvect;
  CQMOMSourceWrapper* d_cqmomSource;

  ArchesParticlesHelper* _particlesHelper;

  // Pressure Eqn Solver
  PressureSolver* d_pressSolver;
  SolverInterface* d_hypreSolver;             // infrastructure hypre solver

  EfficiencyCalculator* d_eff_calculator;
  RadPropertyCalculator* d_rad_prop_calc;

  //Diagnostics
  bool d_printTotalKE;
  double d_ke_limit;

  const VarLabel* d_x_vel_label;
  const VarLabel* d_y_vel_label;
  const VarLabel* d_z_vel_label;
  const VarLabel* d_viscos_label;
  const VarLabel* d_rho_label;
  const VarLabel* d_celltype_label;
  int d_archesLevelIndex;
  int d_rad_calc_frequency{1};
  int d_num_taskgraphs{1};

  std::map<int,WBCHelper*> m_bcHelper;

}; // End class ExplicitSolver
} // End namespace Uintah


#endif
