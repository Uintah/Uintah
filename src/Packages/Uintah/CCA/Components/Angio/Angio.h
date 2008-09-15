#ifndef UINTAH_HOMEBREW_ANGIO_H
#define UINTAH_HOMEBREW_ANGIO_H

#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/CCA/Ports/Output.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/CCA/Components/Angio/AngioFlags.h>
#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleVariable.h>
#include <Packages/Uintah/Core/Labels/AngioLabel.h>
#include <Core/Geometry/Vector.h>
#include <Core/Util/DebugStream.h>

namespace Uintah {

using namespace SCIRun;

class AngioParticleCreator;

/**************************************

CLASS
   Angio
   
   Discrete model for angiogenesis, to be coupled with MPM

GENERAL INFORMATION

   Angio.h

   Jim Guilkey
   Dept. of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Angiogenesis

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

class UINTAHSHARE Angio : public SimulationInterface, public UintahParallelComponent {
public:
  Angio(const ProcessorGroup* myworld);
  virtual ~Angio();

  //////////
  // Insert Documentation Here:
  virtual void problemSetup(const ProblemSpecP& params, 
                            const ProblemSpecP& restart_prob_spec, GridP&,
                            SimulationStateP&);

  virtual void outputProblemSpec(ProblemSpecP& ps);

  virtual void scheduleInitialize(const LevelP& level,
                                  SchedulerP&);

  void schedulePrintParticleCount(const LevelP& level, SchedulerP& sched);
  //////////
  // Insert Documentation Here:
  virtual void scheduleComputeStableTimestep(const LevelP& level, SchedulerP&);

  //////////
  // Insert Documentation Here:
  virtual void scheduleTimeAdvance(const LevelP& level, SchedulerP&);

  void setSharedState(SimulationStateP& ssp);

  void setAngioLabel(AngioLabel* Alb) {
        delete lb;
        lb = Alb;
  };

  enum bctype { NONE=0,
                FIXED,
                SYMMETRY,
                NEIGHBOR };

  enum IntegratorType {
    Explicit,
    Implicit,
  };

  // Unused AMR related tasks
  ///////////////////////////////////////////////////////////////////////////
  virtual void scheduleRefine(const PatchSet* patches, SchedulerP& scheduler);

  virtual void scheduleRefineInterface(const LevelP& fineLevel,
                                       SchedulerP& scheduler,
                                       bool needCoarse, bool needFine);

  virtual void scheduleCoarsen(const LevelP& coarseLevel, SchedulerP& sched);

  /// Schedule to mark flags for AMR regridding
  virtual void scheduleErrorEstimate(const LevelP& coarseLevel, 
                                     SchedulerP& sched);
  
  /// Schedule to mark initial flags for AMR regridding
  void scheduleInitialErrorEstimate(const LevelP& coarseLevel,
                                    SchedulerP& sched);
  ///////////////////////////////////////////////////////////////////////////

protected:
  //////////
  // Insert Documentation Here:
 
  virtual void actuallyInitialize(const ProcessorGroup*,
                                  const PatchSubset* patches,
                                  const MaterialSubset* matls,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw);

  void printParticleCount(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw);

  //////////
  // Initialize particle data with a default values in the
  // new datawarehouse
  void setParticleDefault(ParticleVariable<double>& pvar,
                          const VarLabel* label,
                          ParticleSubset* pset,
                          DataWarehouse* new_dw,
                          double val);
  void setParticleDefault(ParticleVariable<Vector>& pvar,
                          const VarLabel* label, 
                          ParticleSubset* pset,
                          DataWarehouse* new_dw,
                          const Vector& val);
  void setParticleDefault(ParticleVariable<Matrix3>& pvar,
                          const VarLabel* label, 
                          ParticleSubset* pset,
                          DataWarehouse* new_dw,
                          const Matrix3& val);

  // Insert Documentation Here:
  void actuallyComputeStableTimestep(const ProcessorGroup*,
                                     const PatchSubset* patches,
                                     const MaterialSubset* matls,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  virtual void carryForwardOldSegments(const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  virtual void growAtTips(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  virtual void setBCsInterpolated(const ProcessorGroup*,
                                  const PatchSubset* patches,
                                  const MaterialSubset* matls,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw);

  virtual void computeInternalForce(const ProcessorGroup*,
                              const PatchSubset* patches,
                              const MaterialSubset* matls,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  virtual void solveEquationsMotion(const ProcessorGroup*,
                                    const PatchSubset* patches,
                                    const MaterialSubset* matls,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  virtual void integrateAcceleration(const ProcessorGroup*,
                                     const PatchSubset* patches,
                                     const MaterialSubset* matls,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:                            
  void setGridBoundaryConditions(const ProcessorGroup*,
                                 const PatchSubset* patches,
                                 const MaterialSubset* ,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw);

  void addNewParticles(const ProcessorGroup*,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse* old_dw,
                       DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  virtual void interpolateToParticlesAndUpdate(const ProcessorGroup*,
                                               const PatchSubset* patches,
                                               const MaterialSubset* matls,
                                               DataWarehouse* old_dw,
                                               DataWarehouse* new_dw);

  // Unused AMR related tasks
  ///////////////////////////////////////////////////////////////////////////
  void refine(const ProcessorGroup*,
              const PatchSubset* patches,
              const MaterialSubset* matls,
              DataWarehouse*,
              DataWarehouse* new_dw);

  void errorEstimate(const ProcessorGroup*,
                     const PatchSubset* patches,
                     const MaterialSubset* matls,
                     DataWarehouse*,
                     DataWarehouse* new_dw);

  void initialErrorEstimate(const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse*,
                            DataWarehouse* new_dw);
  ///////////////////////////////////////////////////////////////////////////

  virtual void scheduleCarryForwardOldSegments(SchedulerP&, const PatchSet*,
                                               const MaterialSet*);

  virtual void scheduleGrowAtTips(SchedulerP&, const PatchSet*,
                                  const MaterialSet*);

  virtual void scheduleSetBCsInterpolated(SchedulerP&, const PatchSet*,
                                          const MaterialSet*);

  virtual void scheduleComputeInternalForce(SchedulerP&, const PatchSet*,
                                            const MaterialSet*);

  virtual void scheduleSolveEquationsMotion(SchedulerP&, const PatchSet*,
                                            const MaterialSet*);

  virtual void scheduleIntegrateAcceleration(SchedulerP&, const PatchSet*,
                                             const MaterialSet*);

  void scheduleSetGridBoundaryConditions(SchedulerP&, const PatchSet*,
                                         const MaterialSet* matls);
                                                 
  virtual void scheduleInterpolateToParticlesAndUpdate(SchedulerP&, 
                                                       const PatchSet*,
                                                       const MaterialSet*);

  void scheduleAddNewParticles(SchedulerP&, const PatchSet*,const MaterialSet*);

  bool needRecompile(double time, double dt, const GridP& grid);

  void printSchedule(const PatchSet* patches,
                     DebugStream& dbg,
                     const string& where);

  void printSchedule(const LevelP& level,
                     DebugStream& dbg,
                     const string& where);

  void printTask(const PatchSubset* patches,
                 const Patch* patch,
                 DebugStream& dbg,
                 const string& where);

  void materialProblemSetup(const ProblemSpecP& prob_spec,
                            SimulationStateP& sharedState,
                            AngioFlags* flags);

  double findLength(const double& delt);

  double findAngle(const double& old_ang);

  void adjustXNewForPeriodic(Point &x, const Point min, const Point max);


  SimulationStateP d_sharedState;
  AngioLabel* lb;
  AngioFlags* flags;
  Output* dataArchiver;

  double           d_nextOutputTime;
  double           d_outputInterval;
  double           d_SMALL_NUM_Angio;
  double           d_PI;
  int              NGP;      // Number of ghost particles needed.
  int              NGN;      // Number of ghost nodes     needed.
  IntegratorType d_integrator;
  
  bool             d_recompile;
private:

  Angio(const Angio&);
  Angio& operator=(const Angio&);
};
      
} // end namespace Uintah

#endif
