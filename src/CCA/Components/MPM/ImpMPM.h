/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#ifndef UINTAH_HOMEBREW_IMP_MPM_H
#define UINTAH_HOMEBREW_IMP_MPM_H

#include <sci_defs/petsc_defs.h>

#include <Core/Geometry/Vector.h>

#include <Core/Parallel/UintahParallelComponent.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <CCA/Ports/SimulationInterface.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Patch.h>
#include <Core/Labels/MPMLabel.h>
#include <CCA/Components/MPM/ImpMPMFlags.h>
#include <CCA/Components/MPM/MPMCommon.h>
#include <CCA/Components/MPM/Solver.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <CCA/Ports/SwitchingCriteria.h>


#include <map>
#include <vector>
#include <list>

namespace Uintah {

using namespace SCIRun;
 class DataWarehouse;
 class MPMLabel;
 class ProcessorGroup;
 class VarLabel;
 class Task; 
 class ImplicitHeatConduction;
 class ThermalContact;

/**************************************

CLASS
   ImpMPM
   
   Short description...

GENERAL INFORMATION

   ImpMPM.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   
KEYWORDS
   ImpMPM

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

class ImpMPM : public MPMCommon, public UintahParallelComponent, 
  public SimulationInterface {
public:
  ImpMPM(const ProcessorGroup* myworld);
  virtual ~ImpMPM();

  //////////
  // Insert Documentation Here:
  virtual void problemSetup(const ProblemSpecP& params, 
                            const ProblemSpecP& mat_ps,
                            GridP& grid, SimulationStateP&);

  virtual void outputProblemSpec(ProblemSpecP& ps);

  virtual void scheduleInitialize(const LevelP& level, SchedulerP&);

  virtual void switchInitialize(const LevelP& level, SchedulerP&);

  //////////
  // Insert Documentation Here:
  virtual void scheduleComputeStableTimestep(const LevelP& level, SchedulerP&);

  //////////
  // Insert Documentation Here:
  virtual void scheduleTimeAdvance(          const LevelP& level, SchedulerP&);
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

  virtual bool restartableTimesteps();
  virtual double recomputeTimestep(double new_dt);

  void scheduleSwitchTest(const LevelP& level, SchedulerP& sched);
  
  void setSharedState(SimulationStateP& ssp);

  void setMPMLabel(MPMLabel* Mlb)
  {

        delete lb;
        lb = Mlb;
  };

  enum IntegratorType {
    Explicit,
    Implicit 
  };

private:
  //////////
  // Insert Documentation Here:
  MaterialSubset* one_matl;

  friend class MPMICE;

  inline bool compare(double num1, double num2)
    {
      double EPSILON=1.e-16;
      
      return (fabs(num1-num2) <= EPSILON);
    };


  void actuallyInitialize(             const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  void printParticleCount(             const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  void scheduleInitializeHeatFluxBCs(const LevelP& level,
                                     SchedulerP&);

  void scheduleInitializePressureBCs(const LevelP& level, SchedulerP&);


  void countMaterialPointsPerLoadCurve(const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);
                                       
  void initializePressureBC(const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw);

  void initializeHeatFluxBC(const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  void actuallyComputeStableTimestep(  const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  void applyExternalLoads(             const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* ,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  void interpolateParticlesToGrid(     const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  void projectCCHeatSourceToNodes(     const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  void computeCCVolume(                const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  void rigidBody(                      const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  void destroyMatrix(                  const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw,
                                       bool recursion);

  void createMatrix(                   const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  void applyBoundaryConditions(        const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  void computeContact(                 const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  void findFixedDOF(                   const ProcessorGroup*, 
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls, 
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  // This is for the computation with the 24 x 24 matrix
  void computeStressTensorImplicit(    const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw,
                                       bool recursion);

  // No matrix calculations are performed.
  void computeStressTensorImplicit(    const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  void formStiffnessMatrix(            const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);
  //////////
  // Insert Documentation Here:
  void computeInternalForce(           const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  void iterate(                        const ProcessorGroup* pg,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw,
                                       LevelP level, Scheduler* sched);

  void formQ(                          const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  void solveForDuCG(                   const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  void solveForTemp(                   const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  void getDisplacementIncrement(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw);

  void getTemperatureIncrement(const ProcessorGroup*,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw);


  void updateGridKinematics(           const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  void checkConvergence(               const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  void updateTotalDisplacement(        const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  void computeAcceleration(            const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  void interpolateToParticlesAndUpdate(const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  void refine(                         const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse*,
                                       DataWarehouse* new_dw);
                                                                                
  void errorEstimate(                  const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse*,
                                       DataWarehouse* new_dw);
                                                                                
  void initialErrorEstimate(           const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse*,
                                       DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  void interpolateStressToGrid(        const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

  void scheduleComputeStressTensor( SchedulerP&, const PatchSet*,
                                    const MaterialSet*, const bool recursion);

  void scheduleFormStiffnessMatrix( SchedulerP&, const PatchSet*,
                                    const MaterialSet*);

  void scheduleFormHCStiffnessMatrix( SchedulerP&, const PatchSet*,
                                      const MaterialSet*);

  void scheduleComputeInternalForce(SchedulerP&, const PatchSet*,
                                       const MaterialSet*);

  void scheduleFormQ(               SchedulerP&, const PatchSet*,
                                    const MaterialSet*);

  void scheduleFormHCQ(             SchedulerP&, const PatchSet*,
                                    const MaterialSet*);

  void scheduleAdjustHCQAndHCKForBCs(SchedulerP&, const PatchSet*,
                                     const MaterialSet*);

  void scheduleUpdateGridKinematics(SchedulerP&, const PatchSet*, 
                                       const MaterialSet*);

  void scheduleApplyExternalLoads(             SchedulerP&, const PatchSet*,
                                               const MaterialSet*);

  void scheduleInterpolateParticlesToGrid(     SchedulerP&, const PatchSet*,
                                               const MaterialSubset*,
                                               const MaterialSet*);

  void scheduleProjectCCHeatSourceToNodes(     SchedulerP&, const PatchSet*,
                                               const MaterialSubset*,
                                               const MaterialSet*);

  void scheduleComputeCCVolume(                SchedulerP&, const PatchSet*,
                                               const MaterialSubset*,
                                               const MaterialSet*);

  void scheduleComputeHeatExchange(SchedulerP&, const PatchSet*,
                                   const MaterialSet*);
  
  void scheduleRigidBody(                      SchedulerP&, const PatchSet*,
                                               const MaterialSet*);

  void scheduleDestroyMatrix(      SchedulerP&, const PatchSet*,
                                   const MaterialSet*, const bool recursion);

  void scheduleDestroyHCMatrix(    SchedulerP&, const PatchSet*,
                                   const MaterialSet*);

  void scheduleCreateMatrix(       SchedulerP&, const PatchSet*,
                                   const MaterialSet*);

  void scheduleCreateHCMatrix(     SchedulerP&, const PatchSet*,
                                   const MaterialSet*);

  void scheduleApplyBoundaryConditions(        SchedulerP&, const PatchSet*,
                                               const MaterialSet*);

  void scheduleApplyHCBoundaryConditions(      SchedulerP&, const PatchSet*,
                                               const MaterialSet*);

  void scheduleComputeContact(                 SchedulerP&, const PatchSet*,
                                               const MaterialSet*);

  void scheduleFindFixedDOF(                   SchedulerP&, const PatchSet*, 
                                               const MaterialSet*);

  void scheduleFindFixedHCDOF(                 SchedulerP&, const PatchSet*, 
                                               const MaterialSet*);

  void scheduleComputeStressTensorImplicit(    SchedulerP&, const PatchSet*,
                                               const MaterialSet*);

  void scheduleComputeInternalHeatRate(        SchedulerP&, const PatchSet*,
                                               const MaterialSet*);

  void scheduleSolveHeatEquations(             SchedulerP&, const PatchSet*,
                                               const MaterialSet*);

  void scheduleIntegrateTemperatureRate(       SchedulerP&, const PatchSet*,
                                               const MaterialSet*);

  void scheduleSolveForDuCG(                   SchedulerP&,const PatchSet*,
                                               const MaterialSet*);

  void scheduleSolveForTemp(                   SchedulerP&,const PatchSet*,
                                               const MaterialSet*);

  void scheduleGetDisplacementIncrement(       SchedulerP&, const PatchSet*,
                                               const MaterialSet*);

  void scheduleGetTemperatureIncrement(        SchedulerP&, const PatchSet*,
                                               const MaterialSet*);

  void scheduleUpdateTotalDisplacement(        SchedulerP&, const PatchSet*,
                                               const MaterialSet*);

  void scheduleComputeAcceleration(            SchedulerP&, const PatchSet*,
                                               const MaterialSet*);

  void scheduleInterpolateToParticlesAndUpdate(SchedulerP&, const PatchSet*,
                                       const MaterialSet*);

  void scheduleInterpolateStressToGrid(        SchedulerP&, const PatchSet*,
                                               const MaterialSet*);

  void scheduleIterate(             SchedulerP&, const LevelP&,const PatchSet*, 
                                    const MaterialSet*);
  
  void scheduleCheckConvergence(    SchedulerP&, const LevelP&, const PatchSet*,
                                    const MaterialSet*);


  ImpMPM(const ImpMPM&);
  ImpMPM& operator=(const ImpMPM&);

  SimulationStateP d_sharedState;
  MPMLabel* lb;
  ImpMPMFlags* flags;

  ImplicitHeatConduction* heatConductionModel;
  ThermalContact* thermalContactModel;

  SwitchingCriteria* d_switchCriteria;

  double           d_nextOutputTime;
  double           d_SMALL_NUM_MPM;
  int              NGP;      // Number of ghost particles needed.
  int              NGN;      // Number of ghost nodes     needed.
  double           d_initialDt;
  int              d_numIterations;
  Vector           d_contact_dirs; // For rigid body contact
  std::string      d_con_type;
  double           d_stop_time;     // for rigid contact
  Vector           d_vel_after_stop;     // for rigid contact

  std::list<Patch::FaceType>  d_bndy_traction_faces; // list of xminus, xplus, ...

  const PatchSet* d_perproc_patches;

  Solver* d_solver;
  bool d_rigid_body;
  bool d_single_velocity;

  // stuff for not having to recompile the iterative scheduler every timstep
  SchedulerP d_subsched;
  bool d_recompileSubsched;
  
  MaterialSubset*  d_loadCurveIndex;

};
      

 struct particleTempShape {
   double particleTemps;
   vector<IntVector> cellNodes;
   vector<double> shapeValues;
 };
 
 typedef struct particleTempShape particleTempShape;
 
} // end namespace Uintah

#endif
