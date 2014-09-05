/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef UINTAH_HOMEBREW_AMRMPM_H
#define UINTAH_HOMEBREW_AMRMPM_H

#include <Core/Parallel/UintahParallelComponent.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <CCA/Ports/Output.h>
#include <CCA/Ports/SimulationInterface.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <CCA/Components/MPM/SerialMPM.h>
// put here to avoid template problems
#include <Core/Math/Matrix3.h>
#include <Core/Math/Short27.h>
#include <Core/Labels/MPMLabel.h>
#include <CCA/Components/MPM/MPMCommon.h>
#include <Core/Geometry/Vector.h>
#include <CCA/Components/MPM/MPMFlags.h>
#include <Core/Grid/Variables/ParticleVariable.h>

#include <CCA/Components/MPM/uintahshare.h>

namespace Uintah {

using namespace SCIRun;

class UINTAHSHARE AMRMPM : public SerialMPM {

public:
  AMRMPM(const ProcessorGroup* myworld);
  virtual ~AMRMPM();

  virtual void problemSetup(const ProblemSpecP& params, 
                            const ProblemSpecP& restart_prob_spec,
                            GridP&,
                            SimulationStateP&);

  virtual void outputProblemSpec(ProblemSpecP& ps);
         
  virtual void scheduleInitialize(const LevelP& level,
                                  SchedulerP&);

  void schedulePrintParticleCount(const LevelP& level, 
                                  SchedulerP& sched);

  virtual void scheduleComputeStableTimestep(const LevelP& level,
                                             SchedulerP&);
         
  virtual void scheduleTimeAdvance(const LevelP& level, 
                                   SchedulerP&);

  virtual void scheduleFinalizeTimestep(const LevelP& level,
                                        SchedulerP&);

  virtual void scheduleRefine(const PatchSet* patches, 
                              SchedulerP& scheduler);

  virtual void scheduleRefineInterface(const LevelP& fineLevel, 
                                       SchedulerP& scheduler,
                                       bool needCoarse, 
                                       bool needFine);

  virtual void scheduleCoarsen(const LevelP& coarseLevel, SchedulerP& sched);

  /// Schedule to mark flags for AMR regridding
  virtual void scheduleErrorEstimate(const LevelP& coarseLevel, 
                                     SchedulerP& sched);
  
  /// Schedule to mark initial flags for AMR regridding
  void scheduleInitialErrorEstimate(const LevelP& coarseLevel, SchedulerP& sched);

  void setSharedState(SimulationStateP& ssp);


  void setMPMLabel(MPMLabel* Mlb)
  {
    delete lb;
    lb = Mlb;
  };

  void setWithICE()
  {
    flags->d_with_ice = true;
  };

  enum bctype { NONE=0,
                FIXED,
                SYMMETRY,
                NEIGHBOR };

  enum IntegratorType {
    Explicit,
    Implicit,
    Fracture
  };

protected:

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

  void printParticleLabels(vector<const VarLabel*> label,
                           DataWarehouse* dw,
                           int dwi, 
                           const Patch* patch);

  void actuallyComputeStableTimestep(const ProcessorGroup*,
                                     const PatchSubset* patches,
                                     const MaterialSubset* matls,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw);

  virtual void computeZoneOfInfluence(const ProcessorGroup*,
                                      const PatchSubset* patches,
                                      const MaterialSubset* matls,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw);

  virtual void interpolateParticlesToGrid(const ProcessorGroup*,
                                          const PatchSubset* patches,
                                          const MaterialSubset* matls,
                                          DataWarehouse* old_dw,
                                          DataWarehouse* new_dw);

  virtual void computeStressTensor(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw);

  void updateErosionParameter(const ProcessorGroup*,
                              const PatchSubset* patches,
                              const MaterialSubset* ,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw);

  virtual void computeInternalForce(const ProcessorGroup*,
                                    const PatchSubset* patches,  
                                    const MaterialSubset* matls, 
                                    DataWarehouse* old_dw,       
                                    DataWarehouse* new_dw);      


  virtual void computeAndIntegrateAcceleration(const ProcessorGroup*,
                                               const PatchSubset* patches,
                                               const MaterialSubset* matls,
                                               DataWarehouse* old_dw,
                                               DataWarehouse* new_dw);

                          
  void setGridBoundaryConditions(const ProcessorGroup*,
                                 const PatchSubset* patches,
                                 const MaterialSubset* ,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw);
  //////////
  // This task is to be used for setting particle external force
  // and external heat rate.  I'm creating a separate task so that
  // user defined schemes for setting these can be implemented without
  // editing the core routines
  void applyExternalLoads(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* ,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw);

  virtual void interpolateToParticlesAndUpdate(const ProcessorGroup*,
                                               const PatchSubset* patches,
                                               const MaterialSubset* matls,
                                               DataWarehouse* old_dw,
                                               DataWarehouse* new_dw);

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

  virtual void scheduleComputeZoneOfInfluence(SchedulerP&, 
                                              const PatchSet*,
                                              const MaterialSet*);

  virtual void scheduleInterpolateParticlesToGrid(SchedulerP&, 
                                                  const PatchSet*,
                                                  const MaterialSet*);

  virtual void scheduleComputeStressTensor(SchedulerP&, 
                                           const PatchSet*,
                                           const MaterialSet*);
  
  void scheduleUpdateErosionParameter(SchedulerP& sched,
                                      const PatchSet* patches,
                                      const MaterialSet* matls);

  virtual void scheduleComputeInternalForce(SchedulerP&, 
                                            const PatchSet*,
                                            const MaterialSet*);

  virtual void scheduleComputeAndIntegrateAcceleration(SchedulerP&,
                                                       const PatchSet*,
                                                       const MaterialSet*);

  void scheduleSetGridBoundaryConditions(SchedulerP&, 
                                         const PatchSet*,
                                         const MaterialSet* matls);
                                                 
  void scheduleApplyExternalLoads(SchedulerP&, 
                                  const PatchSet*,
                                  const MaterialSet*);

  virtual void scheduleInterpolateToParticlesAndUpdate(SchedulerP&, 
                                                       const PatchSet*,
                                                       const MaterialSet*);

  void scheduleCheckNeedAddMPMMaterial(SchedulerP&,
                                        const PatchSet* patches,
                                        const MaterialSet*);
                                                                             
  void checkNeedAddMPMMaterial(const ProcessorGroup*,
                               const PatchSubset* patches,         
                               const MaterialSubset* matls,        
                               DataWarehouse* old_dw,              
                               DataWarehouse* new_dw);             

  void scheduleSetNeedAddMaterialFlag(SchedulerP&,
                                      const LevelP& level,        
                                      const MaterialSet*);        
  
  
  void setNeedAddMaterialFlag(const ProcessorGroup*,
                              const PatchSubset* patches,         
                              const MaterialSubset* matls,        
                              DataWarehouse*,                     
                              DataWarehouse*);                    
  
  bool needRecompile(double time, 
                     double dt,
                     const GridP& grid);
  
  
  virtual void scheduleSwitchTest(const LevelP& level, 
                                  SchedulerP& sched);
  
  virtual void switchTest(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse*,
                          DataWarehouse*);
  
  SimulationStateP d_sharedState;
  MPMLabel* lb;
  MPMFlags* flags;
  Output* dataArchiver;

  double   d_nextOutputTime;
  double   d_outputInterval;
  double   d_SMALL_NUM_MPM;
  int      NGP;      // Number of ghost particles needed.
  int      NGN;      // Number of ghost nodes     needed.
  
  list<Patch::FaceType>  d_bndy_traction_faces; // list of xminus, xplus, yminus, ...
  vector<MPMPhysicalBC*> d_physicalBCs;
  bool             d_recompile;
  IntegratorType d_integrator;

private:

  AMRMPM(const AMRMPM&);
  AMRMPM& operator=(const AMRMPM&);
         
};
      
} // end namespace Uintah

#endif
