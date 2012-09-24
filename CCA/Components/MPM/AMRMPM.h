/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
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


namespace Uintah {

using namespace SCIRun;
class GeometryObject;

class AMRMPM : public SerialMPM {

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


  void setMPMLabel(MPMLabel* Mlb)
  {
    delete lb;
    lb = Mlb;
  };

  enum IntegratorType {
    Explicit,
    Implicit,
  };

protected:
  enum coarsenFlag{
    coarsenData,
    zeroData,
  };

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

  
  void printParticleLabels(vector<const VarLabel*> label,
                           DataWarehouse* dw,
                           int dwi, 
                           const Patch* patch);

  void actuallyComputeStableTimestep(const ProcessorGroup*,
                                     const PatchSubset* patches,
                                     const MaterialSubset* matls,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw);
                                     
  void partitionOfUnity(const ProcessorGroup*,
                        const PatchSubset* patches,
                        const MaterialSubset* ,
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
  // At Coarse Fine interface
  void interpolateParticlesToGrid_CFI(const ProcessorGroup*,
                                      const PatchSubset* patches,
                                      const MaterialSubset* matls,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw);
                                      
  void coarsenNodalData_CFI(const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw,
                            const coarsenFlag flag);
                            
  void coarsenNodalData_CFI2(const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw);
                            
  void Nodal_velocity_temperature(const ProcessorGroup*,
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
                                    
  void computeInternalForce_CFI(const ProcessorGroup*,
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
  // At Coarse Fine interface
  void interpolateToParticlesAndUpdate_CFI(const ProcessorGroup*,
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

  //______________________________________________________________________
  //
  void schedulePartitionOfUnity(SchedulerP&, 
                                const PatchSet*,
                                const MaterialSet*);
                                  
  virtual void scheduleComputeZoneOfInfluence(SchedulerP&, 
                                              const PatchSet*,
                                              const MaterialSet*);

  virtual void scheduleInterpolateParticlesToGrid(SchedulerP&, 
                                                  const PatchSet*,
                                                  const MaterialSet*);
                                                  
  void scheduleInterpolateParticlesToGrid_CFI(SchedulerP&, 
                                              const PatchSet*,
                                              const MaterialSet*);
                                              
  void scheduleCoarsenNodalData_CFI(SchedulerP&, 
                                    const PatchSet*,
                                    const MaterialSet*,
                                    const coarsenFlag flag);

  void scheduleCoarsenNodalData_CFI2(SchedulerP&, 
                                     const PatchSet*,
                                     const MaterialSet*);
                                    
  void scheduleNodal_velocity_temperature(SchedulerP&, 
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
                                            
  void scheduleComputeInternalForce_CFI(SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSet* matls);

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
                                                       
  void scheduleInterpolateToParticlesAndUpdate_CFI(SchedulerP&, 
                                                   const PatchSet*,
                                                   const MaterialSet*);
  
  //
  //  count the total number of particles in the domain
  //
  void scheduleCountParticles(const PatchSet* patches,
                               SchedulerP& sched);
                                                                                         
  void countParticles(const ProcessorGroup*,
                      const PatchSubset* patches,                            
                      const MaterialSubset*,                   
                      DataWarehouse* old_dw,                                 
                      DataWarehouse* new_dw);
                      
  void scheduleDebug_CFI(SchedulerP&, 
                         const PatchSet*,
                         const MaterialSet*);
                                                                                         
  void debug_CFI(const ProcessorGroup*,
                 const PatchSubset* patches,                            
                 const MaterialSubset*,                   
                 DataWarehouse* old_dw,                                
                 DataWarehouse* new_dw); 
  //
  // returns does coarse patches have a CFI               
  void coarseLevelCFI_Patches(const PatchSubset* patches,
                               Level::selectType& CFI_patches );
  
  SimulationStateP d_sharedState;
  MPMLabel* lb;
  MPMFlags* flags;
  Output* dataArchiver;

  double   d_SMALL_NUM_MPM;
  int      NGP;                     // Number of ghost particles needed.
  int      NGN;                     // Number of ghost nodes  needed.
  int      d_nPaddingCells_Coarse;  // Number of cells on the coarse level that contain particles and surround a fine patch.
                                    // Coarse level particles are used in the task interpolateToParticlesAndUpdate_CFI.
                                   
  Vector   d_acc_ans;               // debugging code used to check the answers (acceleration)
  Vector   d_vel_ans;               // debugging code used to check the answers (velocity) 

  const VarLabel* pDbgLabel;        // debugging labels
  const VarLabel* gSumSLabel;                   
                                   
  vector<MPMPhysicalBC*> d_physicalBCs;
  IntegratorType d_integrator;

private:

  std::vector<GeometryObject*> d_refine_geom_objs;
  AMRMPM(const AMRMPM&);
  AMRMPM& operator=(const AMRMPM&);
         
         
  //______________________________________________________________________
  // Bulletproofing machinery to keep track of 
  // how many nodes on a CFI have been 'touched'
  inline void clearFaceMarks(const int whichMap, const Patch* patch) {
    faceMarks_map[whichMap].erase(patch);
  }
    
  inline int getFaceMark(int whichMap, 
                         const Patch* patch, 
                         Patch::FaceType face)
  {
    ASSERT(whichMap>=0 && whichMap<2);
    return faceMarks_map[whichMap][patch][face];
  };

  inline void setFaceMark(int whichMap, 
                          const Patch* patch, 
                          Patch::FaceType face, 
                          int value) 
  {
    ASSERT(whichMap>=0 && whichMap<2);
    faceMarks_map[whichMap][patch][face]=value;
  };
    
  
  struct faceMarks {
    int marks[Patch::numFaces];
    int& operator[](Patch::FaceType face)
    {
      return marks[static_cast<int>(face)];
    }
    faceMarks()
    {
      marks[0]=0;
      marks[1]=0;
      marks[2]=0;
      marks[3]=0;
      marks[4]=0;
      marks[5]=0;
    }
  };
  map<const Patch*,faceMarks> faceMarks_map[2];         
         
         
};
      
} // end namespace Uintah

#endif
