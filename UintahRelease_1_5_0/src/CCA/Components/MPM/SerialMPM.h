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


#ifndef UINTAH_HOMEBREW_SERIALMPM_H
#define UINTAH_HOMEBREW_SERIALMPM_H

#include <Core/Parallel/UintahParallelComponent.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <CCA/Ports/Output.h>
#include <CCA/Ports/SimulationInterface.h>
#include <CCA/Ports/SwitchingCriteria.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Variables/ComputeSet.h>
// put here to avoid template problems
#include <Core/Math/Matrix3.h>
#include <Core/Math/Short27.h>
#include <Core/Labels/MPMLabel.h>
#include <CCA/Components/MPM/Contact/Contact.h>
#include <CCA/Components/MPM/MPMCommon.h>
#include <Core/Geometry/Vector.h>
#include <CCA/Components/MPM/MPMFlags.h>
#include <CCA/Components/MPM/PhysicalBC/MPMPhysicalBC.h>
#include <CCA/Components/MPM/PhysicalBC/LoadCurve.h>
#include <CCA/Components/OnTheFlyAnalysis/AnalysisModule.h>
#include <Core/Grid/Variables/ParticleVariable.h>



namespace Uintah {

using namespace SCIRun;

class ThermalContact;
class HeatConduction;
class AnalysisModule;

/**************************************

CLASS
   SerialMPM
   
   Short description...

GENERAL INFORMATION

   SerialMPM.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   SerialMPM

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

class SerialMPM : public MPMCommon, public SimulationInterface, public UintahParallelComponent {
public:
  SerialMPM(const ProcessorGroup* myworld);
  virtual ~SerialMPM();

  Contact*         contactModel;
  ThermalContact*  thermalContactModel;
  HeatConduction* heatConductionModel;
 
  //////////
  // Insert Documentation Here:
  virtual void problemSetup(const ProblemSpecP& params, 
                            const ProblemSpecP& restart_prob_spec, GridP&,
                            SimulationStateP&);

  virtual void outputProblemSpec(ProblemSpecP& ps);

  virtual void scheduleInitialize(const LevelP& level,
                                  SchedulerP&);

  virtual void addMaterial(const ProblemSpecP& params,
                           GridP& grid,
                           SimulationStateP&);

  virtual void scheduleInitializeAddedMaterial(const LevelP& level, SchedulerP&);

  virtual void restartInitialize();

  void schedulePrintParticleCount(const LevelP& level, SchedulerP& sched);
  
  void scheduleTotalParticleCount(SchedulerP& sched,
                                 const PatchSet* patches,
                                 const MaterialSet* matls);
  //////////
  // Insert Documentation Here:
  virtual void scheduleComputeStableTimestep(const LevelP& level, SchedulerP&);

  //////////
  // Insert Documentation Here:
  virtual void scheduleTimeAdvance(const LevelP& level, SchedulerP&);

  virtual void scheduleRefine(const PatchSet* patches, SchedulerP& scheduler);

  virtual void scheduleRefineInterface(const LevelP& fineLevel, SchedulerP& scheduler,
                                       bool needCoarse, bool needFine);

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

  void setWithICE()
  {
        flags->d_with_ice = true;
  };

  void setWithARCHES()
  {
        flags->d_with_arches = true;
  };

  enum IntegratorType {
    Explicit,
    Implicit,
    Fracture
  };


protected:
  //////////
  // Insert Documentation Here:
  friend class MPMICE;
  friend class MPMArches;
 
  virtual void actuallyInitialize(const ProcessorGroup*,
                                  const PatchSubset* patches,
                                  const MaterialSubset* matls,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw);

  virtual void actuallyInitializeAddedMaterial(const ProcessorGroup*,
                                               const PatchSubset* patches,
                                               const MaterialSubset* matls,
                                               DataWarehouse* old_dw,
                                               DataWarehouse* new_dw);

  void printParticleCount(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw);
                          
  void totalParticleCount(const ProcessorGroup*,
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

  void printParticleLabels(vector<const VarLabel*> label,DataWarehouse* dw,
                          int dwi, const Patch* patch);

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

  //////////
  // Insert Documentation Here:
  void actuallyComputeStableTimestep(const ProcessorGroup*,
                                     const PatchSubset* patches,
                                     const MaterialSubset* matls,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  virtual void interpolateParticlesToGrid(const ProcessorGroup*,
                                          const PatchSubset* patches,
                                          const MaterialSubset* matls,
                                          DataWarehouse* old_dw,
                                          DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  virtual void addCohesiveZoneForces(const ProcessorGroup*,
                                     const PatchSubset* patches,
                                     const MaterialSubset* matls,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  virtual void computeStressTensor(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw);

  /*! Update the erosion parameter if mass is to be removed */
  void updateErosionParameter(const ProcessorGroup*,
                              const PatchSubset* patches,
                              const MaterialSubset* ,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw);

  /*! Find particles that should be deleted */
  void findRogueParticles(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* ,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw);

  //////////
  // Compute Accumulated Strain Energy
  void computeAccStrainEnergy(const ProcessorGroup*,
                              const PatchSubset*,
                              const MaterialSubset*,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  virtual void computeContactArea(const ProcessorGroup*,
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
  virtual void computeAndIntegrateAcceleration(const ProcessorGroup*,
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

  void addNewParticles(const ProcessorGroup*,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse* old_dw,
                       DataWarehouse* new_dw);


  /*!  Convert the localized particles into particles of a new material
       with a different velocity field */
  void convertLocalizedParticles(const ProcessorGroup*,
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

  //////////
  // Insert Documentation Here:
  virtual void updateCohesiveZones(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  virtual void setPrescribedMotion(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw);

  //////////
  // Allow blocks of particles to be moved according to a prescribed schedule:
  virtual void insertParticles(const ProcessorGroup*,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw);

  // Used to compute the particles initial physical size
  // for use in deformed particle visualization
  virtual void computeParticleScaleFactor(const ProcessorGroup*,
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

  virtual void scheduleInterpolateParticlesToGrid(SchedulerP&, const PatchSet*,
                                                  const MaterialSet*);

  virtual void scheduleAddCohesiveZoneForces(SchedulerP&, 
                                             const PatchSet*,
                                             const MaterialSubset*,
                                             const MaterialSubset*,
                                             const MaterialSet*);

  virtual void scheduleComputeHeatExchange(SchedulerP&, const PatchSet*,
                                           const MaterialSet*);

  virtual void scheduleExMomInterpolated(SchedulerP&, const PatchSet*,
                                         const MaterialSet*);

  virtual void scheduleComputeStressTensor(SchedulerP&, const PatchSet*,
                                           const MaterialSet*);
  

  void scheduleUpdateErosionParameter(SchedulerP& sched,
                                      const PatchSet* patches,
                                      const MaterialSet* matls);

  void scheduleFindRogueParticles(SchedulerP& sched,
                                  const PatchSet* patches,
                                  const MaterialSet* matls);

  void scheduleComputeAccStrainEnergy(SchedulerP&, const PatchSet*,
                                      const MaterialSet*);

  virtual void scheduleComputeContactArea(SchedulerP&, const PatchSet*,
                                          const MaterialSet*);
  
  virtual void scheduleComputeInternalForce(SchedulerP&, const PatchSet*,
                                            const MaterialSet*);

  virtual void scheduleComputeInternalHeatRate(SchedulerP&, const PatchSet*,
                                               const MaterialSet*);
                                          
  virtual void scheduleComputeNodalHeatFlux(SchedulerP&, const PatchSet*,
                                            const MaterialSet*);

  virtual void scheduleSolveHeatEquations(SchedulerP&, const PatchSet*,
                                          const MaterialSet*);

  virtual void scheduleComputeAndIntegrateAcceleration(SchedulerP&,
                                                       const PatchSet*,
                                                       const MaterialSet*);

  virtual void scheduleIntegrateTemperatureRate(SchedulerP&, const PatchSet*,
                                                const MaterialSet*);

  virtual void scheduleExMomIntegrated(SchedulerP&, const PatchSet*,
                                       const MaterialSet*);

  void scheduleSetGridBoundaryConditions(SchedulerP&, const PatchSet*,
                                         const MaterialSet* matls);
                                                 
  void scheduleApplyExternalLoads(SchedulerP&, const PatchSet*,
                                  const MaterialSet*);

  virtual void scheduleInterpolateToParticlesAndUpdate(SchedulerP&, 
                                                       const PatchSet*,
                                                       const MaterialSet*);

  virtual void scheduleUpdateCohesiveZones(SchedulerP&, 
                                           const PatchSet*,
                                           const MaterialSubset*,
                                           const MaterialSubset*,
                                           const MaterialSet*);

  virtual void scheduleSetPrescribedMotion(SchedulerP&, 
                                           const PatchSet*,
                                           const MaterialSet*);

  void scheduleAddNewParticles(SchedulerP&, const PatchSet*,const MaterialSet*);

  void scheduleConvertLocalizedParticles(SchedulerP&, const PatchSet*,
                                         const MaterialSet*);

  void scheduleCheckNeedAddMPMMaterial(SchedulerP&,
                                       const PatchSet* patches,
                                       const MaterialSet*);

  virtual void scheduleInterpolateToParticlesAndUpdateMom1(SchedulerP&, 
                                                           const PatchSet*,
                                                           const MaterialSet*);

  virtual void scheduleInterpolateParticleVelToGridMom(SchedulerP&, 
                                                       const PatchSet*,
                                                       const MaterialSet*);

  virtual void scheduleInterpolateToParticlesAndUpdateMom2(SchedulerP&, 
                                                           const PatchSet*,
                                                           const MaterialSet*);

  virtual void scheduleInsertParticles(SchedulerP&, 
                                       const PatchSet*,
                                       const MaterialSet*);

  virtual void scheduleComputeParticleScaleFactor(SchedulerP&, 
                                                  const PatchSet*,
                                                  const MaterialSet*);

  //////////
  // Insert Documentation Here:
  virtual void interpolateToParticlesAndUpdateMom1(const ProcessorGroup*,
                                                   const PatchSubset* patches,
                                                   const MaterialSubset* matls,
                                                   DataWarehouse* old_dw,
                                                   DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  virtual void interpolateToParticlesAndUpdateMom2(const ProcessorGroup*,
                                                   const PatchSubset* patches,
                                                   const MaterialSubset* matls,
                                                   DataWarehouse* old_dw,
                                                   DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  virtual void interpolateParticleVelToGridMom(const ProcessorGroup*,
                                               const PatchSubset* patches,
                                               const MaterialSubset* matls,
                                               DataWarehouse* old_dw,
                                               DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
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
  
  bool needRecompile(double time, double dt,
                     const GridP& grid);

  void readPrescribedDeformations(string filename);

  void readInsertParticlesFile(string filename);
  
  virtual void scheduleSwitchTest(const LevelP& level, SchedulerP& sched);
                   

  
  SimulationStateP d_sharedState;
  MPMLabel* lb;
  MPMFlags* flags;
  Output* dataArchiver;

  double           d_nextOutputTime;
  double           d_SMALL_NUM_MPM;
  int              NGP;      // Number of ghost particles needed.
  int              NGN;      // Number of ghost nodes     needed.
  
  std::list<Patch::FaceType>  d_bndy_traction_faces; // list of xminus, xplus, yminus, ...
  std::vector<MPMPhysicalBC*> d_physicalBCs;

  std::vector<double>   d_prescribedTimes;    // These three are used only if
  std::vector<double>  d_prescribedAngle;  // d_prescribeDeformation
  std::vector<Vector>  d_prescribedRotationAxis; // is "true".  It is "false" by default.
  std::vector<Matrix3>  d_prescribedF;

  // The following are used iff the d_insertParticles flag is true.
  std::vector<double> d_IPTimes;
  std::vector<double> d_IPColor;
  std::vector<Vector> d_IPTranslate;
  std::vector<Vector> d_IPVelNew;



  bool             d_fracture;
  bool             d_recompile;
  IntegratorType   d_integrator;
  MaterialSubset*  d_loadCurveIndex;
  
  std::vector<AnalysisModule*> d_analysisModules;
  SwitchingCriteria* d_switchCriteria;
  
private:

  SerialMPM(const SerialMPM&);
  SerialMPM& operator=(const SerialMPM&);
};
      
} // end namespace Uintah

#endif
