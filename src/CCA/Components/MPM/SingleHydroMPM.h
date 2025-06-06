/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
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

#ifndef UINTAH_HOMEBREW_SingleHydroMPM_H
#define UINTAH_HOMEBREW_SingleHydroMPM_H

#include <CCA/Ports/DataWarehouseP.h>
#include <CCA/Ports/Output.h>
#include <CCA/Ports/SwitchingCriteria.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Variables/ComputeSet.h>
// put here to avoid template problems
#include <Core/Math/Matrix3.h>
#include <Core/Math/Short27.h>
#include <CCA/Components/Application/ApplicationCommon.h>
#include <CCA/Components/MPM/Materials/Contact/Contact.h>
#include <CCA/Components/MPM/MPMCommon.h>
#include <Core/Geometry/Vector.h>
#include <CCA/Components/MPM/Core/MPMFlags.h>
#include <CCA/Components/MPM/PhysicalBC/MPMPhysicalBC.h>
#include <CCA/Components/MPM/PhysicalBC/LoadCurve.h>
#include <CCA/Components/OnTheFlyAnalysis/AnalysisModule.h>
#include <Core/Grid/Variables/ParticleVariable.h>

#include <CCA/Components/MPM/SerialMPM.h>

namespace Uintah {

class ThermalContact;
class HeatConduction;
class AnalysisModule;
class SDInterfaceModel;
class FluxBCModel;

class HydroMPMLabel;
class CZLabel;
class FluidContact;

/**************************************

CLASS
   SingleHydroMPM
   
GENERAL INFORMATION

   SingleHydroMPM.h   (copied from SerialMPM)

   Quoc Anh Tran
   Department of Civil and Environmental Engineering
   Norwegian University of Science and Technology 
   
KEYWORDS
   SingleHydroMPM

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class SingleHydroMPM : public ApplicationCommon, public MPMCommon {
public:
    SingleHydroMPM(const ProcessorGroup* myworld,
                   const MaterialManagerP materialManager);

  virtual ~SingleHydroMPM();

  Contact*          contactModel;
  FluidContact*     fluidContactModel;
  ThermalContact*   thermalContactModel;
  HeatConduction*   heatConductionModel;
  SDInterfaceModel* d_sdInterfaceModel;
  //////////
  // Insert Documentation Here:
  virtual double recomputeDelT(const double delT);

  virtual void problemSetup(const ProblemSpecP& params, 
                            const ProblemSpecP& restart_prob_spec,
                            GridP&);

  virtual void outputProblemSpec(ProblemSpecP& ps);

  virtual void scheduleInitialize(const LevelP& level,
                                  SchedulerP&);
                                  
  virtual void scheduleDeleteGeometryObjects(const LevelP& level,
                                             SchedulerP& sched);

  virtual void scheduleRestartInitialize(const LevelP& level,
                                         SchedulerP& sched);

  void schedulePrintParticleCount(const LevelP& level, SchedulerP& sched);
  
  void scheduleTotalParticleCount(SchedulerP& sched,
                                 const PatchSet* patches,
                                 const MaterialSet* matls);
  //////////
  // Insert Documentation Here:
  virtual void scheduleComputeStableTimeStep(const LevelP& level, SchedulerP&);

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

  void setWithICE()
  {
        flags->d_with_ice = true;
  };

  /*
  void setWithARCHES()
  {computeCurrentParticleSize
        flags->d_with_arches = true;
  };
  */

  enum IntegratorType {
    Explicit,
    Implicit,
    Fracture
  };


protected:
  //////////
  // Insert Documentation Here:
  friend class MPMICE;
  friend class MPMICE2;
  friend class MPMICE2explicit;
  friend class MPMICEexplicit;
  friend class MPMArches;

  SerialMPM* d_mpm;
  int              d_8or27;

  MaterialSubset* d_one_matl;         // matlsubset for zone of influence

  FluxBCModel*  d_fluxBC;
 
  virtual void actuallyInitialize(const ProcessorGroup*,
                                  const PatchSubset* patches,
                                  const MaterialSubset* matls,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw);

  void deleteGeometryObjects(const ProcessorGroup*,
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

  void printParticleLabels(std::vector<const VarLabel*> label,DataWarehouse* dw,
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
  virtual void interpolateParticlesToGrid(const ProcessorGroup*,
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

  //////////
  // Compute Accumulated Strain Energy
  void computeAccStrainEnergy(const ProcessorGroup*,
                              const PatchSubset*,
                              const MaterialSubset*,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw);

  virtual void computeInternalForce(const ProcessorGroup*,
                              const PatchSubset* patches,
                              const MaterialSubset* matls,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw);

  //////////
 // F_drag = Q * (w-v)
  virtual void computeFluidDragForce(const ProcessorGroup*,
      const PatchSubset* patches,
      const MaterialSubset*,
      DataWarehouse* old_dw,
      DataWarehouse* new_dw);
 
  //////////
 // For hydro-mechanical coupling MPM, compute fluid velocities and acceleration
  virtual void computeAndIntegrateFluidAcceleration(const ProcessorGroup*,
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
  void applyExternalFluidLoads(const ProcessorGroup*,
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

  virtual void computeParticleGradients(const ProcessorGroup*,
                                        const PatchSubset* patches,
                                        const MaterialSubset* matls,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  virtual void finalParticleUpdate(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw);

  virtual void InterpolateParticleToGridFilter(const ProcessorGroup*,
      const PatchSubset* patches,
      const MaterialSubset* matls,
      DataWarehouse* old_dw,
      DataWarehouse* new_dw);

  virtual void InterpolateGridToParticleFilter(const ProcessorGroup*,
      const PatchSubset* patches,
      const MaterialSubset* matls,
      DataWarehouse* old_dw,
      DataWarehouse* new_dw);

  virtual void insertParticles(const ProcessorGroup*,
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

  virtual void scheduleComputeHeatExchange(SchedulerP&, const PatchSet*,
                                           const MaterialSet*);

  virtual void scheduleExMomInterpolated(SchedulerP&, const PatchSet*,
                                         const MaterialSet*);

  virtual void scheduleFluidExMomInterpolated(SchedulerP&, const PatchSet*,
      const MaterialSet*);

  virtual void scheduleComputeStressTensor(SchedulerP&, const PatchSet*,
                                           const MaterialSet*);

  void scheduleComputeAccStrainEnergy(SchedulerP&, const PatchSet*,
                                      const MaterialSet*);

  virtual void scheduleComputeInternalForce(SchedulerP&, const PatchSet*,
                                            const MaterialSet*);

  virtual void scheduleComputeInternalHeatRate(SchedulerP&, const PatchSet*,
                                               const MaterialSet*);
                                          
  virtual void scheduleComputeNodalHeatFlux(SchedulerP&, const PatchSet*,
                                            const MaterialSet*);

  virtual void scheduleSolveHeatEquations(SchedulerP&, const PatchSet*,
                                          const MaterialSet*);

  virtual void scheduleComputeFluidDragForce(SchedulerP&, const PatchSet*,
      const MaterialSet*);
  
  virtual void scheduleComputeAndIntegrateFluidAcceleration(SchedulerP&,
      const PatchSet*,
      const MaterialSet*);

  virtual void scheduleComputeAndIntegrateAcceleration(SchedulerP&,
                                                       const PatchSet*,
                                                       const MaterialSet*);

  virtual void scheduleIntegrateTemperatureRate(SchedulerP&, const PatchSet*,
                                                const MaterialSet*);

  virtual void scheduleExMomIntegrated(SchedulerP&, const PatchSet*,
                                       const MaterialSet*);

  virtual void scheduleFluidExMomIntegrated(SchedulerP&, const PatchSet*,
      const MaterialSet*);

  void scheduleSetGridBoundaryConditions(SchedulerP&, const PatchSet*,
                                         const MaterialSet* matls);
                                                 
  void scheduleApplyExternalFluidLoads(SchedulerP&, const PatchSet*,
                                  const MaterialSet*);

  virtual void scheduleInterpolateToParticlesAndUpdate(SchedulerP&, 
                                                       const PatchSet*,
                                                       const MaterialSet*);

  virtual void scheduleComputeParticleGradients(SchedulerP&, 
                                                const PatchSet*,
                                                const MaterialSet*);

  virtual void scheduleFinalParticleUpdate(SchedulerP&, 
                                           const PatchSet*,
                                           const MaterialSet*);

  // Null space filter for pore pressure
  virtual void scheduleInterpolateParticleToGridFilter(SchedulerP&,
      const PatchSet*,
      const MaterialSet*);

  virtual void scheduleInterpolateGridToParticleFilter(SchedulerP&,
      const PatchSet*,
      const MaterialSet*);

  virtual void scheduleInsertParticles(SchedulerP&,
      const PatchSet*,
      const MaterialSet*);

  // JBH -- Scalar Diffusion Related

  void readPrescribedDeformations(std::string filename);

  void readInsertParticlesFile(std::string filename);
  
  virtual void scheduleSwitchTest(const LevelP& level, SchedulerP& sched);

  // Check for rigid material and coupled flow
  bool hasFluid(int m);

  //__________________________________
  // refinement criteria threshold knobs
  struct thresholdVar {
    std::string name;
    int matl;
    double value;
  };
  std::vector<thresholdVar> d_thresholdVars;
                   
  inline void computeVelocityGradient(Matrix3& velGrad,
                                    std::vector<IntVector>& ni,
                                    std::vector<Vector>& d_S,
                                    const double* oodx,
                                    constNCVariable<Vector>& gVelocity,
                                    const int NN)
  {
    for(int k = 0; k < NN; k++) {
      const Vector& gvel = gVelocity[ni[k]];
      for (int j = 0; j<3; j++){
        double d_SXoodx = d_S[k][j]*oodx[j];
        for (int i = 0; i<3; i++) {
          velGrad(i,j) += gvel[i] * d_SXoodx;
        }
      }
    }
  };


  inline void computeAxiSymVelocityGradient(Matrix3& velGrad,
                                           std::vector<IntVector>& ni,
                                           std::vector<Vector>& d_S,
                                           std::vector<double>& S,
                                           const double* oodx,
                                           constNCVariable<Vector>& gVelocity,
                                           const Point& px, const int NN)
  {
    // x -> r, y -> z, z -> theta
    for(int k = 0; k < NN; k++) {
      Vector gvel = gVelocity[ni[k]];
      for (int j = 0; j<2; j++){
        for (int i = 0; i<2; i++) {
          velGrad(i,j)+=gvel[i] * d_S[k][j] * oodx[j];
        }
      }
      velGrad(2,2) += gvel.x()*d_S[k].z();
    }
  };
  
  MPMFlags* flags;

  double           d_nextOutputTime;
  double           d_SMALL_NUM_MPM;
  int              NGP;      // Number of ghost particles needed.
  int              NGN;      // Number of ghost nodes     needed.
  
  std::list<Patch::FaceType>  d_bndy_traction_faces; // list of xminus, xplus, yminus, ...
  std::vector<MPMPhysicalBC*> d_physicalBCs;

  std::vector<double>  d_prescribedTimes;    // These three are used only if
  std::vector<double>  d_prescribedAngle;    // d_prescribeDeformation
  std::vector<Vector>  d_prescribedRotationAxis; // is "true".  It is "false" by default.
  std::vector<Matrix3>  d_prescribedF;

  // The following are used iff the d_insertParticles flag is true.
  std::vector<double> d_IPTimes;
  std::vector<double> d_IPColor;
  std::vector<Vector> d_IPTranslate;
  std::vector<Vector> d_IPVelNew;


  bool             d_fracture;
  MaterialSubset*  d_loadCurveIndex;
  
  std::vector<AnalysisModule*> d_analysisModules;
  SwitchingCriteria* d_switchCriteria;
  
  HydroMPMLabel* Hlb{ nullptr };
  CZLabel* Cl{ nullptr };
  CohesiveZoneTasks* cohesiveZoneTasks;

private:

  SingleHydroMPM(const SingleHydroMPM&);
  SingleHydroMPM& operator=(const SingleHydroMPM&);
};
      
} // end namespace Uintah

#endif
