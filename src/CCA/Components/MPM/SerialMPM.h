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
#include <CCA/Components/MPM/PhysicalBC/FluxBCModel.h>
#include <CCA/Components/OnTheFlyAnalysis/AnalysisModule.h>
#include <Core/Grid/Variables/ParticleVariable.h>



namespace Uintah {

class ThermalContact;
class HeatConduction;
class AnalysisModule;
class SDInterfaceModel;
class FluxBCModel;

//-----------------------------------
class SerialMPM : public MPMCommon,
                  public SimulationInterface,
                  public UintahParallelComponent
{
public:
           SerialMPM(const ProcessorGroup* myworld);
  virtual ~SerialMPM();

  enum IntegratorType
  {
    Explicit,
    Implicit,
    Fracture
  };

  Contact*          contactModel;
  ThermalContact*   thermalContactModel;
  HeatConduction*   heatConductionModel;
  SDInterfaceModel* d_sdInterfaceModel;

protected:
  friend class MPMICE;
  friend class MPMArches;

  SimulationStateP  d_sharedState;
  MPMLabel*         lb;
  MPMFlags*         flags;
  Output*           dataArchiver;
  FluxBCModel*      d_fluxbc;

  double            d_nextOutputTime;
  double            d_SMALL_NUM_MPM;
  int               NGP;                // Number of ghost particles needed.
  int               NGN;                // Number of ghost nodes     needed.
  int               d_ndim;             // Num. of dimensions, 2 or 3.
                                        //   If 2, assume x-y

  // list of xminus, xplus, yminus, ...
  std::list<Patch::FaceType>    d_bndy_traction_faces;

  std::vector<MPMPhysicalBC*>   d_physicalBCs;
  std::vector<AnalysisModule*>  d_analysisModules;

  std::vector<double>   d_prescribedTimes;        // These three are used only
  std::vector<double>   d_prescribedAngle;        //  if d_prescribeDeformation
  std::vector<Vector>   d_prescribedRotationAxis; //  is "true".  It is "false"
  std::vector<Matrix3>  d_prescribedF;            //  by default.

  // The following are used iff the d_insertParticles flag is true.
  std::vector<double>   d_IPTimes;
  std::vector<double>   d_IPColor;
  std::vector<Vector>   d_IPTranslate;
  std::vector<Vector>   d_IPVelNew;

  bool                  d_fracture;
  bool                  d_recompile;
  IntegratorType        d_integrator;
  MaterialSubset*       d_loadCurveIndex;

  SwitchingCriteria*    d_switchCriteria;

  // Diffusion model members
  //SDInterfaceModel*     d_sdInterfaceModel;

private:
  SerialMPM(const SerialMPM&);
  SerialMPM& operator=(const SerialMPM&);


// Methods inherited from SimulationInterface
public:
 
  virtual void problemSetup(const ProblemSpecP      & params            ,
                            const ProblemSpecP      & restart_prob_spec ,
                                  GridP             &                   ,
                                  SimulationStateP  &                   );

  virtual void outputProblemSpec(ProblemSpecP& ps);


  virtual void scheduleInitialize(const LevelP      & level ,
                                        SchedulerP  &       );
  // JBH -- Why does SerialMPM have a restartInitialize() method but doesn't
  //        schedule it? TODO FIXME
  virtual void scheduleRestartInitialize(const LevelP     & level ,
                                               SchedulerP & sched );

  virtual void restartInitialize();

  virtual void scheduleComputeStableTimestep(const LevelP     & level ,
                                                   SchedulerP &       );

  virtual void scheduleTimeAdvance(const LevelP     & level ,
                                         SchedulerP &       );

  virtual void scheduleRefine(const PatchSet    * patches   ,
                                    SchedulerP  & scheduler );

  virtual void scheduleRefineInterface(const LevelP     & fineLevel   ,
                                             SchedulerP & scheduler   ,
                                             bool         needCoarse  ,
                                             bool         needFine    );

  virtual void scheduleCoarsen(const LevelP     & coarseLevel ,
                                     SchedulerP & sched       );

  /// Schedule to mark flags for AMR regridding
  virtual void scheduleErrorEstimate(const LevelP     & coarseLevel ,
                                           SchedulerP & sched       );

  /// Schedule to mark initial flags for AMR regridding
  void scheduleInitialErrorEstimate(const LevelP      & coarseLevel ,
                                          SchedulerP  & sched       );

protected:
  bool needRecompile(      double   time  ,
                           double   dt    ,
                     const GridP  & grid  );

  virtual void scheduleSwitchTest(const LevelP      & level ,
                                        SchedulerP  & sched );
//  ---- End methods from SimulationInterface

// Methods native to SerialMPM
public:
  void schedulePrintParticleCount(const LevelP      & level ,
                                        SchedulerP  & sched );

  void scheduleTotalParticleCount(SchedulerP& sched,
                                 const PatchSet* patches,
                                 const MaterialSet* matls);

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

protected:

  void printParticleCount(const ProcessorGroup  *         ,
                          const PatchSubset     * patches ,
                          const MaterialSubset  * matls   ,
                                DataWarehouse   * old_dw  ,
                                DataWarehouse   * new_dw  );

  void printParticleLabels(      std::vector<const VarLabel*>   label ,
                                 DataWarehouse                * dw    ,
                                 int                            dwi   ,
                           const Patch                        * patch );

  void actuallyComputeStableTimestep(const ProcessorGroup *         ,
                                     const PatchSubset    * patches ,
                                     const MaterialSubset * matls   ,
                                           DataWarehouse  * old_dw  ,
                                           DataWarehouse  * new_dw  );

  void setGridBoundaryConditions(const ProcessorGroup   *         ,
                                 const PatchSubset      * patches ,
                                 const MaterialSubset   *         ,
                                       DataWarehouse    * old_dw  ,
                                       DataWarehouse    * new_dw  );

  void totalParticleCount(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw);

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

  // Compute Accumulated Strain Energy
  void computeAccStrainEnergy(const ProcessorGroup*,
                              const PatchSubset*,
                              const MaterialSubset*,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw);

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

  void scheduleComputeAccStrainEnergy(SchedulerP&, const PatchSet*,
                                      const MaterialSet*);
  void scheduleSetGridBoundaryConditions(SchedulerP&, const PatchSet*,
                                         const MaterialSet* matls);

  void scheduleApplyExternalLoads(SchedulerP&, const PatchSet*,
                                  const MaterialSet*);

  void readPrescribedDeformations(std::string filename);

  void readInsertParticlesFile(std::string filename);
  
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

// Methods native to SerialMPM

// Inheritable functions for classes derived from Serial MPM
protected:
  virtual void actuallyInitialize(const ProcessorGroup  *         ,
                                  const PatchSubset     * patches ,
                                  const MaterialSubset  * matls   ,
                                        DataWarehouse   * old_dw  ,
                                        DataWarehouse   * new_dw  );

  virtual void interpolateParticlesToGrid(const ProcessorGroup  *         ,
                                          const PatchSubset     * patches ,
                                          const MaterialSubset  * matls   ,
                                                DataWarehouse   * old_dw  ,
                                                DataWarehouse   * new_dw  );

  virtual void computeStressTensor(const ProcessorGroup *         ,
                                   const PatchSubset    * patches ,
                                   const MaterialSubset * matls   ,
                                         DataWarehouse  * old_dw  ,
                                         DataWarehouse  * new_dw  );

  virtual void computeInternalForce(const ProcessorGroup  *         ,
                                    const PatchSubset     * patches ,
                                    const MaterialSubset  * matls   ,
                                          DataWarehouse   * old_dw  ,
                                          DataWarehouse   * new_dw  );

  virtual void computeAndIntegrateAcceleration(const ProcessorGroup *         ,
                                               const PatchSubset    * patches ,
                                               const MaterialSubset * matls   ,
                                                     DataWarehouse  * old_dw  ,
                                                     DataWarehouse  * new_dw  );


  virtual void interpolateToParticlesAndUpdate(const ProcessorGroup *         ,
                                               const PatchSubset    * patches ,
                                               const MaterialSubset * matls   ,
                                                     DataWarehouse  * old_dw  ,
                                                     DataWarehouse  * new_dw  );
  // Used to compute the particles initial physical size
  // for use in deformed particle visualization
  virtual void computeParticleScaleFactor(const ProcessorGroup  *         ,
                                          const PatchSubset     * patches ,
                                          const MaterialSubset  * matls   ,
                                                DataWarehouse   * old_dw  ,
                                                DataWarehouse   * new_dw  );

  virtual void finalParticleUpdate(const ProcessorGroup *         ,
                                   const PatchSubset    * patches ,
                                   const MaterialSubset * matls   ,
                                         DataWarehouse  * old_dw  ,
                                         DataWarehouse  * new_dw  );

  //////////
  // Add new particles to the simulation based on criteria TBD:
  virtual void addParticles(const ProcessorGroup  *         ,
                            const PatchSubset     * patches ,
                            const MaterialSubset  * matls   ,
                                  DataWarehouse   * old_dw  ,
                                  DataWarehouse   * new_dw  );

  virtual void addCohesiveZoneForces(const ProcessorGroup*,
                                     const PatchSubset* patches,
                                     const MaterialSubset* matls,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw);

  virtual void computeContactArea(const ProcessorGroup*,
                                  const PatchSubset* patches,
                                  const MaterialSubset* matls,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw);

  virtual void updateCohesiveZones(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw);

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


  virtual void scheduleInterpolateToParticlesAndUpdate(SchedulerP&,
                                                       const PatchSet*,
                                                       const MaterialSet*);

  virtual void scheduleFinalParticleUpdate(SchedulerP&,
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

#if 0
  virtual void scheduleInterpolateToParticlesAndUpdateMom1(SchedulerP&,
                                                           const PatchSet*,
                                                           const MaterialSet*);

  virtual void scheduleInterpolateToParticlesAndUpdateMom2(SchedulerP&,
                                                           const PatchSet*,
                                                           const MaterialSet*);
#endif

  virtual void scheduleInterpolateParticleVelToGridMom(SchedulerP&,
                                                       const PatchSet*,
                                                       const MaterialSet*);

  virtual void scheduleInsertParticles(SchedulerP&,
                                       const PatchSet*,
                                       const MaterialSet*);

  virtual void scheduleAddParticles(SchedulerP&,
                                    const PatchSet*,
                                    const MaterialSet*);

  virtual void scheduleComputeParticleScaleFactor(SchedulerP&,
                                                  const PatchSet*,
                                                  const MaterialSet*);

  virtual void interpolateToParticlesAndUpdateMom1(const ProcessorGroup*,
                                                   const PatchSubset* patches,
                                                   const MaterialSubset* matls,
                                                   DataWarehouse* old_dw,
                                                   DataWarehouse* new_dw);

  virtual void interpolateToParticlesAndUpdateMom2(const ProcessorGroup*,
                                                   const PatchSubset* patches,
                                                   const MaterialSubset* matls,
                                                   DataWarehouse* old_dw,
                                                   DataWarehouse* new_dw);

  virtual void interpolateParticleVelToGridMom(const ProcessorGroup*,
                                               const PatchSubset* patches,
                                               const MaterialSubset* matls,
                                               DataWarehouse* old_dw,
                                               DataWarehouse* new_dw);

//  // Routines for phase-field diffusion
  virtual void scheduleConcInterpolated(      SchedulerP  & sched   ,
                                        const PatchSet    * patches ,
                                        const MaterialSet * matls   );

  virtual void scheduleComputeFlux(
                                         SchedulerP   &         ,
                                   const PatchSet     * patches ,
                                   const MaterialSet  * matls   );

  virtual void computeFlux(const ProcessorGroup *         ,
                           const PatchSubset    * patches ,
                           const MaterialSubset * matls   ,
                                 DataWarehouse  * old_dw  ,
                                 DataWarehouse  * new_dw  );

  virtual void scheduleComputeDivergence(      SchedulerP   &         ,
                                         const PatchSet     * patches ,
                                         const MaterialSet  * matls   );

  virtual void computeDivergence(const ProcessorGroup *         ,
                                 const PatchSubset    * patches ,
                                 const MaterialSubset * matls   ,
                                       DataWarehouse  * old_dw  ,
                                       DataWarehouse  * new_dw  );

  virtual void scheduleDiffusionInterfaceDiv(      SchedulerP   & sched   ,
                                             const PatchSet     * patches ,
                                             const MaterialSet  * matls   );

  virtual void scheduleComputeChemicalPotential(      SchedulerP  & sched   ,
                                                const PatchSet    * patches ,
                                                const MaterialSet * matls   );

  virtual void computeChemicalPotential(
                                        const ProcessorGroup  *         ,
                                        const PatchSubset     * patches ,
                                        const MaterialSubset  * matls   ,
                                              DataWarehouse   * old_dw  ,
                                              DataWarehouse   * new_dw
                                       );

// Inheritable functions for classes derived from SerialMPM

};
      
} // end namespace Uintah

#endif
