/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
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


#ifndef UINTAH_HOMEBREW_FRACTUREMPM_H
#define UINTAH_HOMEBREW_FRACTUREMPM_H

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/CCA/Ports/Output.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/CCA/Components/MPM/Crack/Crack.h>
// put here to avoid template problems
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/Short27.h>
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Packages/Uintah/CCA/Components/MPM/Contact/Contact.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMFlags.h>
#include <Packages/Uintah/CCA/Components/MPM/SerialMPM.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/MPMPhysicalBC.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleVariable.h>

#include <Packages/Uintah/CCA/Components/MPM/uintahshare.h>
namespace Uintah {

using namespace SCIRun;

class Crack;
class ThermalContact;
class HeatConduction;

/**************************************

CLASS
   FractureMPM
   
   Short description...

GENERAL INFORMATION

   FractureMPM.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   FractureMPM

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

class UINTAHSHARE FractureMPM : public SerialMPM {
public:
  FractureMPM(const ProcessorGroup* myworld);
   virtual ~FractureMPM();
   
  Crack*          crackModel; // for Fracture
	 
  //////////
  // Insert Documentation Here:
  virtual void problemSetup(const ProblemSpecP& params, 
                            const ProblemSpecP& restart_prob_spec,
                            GridP& grid, SimulationStateP&);
	 
  virtual void scheduleInitialize(const LevelP& level,
				  SchedulerP&);

  virtual void scheduleInitializeAddedMaterial(const LevelP& level,
                                               SchedulerP&);

  //////////
  // Insert Documentation Here:
  virtual void scheduleComputeStableTimestep(const LevelP& level,
					     SchedulerP&);
	 
  //////////
  // Insert Documentation Here:
  virtual void scheduleTimeAdvance(const LevelP& level, 
				   SchedulerP&);

  void scheduleRefine(const PatchSet* patches, SchedulerP& scheduler);

  void scheduleRefineInterface(const LevelP& fineLevel, SchedulerP& scheduler,
                               bool, bool);

  void scheduleCoarsen(const LevelP& coarseLevel, SchedulerP& sched);

  /// Schedule to mark flags for AMR regridding
  void scheduleErrorEstimate(const LevelP& coarseLevel, SchedulerP& sched);

  /// Schedule to mark initial flags for AMR regridding
  void scheduleInitialErrorEstimate(const LevelP& coarseLevel, SchedulerP& sched);


protected:
  //////////
  // Insert Documentation Here:
  friend class MPMICE;
  friend class MPMArches;

  virtual void materialProblemSetup(const ProblemSpecP& prob_spec,
                                    SimulationStateP& sharedState,
                                    MPMFlags* flags);

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

  void scheduleInitializePressureBCs(const LevelP& level,
                                     SchedulerP&);

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
  virtual void computeStressTensor(const ProcessorGroup*,
			           const PatchSubset* patches,
			           const MaterialSubset* matls,
			           DataWarehouse* old_dw,
			           DataWarehouse* new_dw);
  //////////
  // Insert Documentation Here: for thermal stress analysis
  virtual void computeParticleTempFromGrid(const ProcessorGroup*,
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

  //////////
  // Insert Documentation Here:
  virtual void computeContactArea(const ProcessorGroup*,
                                  const PatchSubset* patches,
                                  const MaterialSubset* matls,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  virtual void computeInternalForce(const ProcessorGroup*,
			    const PatchSubset* patches,
			    const MaterialSubset* matls,
			    DataWarehouse* old_dw,
			    DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  void computeArtificialViscosity(const ProcessorGroup*,
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

  //////////
  // Calculate the rate of evolution of the damping coefficient
  void calculateDampingRate(const ProcessorGroup*,
			    const PatchSubset* patches,
			    const MaterialSubset* matls,
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

  virtual void scheduleComputeStressTensor(SchedulerP&, const PatchSet*,
				           const MaterialSet*);

  // for thermal stress analysis
  virtual void scheduleComputeParticleTempFromGrid(SchedulerP&, const PatchSet*,
                                                   const MaterialSet*);

  void scheduleComputeAccStrainEnergy(SchedulerP&, const PatchSet*,
                                      const MaterialSet*);

  virtual void scheduleComputeContactArea(SchedulerP&, const PatchSet*,
                                          const MaterialSet*);

  virtual void scheduleComputeInternalForce(SchedulerP&, const PatchSet*,
				            const MaterialSet*);

  void scheduleComputeArtificialViscosity(SchedulerP&, const PatchSet*,
				          const MaterialSet*);

  virtual void scheduleComputeInternalHeatRate(SchedulerP&, const PatchSet*,
				               const MaterialSet*);

  virtual void scheduleSolveEquationsMotion(SchedulerP&, const PatchSet*,
				            const MaterialSet*);

  virtual void scheduleSolveHeatEquations(SchedulerP&, const PatchSet*,
				          const MaterialSet*);

  virtual void scheduleIntegrateAcceleration(SchedulerP&, const PatchSet*,
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

  void scheduleAddNewParticles(SchedulerP&, const PatchSet*,
                               const MaterialSet*);

  void scheduleConvertLocalizedParticles(SchedulerP&, const PatchSet*,
                                         const MaterialSet*);

  void scheduleCalculateDampingRate(SchedulerP&, const PatchSet*,
				    const MaterialSet*);

  // for Farcture ----------------------------------
  virtual void scheduleParticleVelocityField(SchedulerP& sched,
                                             const PatchSet* patches,
                                             const MaterialSet* matls);
  
  virtual void scheduleAdjustCrackContactInterpolated(SchedulerP& sched,
                                                      const PatchSet* patches,
                                                      const MaterialSet* matls);
  
  virtual void scheduleAdjustCrackContactIntegrated(SchedulerP& sched,    
                                                    const PatchSet* patches,
                                                    const MaterialSet* matls);
  
  virtual void scheduleCalculateFractureParameters(SchedulerP& sched,
                                                   const PatchSet* patches,
                                                   const MaterialSet* matls);
  
  virtual void scheduleDoCrackPropagation(SchedulerP& sched,
                                          const PatchSet* patches,
                                          const MaterialSet* matls);
  
  virtual void scheduleMoveCracks(SchedulerP& sched,             
                                  const PatchSet* patches,
                                  const MaterialSet* matls);
  
  virtual void scheduleUpdateCrackFront(SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSet* matls);

private:

  FractureMPM(const FractureMPM&);
  FractureMPM& operator=(const FractureMPM&);
	 
};
      
} // end namespace Uintah

#endif
