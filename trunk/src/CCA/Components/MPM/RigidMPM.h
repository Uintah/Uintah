#ifndef UINTAH_HOMEBREW_RIGIDMPM_H
#define UINTAH_HOMEBREW_RIGIDMPM_H

#include <Core/Parallel/UintahParallelComponent.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <CCA/Ports/SimulationInterface.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <CCA/Components/MPM/Contact/Contact.h>
#include <CCA/Components/MPM/SerialMPM.h>
#include <SCIRun/Core/Geometry/Vector.h>
#include <CCA/Components/MPM/PhysicalBC/MPMPhysicalBC.h>
#include <Core/Grid/Variables/ParticleVariable.h>

#include <CCA/Components/MPM/uintahshare.h>
namespace Uintah {

using namespace SCIRun;

class ThermalContact;

/**************************************

CLASS
   RigidMPM
   
   Short description...

GENERAL INFORMATION

   RigidMPM.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   RigidMPM

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

class UINTAHSHARE RigidMPM : public SerialMPM {
public:
  RigidMPM(const ProcessorGroup* myworld);
  virtual ~RigidMPM();

  //////////
  // Insert Documentation Here:
  virtual void problemSetup(const ProblemSpecP& params, 
                            const ProblemSpecP& restart_prob_spec, 
                            GridP& grid, SimulationStateP&);
	 
  //////////
  // Insert Documentation Here:
  friend class MPMICE;
  friend class MPMArches;

  //////////
  // Insert Documentation Here:
  void computeStressTensor(const ProcessorGroup*,
			   const PatchSubset* patches,
			   const MaterialSubset* matls,
			   DataWarehouse* old_dw,
			   DataWarehouse* new_dw);

  //////////
  // Insert Documentation Here:
  void scheduleComputeInternalForce(           SchedulerP&, const PatchSet*,
                                               const MaterialSet*);


  void computeInternalForce(const ProcessorGroup*,
			    const PatchSubset* patches,
			    const MaterialSubset* matls,
			    DataWarehouse* old_dw,
			    DataWarehouse* new_dw);

  void scheduleSolveEquationsMotion(      SchedulerP&, const PatchSet*,
                                          const MaterialSet*);


  // Insert Documentation Here:
  void solveEquationsMotion(const ProcessorGroup*,
			    const PatchSubset* patches,
			    const MaterialSubset* matls,
			    DataWarehouse* old_dw,
			    DataWarehouse* new_dw);

  void scheduleInterpolateToParticlesAndUpdate(SchedulerP&,
                                                       const PatchSet*,
                                                       const MaterialSet*);

  void interpolateToParticlesAndUpdate(const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

private:
  RigidMPM(const RigidMPM&);
  RigidMPM& operator=(const RigidMPM&);
};
      
} // end namespace Uintah

#endif
