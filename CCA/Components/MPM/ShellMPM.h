#ifndef UINTAH_HOMEBREW_SHELLMPM_H
#define UINTAH_HOMEBREW_SHELLMPM_H

#include <Packages/Uintah/CCA/Components/MPM/SerialMPM.h>

namespace Uintah {

using namespace SCIRun;

/**************************************

CLASS
   ShellMPM
   
   MPM with extra stuff for shell formulation

GENERAL INFORMATION

   ShellMPM.h

   Biswajit Banerjee
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
   University of Utah
   Copyright (C) 2003 University of Utah

KEYWORDS
   ShellMPM

DESCRIPTION
   Adds shell specific stuff and extends SerialMPM
  
WARNING
  
****************************************/

class ShellMPM : public SerialMPM {

public:

  ///////////////////////////////////////////////////////////////////////////
  //
  // Constructor:  Uses the constructor of SerialMPM.  Changes to
  // The constructor of SerialMPM should be reflected here.
  //
  ShellMPM(const ProcessorGroup* myworld);
  virtual ~ShellMPM();

  ///////////////////////////////////////////////////////////////////////////
  //
  // Setup problem -- additional set-up parameters may be added here
  // for the shell problem
  //
  virtual void problemSetup(const ProblemSpecP& params, 
                            GridP& grid,
			    SimulationStateP&);
	 
  ///////////////////////////////////////////////////////////////////////////
  //
  // Schedule general initialization
  //
  virtual void scheduleInitialize(const LevelP& level,
				  SchedulerP&);
	 
protected:

  ///////////////////////////////////////////////////////////////////////////
  //
  // Initializate Shell Variables
  //
  void initializeShellVariables(const ProcessorGroup*,
				const PatchSubset* patches,
				const MaterialSubset* matls,
				DataWarehouse*,
				DataWarehouse* new_dw);

  ///////////////////////////////////////////////////////////////////////////
  //
  // Schedule interpolation from particles to the grid
  //
  virtual void scheduleInterpolateParticlesToGrid(SchedulerP& sched,
						  const PatchSet* patches,
						  const MaterialSet* matls);

  ///////////////////////////////////////////////////////////////////////////
  //
  // Schedule interpolation of rotation from particles to the grid
  //
  void schedInterpolateParticleRotToGrid(SchedulerP& sched,
					 const PatchSet* patches,
					 const MaterialSet* matls);

  ///////////////////////////////////////////////////////////////////////////
  //
  // Actually interpolate normal rotation from particles to the grid
  //
  void interpolateParticleRotToGrid(const ProcessorGroup*,
				    const PatchSubset* patches,
				    const MaterialSubset* ,
				    DataWarehouse* old_dw,
				    DataWarehouse* new_dw);

  ///////////////////////////////////////////////////////////////////////////
  //
  // Schedule computation of stress tensor .. This is a copy of the
  // SerialMPM one + extra variables for shells
  //
  virtual void scheduleComputeStressTensor(SchedulerP& sched,
					   const PatchSet* patches,
					   const MaterialSet* matls);

  ///////////////////////////////////////////////////////////////////////////
  //
  // Actually compute of stress tensor using the constitutive model
  //
  virtual void computeStressTensor(const ProcessorGroup* pg,
				   const PatchSubset* patches,
				   const MaterialSubset* matl,
				   DataWarehouse* old_dw,
				   DataWarehouse* new_dw);

  ///////////////////////////////////////////////////////////////////////////
  //
  // Schedule computation of Internal Force
  //
  virtual void scheduleComputeInternalForce(SchedulerP& sched,
					    const PatchSet* patches,
					    const MaterialSet* matls);

  ///////////////////////////////////////////////////////////////////////////
  //
  // Schedule computation of rotational internal moment
  //
  void schedComputeRotInternalMoment(SchedulerP& sched,
				     const PatchSet* patches,
				     const MaterialSet* matls);

  ///////////////////////////////////////////////////////////////////////////
  //
  // Actually compute rotational Internal moment
  //
  void computeRotInternalMoment(const ProcessorGroup*,
				const PatchSubset* patches,
				const MaterialSubset* ,
				DataWarehouse* old_dw,
				DataWarehouse* new_dw);

  ///////////////////////////////////////////////////////////////////////////
  //
  // Schedule Calculation of acceleration
  //
  virtual void scheduleSolveEquationsMotion(SchedulerP& sched,
					    const PatchSet* patches,
					    const MaterialSet* matls);

  ///////////////////////////////////////////////////////////////////////////
  //
  // Schedule calculation of rotational acceleration of shell normal
  //
  void schedComputeRotAcceleration(SchedulerP& sched,
				   const PatchSet* patches,
				   const MaterialSet* matls);

  ///////////////////////////////////////////////////////////////////////////
  //
  // Actually calculate of rotational acceleration of shell normal
  //
  void computeRotAcceleration(const ProcessorGroup*,
			      const PatchSubset* patches,
			      const MaterialSubset*,
			      DataWarehouse* old_dw,
			      DataWarehouse* new_dw);

  ///////////////////////////////////////////////////////////////////////////
  //
  // Schedule interpolation from grid to particles and update
  //
  void scheduleInterpolateToParticlesAndUpdate(SchedulerP& sched,
					       const PatchSet* patches,
					       const MaterialSet* matls);

  ///////////////////////////////////////////////////////////////////////////
  //
  // Schedule interpolation of rotation rate from grid to particles and update
  //
  void schedInterpolateRotToParticlesAndUpdate(SchedulerP& sched,
					       const PatchSet* patches,
					       const MaterialSet* matls);

  ///////////////////////////////////////////////////////////////////////////
  //
  // Actually interpolate rotation rate from grid to particles and update
  //
  void interpolateRotToParticlesAndUpdate(const ProcessorGroup*,
					  const PatchSubset* patches,
					  const MaterialSubset* ,
					  DataWarehouse* old_dw,
					  DataWarehouse* new_dw);


private:

  ///////////////////////////////////////////////////////////////////////////
  //
  // Forbid copying of this class
  //
  ShellMPM(const ShellMPM&);
  ShellMPM& operator=(const ShellMPM&);
	 
};
      
} // end namespace Uintah

#endif
