#include <Packages/Uintah/CCA/Components/MPM/ShellMPM.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ShellMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/CCA/Components/MPM/Contact/Contact.h>
#include <Packages/Uintah/CCA/Components/MPM/Contact/ContactFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/ParticleCreator/ShellParticleCreator.h>
#include <Packages/Uintah/CCA/Components/MPM/ThermalContact/ThermalContact.h>
#include <Packages/Uintah/CCA/Components/MPM/ThermalContact/ThermalContactFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/ForceBC.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/PressureBC.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>

#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SoleVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/BoundCond.h>
#include <Packages/Uintah/Core/Grid/VelocityBoundCond.h>
#include <Packages/Uintah/Core/Grid/SymmetryBoundCond.h>
#include <Packages/Uintah/Core/Grid/TemperatureBoundCond.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Grid/fillFace.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/MinMax.h>
#include <Core/Util/NotFinished.h>
#include <Core/Util/DebugStream.h>

#include <iostream>
#include <fstream>

#undef KUMAR
//#define KUMAR

using namespace Uintah;
using namespace SCIRun;

using namespace std;

#define MAX_BASIS 27
#undef INTEGRAL_TRACTION

static DebugStream cout_doing("MPM", false);

// From ThreadPool.cc:  Used for syncing cerr'ing so it is easier to read.
extern Mutex cerrLock;

///////////////////////////////////////////////////////////////////////////
//
// Construct ShellMPM using the SerialMPM constructor
//
ShellMPM::ShellMPM(const ProcessorGroup* myworld):SerialMPM(myworld)
{
}

///////////////////////////////////////////////////////////////////////////
//
// Destruct ShellMPM using the SerialMPM destructor
//
ShellMPM::~ShellMPM()
{
}

///////////////////////////////////////////////////////////////////////////
//
// Setup problem -- additional set-up parameters may be added here
// for the shell problem
//
void 
ShellMPM::problemSetup(const ProblemSpecP& prob_spec, GridP& grid,
		       SimulationStateP& sharedState)
{
  SerialMPM::problemSetup(prob_spec, grid, sharedState);
}

///////////////////////////////////////////////////////////////////////////
//
// Setup material part of the problem specific to the shell formulation
// Nothing special right now .. but option of adding stuff is made available
//
void 
ShellMPM::materialProblemSetup(const ProblemSpecP& prob_spec, 
			       SimulationStateP& sharedState,
			       MPMLabel* lb, int /*n8or27*/,
			       string integrator, bool /*haveLoadCurve*/,
			       bool /*doErosion*/)
{
  //Search for the MaterialProperties block and then get the MPM section
  ProblemSpecP mat_ps =  prob_spec->findBlock("MaterialProperties");
  ProblemSpecP mpm_mat_ps = mat_ps->findBlock("MPM");
  for (ProblemSpecP ps = mpm_mat_ps->findBlock("material"); ps != 0;
       ps = ps->findNextBlock("material") ) {
    MPMMaterial *mat = scinew MPMMaterial(ps, lb, d_8or27,integrator,
					  d_useLoadCurves, d_doErosion);
    //register as an MPM material
    sharedState->registerMPMMaterial(mat);
  }
}

///////////////////////////////////////////////////////////////////////////
//
// Schedule interpolation from particles to the grid
//
void 
ShellMPM::scheduleInterpolateParticlesToGrid(SchedulerP& sched,
					     const PatchSet* patches,
					     const MaterialSet* matls)
{
  // First do the standard SerialMPM stuff
  SerialMPM::scheduleInterpolateParticlesToGrid(sched, patches, matls);

  // Then add a task for interpolating shell normal rotations to the grid
  schedInterpolateParticleRotToGrid(sched, patches, matls);
}

///////////////////////////////////////////////////////////////////////////
//
// Schedule interpolation of rotation from particles to the grid
//
void 
ShellMPM::schedInterpolateParticleRotToGrid(SchedulerP& sched,
					    const PatchSet* patches,
					    const MaterialSet* matls)
{
  Task* t = scinew Task("ShellMPM::interpolateParticleRotToGrid",
			this,&ShellMPM::interpolateParticleRotToGrid);
  int numMatls = d_sharedState->getNumMPMMatls();
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    ShellMaterial* smcm = dynamic_cast<ShellMaterial*>(cm);
    if (smcm) smcm->addComputesRequiresParticleRotToGrid(t, mpm_matl, patches);
  }
  sched->addTask(t, patches, matls);
}

///////////////////////////////////////////////////////////////////////////
//
// Actually interpolate normal rotation from particles to the grid
//
void 
ShellMPM::interpolateParticleRotToGrid(const ProcessorGroup*,
				       const PatchSubset* patches,
				       const MaterialSubset* ,
				       DataWarehouse* old_dw,
				       DataWarehouse* new_dw)
{
  int numMatls = d_sharedState->getNumMPMMatls();
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    ShellMaterial* smcm = dynamic_cast<ShellMaterial*>(cm);
    if (smcm) smcm->interpolateParticleRotToGrid(patches, mpm_matl, 
                                                 old_dw, new_dw);
  }
}

///////////////////////////////////////////////////////////////////////////
//
// Schedule computation of Internal Force
//
void 
ShellMPM::scheduleComputeInternalForce(SchedulerP& sched,
				       const PatchSet* patches,
				       const MaterialSet* matls)
{
  // Call the SerialMPM version first
  SerialMPM::scheduleComputeInternalForce(sched, patches, matls);

  // Add task for computing internal moment for the shell particles
  schedComputeRotInternalMoment(sched, patches, matls);
}

///////////////////////////////////////////////////////////////////////////
//
// Schedule computation of rotational internal moment
//
void 
ShellMPM::schedComputeRotInternalMoment(SchedulerP& sched,
				        const PatchSet* patches,
				        const MaterialSet* matls)
{
  Task* t = scinew Task("MPM::computeRotInternalMoment",
		        this, &ShellMPM::computeRotInternalMoment);
  int numMatls = d_sharedState->getNumMPMMatls();
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    ShellMaterial* smcm = dynamic_cast<ShellMaterial*>(cm);
    if (smcm) smcm->addComputesRequiresRotInternalMoment(t, mpm_matl, patches);
  }
  sched->addTask(t, patches, matls);
}

///////////////////////////////////////////////////////////////////////////
//
// Actually compute rotational Internal moment
//
void 
ShellMPM::computeRotInternalMoment(const ProcessorGroup*,
				   const PatchSubset* patches,
				   const MaterialSubset* ,
				   DataWarehouse* old_dw,
				   DataWarehouse* new_dw)
{
  // Loop over materials
  int numMatls = d_sharedState->getNumMPMMatls();
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    ShellMaterial* smcm = dynamic_cast<ShellMaterial*>(cm);
    if (smcm) smcm->computeRotInternalMoment(patches, mpm_matl, old_dw, new_dw);
  } 
}

///////////////////////////////////////////////////////////////////////////
//
// Schedule Calculation of acceleration
//
void 
ShellMPM::scheduleSolveEquationsMotion(SchedulerP& sched,
				       const PatchSet* patches,
				       const MaterialSet* matls)
{
  // Call SerialMPM version first
  SerialMPM::scheduleSolveEquationsMotion(sched, patches, matls);

  // Add a task for the rotational acceleration for the shell
  schedComputeRotAcceleration(sched, patches, matls);
}

///////////////////////////////////////////////////////////////////////////
//
// Schedule calculation of rotational acceleration of shell normal
//
void 
ShellMPM::schedComputeRotAcceleration(SchedulerP& sched,
				      const PatchSet* patches,
				      const MaterialSet* matls)
{
  Task* t = scinew Task("MPM::computeRotAcceleration",
		        this, &ShellMPM::computeRotAcceleration);

  int numMatls = d_sharedState->getNumMPMMatls();
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    ShellMaterial* smcm = dynamic_cast<ShellMaterial*>(cm);
    if (smcm) smcm->addComputesRequiresRotAcceleration(t, mpm_matl, patches);
  }

  sched->addTask(t, patches, matls);
}

///////////////////////////////////////////////////////////////////////////
//
// Actually calculate of rotational acceleration of shell normal
//
void 
ShellMPM::computeRotAcceleration(const ProcessorGroup*,
			         const PatchSubset* patches,
			         const MaterialSubset*,
			         DataWarehouse* old_dw,
			         DataWarehouse* new_dw)
{
  // Loop over materials
  int numMatls = d_sharedState->getNumMPMMatls();
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    ShellMaterial* smcm = dynamic_cast<ShellMaterial*>(cm);
    if (smcm) smcm->computeRotAcceleration(patches, mpm_matl, old_dw, new_dw);
  } 
}

///////////////////////////////////////////////////////////////////////////
//
// Schedule interpolation from grid to particles and update
//
void 
ShellMPM::scheduleInterpolateToParticlesAndUpdate(SchedulerP& sched,
						  const PatchSet* patches,
						  const MaterialSet* matls)
{
  // First Schedule update of the rate of shell normal rotation
  schedParticleNormalRotRateUpdate(sched, patches, matls);

  // Schedule update of the rest using SerialMPM
  SerialMPM::scheduleInterpolateToParticlesAndUpdate(sched, patches, matls);
}

///////////////////////////////////////////////////////////////////////////
//
// Schedule update of the particle normal rotation rate 
//
void 
ShellMPM::schedParticleNormalRotRateUpdate(SchedulerP& sched,
					   const PatchSet* patches,
					   const MaterialSet* matls)
{
  Task* t=scinew Task("ShellMPM::schedParticleNormalRotRateUpdate",
		      this, &ShellMPM::particleNormalRotRateUpdate);

  int numMatls = d_sharedState->getNumMPMMatls();
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    ShellMaterial* smcm = dynamic_cast<ShellMaterial*>(cm);
    if (smcm) smcm->addComputesRequiresRotRateUpdate(t, mpm_matl, patches);
  }
  sched->addTask(t, patches, matls);
}

///////////////////////////////////////////////////////////////////////////
//
// Actually update the particle normal rotation rate 
//
void 
ShellMPM::particleNormalRotRateUpdate(const ProcessorGroup*,
				      const PatchSubset* patches,
				      const MaterialSubset* ,
				      DataWarehouse* old_dw,
				      DataWarehouse* new_dw)
{
  // Loop over materials
  int numMatls = d_sharedState->getNumMPMMatls();
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    ShellMaterial* smcm = dynamic_cast<ShellMaterial*>(cm);
    if (smcm) 
      smcm->particleNormalRotRateUpdate(patches, mpm_matl, old_dw, new_dw);
  } 
}

