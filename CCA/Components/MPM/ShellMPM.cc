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
//
void 
ShellMPM::materialProblemSetup(const ProblemSpecP& prob_spec, 
			        SimulationStateP& sharedState,
                                MPMLabel* lb, int n8or27,
			        string integrator, bool haveLoadCurve,
			        bool doErosion)
{
  bool useShell = true;

  //Search for the MaterialProperties block and then get the MPM section
  ProblemSpecP mat_ps =  prob_spec->findBlock("MaterialProperties");
  ProblemSpecP mpm_mat_ps = mat_ps->findBlock("MPM");
  for (ProblemSpecP ps = mpm_mat_ps->findBlock("material"); ps != 0;
       ps = ps->findNextBlock("material") ) {
    MPMMaterial *mat = scinew MPMMaterial(ps, lb, d_8or27,integrator,
					  d_useLoadCurves, d_doErosion,
                                          useShell);
    //register as an MPM material
    sharedState->registerMPMMaterial(mat);
  }
}

///////////////////////////////////////////////////////////////////////////
//
// Schedule general initialization
//
void 
ShellMPM::scheduleInitialize(const LevelP& level,
			     SchedulerP& sched)
{
  Task* t = scinew Task("ShellMPM::actuallyInitialize",
			this, &ShellMPM::actuallyInitialize);

  MaterialSubset* zeroth_matl = scinew MaterialSubset();
  zeroth_matl->add(0);
  zeroth_matl->addReference();

  t->computes(lb->partCountLabel);
  t->computes(lb->pXLabel);
  t->computes(lb->pDispLabel);
  t->computes(lb->pMassLabel);
  t->computes(lb->pVolumeLabel);
  t->computes(lb->pTemperatureLabel);
  t->computes(lb->pVelocityLabel);
  t->computes(lb->pExternalForceLabel);
  t->computes(lb->pParticleIDLabel);
  t->computes(lb->pDeformationMeasureLabel);
  t->computes(lb->pStressLabel);
  t->computes(lb->pSizeLabel);
  t->computes(d_sharedState->get_delt_label());
  t->computes(lb->pCellNAPIDLabel,zeroth_matl);

  if (d_useLoadCurves) t->computes(lb->pLoadCurveIDLabel);
  if (d_accStrainEnergy) t->computes(lb->AccStrainEnergyLabel);
  if (d_artificialDampCoeff > 0.0) {
     t->computes(lb->pDampingRateLabel); t->computes(lb->pDampingCoeffLabel); 
  }

  // Schedule the initialization of shell variables
  t->computes(lb->pInitialThickTopLabel);
  t->computes(lb->pInitialThickBotLabel);
  t->computes(lb->pInitialNormalLabel);
  t->computes(lb->pThickTopLabel);
  t->computes(lb->pThickBotLabel);
  t->computes(lb->pNormalLabel);
  t->computes(lb->pNormalRotRateLabel);
  t->computes(lb->pRotationLabel);
  t->computes(lb->pDefGradTopLabel);
  t->computes(lb->pDefGradCenLabel);
  t->computes(lb->pDefGradBotLabel);
  t->computes(lb->pStressTopLabel);
  t->computes(lb->pStressCenLabel);
  t->computes(lb->pStressBotLabel);

  // Schedule the initialization of constitutive models
  int numMPM = d_sharedState->getNumMPMMatls();
  const PatchSet* patches = level->eachPatch();
  if (d_doErosion) {
    for(int m = 0; m < numMPM; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
      ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
      cm->addInitialComputesAndRequiresWithErosion(t, mpm_matl, patches,
                                                   d_erosionAlgorithm);
    }
  } else {
    for(int m = 0; m < numMPM; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
      ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
      cm->addInitialComputesAndRequires(t, mpm_matl, patches);
    }
  }

  sched->addTask(t, level->eachPatch(), d_sharedState->allMPMMaterials());

  schedulePrintParticleCount(level, sched);

  if (zeroth_matl->removeReference()) delete zeroth_matl; 
  if (d_useLoadCurves) scheduleInitializePressureBCs(level, sched);
}

///////////////////////////////////////////////////////////////////////////
//
// Actually initialize
//
void 
ShellMPM::actuallyInitialize(const ProcessorGroup* pg,
			     const PatchSubset* patches,
			     const MaterialSubset* matls,
			     DataWarehouse* old_dw,
			     DataWarehouse* new_dw)
{
  SerialMPM::actuallyInitialize(pg, patches, matls, old_dw, new_dw);


  // Initialization of the shell variables is done in the
  // particle creator only if the material is a shell material.
  // If the material is not a shell material, the shell variables
  // are registered and initialized here.
  int numMPM = d_sharedState->getNumMPMMatls();
  for(int m = 0; m < numMPM; m++){

    // Initialize shell-related particle variables for non-shell materials
    const MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    ShellMaterial* scm = dynamic_cast<ShellMaterial*>(cm);
    if (!scm) {
      for(int p=0;p<patches->size();p++){
        const Patch* patch = patches->get(p);
        initializeShellVariables(patch, mpm_matl, new_dw);
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////
//
// Initialize shell related variables for non shell materials
//
void 
ShellMPM::initializeShellVariables(const Patch* patch,
			           const MPMMaterial* matl,
			           DataWarehouse* new_dw)
{
  int index = matl->getDWIndex();
  ParticleSubset* pset = new_dw->getParticleSubset(index, patch);

  ParticleVariable<double>  pThickTop0, pThickBot0, pThickTop, pThickBot;
  ParticleVariable<Vector>  pNormal0, pNormal, pRotRate; 
  ParticleVariable<Matrix3> pRotation, pDefGradTop, pDefGradCen, pDefGradBot, 
                            pStressTop, pStressCen, pStressBot;
  new_dw->allocateAndPut(pThickTop0,  lb->pInitialThickTopLabel, pset);
  new_dw->allocateAndPut(pThickBot0,  lb->pInitialThickBotLabel, pset);
  new_dw->allocateAndPut(pNormal0,    lb->pInitialNormalLabel,   pset);
  new_dw->allocateAndPut(pThickTop,   lb->pThickTopLabel,        pset);
  new_dw->allocateAndPut(pThickBot,   lb->pThickBotLabel,        pset);
  new_dw->allocateAndPut(pNormal,     lb->pNormalLabel,          pset);
  new_dw->allocateAndPut(pRotRate,    lb->pNormalRotRateLabel,   pset);
  new_dw->allocateAndPut(pRotation,   lb->pRotationLabel,        pset);
  new_dw->allocateAndPut(pDefGradTop, lb->pDefGradTopLabel,      pset);
  new_dw->allocateAndPut(pDefGradCen, lb->pDefGradCenLabel,      pset);
  new_dw->allocateAndPut(pDefGradBot, lb->pDefGradBotLabel,      pset);
  new_dw->allocateAndPut(pStressTop,  lb->pStressTopLabel,       pset);
  new_dw->allocateAndPut(pStressCen,  lb->pStressCenLabel,       pset);
  new_dw->allocateAndPut(pStressBot,  lb->pStressBotLabel,       pset);

  Matrix3 One; One.Identity();
  Matrix3 Zero(0.0);
  ParticleSubset::iterator iter = pset->begin();
  for (; iter != pset->end(); iter++) {
    particleIndex pidx = *iter;
    pThickTop0[pidx]  = 0.0;
    pThickBot0[pidx]  = 0.0;
    pNormal0[pidx]    = Vector(0.0,1.0,0.0);
    pThickTop[pidx]   = 0.0;
    pThickBot[pidx]   = 0.0;
    pNormal[pidx]     = Vector(0.0,1.0,0.0);
    pRotRate[pidx]    = Vector(0.0,0.0,0.0);
    pRotation[pidx]   = One;
    pDefGradTop[pidx] = One;
    pDefGradCen[pidx] = One;
    pDefGradBot[pidx] = One;
    pStressTop[pidx]  = Zero;
    pStressCen[pidx]  = Zero;
    pStressBot[pidx]  = Zero;
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
  Ghost::GhostType  gan = Ghost::AroundNodes;
  t->requires(Task::OldDW,   lb->pMassLabel,             gan,NGP);
  t->requires(Task::OldDW,   lb->pNormalRotRateLabel,    gan,NGP);
  t->requires(Task::OldDW,   lb->pXLabel,                gan,NGP);
  if (d_8or27==27) 
    t->requires(Task::OldDW, lb->pSizeLabel,             gan,NGP);
  t->requires(Task::NewDW,   lb->gMassLabel,             gan,NGP);

  t->computes(lb->gNormalRotRateLabel);
  
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
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int numMatls = d_sharedState->getNumMPMMatls();
    Ghost::GhostType  gan = Ghost::AroundNodes;
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch, gan, NGP, 
                                                       lb->pXLabel);

      // Create arrays for the particle data
      constParticleVariable<double> pMass;
      constParticleVariable<Point>  pX;
      constParticleVariable<Vector> pRotRate, pSize;
      constNCVariable<double> gMass;
      old_dw->get(pX,             lb->pXLabel,             pset);
      old_dw->get(pMass,          lb->pMassLabel,          pset);
      old_dw->get(pRotRate,       lb->pNormalRotRateLabel, pset);
      if(d_8or27==27)
        old_dw->get(pSize,        lb->pSizeLabel,          pset);
      new_dw->get(gMass,          lb->gMassLabel, dwi,     patch, gan, NGP);

      // Create arrays for the grid data
      NCVariable<Vector> gRotRate;
      new_dw->allocateAndPut(gRotRate, lb->gNormalRotRateLabel, dwi,patch);
      gRotRate.initialize(Vector(0,0,0));

      // Interpolate particle data to Grid data.  Attempt to conserve 
      // angular momentum (I_grid*omega_grid =  S*I_particle*omega_particle).
      IntVector ni[MAX_BASIS];
      double S[MAX_BASIS];
      Vector pMom(0.0,0.0,0.0);
      for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); 
           iter++){
	particleIndex idx = *iter;

	// Get the node indices that surround the cell
	if(d_8or27==27) patch->findCellAndWeights27(pX[idx], ni, S, pSize[idx]);
	else            patch->findCellAndWeights(pX[idx], ni, S);

        // Calculate momentum
	pMom = pRotRate[idx]*pMass[idx];

	// Add each particles contribution to the local mass & velocity 
	// Must use the node indices
	for(int k = 0; k < d_8or27; k++) {
	  if(patch->containsNode(ni[k])) gRotRate[ni[k]] += pMom * S[k];
	}
      } // End of particle loop

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
        gRotRate[*iter] /= gMass[*iter];
      }
    }  // End loop over materials
  }  // End loop over patches
}

///////////////////////////////////////////////////////////////////////////
//
// Schedule computation of stress tensor
//
void 
ShellMPM::scheduleComputeStressTensor(SchedulerP& sched,
				      const PatchSet* patches,
				      const MaterialSet* matls)
{
  Task* t = scinew Task("ShellMPM::computeStressTensor",
			this, &ShellMPM::computeStressTensor);
  int numMatls = d_sharedState->getNumMPMMatls();
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    if (d_doErosion) 
      cm->addComputesAndRequiresWithErosion(t, mpm_matl, patches);
    else
      cm->addComputesAndRequires(t, mpm_matl, patches);
    const MaterialSubset* matlset = mpm_matl->thisMaterial();

    // Begin: Additional stuff for shell materials 
    t->computes(lb->pInitialThickTopLabel_preReloc, matlset);
    t->computes(lb->pInitialThickBotLabel_preReloc, matlset);
    t->computes(lb->pInitialNormalLabel_preReloc,   matlset);
    t->computes(lb->pThickTopLabel_preReloc,        matlset);
    t->computes(lb->pThickBotLabel_preReloc,        matlset);
    t->computes(lb->pNormalLabel_preReloc,          matlset);
    t->computes(lb->pRotationLabel_preReloc,        matlset);
    t->computes(lb->pDefGradTopLabel_preReloc,      matlset);
    t->computes(lb->pDefGradCenLabel_preReloc,      matlset);
    t->computes(lb->pDefGradBotLabel_preReloc,      matlset);
    t->computes(lb->pStressTopLabel_preReloc,       matlset);
    t->computes(lb->pStressCenLabel_preReloc,       matlset);
    t->computes(lb->pStressBotLabel_preReloc,       matlset);
    // End: Additional stuff for shell materials 
  }
  t->computes(d_sharedState->get_delt_label());
  t->computes(lb->StrainEnergyLabel);

  sched->addTask(t, patches, matls);

  // Compute the accumulated strain energy
  if(d_accStrainEnergy) 
    SerialMPM::scheduleComputeAccStrainEnergy(sched, patches, matls);

  // Compute artificial viscosity
  if(d_artificial_viscosity)
    SerialMPM::scheduleComputeArtificialViscosity(sched, patches, matls);
}

///////////////////////////////////////////////////////////////////////////
//
// Actually compute of stress tensor using the constitutive model
//
void ShellMPM::computeStressTensor(const ProcessorGroup* pg,
				    const PatchSubset* patches,
				    const MaterialSubset* matl,
				    DataWarehouse* old_dw,
				    DataWarehouse* new_dw)
{
  // First call the standard SerialMPM version
  SerialMPM::computeStressTensor(pg, patches, matl, old_dw, new_dw);

  // Additional stuff for shell formulation.  The shell "compute stress
  // tensor" does the correct thing for shells.  The following sets the
  // shell specific variables to default values for non-shell materials.
  Matrix3 One; One.Identity();
  Matrix3 Zero(0.0);

  // Loop thru patches
  for (int p = 0; p < patches->size(); ++p) {
    const Patch* patch = patches->get(p);

    // Loop thru materials
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
      ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
      ShellMaterial* smcm = dynamic_cast<ShellMaterial*>(cm);
    
      // If the material is a shell material do nothing
      // Shell material variables are evaluated inside constitutive model
      if (smcm) continue;  

      // Non shell material
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
      ParticleVariable<double>  pThickTop, pThickBot,  pThickTop0, pThickBot0;
      ParticleVariable<Vector>  pNormal, pNormal0;
      ParticleVariable<Matrix3> pRotation, 
                                pDefGradTop, pDefGradCen, pDefGradBot,
                                pStressTop, pStressCen, pStressBot;
      new_dw->allocateAndPut(pThickTop0,  lb->pInitialThickTopLabel_preReloc,   
                             pset);
      new_dw->allocateAndPut(pThickBot0,  lb->pInitialThickBotLabel_preReloc,   
                             pset);
      new_dw->allocateAndPut(pThickTop,   lb->pThickTopLabel_preReloc,   pset);
      new_dw->allocateAndPut(pThickBot,   lb->pThickBotLabel_preReloc,   pset);
      new_dw->allocateAndPut(pNormal,     lb->pNormalLabel_preReloc,     pset);
      new_dw->allocateAndPut(pNormal0,    lb->pInitialNormalLabel_preReloc,
                             pset);
      new_dw->allocateAndPut(pRotation,   lb->pRotationLabel_preReloc,   pset);
      new_dw->allocateAndPut(pDefGradTop, lb->pDefGradTopLabel_preReloc, pset);
      new_dw->allocateAndPut(pDefGradCen, lb->pDefGradCenLabel_preReloc, pset);
      new_dw->allocateAndPut(pDefGradBot, lb->pDefGradBotLabel_preReloc, pset);
      new_dw->allocateAndPut(pStressTop,  lb->pStressTopLabel_preReloc,  pset);
      new_dw->allocateAndPut(pStressCen,  lb->pStressCenLabel_preReloc,  pset);
      new_dw->allocateAndPut(pStressBot,  lb->pStressBotLabel_preReloc,  pset);
   
      // Initialize to default values
      ParticleSubset::iterator iter = pset->begin(); 
      for(; iter != pset->end(); iter++){
	particleIndex idx = *iter;
	pThickTop[idx] = 0.0;
	pThickBot[idx] = 0.0;
	pNormal[idx] = Vector(0.0, 1.0, 0.0);
	pThickTop0[idx] = 0.0;
	pThickBot0[idx] = 0.0;
	pNormal0[idx] = Vector(0.0, 1.0, 0.0);
	pRotation[idx] = One;
	pDefGradTop[idx] = One;
	pDefGradCen[idx] = One;
	pDefGradBot[idx] = One;
	pStressTop[idx] = Zero;
	pStressCen[idx] = Zero;
	pStressBot[idx] = Zero;
      }
    }        
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

  Ghost::GhostType  gan   = Ghost::AroundNodes;
  Ghost::GhostType  gnone = Ghost::None;
  t->requires(Task::OldDW,   lb->pMassLabel,                  gan,NGP);
  t->requires(Task::OldDW,   lb->pXLabel,                     gan,NGP);
  if(d_8or27==27) 
    t->requires(Task::OldDW, lb->pSizeLabel,                  gan,NGP);
  t->requires(Task::NewDW,   lb->pThickTopLabel_preReloc,     gan,NGP);
  t->requires(Task::NewDW,   lb->pThickBotLabel_preReloc,     gan,NGP);
  t->requires(Task::NewDW,   lb->pVolumeDeformedLabel,        gan,NGP);
  t->requires(Task::NewDW,   lb->pNormalLabel_preReloc,       gan,NGP);
  t->requires(Task::NewDW,   lb->pStressCenLabel_preReloc,    gan,NGP);
  t->requires(Task::NewDW,   lb->pStressTopLabel_preReloc,    gan,NGP);
  t->requires(Task::NewDW,   lb->pStressBotLabel_preReloc,    gan,NGP);

  t->computes(lb->gNormalRotMassLabel);
  t->computes(lb->gNormalRotMomentLabel);

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
  // Initialize constants
  Matrix3 Id; Id.Identity();
  Ghost::GhostType  gan   = Ghost::AroundNodes;

  // Loop over patches
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    // Loop over materials
    int numMPMMatls = d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch, gan, NGP,
						       lb->pXLabel);

      // Get stuff from datawarehouse
      constParticleVariable<double>  pMass, pVol, pThickTop, pThickBot;
      constParticleVariable<Point>   pX;
      constParticleVariable<Vector>  pSize, pNormal;
      constParticleVariable<Matrix3> pStressTop, pStressCen, pStressBot;

      old_dw->get(pMass,      lb->pMassLabel,                   pset);
      old_dw->get(pX,         lb->pXLabel,                      pset);
      if(d_8or27==27)
        old_dw->get(pSize,    lb->pSizeLabel,                   pset);
      new_dw->get(pThickTop,  lb->pThickTopLabel_preReloc,      pset);
      new_dw->get(pThickBot,  lb->pThickBotLabel_preReloc,      pset);
      new_dw->get(pVol,       lb->pVolumeDeformedLabel,         pset);
      new_dw->get(pNormal,    lb->pNormalLabel_preReloc,        pset);
      new_dw->get(pStressTop, lb->pStressTopLabel_preReloc,     pset);
      new_dw->get(pStressCen, lb->pStressCenLabel_preReloc,     pset);
      new_dw->get(pStressBot, lb->pStressBotLabel_preReloc,     pset);

      // Allocate stuff to be written to datawarehouse
      NCVariable<Vector> gRotMoment;
      NCVariable<double> gRotMass;
      new_dw->allocateAndPut(gRotMoment, lb->gNormalRotMomentLabel,  
                             dwi, patch);
      new_dw->allocateAndPut(gRotMass,   lb->gNormalRotMassLabel,  
                             dwi, patch);
      gRotMoment.initialize(Vector(0,0,0));
      gRotMass.initialize(0.0);

      // Loop thru particles
      IntVector ni[MAX_BASIS]; double S[MAX_BASIS]; Vector d_S[MAX_BASIS];
      for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); 
	   iter++){
	particleIndex idx = *iter;
  
	// Get the node indices that surround the cell and the derivatives
        // of the interpolation functions
	if(d_8or27==27)
	  patch->findCellAndWeightsAndShapeDerivatives27(pX[idx], ni, S, d_S,
							 pSize[idx]);
        else
	  patch->findCellAndWeightsAndShapeDerivatives(pX[idx], ni, S, d_S);

        // Calculate the in-surface identity tensor
        Matrix3 nn(pNormal[idx], pNormal[idx]);
        Matrix3 Is = Id - nn;

        // Calculate the average stress over the thickness of the shell
        // using the trapezoidal rule
        double ht = pThickTop[idx]; double hb = pThickBot[idx];
        double h = (ht+hb)*2.0+1.0e-100;
        Matrix3 avStress = ((pStressTop[idx]+pStressCen[idx])*ht +
                            (pStressBot[idx]+pStressCen[idx])*hb)/h;
        Vector nSig = pNormal[idx]*avStress*Is;

        // Calculate the average moment over the thickness of the shell
        // Loop thru nodes
        Matrix3 avMoment = (pStressTop[idx]*(ht*ht) - 
                            pStressBot[idx]*(hb*hb))*(0.5/h);
        avMoment = Is*avMoment*Is;

        // Calculate inertia term
        double pRotMass = pMass[idx]*h*h/12.0;

	for (int k = 0; k < d_8or27; k++){
	  if(patch->containsNode(ni[k])){
	    Vector gradS(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],
		       d_S[k].z()*oodx[2]);
            gradS = gradS*Is; // In-surface gradient of S
	    gRotMoment[ni[k]] -= ((gradS*avMoment)-(nSig*S[k]))*pVol[idx];
            gRotMass[ni[k]] += (pRotMass*S[k]);
	  }
	}
      }

      // Symmetry boundary conditions
      for(Patch::FaceType face = Patch::startFace;
	  face <= Patch::endFace; face=Patch::nextFace(face)){
	const BoundCondBase *sym_bcs;
	if (patch->getBCType(face) != Patch::None) continue;
	if (patch->getBCValues(dwi,"Symmetric",face)) {
	  IntVector offset(0,0,0);
	  fillFaceNormal(gRotMoment, patch, face, offset);
	}
      }
    }
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

  Ghost::GhostType  gnone = Ghost::None;
  t->requires(Task::NewDW, lb->gNormalRotMomentLabel, gnone);
  t->requires(Task::NewDW, lb->gNormalRotMassLabel,   gnone);

  t->computes(lb->gNormalRotAccLabel);
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
  // Constants
  Ghost::GhostType  gnone = Ghost::None;

  // Loop thru patches
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
 
    // Loop thru materials
    int numMPMMatls = d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      // Get stuff from datawarehouse
      constNCVariable<Vector> gRotMoment;
      constNCVariable<double> gRotMass;
      new_dw->get(gRotMoment, lb->gNormalRotMomentLabel, dwi, patch, gnone, 0);
      new_dw->get(gRotMass,   lb->gNormalRotMassLabel,   dwi, patch, gnone, 0);

      // Create variables for the results
      NCVariable<Vector> gRotAcc;
      new_dw->allocateAndPut(gRotAcc,  lb->gNormalRotAccLabel, dwi, patch);
      gRotAcc.initialize(Vector(0.,0.,0.));

      // Loop thru nodes
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
	IntVector c = *iter;
	gRotAcc[c] = gRotMoment[c]/(gRotMass[c]+1.0e-100);
      }
    }
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
  schedInterpolateRotToParticlesAndUpdate(sched, patches, matls);

  // Schedule update of the rest using SerialMPM
  SerialMPM::scheduleInterpolateToParticlesAndUpdate(sched, patches, matls);
}

///////////////////////////////////////////////////////////////////////////
//
// Schedule interpolation of rotation rate from grid to particles and update
//
void 
ShellMPM::schedInterpolateRotToParticlesAndUpdate(SchedulerP& sched,
					          const PatchSet* patches,
					          const MaterialSet* matls)
{
  Task* t=scinew Task("ShellMPM::interpolateRotToParticlesAndUpdate",
		      this, &ShellMPM::interpolateRotToParticlesAndUpdate);

  Ghost::GhostType gnone = Ghost::None;
  Ghost::GhostType gac   = Ghost::AroundCells;
  t->requires(Task::OldDW,   d_sharedState->get_delt_label());
  t->requires(Task::OldDW,   lb->pXLabel,                 gnone);
  if(d_8or27==27)
    t->requires(Task::OldDW, lb->pSizeLabel,              gnone);
  t->requires(Task::OldDW,   lb->pNormalRotRateLabel,     gnone);
  t->requires(Task::NewDW,   lb->pRotationLabel_preReloc, gnone);
  t->requires(Task::NewDW,   lb->gNormalRotAccLabel,      gac,NGN);

  t->computes(lb->pNormalRotRateLabel_preReloc);
  sched->addTask(t, patches, matls);
}

///////////////////////////////////////////////////////////////////////////
//
// Actually interpolate rotation rate from grid to particles and update
//
void 
ShellMPM::interpolateRotToParticlesAndUpdate(const ProcessorGroup*,
					     const PatchSubset* patches,
					     const MaterialSubset* ,
					     DataWarehouse* old_dw,
					     DataWarehouse* new_dw)
{
  Ghost::GhostType  gac = Ghost::AroundCells;

  // Loop thru patches
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label() );

    // Loop thru materials
    int numMPMMatls=d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      // Get the needed particle and grid variables
      constParticleVariable<Point>   pX;
      constParticleVariable<Vector>  pSize, pRotRate;
      constParticleVariable<Matrix3> pRotation;
      constNCVariable<Vector>        gRotAcc;
      old_dw->get(pX,        lb->pXLabel,                 pset);
      if(d_8or27==27)
	old_dw->get(pSize,   lb->pSizeLabel,              pset);
      old_dw->get(pRotRate,  lb->pNormalRotRateLabel,     pset);
      new_dw->get(pRotation, lb->pRotationLabel_preReloc, pset);
      new_dw->get(gRotAcc,   lb->gNormalRotAccLabel,      dwi,patch,gac,NGP);

      // Create the updated particle variables
      ParticleVariable<Vector> pRotRate_new;
      new_dw->allocateAndPut(pRotRate_new, lb->pNormalRotRateLabel_preReloc,   
                             pset);

      IntVector ni[MAX_BASIS];
      double S[MAX_BASIS];
      Vector d_S[MAX_BASIS];
      // Loop over particles
      for(ParticleSubset::iterator iter = pset->begin();
	  iter != pset->end(); iter++){
	particleIndex idx = *iter;

	// Get the node indices that surround the cell
	if(d_8or27==27)
	  patch->findCellAndWeightsAndShapeDerivatives27(pX[idx], ni, S, d_S,
							 pSize[idx]);
        else
	  patch->findCellAndWeightsAndShapeDerivatives(pX[idx], ni, S, d_S);

	// Accumulate the contribution from each surrounding vertex
	Vector acc(0.0,0.0,0.0);
	for (int k = 0; k < d_8or27; k++) acc += gRotAcc[ni[k]]*S[k];
        Vector rotRateTilde = acc*delT;

	// Update the particle's rotational velocity
	pRotRate_new[idx] = pRotRate[idx] + pRotation[idx]*rotRateTilde;
      }
    }
  }
}

