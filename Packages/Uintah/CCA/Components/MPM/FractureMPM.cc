/* Synced with version 1.141 of SerialMPM (4/2/2003)*/
#include <Packages/Uintah/CCA/Components/MPM/FractureMPM.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/CCA/Components/MPM/Contact/Contact.h>
#include <Packages/Uintah/CCA/Components/MPM/Contact/ContactFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/ThermalContact/ThermalContact.h>
#include <Packages/Uintah/CCA/Components/MPM/ThermalContact/ThermalContactFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/ForceBC.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/PressureBC.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/Short27.h> // for Farcture
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SoleVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMBoundCond.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/MinMax.h>
#include <Core/Util/NotFinished.h>
#include <Core/Util/DebugStream.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <fstream>
#include <sgi_stl_warnings_on.h>

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

FractureMPM::FractureMPM(const ProcessorGroup* myworld) :
  UintahParallelComponent(myworld)
{
  lb = scinew MPMLabel();
  d_nextOutputTime=0.;
  d_SMALL_NUM_MPM=1e-200;
  d_with_ice    = false;
  d_with_arches = false;
  contactModel        = 0;
  thermalContactModel = 0;
  crackMethod         = 0; // for Fracture
  d_8or27 = 8;
  d_min_part_mass = 3.e-15;
  d_max_vel = 3.e105;
  d_artificial_viscosity = false;
  NGP     = 1;
  NGN     = 1;

  d_artificialDampCoeff = 0.0;
  d_accStrainEnergy = false; // Flag for accumulating strain energy
  d_useLoadCurves = false; // Flag for using load curves
  d_doErosion = false; // Default is no erosion
  d_erosionAlgorithm = "none"; // Default algorithm is none
}

FractureMPM::~FractureMPM()
{
  delete lb;
  delete contactModel;
  delete thermalContactModel;
  MPMPhysicalBCFactory::clean();
}

void FractureMPM::problemSetup(const ProblemSpecP& prob_spec, GridP& /*grid*/,
			     SimulationStateP& sharedState)
{
   d_sharedState = sharedState;

   ProblemSpecP mpm_soln_ps = prob_spec->findBlock("MPM");

   if(mpm_soln_ps) {
     mpm_soln_ps->get("nodes8or27", d_8or27);
     mpm_soln_ps->get("minimum_particle_mass",    d_min_part_mass);
     mpm_soln_ps->get("maximum_particle_velocity",d_max_vel);
     mpm_soln_ps->get("artificial_damping_coeff", d_artificialDampCoeff);
     mpm_soln_ps->get("artificial_viscosity",     d_artificial_viscosity);
     mpm_soln_ps->get("accumulate_strain_energy", d_accStrainEnergy);
     mpm_soln_ps->get("use_load_curves", d_useLoadCurves);
     ProblemSpecP erosion_ps = mpm_soln_ps->findBlock("erosion");
     if (erosion_ps) {
       if (erosion_ps->getAttribute("algorithm", d_erosionAlgorithm)) {
          if (d_erosionAlgorithm == "none") d_doErosion = false;
          else d_doErosion = true;
       }
     }
   }

   if(d_8or27==8){
     NGP=1;
     NGN=1;
   } else if(d_8or27==MAX_BASIS){
     NGP=2;
     NGN=2;
   }

  //__________________________________
  // Grab time_integrator, default is explicit
   string integrator_type = "explicit";
   d_integrator = Explicit;
   if (mpm_soln_ps ) {
     mpm_soln_ps->get("time_integrator",integrator_type);
     if (integrator_type == "implicit"){
       d_integrator = Implicit;
     }
     if (integrator_type == "explicit") {
       d_integrator = Explicit;
     }
     if (integrator_type == "fracture") {
       d_integrator = Fracture;
     }
   }
   
   MPMPhysicalBCFactory::create(prob_spec);

   contactModel = ContactFactory::create(prob_spec,sharedState, lb, d_8or27);
   thermalContactModel =
		 ThermalContactFactory::create(prob_spec, sharedState, lb);

   // for Fracture 
   crackMethod = scinew Crack(prob_spec,sharedState,lb,d_8or27);
  
   ProblemSpecP p = prob_spec->findBlock("DataArchiver");
   if(!p->get("outputInterval", d_outputInterval))
      d_outputInterval = 1.0;
   crackMethod->d_outputInterval=d_outputInterval; 

   //Search for the MaterialProperties block and then get the MPM section

   ProblemSpecP mat_ps =  prob_spec->findBlock("MaterialProperties");

   ProblemSpecP mpm_mat_ps = mat_ps->findBlock("MPM");

   for (ProblemSpecP ps = mpm_mat_ps->findBlock("material"); ps != 0;
       ps = ps->findNextBlock("material") ) {
     MPMMaterial *mat = scinew MPMMaterial(ps, lb, d_8or27,integrator_type,
                                           d_useLoadCurves, d_doErosion);
     //register as an MPM material
     sharedState->registerMPMMaterial(mat);
   }
}

void FractureMPM::scheduleInitialize(const LevelP& level,
				   SchedulerP& sched)
{
  Task* t = scinew Task("FractureMPM::actuallyInitialize",
			this, &FractureMPM::actuallyInitialize);

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

  if (d_useLoadCurves) {
    // Computes the load curve ID associated with each particle
    t->computes(lb->pLoadCurveIDLabel);
  }

  if (d_accStrainEnergy) {
    // Computes accumulated strain energy
    t->computes(lb->AccStrainEnergyLabel);
  }

  // artificial damping coeff initialized to 0.0
  cout_doing << "Artificial Damping Coeff = " << d_artificialDampCoeff 
       << " 8 or 27 = " << d_8or27 << endl;
  if (d_artificialDampCoeff > 0.0) {
     t->computes(lb->pDampingRateLabel); 
     t->computes(lb->pDampingCoeffLabel); 
  }

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

  t = scinew Task("FractureMPM::printParticleCount",
		  this, &FractureMPM::printParticleCount);
  t->requires(Task::NewDW, lb->partCountLabel);
  sched->addTask(t, level->eachPatch(), d_sharedState->allMPMMaterials());
    
  // Descritize crack plane into triangular elements
  t = scinew Task("Crack:CrackDiscretization",
                   crackMethod, &Crack::CrackDiscretization);
  crackMethod->addComputesAndRequiresCrackDiscretization(t,
                  level->eachPatch(), d_sharedState->allMPMMaterials());
  sched->addTask(t, level->eachPatch(), d_sharedState->allMPMMaterials());

  // The task will have a reference to zeroth_matl
  if (zeroth_matl->removeReference())
    delete zeroth_matl; // shouln't happen, but...

  if (d_useLoadCurves) {
    // Schedule the initialization of pressure BCs per particle
    scheduleInitializePressureBCs(level, sched);
  }
}

void FractureMPM::scheduleInitializePressureBCs(const LevelP& level,
				              SchedulerP& sched)
{
  MaterialSubset* loadCurveIndex = scinew MaterialSubset();
  int nofPressureBCs = 0;
  for (int ii = 0; ii<(int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); ii++){
    string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
    if (bcs_type == "Pressure") loadCurveIndex->add(nofPressureBCs++);
  }
  //  cout << "nofPressureBCs: " << nofPressureBCs << "\n"; //for MPI
  if (nofPressureBCs > 0) {

    // Create a task that calculates the total number of particles
    // associated with each load curve.  
    Task* t = scinew Task("FractureMPM::countMaterialPointsPerLoadCurve",
		  this, &FractureMPM::countMaterialPointsPerLoadCurve);
    t->requires(Task::NewDW, lb->pLoadCurveIDLabel, Ghost::None);
    t->computes(lb->materialPointsPerLoadCurveLabel, loadCurveIndex,
                                                 Task::OutOfDomain);
    //t->computes(lb->materialPointsPerLoadCurveLabel);//for MPI
    sched->addTask(t, level->eachPatch(), d_sharedState->allMPMMaterials());

    // Create a task that calculates the force to be associated with
    // each particle based on the pressure BCs
    t = scinew Task("FractureMPM::initializePressureBC",
		  this, &FractureMPM::initializePressureBC);
    t->requires(Task::NewDW, lb->pXLabel, Ghost::None);
    t->requires(Task::NewDW, lb->pLoadCurveIDLabel, Ghost::None);
    t->requires(Task::NewDW, lb->materialPointsPerLoadCurveLabel);
    t->modifies(lb->pExternalForceLabel);
    sched->addTask(t, level->eachPatch(), d_sharedState->allMPMMaterials());
  }
}

void FractureMPM::scheduleComputeStableTimestep(const LevelP&,
					      SchedulerP&)
{
   // Nothing to do here - delt is computed as a by-product of the
   // consitutive model
}

void
FractureMPM::scheduleTimeAdvance(const LevelP & level,
			       SchedulerP   & sched,
			       int, int ) // AMR Parameters
{
  const PatchSet* patches = level->eachPatch();
  const MaterialSet* matls = d_sharedState->allMPMMaterials();

  scheduleParticleVelocityField(          sched, patches, matls); 
  scheduleInterpolateParticlesToGrid(     sched, patches, matls);
  scheduleComputeHeatExchange(            sched, patches, matls);
  scheduleAdjustCrackContactInterpolated( sched, patches, matls);
  scheduleExMomInterpolated(              sched, patches, matls);
  scheduleComputeStressTensor(            sched, patches, matls);
  scheduleComputeInternalForce(           sched, patches, matls);
  scheduleComputeInternalHeatRate(        sched, patches, matls);
  scheduleSolveEquationsMotion(           sched, patches, matls);
  scheduleSolveHeatEquations(             sched, patches, matls);
  scheduleIntegrateAcceleration(          sched, patches, matls);
  scheduleIntegrateTemperatureRate(       sched, patches, matls);
  scheduleAdjustCrackContactIntegrated(   sched, patches, matls);
  scheduleExMomIntegrated(                sched, patches, matls);
  scheduleSetGridBoundaryConditions(      sched, patches, matls);
  scheduleApplyExternalLoads(             sched, patches, matls);
  scheduleCalculateDampingRate(           sched, patches, matls);
  scheduleInterpolateToParticlesAndUpdate(sched, patches, matls);
  scheduleCalculateFractureParameters(    sched, patches, matls);
  scheduleDoCrackPropagation(             sched, patches, matls);
  scheduleMoveCracks(                     sched, patches, matls);

  sched->scheduleParticleRelocation(level, lb->pXLabel_preReloc,
				    lb->d_particleState_preReloc,
				    lb->pXLabel, lb->d_particleState,
				    lb->pParticleIDLabel, matls);
}

// determine if particles are above, below or in same side with nodes
void FractureMPM::scheduleParticleVelocityField(SchedulerP& sched,
                                                  const PatchSet* patches,
                                                  const MaterialSet* matls)
{
  Task* t = scinew Task("Crack::ParticleVelocityField", crackMethod,
                        &Crack::ParticleVelocityField);

  crackMethod->addComputesAndRequiresParticleVelocityField(t, patches, matls);
  sched->addTask(t, patches, matls);
}

void FractureMPM::scheduleInterpolateParticlesToGrid(SchedulerP& sched,
						   const PatchSet* patches,
						   const MaterialSet* matls)
{
  /* interpolateParticlesToGrid
   *   in(P.MASS, P.VELOCITY, P.NAT_X)
   *   operation(interpolate the P.MASS and P.VEL to the grid
   *             using P.NAT_X and some shape function evaluations)
   *   out(G.MASS, G.VELOCITY) */


  Task* t = scinew Task("FractureMPM::interpolateParticlesToGrid",
			this,&FractureMPM::interpolateParticlesToGrid);
  Ghost::GhostType  gan = Ghost::AroundNodes;
  t->requires(Task::OldDW, lb->pMassLabel,             gan,NGP);
  t->requires(Task::OldDW, lb->pVolumeLabel,           gan,NGP);
  
  t->requires(Task::OldDW, lb->pVelocityLabel,         gan,NGP);
  t->requires(Task::OldDW, lb->pXLabel,                gan,NGP);
  t->requires(Task::OldDW, lb->pExternalForceLabel,    gan,NGP);
  t->requires(Task::OldDW, lb->pTemperatureLabel,      gan,NGP);
  t->requires(Task::OldDW, lb->pSp_volLabel,           gan,NGP); 
  
  if(d_8or27==27){
   t->requires(Task::OldDW,lb->pSizeLabel,             gan,NGP);
  }
//t->requires(Task::OldDW, lb->pExternalHeatRateLabel, gan,NGP);

  if (d_doErosion) {
    t->requires(Task::OldDW, lb->pErosionLabel, gan, NGP);
  }

  t->computes(lb->gMassLabel);
  t->computes(lb->gMassLabel,        d_sharedState->getAllInOneMatl(),
	      Task::OutOfDomain);
  t->computes(lb->gTemperatureLabel, d_sharedState->getAllInOneMatl(),
	      Task::OutOfDomain);
  t->computes(lb->gVelocityLabel,    d_sharedState->getAllInOneMatl(),
	      Task::OutOfDomain);
  t->computes(lb->gSp_volLabel);
  t->computes(lb->gVolumeLabel);
  t->computes(lb->gVelocityLabel);
  t->computes(lb->gExternalForceLabel);
  t->computes(lb->gTemperatureLabel);
  t->computes(lb->gTemperatureNoBCLabel);
  t->computes(lb->gExternalHeatRateLabel);
  t->computes(lb->gNumNearParticlesLabel);
  t->computes(lb->TotalMassLabel);
 
  // for Fracture
  t->requires(Task::OldDW, lb->pDispLabel,  gan, NGP);
  t->requires(Task::NewDW, lb->pgCodeLabel, gan, NGP);

  t->computes(lb->GMassLabel);
  t->computes(lb->GSp_volLabel);
  t->computes(lb->GVolumeLabel);
  t->computes(lb->GVelocityLabel);
  t->computes(lb->GExternalForceLabel);
  t->computes(lb->GTemperatureLabel);
  t->computes(lb->GTemperatureNoBCLabel);
  t->computes(lb->GExternalHeatRateLabel);
  t->computes(lb->gDisplacementLabel);
  t->computes(lb->GDisplacementLabel);

  sched->addTask(t, patches, matls);
}

void FractureMPM::scheduleComputeHeatExchange(SchedulerP& sched,
					    const PatchSet* patches,
					    const MaterialSet* matls)
{
 /* computeHeatExchange
  *   in(G.MASS, G.TEMPERATURE, G.EXTERNAL_HEAT_RATE)
  *   operation(peform heat exchange which will cause each of
  *   velocity fields to exchange heat according to 
  *   the temperature differences)
  *   out(G.EXTERNAL_HEAT_RATE) */


  Task* t = scinew Task("ThermalContact::computeHeatExchange",
		        thermalContactModel,
		        &ThermalContact::computeHeatExchange);

  thermalContactModel->addComputesAndRequires(t, patches, matls);
  sched->addTask(t, patches, matls);
}

// ckeck crack contact and make adjustments on velocity field
void FractureMPM::scheduleAdjustCrackContactInterpolated(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)
{
  Task* t = scinew Task("Crack::AdjustCrackContactInterpolated",
                    crackMethod,&Crack::AdjustCrackContactInterpolated);

  crackMethod->addComputesAndRequiresAdjustCrackContactInterpolated(t,
                                                     patches, matls);

  sched->addTask(t, patches, matls);
}

void FractureMPM::scheduleExMomInterpolated(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls)
{
  Task* t = scinew Task("Contact::exMomInterpolated",
		    contactModel,
		    &Contact::exMomInterpolated);

  contactModel->addComputesAndRequiresInterpolated(t, patches, matls);
  sched->addTask(t, patches, matls);
}

void FractureMPM::scheduleComputeStressTensor(SchedulerP& sched,
					    const PatchSet* patches,
					    const MaterialSet* matls)
{
  int numMatls = d_sharedState->getNumMPMMatls();
  Task* t = scinew Task("FractureMPM::computeStressTensor",
		    this, &FractureMPM::computeStressTensor);
  if (d_doErosion) {
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
      ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
      cm->addComputesAndRequiresWithErosion(t, mpm_matl, patches);
    }
  } else {
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
      ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
      cm->addComputesAndRequires(t, mpm_matl, patches);
    }
  }

  t->computes(d_sharedState->get_delt_label());
  t->computes(lb->StrainEnergyLabel);

  sched->addTask(t, patches, matls);

  if (d_accStrainEnergy) {
    // Compute the accumulated strain energy
    t = scinew Task("FractureMPM::computeAccStrainEnergy",
		    this, &FractureMPM::computeAccStrainEnergy);
    t->requires(Task::OldDW, lb->AccStrainEnergyLabel);
    t->requires(Task::NewDW, lb->StrainEnergyLabel);
    t->computes(lb->AccStrainEnergyLabel);
    sched->addTask(t, patches, matls);
  }

  if(d_artificial_viscosity){
    scheduleComputeArtificialViscosity(   sched, patches, matls);
  }
}

void FractureMPM::scheduleComputeArtificialViscosity(SchedulerP& sched,
                                                   const PatchSet* patches,
                                                   const MaterialSet* matls)
{
  Task* t = scinew Task("FractureMPM::computeArtificialViscosity",
		    this, &FractureMPM::computeArtificialViscosity);

  Ghost::GhostType  gac = Ghost::AroundCells;
  t->requires(Task::OldDW, lb->pXLabel,                 Ghost::None);
  t->requires(Task::OldDW, lb->pMassLabel,              Ghost::None);
  t->requires(Task::NewDW, lb->pVolumeDeformedLabel,    Ghost::None);

  if(d_8or27==27){
    t->requires(Task::OldDW,lb->pSizeLabel,             Ghost::None);
  }

  t->requires(Task::NewDW,lb->gVelocityLabel, gac, NGN);
  t->requires(Task::NewDW,lb->GVelocityLabel, gac, NGN);  //additional field

  t->computes(lb->p_qLabel);

  sched->addTask(t, patches, matls);
}

void FractureMPM::scheduleComputeInternalForce(SchedulerP& sched,
					     const PatchSet* patches,
					     const MaterialSet* matls)
{
 /*
  * computeInternalForce
  *   in(P.CONMOD, P.NAT_X, P.VOLUME)
  *   operation(evaluate the divergence of the stress (stored in
  *   P.CONMOD) using P.NAT_X and the gradients of the
  *   shape functions)
  * out(G.F_INTERNAL) */

  Task* t = scinew Task("FractureMPM::computeInternalForce",
		    this, &FractureMPM::computeInternalForce);

  Ghost::GhostType  gan   = Ghost::AroundNodes;
  Ghost::GhostType  gnone = Ghost::None;
  t->requires(Task::NewDW,lb->gMassLabel, gnone);
  t->requires(Task::NewDW,lb->gMassLabel, d_sharedState->getAllInOneMatl(),
						Task::OutOfDomain, gnone);
  t->requires(Task::NewDW,lb->pStressLabel_preReloc,      gan,NGP);
  t->requires(Task::NewDW,lb->pVolumeDeformedLabel,       gan,NGP);
  t->requires(Task::OldDW,lb->pXLabel,                    gan,NGP);
  t->requires(Task::OldDW,lb->pMassLabel,                 gan,NGP);
  if(d_8or27==27){
   t->requires(Task::OldDW, lb->pSizeLabel,               gan,NGP);
  }

  // for Fracture
  t->requires(Task::NewDW,lb->pgCodeLabel,                gan,NGP);
  t->requires(Task::NewDW,lb->GMassLabel, gnone); 
  t->computes(lb->GInternalForceLabel);

  if(d_with_ice){
    t->requires(Task::NewDW, lb->pPressureLabel,          gan,NGP);
  }
  if(d_artificial_viscosity){
    t->requires(Task::NewDW, lb->p_qLabel,                gan,NGP);
  }

  if (d_doErosion) {
    t->requires(Task::OldDW, lb->pErosionLabel, gan, NGP);
  }

  t->computes(lb->gInternalForceLabel);
#ifdef INTEGRAL_TRACTION
  t->computes(lb->NTractionZMinusLabel);
#endif
  t->computes(lb->gStressForSavingLabel);
  t->computes(lb->gStressForSavingLabel, d_sharedState->getAllInOneMatl(),
	      Task::OutOfDomain);

  sched->addTask(t, patches, matls);
}

void FractureMPM::scheduleComputeInternalHeatRate(SchedulerP& sched,
						const PatchSet* patches,
						const MaterialSet* matls)
{  
  /*
   * computeInternalHeatRate
   * out(G.INTERNALHEATRATE) */

  Task* t = scinew Task("FractureMPM::computeInternalHeatRate",
			this, &FractureMPM::computeInternalHeatRate);

  Ghost::GhostType  gan = Ghost::AroundNodes;
  Ghost::GhostType  gac = Ghost::AroundCells;
  t->requires(Task::OldDW, lb->pXLabel,              gan, NGP);
  if(d_8or27==27){
   t->requires(Task::OldDW, lb->pSizeLabel,          gan, NGP);
  }
  t->requires(Task::NewDW, lb->pVolumeDeformedLabel, gan, NGP);
  t->requires(Task::NewDW, lb->gTemperatureLabel,    gac, 2*NGP);

  if (d_doErosion) {
    t->requires(Task::OldDW, lb->pErosionLabel, gan, NGP);
  }

  // for Fracture
  t->requires(Task::NewDW, lb->pgCodeLabel,          gan, NGP);
  t->requires(Task::NewDW, lb->GTemperatureLabel,    gac, 2*NGP);
  t->computes(lb->GInternalHeatRateLabel);

  t->computes(lb->gInternalHeatRateLabel);
  sched->addTask(t, patches, matls);
}

void FractureMPM::scheduleSolveEquationsMotion(SchedulerP& sched,
					     const PatchSet* patches,
					     const MaterialSet* matls)
{
  /* solveEquationsMotion
   *   in(G.MASS, G.F_INTERNAL)
   *   operation(acceleration = f/m)
   *   out(G.ACCELERATION) */

  Task* t = scinew Task("FractureMPM::solveEquationsMotion",
		    this, &FractureMPM::solveEquationsMotion);

  t->requires(Task::OldDW, d_sharedState->get_delt_label());

  t->requires(Task::NewDW, lb->gMassLabel,          Ghost::None);
  t->requires(Task::NewDW, lb->gInternalForceLabel, Ghost::None);
  t->requires(Task::NewDW, lb->gExternalForceLabel, Ghost::None);
//Uncomment  the next line to use damping
//t->requires(Task::NewDW, lb->gVelocityLabel,      Ghost::None);     

  if(d_with_ice){
    t->requires(Task::NewDW, lb->gradPAccNCLabel,   Ghost::None);
  }
  if(d_with_arches){
    t->requires(Task::NewDW, lb->AccArchesNCLabel,  Ghost::None);
  }

  // for Fracture
  t->requires(Task::NewDW, lb->GMassLabel,          Ghost::None);
  t->requires(Task::NewDW, lb->GInternalForceLabel, Ghost::None);
  t->requires(Task::NewDW, lb->GExternalForceLabel, Ghost::None);
  t->computes(lb->GAccelerationLabel);

  t->computes(lb->gAccelerationLabel);
  sched->addTask(t, patches, matls);
}

void FractureMPM::scheduleSolveHeatEquations(SchedulerP& sched,
					   const PatchSet* patches,
					   const MaterialSet* matls)
{
  /* solveHeatEquations
   *   in(G.MASS, G.INTERNALHEATRATE, G.EXTERNALHEATRATE)
   *   out(G.TEMPERATURERATE) */

  Task* t = scinew Task("FractureMPM::solveHeatEquations",
			    this, &FractureMPM::solveHeatEquations);

  const MaterialSubset* mss = matls->getUnion();

  Ghost::GhostType  gnone = Ghost::None;
  t->requires(Task::NewDW, lb->gMassLabel,                           gnone);
  t->requires(Task::NewDW, lb->gVolumeLabel,                         gnone);
  t->requires(Task::NewDW, lb->gExternalHeatRateLabel,               gnone);
  t->modifies(             lb->gInternalHeatRateLabel,               mss);
  t->requires(Task::NewDW, lb->gThermalContactHeatExchangeRateLabel, gnone);
		
  if(d_with_arches){
    t->requires(Task::NewDW, lb->heaTranSolid_NCLabel, gnone);
  }

  // for Fracture
  t->requires(Task::NewDW, lb->GMassLabel,                           gnone);
  t->requires(Task::NewDW, lb->GVolumeLabel,                         gnone);
  t->requires(Task::NewDW, lb->GExternalHeatRateLabel,               gnone);
  t->modifies(             lb->GInternalHeatRateLabel,               mss);
  t->requires(Task::NewDW, lb->GThermalContactHeatExchangeRateLabel, gnone);
  t->computes(lb->GTemperatureRateLabel);
		
  t->computes(lb->gTemperatureRateLabel);

  sched->addTask(t, patches, matls);
}

void FractureMPM::scheduleIntegrateAcceleration(SchedulerP& sched,
					      const PatchSet* patches,
					      const MaterialSet* matls)
{
  /* integrateAcceleration
   *   in(G.ACCELERATION, G.VELOCITY)
   *   operation(v* = v + a*dt)
   *   out(G.VELOCITY_STAR) */

  Task* t = scinew Task("FractureMPM::integrateAcceleration",
			    this, &FractureMPM::integrateAcceleration);

  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  t->requires(Task::NewDW, lb->gAccelerationLabel,      Ghost::None);
  t->requires(Task::NewDW, lb->gVelocityLabel,          Ghost::None);

  t->computes(lb->gVelocityStarLabel);

  // for Fracture
  t->requires(Task::NewDW, lb->GAccelerationLabel,      Ghost::None);
  t->requires(Task::NewDW, lb->GVelocityLabel,          Ghost::None);
  t->computes(lb->GVelocityStarLabel);

  sched->addTask(t, patches, matls);
}

void FractureMPM::scheduleIntegrateTemperatureRate(SchedulerP& sched,
						 const PatchSet* patches,
						 const MaterialSet* matls)
{
  /* integrateTemperatureRate
   *   in(G.TEMPERATURE, G.TEMPERATURERATE)
   *   operation(t* = t + t_rate * dt)
   *   out(G.TEMPERATURE_STAR) */

  Task* t = scinew Task("FractureMPM::integrateTemperatureRate",
		    this, &FractureMPM::integrateTemperatureRate);

  const MaterialSubset* mss = matls->getUnion();

  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  t->requires(Task::NewDW, lb->gTemperatureLabel,     Ghost::None);
  t->requires(Task::NewDW, lb->gTemperatureNoBCLabel, Ghost::None);
  t->modifies(             lb->gTemperatureRateLabel, mss);

  // for Fracture
  t->requires(Task::NewDW, lb->GTemperatureLabel,     Ghost::None);
  t->requires(Task::NewDW, lb->GTemperatureNoBCLabel, Ghost::None);
  t->modifies(             lb->GTemperatureRateLabel, mss);
  t->computes(lb->GTemperatureStarLabel);

  t->computes(lb->gTemperatureStarLabel);
		     
  sched->addTask(t, patches, matls);
}

// check crack contact and adjust nodal velocities and accelerations
void FractureMPM::scheduleAdjustCrackContactIntegrated(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)
{ 
  Task* t = scinew Task("Crack::AdjustCrackContactIntegrated",
                    crackMethod,&Crack::AdjustCrackContactIntegrated);
  
  crackMethod->addComputesAndRequiresAdjustCrackContactIntegrated(t,
                                                      patches, matls);
  
  sched->addTask(t, patches, matls);
}

void FractureMPM::scheduleExMomIntegrated(SchedulerP& sched,
					const PatchSet* patches,
					const MaterialSet* matls)
{
  /* exMomIntegrated
   *   in(G.MASS, G.VELOCITY_STAR, G.ACCELERATION)
   *   operation(peform operations which will cause each of
   *		  velocity fields to feel the influence of the
   *		  the others according to specific rules)
   *   out(G.VELOCITY_STAR, G.ACCELERATION) */

  Task* t = scinew Task("Contact::exMomIntegrated",
		   contactModel,
		   &Contact::exMomIntegrated);

  contactModel->addComputesAndRequiresIntegrated(t, patches, matls);
  sched->addTask(t, patches, matls);
}

void FractureMPM::scheduleSetGridBoundaryConditions(SchedulerP& sched,
						       const PatchSet* patches,
						       const MaterialSet* matls)

{
  Task* t=scinew Task("FractureMPM::setGridBoundaryConditions",
		    this, &FractureMPM::setGridBoundaryConditions);
                  
  const MaterialSubset* mss = matls->getUnion();
  t->requires(Task::OldDW, d_sharedState->get_delt_label() );
  
  t->modifies(             lb->gAccelerationLabel,     mss);
  t->modifies(             lb->gVelocityStarLabel,     mss);

  //for Fracture
  t->modifies(             lb->GAccelerationLabel,     mss);
  t->modifies(             lb->GVelocityStarLabel,     mss);

  sched->addTask(t, patches, matls);
}

void FractureMPM::scheduleApplyExternalLoads(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls)

{
 /*
  * applyExternalLoads
  *   in(p.externalForce, p.externalheatrate)
  *   out(p.externalForceNew, p.externalheatrateNew) */
  Task* t=scinew Task("FractureMPM::applyExternalLoads",
		    this, &FractureMPM::applyExternalLoads);
                  
  t->requires(Task::OldDW, lb->pExternalForceLabel,    Ghost::None);
  t->computes(             lb->pExtForceLabel_preReloc);
  if (d_useLoadCurves) {
    t->requires(Task::OldDW, lb->pXLabel, Ghost::None);
    t->requires(Task::OldDW, lb->pLoadCurveIDLabel,    Ghost::None);
    t->computes(             lb->pLoadCurveIDLabel_preReloc);
  }

//  t->computes(Task::OldDW, lb->pExternalHeatRateLabel_preReloc);

  sched->addTask(t, patches, matls);

}

void FractureMPM::scheduleCalculateDampingRate(SchedulerP& sched,
					     const PatchSet* patches,
					     const MaterialSet* matls)
{
 /*
  * calculateDampingRate
  *   in(G.VELOCITY_STAR, P.X, P.Size)
  *   operation(Calculate the interpolated particle velocity and
  *             sum the squares of the velocities over particles)
  *   out(sum_vartpe(dampingRate)) 
  */
  if (d_artificialDampCoeff > 0.0) {
    Task* t=scinew Task("FractureMPM::calculateDampingRate", this, 
			&FractureMPM::calculateDampingRate);
    t->requires(Task::NewDW, lb->gVelocityStarLabel, Ghost::AroundCells, NGN);
    t->requires(Task::OldDW, lb->pXLabel, Ghost::None);
    if(d_8or27==27) t->requires(Task::OldDW, lb->pSizeLabel, Ghost::None);

    // for Fracture
    t->requires(Task::NewDW, lb->GVelocityStarLabel, Ghost::AroundCells, NGN);
    t->requires(Task::NewDW, lb->pgCodeLabel, Ghost::None);

    t->computes(lb->pDampingRateLabel);
    sched->addTask(t, patches, matls);
  }
}

void FractureMPM::scheduleInterpolateToParticlesAndUpdate(SchedulerP& sched,
						       const PatchSet* patches,
						       const MaterialSet* matls)

{
 /*
  * interpolateToParticlesAndUpdate
  *   in(G.ACCELERATION, G.VELOCITY_STAR, P.NAT_X)
  *   operation(interpolate acceleration and v* to particles and
  *   integrate these to get new particle velocity and position)
  * out(P.VELOCITY, P.X, P.NAT_X) */

  Task* t=scinew Task("FractureMPM::interpolateToParticlesAndUpdate",
		    this, &FractureMPM::interpolateToParticlesAndUpdate);


  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  Ghost::GhostType  gac = Ghost::AroundCells;
  t->requires(Task::NewDW, lb->gAccelerationLabel,     gac,NGN);
  t->requires(Task::NewDW, lb->gVelocityStarLabel,     gac,NGN);
  t->requires(Task::NewDW, lb->gTemperatureRateLabel,  gac,NGN);
  t->requires(Task::NewDW, lb->gTemperatureLabel,      gac,NGN);
  t->requires(Task::NewDW, lb->gTemperatureNoBCLabel,  gac,NGN);
  t->requires(Task::NewDW, lb->gSp_volLabel,           gac,NGN);
  t->requires(Task::NewDW, lb->frictionalWorkLabel,    gac,NGN);
  t->requires(Task::OldDW, lb->pXLabel,                Ghost::None);
  t->requires(Task::OldDW, lb->pMassLabel,             Ghost::None);
  t->requires(Task::OldDW, lb->pParticleIDLabel,       Ghost::None);
  t->requires(Task::OldDW, lb->pTemperatureLabel,      Ghost::None);
  t->requires(Task::OldDW, lb->pSp_volLabel,           Ghost::None); 
  t->requires(Task::OldDW, lb->pVelocityLabel,         Ghost::None);
  t->requires(Task::OldDW, lb->pMassLabel,             Ghost::None);
  if(d_8or27==27){
   t->requires(Task::OldDW, lb->pSizeLabel,            Ghost::None);
  }
  t->requires(Task::NewDW, lb->pVolumeDeformedLabel,   Ghost::None);

  if (d_doErosion) {
    t->requires(Task::OldDW, lb->pErosionLabel, Ghost::None);
  }

  // for Fracture
  t->requires(Task::NewDW, lb->GAccelerationLabel,     gac,NGN);
  t->requires(Task::NewDW, lb->GVelocityStarLabel,     gac,NGN);
  t->requires(Task::NewDW, lb->GTemperatureRateLabel,  gac,NGN);
  t->requires(Task::NewDW, lb->GTemperatureLabel,      gac,NGN);
  t->requires(Task::NewDW, lb->GTemperatureNoBCLabel,  gac,NGN);
  t->requires(Task::NewDW, lb->GSp_volLabel,           gac,NGN);
  t->requires(Task::NewDW, lb->pgCodeLabel,            Ghost::None);
  t->requires(Task::OldDW, lb->pDispLabel,             Ghost::None);
  t->computes(lb->pDispLabel_preReloc);
  t->computes(lb->pKineticEnergyDensityLabel);

  // The dampingCoeff (alpha) is 0.0 for standard usage, otherwise
  // it is determined by the damping rate if the artificial damping
  // coefficient Q is greater than 0.0
  if (d_artificialDampCoeff > 0.0) {
    t->requires(Task::OldDW, lb->pDampingCoeffLabel);
    t->requires(Task::NewDW, lb->pDampingRateLabel);
    t->computes(lb->pDampingCoeffLabel);
  }

  if(d_with_ice){
    t->requires(Task::NewDW, lb->dTdt_NCLabel,         gac,NGN);
    t->requires(Task::NewDW, lb->gSp_vol_srcLabel,     gac,NGN);
    t->requires(Task::NewDW, lb->GSp_vol_srcLabel,     gac,NGN); // for Fracture
    t->requires(Task::NewDW, lb->massBurnFractionLabel,gac,NGN);
  }

  t->computes(lb->pVelocityLabel_preReloc);
  t->computes(lb->pXLabel_preReloc);
  t->computes(lb->pParticleIDLabel_preReloc);
  t->computes(lb->pTemperatureLabel_preReloc);
  t->computes(lb->pSp_volLabel_preReloc);
  t->computes(lb->pMassLabel_preReloc);
  t->computes(lb->pVolumeLabel_preReloc);
  if(d_8or27==27){
    t->computes(lb->pSizeLabel_preReloc);
  }

  t->computes(lb->KineticEnergyLabel);
  t->computes(lb->CenterOfMassPositionLabel);
  t->computes(lb->CenterOfMassVelocityLabel);
  sched->addTask(t, patches, matls);
}

void FractureMPM::scheduleCalculateFractureParameters(SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSet* matls)
{
  // Get nodal solutions
  Task* t = scinew Task("Crack::GetNodalSolutions", crackMethod,
                        &Crack::GetNodalSolutions);
  crackMethod->addComputesAndRequiresGetNodalSolutions(t,patches, matls);
  sched->addTask(t, patches, matls);

  // Compute fracture parameters (J, K,...)
  t = scinew Task("Crack::CalculateFractureParameters", crackMethod,
                        &Crack::CalculateFractureParameters);
  crackMethod->addComputesAndRequiresCalculateFractureParameters(t, 
                                                    patches, matls);
  sched->addTask(t, patches, matls);
}
// Do crack propgation
void FractureMPM::scheduleDoCrackPropagation(SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSet* matls)
{
  Task* t = scinew Task("Crack::PropagateCracks", crackMethod,
                         &Crack::PropagateCracks);
  crackMethod->addComputesAndRequiresPropagateCracks(t, patches, matls);
  sched->addTask(t, patches, matls);
}

void FractureMPM::scheduleMoveCracks(SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSet* matls)
{
  // Subset of crack points 
  Task* t = scinew Task("Crack::CrackPointSubset", crackMethod,
                        &Crack::CrackPointSubset);
  crackMethod->addComputesAndRequiresCrackPointSubset(t, patches, matls);
  sched->addTask(t, patches, matls);

  // Move crack points
  t = scinew Task("Crack::MoveCracks", crackMethod,
                        &Crack::MoveCracks);
  crackMethod->addComputesAndRequiresMoveCracks(t, patches, matls);
  sched->addTask(t, patches, matls);

  // Update crack extent and normals
  t = scinew Task("Crack::UpdateCrackExtentAndNormals", crackMethod,
                        &Crack::UpdateCrackExtentAndNormals);
  crackMethod->addComputesAndRequiresUpdateCrackExtentAndNormals(t,
                                                       patches, matls);
  sched->addTask(t, patches, matls);
}

void FractureMPM::printParticleCount(const ProcessorGroup* pg,
				   const PatchSubset*,
				   const MaterialSubset*,
				   DataWarehouse*,
				   DataWarehouse* new_dw)
{
  sumlong_vartype pcount;
  new_dw->get(pcount, lb->partCountLabel);
  
  if(pg->myrank() == 0){
    static bool printed=false;
    if(!printed){
      cerr << "Created " << (long) pcount << " total particles\n";
      printed=true;
    }
  }
}

void FractureMPM::computeAccStrainEnergy(const ProcessorGroup*,
				    const PatchSubset*,
				    const MaterialSubset*,
				    DataWarehouse* old_dw,
				    DataWarehouse* new_dw)
{
   // Get the totalStrainEnergy from the old datawarehouse
   max_vartype accStrainEnergy;
   old_dw->get(accStrainEnergy, lb->AccStrainEnergyLabel);

   // Get the incremental strain energy from the new datawarehouse
   sum_vartype incStrainEnergy;
   new_dw->get(incStrainEnergy, lb->StrainEnergyLabel);
  
   // Add the two a put into new dw
   double totalStrainEnergy = 
     (double) accStrainEnergy + (double) incStrainEnergy;
   new_dw->put(max_vartype(totalStrainEnergy), lb->AccStrainEnergyLabel);
}

// Calculate the number of material points per load curve
void FractureMPM::countMaterialPointsPerLoadCurve(const ProcessorGroup*,
						const PatchSubset* patches,
						const MaterialSubset*,
						DataWarehouse* ,
						DataWarehouse* new_dw)
{
  // Find the number of pressure BCs in the problem
  int nofPressureBCs = 0;
  for (int ii = 0; ii<(int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); ii++){
    string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
    if (bcs_type == "Pressure") {
      nofPressureBCs++;

      // Loop through the patches and count
      for(int p=0;p<patches->size();p++){
	const Patch* patch = patches->get(p);
	int numMPMMatls=d_sharedState->getNumMPMMatls();
	int numPts = 0;
	for(int m = 0; m < numMPMMatls; m++){
	  MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
	  int dwi = mpm_matl->getDWIndex();

	  ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
	  constParticleVariable<int> pLoadCurveID;
	  new_dw->get(pLoadCurveID, lb->pLoadCurveIDLabel, pset);

	  ParticleSubset::iterator iter = pset->begin();
	  for(;iter != pset->end(); iter++){
	    particleIndex idx = *iter;
	    if (pLoadCurveID[idx] == (nofPressureBCs)) ++numPts;
	  }
	} // matl loop
	new_dw->put(sumlong_vartype(numPts), lb->materialPointsPerLoadCurveLabel                                             , 0, nofPressureBCs-1);
        //new_dw->put(sumlong_vartype(numPts), lb->materialPointsPerLoadCurveLabel);// for MPI

      }  // patch loop
    }
  }
}

// Calculate the number of material points per load curve
void FractureMPM::initializePressureBC(const ProcessorGroup*,
				     const PatchSubset* patches,
				     const MaterialSubset*,
				     DataWarehouse* ,
				     DataWarehouse* new_dw)
{
  // Get the current time
  double time = 0.0;
  cout_doing << "Current Time (Initialize Pressure BC) = " << time << endl;

  // Calculate the force vector at each particle
  int nofPressureBCs = 0;
  for (int ii = 0; ii<(int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); ii++) {
    string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
    if (bcs_type == "Pressure") {

      // Get the material points per load curve
      sumlong_vartype numPart = 0;
      new_dw->get(numPart, lb->materialPointsPerLoadCurveLabel, 0, nofPressureBCs++);
      //new_dw->get(numPart, lb->materialPointsPerLoadCurveLabel); //for MPI

      // Save the material points per load curve in the PressureBC object
      PressureBC* pbc = dynamic_cast<PressureBC*>(MPMPhysicalBCFactory::mpmPhysicalBCs[ii]);
      pbc->numMaterialPoints(numPart);
      cout_doing << "    Load Curve = " << nofPressureBCs 
	         << " Num Particles = " << numPart << endl;

      // Calculate the force per particle at t = 0.0
      double forcePerPart = pbc->forcePerParticle(time);

      // Loop through the patches and calculate the force vector
      // at each particle
      for(int p=0;p<patches->size();p++){
	const Patch* patch = patches->get(p);
	int numMPMMatls=d_sharedState->getNumMPMMatls();
	for(int m = 0; m < numMPMMatls; m++){
	  MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
	  int dwi = mpm_matl->getDWIndex();

	  ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
          constParticleVariable<Point>  px;
          new_dw->get(px, lb->pXLabel,             pset);
	  constParticleVariable<int> pLoadCurveID;
	  new_dw->get(pLoadCurveID, lb->pLoadCurveIDLabel, pset);
	  ParticleVariable<Vector> pExternalForce;
          new_dw->getModifiable(pExternalForce, lb->pExternalForceLabel, pset);

	  ParticleSubset::iterator iter = pset->begin();
	  for(;iter != pset->end(); iter++){
	    particleIndex idx = *iter;
	    if (pLoadCurveID[idx] == nofPressureBCs) {
               pExternalForce[idx] = pbc->getForceVector(px[idx], forcePerPart);
            }
	  }
	} // matl loop
      }  // patch loop
    }
  }
}

void FractureMPM::actuallyInitialize(const ProcessorGroup*,
				   const PatchSubset* patches,
				   const MaterialSubset* matls,
				   DataWarehouse*,
				   DataWarehouse* new_dw)
{
  particleIndex totalParticles=0;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing actuallyInitialize on patch " << patch->getID()
	       <<"\t\t\t MPM"<< endl;

    CCVariable<short int> cellNAPID;
    new_dw->allocateAndPut(cellNAPID, lb->pCellNAPIDLabel, 0, patch);
    cellNAPID.initialize(0);

    for(int m=0;m<matls->size();m++){
      //cerrLock.lock();
      //NOT_FINISHED("not quite right - mapping of matls, use matls->get()");
      //cerrLock.unlock();
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      particleIndex numParticles = mpm_matl->countParticles(patch);
      totalParticles+=numParticles;

      mpm_matl->createParticles(numParticles, cellNAPID, patch, new_dw);

      mpm_matl->getConstitutiveModel()->initializeCMData(patch,
							 mpm_matl,
							 new_dw);
      if (d_doErosion) {
	int index = mpm_matl->getDWIndex();
	ParticleSubset* pset = new_dw->getParticleSubset(index, patch);
	ParticleVariable<double> pErosion;
	setParticleDefault(pErosion, lb->pErosionLabel, pset, new_dw, 1.0);
      }
    }
  }

  if (d_accStrainEnergy) {
    // Initialize the accumulated strain energy
    new_dw->put(max_vartype(0.0), lb->AccStrainEnergyLabel);
  }

  // Initialize the artificial damping ceofficient (alpha) to zero
  if (d_artificialDampCoeff > 0.0) {
    double alpha = 0.0;    
    double alphaDot = 0.0;    
    new_dw->put(max_vartype(alpha), lb->pDampingCoeffLabel);
    new_dw->put(sum_vartype(alphaDot), lb->pDampingRateLabel);
  }

  new_dw->put(sumlong_vartype(totalParticles), lb->partCountLabel);
}


void FractureMPM::actuallyComputeStableTimestep(const ProcessorGroup*,
					      const PatchSubset*,
					      const MaterialSubset*,
					      DataWarehouse*,
					      DataWarehouse*)
{
}

void FractureMPM::interpolateParticlesToGrid(const ProcessorGroup*,
					   const PatchSubset* patches,
					   const MaterialSubset* ,
					   DataWarehouse* old_dw,
					   DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing interpolateParticlesToGrid on patch " << patch->getID()
	       <<"\t\t MPM"<< endl;

    int numMatls = d_sharedState->getNumMPMMatls();

    NCVariable<double> gmassglobal,gtempglobal;
    NCVariable<Vector> gvelglobal;
    new_dw->allocateAndPut(gmassglobal, lb->gMassLabel,
			   d_sharedState->getAllInOneMatl()->get(0), patch);
    new_dw->allocateAndPut(gtempglobal, lb->gTemperatureLabel,
			   d_sharedState->getAllInOneMatl()->get(0), patch);
    new_dw->allocateAndPut(gvelglobal, lb->gVelocityLabel,
			   d_sharedState->getAllInOneMatl()->get(0), patch);
    gmassglobal.initialize(d_SMALL_NUM_MPM);
    gtempglobal.initialize(0.0);
    gvelglobal.initialize(Vector(0.0));

    Ghost::GhostType  gan = Ghost::AroundNodes;
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      // Create arrays for the particle data
      constParticleVariable<Point>  px;
      constParticleVariable<double> pmass, pvolume, pTemperature, pSp_vol;
      constParticleVariable<Vector> pvelocity, pexternalforce, psize, pdisp;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                 gan, NGP, lb->pXLabel);
      old_dw->get(pdisp,          lb->pDispLabel,          pset);
      old_dw->get(px,             lb->pXLabel,             pset);
      old_dw->get(pmass,          lb->pMassLabel,          pset);
      old_dw->get(pvolume,        lb->pVolumeLabel,        pset);
      old_dw->get(pSp_vol,        lb->pSp_volLabel,        pset);
      old_dw->get(pvelocity,      lb->pVelocityLabel,      pset);
      old_dw->get(pTemperature,   lb->pTemperatureLabel,   pset);
      old_dw->get(pexternalforce, lb->pExternalForceLabel, pset);
      if(d_8or27==27){
        old_dw->get(psize,        lb->pSizeLabel,          pset);
      }

      // for Fracture
      constParticleVariable<Short27> pgCode;
      new_dw->get(pgCode,lb->pgCodeLabel,pset);

      // Create arrays for the grid data
      NCVariable<double> gmass;
      NCVariable<double> gvolume;
      NCVariable<Vector> gvelocity;
      NCVariable<Vector> gexternalforce;
      NCVariable<double> gexternalheatrate;
      NCVariable<double> gTemperature;
      NCVariable<double> gSp_vol;
      NCVariable<double> gTemperatureNoBC;
      NCVariable<double> gnumnearparticles;

      new_dw->allocateAndPut(gmass,            lb->gMassLabel,       dwi,patch);   
      new_dw->allocateAndPut(gSp_vol,          lb->gSp_volLabel,     dwi,patch);
      new_dw->allocateAndPut(gvolume,          lb->gVolumeLabel,     dwi,patch);
      new_dw->allocateAndPut(gvelocity,        lb->gVelocityLabel,   dwi,patch);
      new_dw->allocateAndPut(gTemperature,     lb->gTemperatureLabel,dwi,patch);
      new_dw->allocateAndPut(gTemperatureNoBC, lb->gTemperatureNoBCLabel,
			     dwi,patch);
      new_dw->allocateAndPut(gexternalforce,   lb->gExternalForceLabel,
			     dwi,patch);
      new_dw->allocateAndPut(gexternalheatrate,lb->gExternalHeatRateLabel,
			     dwi,patch);
      new_dw->allocateAndPut(gnumnearparticles,lb->gNumNearParticlesLabel,
			     dwi,patch);

      gmass.initialize(d_SMALL_NUM_MPM);
      gvolume.initialize(0);
      gvelocity.initialize(Vector(0,0,0));
      gexternalforce.initialize(Vector(0,0,0));
      gTemperature.initialize(0);
      gTemperatureNoBC.initialize(0);
      gexternalheatrate.initialize(0);
      gnumnearparticles.initialize(0.);
      gSp_vol.initialize(0.);

      // for Fracture -- Create arrays for additional field
      NCVariable<double> Gmass;
      NCVariable<double> Gvolume;
      NCVariable<Vector> Gvelocity;
      NCVariable<Vector> Gexternalforce;
      NCVariable<double> Gexternalheatrate;
      NCVariable<double> GTemperature;
      NCVariable<double> GSp_vol;
      NCVariable<double> GTemperatureNoBC;
      NCVariable<Vector> gdisplacement;
      NCVariable<Vector> Gdisplacement;

      new_dw->allocateAndPut(Gmass,            lb->GMassLabel,       dwi,patch);
      new_dw->allocateAndPut(GSp_vol,          lb->GSp_volLabel,     dwi,patch);
      new_dw->allocateAndPut(Gvolume,          lb->GVolumeLabel,     dwi,patch);
      new_dw->allocateAndPut(Gvelocity,        lb->GVelocityLabel,   dwi,patch);
      new_dw->allocateAndPut(GTemperature,     lb->GTemperatureLabel,dwi,patch);
      new_dw->allocateAndPut(GTemperatureNoBC, lb->GTemperatureNoBCLabel,
                                                                     dwi,patch);
      new_dw->allocateAndPut(Gexternalforce,   lb->GExternalForceLabel,
                                                                     dwi,patch);
      new_dw->allocateAndPut(Gexternalheatrate,lb->GExternalHeatRateLabel,
                                                                     dwi,patch);
      new_dw->allocateAndPut(gdisplacement,   lb->gDisplacementLabel,dwi,patch);
      new_dw->allocateAndPut(Gdisplacement,   lb->GDisplacementLabel,dwi,patch);

      // initialization
      Gmass.initialize(d_SMALL_NUM_MPM);
      Gvolume.initialize(0);
      Gvelocity.initialize(Vector(0,0,0));
      GTemperature.initialize(0);
      GTemperatureNoBC.initialize(0);
      Gexternalforce.initialize(Vector(0,0,0));
      Gexternalheatrate.initialize(0);
      gdisplacement.initialize(Vector(0,0,0));
      Gdisplacement.initialize(Vector(0,0,0));
      GSp_vol.initialize(0.);

      // Get the particle erosion information
      constParticleVariable<double> pErosion;
      if (d_doErosion) {
        old_dw->get(pErosion, lb->pErosionLabel, pset);
      } else {
        setParticleDefaultWithTemp(pErosion, pset, new_dw, 1.0);
      }

      // Interpolate particle data to Grid data.
      // This currently consists of the particle velocity and mass
      // Need to compute the lumped global mass matrix and velocity
      // Vector from the individual mass matrix and velocity vector
      // GridMass * GridVelocity =  S^T*M_D*ParticleVelocity
      
      double totalmass = 0;
      Vector total_mom(0.0,0.0,0.0);

      IntVector ni[MAX_BASIS];
      double S[MAX_BASIS];
      Vector pmom;

      for (ParticleSubset::iterator iter = pset->begin();
                                    iter != pset->end(); 
                                    iter++){
	particleIndex idx = *iter;

	// Get the node indices that surround the cell
	if(d_8or27==8){
	  patch->findCellAndWeights(px[idx], ni, S);
	}
	else if(d_8or27==27){
	  patch->findCellAndWeights27(px[idx], ni, S, psize[idx]);
	}

	pmom = pvelocity[idx]*pmass[idx];
	total_mom += pvelocity[idx]*pmass[idx];

	// Add each particles contribution to the local mass & velocity 
	// Must use the node indices
	for(int k = 0; k < d_8or27; k++) {
	  if(patch->containsNode(ni[k])) {
	    S[k] *= pErosion[idx];
            if(pgCode[idx][k]==1) {   // for primary field
              gdisplacement[ni[k]]  += pdisp[idx] * pmass[idx]        * S[k];
              gmass[ni[k]]          += pmass[idx]                     * S[k];
              gvelocity[ni[k]]      += pvelocity[idx] * pmass[idx]    * S[k];
              gvolume[ni[k]]        += pvolume[idx]                   * S[k];
              gexternalforce[ni[k]] += pexternalforce[idx]            * S[k];
              gTemperature[ni[k]]   += pTemperature[idx] * pmass[idx] * S[k];
              //gexternalheatrate[ni[k]] += pexternalheatrate[idx]      * S[k];
              gnumnearparticles[ni[k]] += 1.0;
              gSp_vol[ni[k]]        += pSp_vol[idx]  * pmass[idx]     *S[k];
            }
            else if(pgCode[idx][k]==2) {  // for additional field
              Gdisplacement[ni[k]]  += pdisp[idx] * pmass[idx]        * S[k];
              Gmass[ni[k]]          += pmass[idx]                     * S[k];
              Gvolume[ni[k]]        += pvolume[idx]                   * S[k];
              Gexternalforce[ni[k]] += pexternalforce[idx]            * S[k];
              Gvelocity[ni[k]]      += pvelocity[idx] * pmass[idx]    * S[k];
              GTemperature[ni[k]]   += pTemperature[idx] * pmass[idx] * S[k];
              //Gexternalheatrate[ni[k]] += pexternalheatrate[idx]      * S[k];
              GSp_vol[ni[k]]        += pSp_vol[idx]  * pmass[idx]     *S[k];
            }
	  }
	}
      } // End of particle loop

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
        IntVector c = *iter; 
        totalmass          += (gmass[c]+Gmass[c]);
        gmassglobal[c]     += (gmass[c]+Gmass[c]);
        gvelglobal[c]      += (gvelocity[c]+Gvelocity[c]);
        gtempglobal[c]     += (gTemperature[c]+GTemperature[c]);

        // for primary field
        gvelocity[c]       /= gmass[c];
        gdisplacement[c]   /= gmass[c];
        gSp_vol[c]         /= gmass[c];
        gTemperature[c]    /= gmass[c];
        gTemperatureNoBC[c] = gTemperature[c];

        // for additional field 
        Gvelocity[c]       /= Gmass[c];
        Gdisplacement[c]   /= Gmass[c];
        GSp_vol[c]         /= Gmass[c];
        GTemperature[c]    /= Gmass[c];
        GTemperatureNoBC[c] = GTemperature[c];
      }

      // Apply grid boundary conditions to the velocity before storing the data

      MPMBoundCond bc;
      bc.setBoundaryCondition(patch,dwi,"Velocity",gvelocity);
      bc.setBoundaryCondition(patch,dwi,"Velocity",Gvelocity);
      bc.setBoundaryCondition(patch,dwi,"Symmetric",gvelocity);
      bc.setBoundaryCondition(patch,dwi,"Symmetric",Gvelocity);
      bc.setBoundaryCondition(patch,dwi,"Temperature",gTemperature);
      bc.setBoundaryCondition(patch,dwi,"Temperature",GTemperature);

      new_dw->put(sum_vartype(totalmass), lb->TotalMassLabel);

    }  // End loop over materials

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
      IntVector c = *iter;
      gtempglobal[c] /= gmassglobal[c];
      gvelglobal[c] /= gmassglobal[c];
    }
  }  // End loop over patches
}

void FractureMPM::computeStressTensor(const ProcessorGroup*,
				    const PatchSubset* patches,
				    const MaterialSubset* ,
				    DataWarehouse* old_dw,
				    DataWarehouse* new_dw)
{

  cout_doing <<"Doing computeStressTensor " <<"\t\t\t\t MPM"<< endl;

  if (d_doErosion) {
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
      ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
      cm->computeStressTensorWithErosion(patches, mpm_matl, old_dw, new_dw);
    }
  } else {
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
      ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
      cm->computeStressTensor(patches, mpm_matl, old_dw, new_dw);
    }
  }
}

void FractureMPM::computeArtificialViscosity(const ProcessorGroup*,
                                           const PatchSubset* patches,
                                           const MaterialSubset* ,
                                           DataWarehouse* old_dw,
                                           DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing computeArtificialViscosity on patch " << patch->getID()
	       <<"\t\t MPM"<< endl;

    Ghost::GhostType  gac   = Ghost::AroundCells;

    int numMatls = d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      constNCVariable<Vector> gvelocity;
      ParticleVariable<double> p_q;
      constParticleVariable<Vector> psize;
      constParticleVariable<Point> px;
      constParticleVariable<double> pmass,pvol_def;
      new_dw->get(gvelocity, lb->gVelocityLabel, dwi,patch, gac, NGN);
      old_dw->get(px,        lb->pXLabel,                      pset);
      old_dw->get(pmass,     lb->pMassLabel,                   pset);
      new_dw->get(pvol_def,  lb->pVolumeDeformedLabel,         pset);
      new_dw->allocateAndPut(p_q,    lb->p_qLabel,             pset);
      if(d_8or27==27){
        old_dw->get(psize,   lb->pSizeLabel,                   pset);
      }

      // for Fracture
      constNCVariable<Vector> Gvelocity;
      new_dw->get(Gvelocity, lb->GVelocityLabel, dwi,patch, gac, NGN);
      constParticleVariable<Short27> pgCode;
      new_dw->get(pgCode,    lb->pgCodeLabel,                  pset);

      Matrix3 velGrad;
      Vector dx = patch->dCell();
      double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};
      double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;

      double K = 1./mpm_matl->getConstitutiveModel()->getCompressibility();
      double c_dil;

      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
	particleIndex idx = *iter;

	// Get the node indices that surround the cell
	IntVector ni[MAX_BASIS];
	Vector d_S[MAX_BASIS];

	if(d_8or27==8){
          patch->findCellAndShapeDerivatives(px[idx], ni, d_S);
	}
	else if(d_8or27==27){
	  patch->findCellAndShapeDerivatives27(px[idx], ni, d_S,psize[idx]);
	}

        // get particle's velocity gradients 
        Vector gvel;
	velGrad.set(0.0);
	for(int k = 0; k < d_8or27; k++) {
          if(pgCode[idx][k]==1) gvel = gvelocity[ni[k]];
          if(pgCode[idx][k]==2) gvel = Gvelocity[ni[k]];
	  for(int j = 0; j<3; j++){
            double d_SXoodx = d_S[k][j] * oodx[j];
            for(int i = 0; i<3; i++) {
	      velGrad(i+1,j+1) += gvel[i] * d_SXoodx;
            }
	  }
	}

	Matrix3 D = (velGrad + velGrad.Transpose())*.5;

	double DTrace = D.Trace();
	p_q[idx] = 0.0;
	if(DTrace<0.){
	  c_dil = sqrt(K*pvol_def[idx]/pmass[idx]);
	  p_q[idx] = (.2*fabs(c_dil*DTrace*dx_ave) +
                      2.*(DTrace*DTrace*dx_ave*dx_ave))*
	    (pmass[idx]/pvol_def[idx]);
	}

      }
    }
  }

}

void FractureMPM::computeInternalForce(const ProcessorGroup*,
				     const PatchSubset* patches,
				     const MaterialSubset* ,
				     DataWarehouse* old_dw,
				     DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing computeInternalForce on patch " << patch->getID()
	       <<"\t\t\t MPM"<< endl;

    Vector dx = patch->dCell();
    double oodx[3];
    oodx[0] = 1.0/dx.x();
    oodx[1] = 1.0/dx.y();
    oodx[2] = 1.0/dx.z();
    Matrix3 Id;
    Id.Identity();

    int numMPMMatls = d_sharedState->getNumMPMMatls();

#ifdef INTEGRAL_TRACTION
    double integralTraction = 0.;
    double integralArea = 0.;
#endif

    NCVariable<Matrix3>       gstressglobal;
    constNCVariable<double>   gmassglobal;
    new_dw->get(gmassglobal,  lb->gMassLabel,
		d_sharedState->getAllInOneMatl()->get(0), patch, Ghost::None,0);
    new_dw->allocateAndPut(gstressglobal, lb->gStressForSavingLabel, 
			   d_sharedState->getAllInOneMatl()->get(0), patch);
    gstressglobal.initialize(Matrix3(0.));

    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      // Create arrays for the particle position, volume
      // and the constitutive model
      constParticleVariable<Point>   px;
      constParticleVariable<double>  pvol, pmass;
      constParticleVariable<double>  p_pressure;
      constParticleVariable<double>  p_q;
      constParticleVariable<Matrix3> pstress;
      constParticleVariable<Vector>  psize;
      NCVariable<Vector>             internalforce;
      NCVariable<Matrix3>            gstress;
      constNCVariable<double>        gmass;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
						       Ghost::AroundNodes, NGP,
						       lb->pXLabel);

      old_dw->get(px,      lb->pXLabel,                      pset);
      old_dw->get(pmass,   lb->pMassLabel,                   pset);
      new_dw->get(pvol,    lb->pVolumeDeformedLabel,         pset);
      new_dw->get(pstress, lb->pStressLabel_preReloc,        pset);
      if(d_8or27==27){
        old_dw->get(psize, lb->pSizeLabel,                   pset);
      }
      new_dw->get(gmass,   lb->gMassLabel, dwi, patch, Ghost::None, 0);

      new_dw->allocateAndPut(gstress,      lb->gStressForSavingLabel,dwi,patch);
      new_dw->allocateAndPut(internalforce,lb->gInternalForceLabel,  dwi,patch);
      gstress.initialize(Matrix3(0.));  
      internalforce.initialize(Vector(0,0,0));
 
      // for Fracture
      constParticleVariable<Short27> pgCode;
      new_dw->get(pgCode, lb->pgCodeLabel, pset);
      constNCVariable<double> Gmass;
      new_dw->get(Gmass, lb->GMassLabel, dwi, patch, Ghost::None, 0);
      NCVariable<Vector> Ginternalforce;
      new_dw->allocateAndPut(Ginternalforce,lb->GInternalForceLabel, dwi,patch);
      Ginternalforce.initialize(Vector(0,0,0));

      if(d_with_ice){
        new_dw->get(p_pressure,lb->pPressureLabel, pset);
      }
      else {
	ParticleVariable<double>  p_pressure_create;
	new_dw->allocateTemporary(p_pressure_create,  pset);
	for(ParticleSubset::iterator it = pset->begin();it != pset->end();it++){
	  p_pressure_create[*it]=0.0;
	}
	p_pressure = p_pressure_create; // reference created data
      }

      if(d_artificial_viscosity){
        new_dw->get(p_q,lb->p_qLabel, pset);
      }
      else {
	ParticleVariable<double>  p_q_create;
	new_dw->allocateTemporary(p_q_create,  pset);
	for(ParticleSubset::iterator it = pset->begin();it != pset->end();it++){
	  p_q_create[*it]=0.0;
	}
	p_q = p_q_create; // reference created data
      }

      // Get the particle erosion information
      constParticleVariable<double> pErosion;
      if (d_doErosion) {
        old_dw->get(pErosion, lb->pErosionLabel, pset);
      } else {
        setParticleDefaultWithTemp(pErosion, pset, new_dw, 1.0);
      }

      IntVector ni[MAX_BASIS];
      double S[MAX_BASIS];
      Vector d_S[MAX_BASIS];
      Matrix3 stressmass;
      Matrix3 stresspress;

      for (ParticleSubset::iterator iter = pset->begin();
                                    iter != pset->end(); 
                                    iter++){
	particleIndex idx = *iter;
  
	// Get the node indices that surround the cell
	if(d_8or27==8){
	  patch->findCellAndWeightsAndShapeDerivatives(px[idx], ni, S,  d_S);
	}
	else if(d_8or27==27){
	  patch->findCellAndWeightsAndShapeDerivatives27(px[idx], ni, S,d_S,
							 psize[idx]);
	}

	stressmass  = pstress[idx]*pmass[idx];
	stresspress = pstress[idx] + Id*p_pressure[idx] - Id*p_q[idx];

	for (int k = 0; k < d_8or27; k++){
	  if(patch->containsNode(ni[k])){
	    Vector div(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],
		       d_S[k].z()*oodx[2]);
	    div *= pErosion[idx];
            if(pgCode[idx][k]==1) {
              internalforce[ni[k]] -=
                  (div * (pstress[idx] + Id*p_pressure[idx]-Id*p_q[idx]) * pvol[idx]);
            }
            else if(pgCode[idx][k]==2) {
              Ginternalforce[ni[k]] -=
                  (div * (pstress[idx] + Id*p_pressure[idx]-Id*p_q[idx]) * pvol[idx]);
            }
            gstress[ni[k]] += pstress[idx] * pmass[idx] * S[k];
	  }
	}
      }

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
        IntVector c = *iter;
        gstressglobal[c] += gstress[c];
        gstress[c] /= (gmass[c]+Gmass[c]); //add in addtional field
      }

#ifdef INTEGRAL_TRACTION
      IntVector low = patch-> getInteriorNodeLowIndex();
      IntVector hi  = patch-> getInteriorNodeHighIndex();
      for(Patch::FaceType face = Patch::startFace;
	  face <= Patch::endFace; face=Patch::nextFace(face)){

        // I assume we have the patch variable
        // Check if the face is on the boundary
	Patch::BCType bc_type = patch->getBCType(face);
	if (bc_type == Patch::None) {
	  // We are on the boundary, i.e. not on an interior patch
	  // boundary, so do the traction accumulation . . .
	  if(face==Patch::zminus){
	    int K=low.z();
	    for (int i = low.x(); i<hi.x(); i++) {
              for (int j = low.y(); j<hi.y(); j++) {
		integralTraction +=
		  gstress[IntVector(i,j,K)](3,3)*dx.x()*dx.y();
		if(fabs(gstress[IntVector(i,j,K)](3,3)) > 1.e-12){
		  integralArea+=dx.x()*dx.y();
                }
              }
	    }
	  }
        } // end of if (bc_type == Patch::None)
      }
#endif
      //__________________________________
      // Set internal force = 0 on symmetric boundaries

      MPMBoundCond bc;
      bc.setBoundaryCondition(patch,dwi,"Symmetric",internalforce);
      bc.setBoundaryCondition(patch,dwi,"Symmetric",Ginternalforce);

#ifdef KUMAR
      internalforce.initialize(Vector(0,0,0));
#endif
    }
#ifdef INTEGRAL_TRACTION
    if(integralArea > 0.){
      integralTraction=integralTraction/integralArea;
    }
    else{
      integralTraction=0.;
    }
    new_dw->put(sum_vartype(integralTraction), lb->NTractionZMinusLabel);
#endif

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
      IntVector c = *iter;
      gstressglobal[c] /= gmassglobal[c];
    }
  }
}

void FractureMPM::computeInternalHeatRate(const ProcessorGroup*,
				        const PatchSubset* patches,
					const MaterialSubset* ,
				        DataWarehouse* old_dw,
				        DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing computeInternalHeatRate on patch " << patch->getID()
	       <<"\t\t MPM"<< endl;

    Vector dx = patch->dCell();
    double oodx[3];
    oodx[0] = 1.0/dx.x();
    oodx[1] = 1.0/dx.y();
    oodx[2] = 1.0/dx.z();

    Ghost::GhostType  gac = Ghost::AroundCells;
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      double thermalConductivity = mpm_matl->getThermalConductivity();
      
      constParticleVariable<Point>  px;
      constParticleVariable<double> pvol;
      constParticleVariable<Vector> psize;
      ParticleVariable<Vector>      pTemperatureGradient;
      constNCVariable<double>       gTemperature;
      NCVariable<double>            internalHeatRate;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
						       Ghost::AroundNodes, NGP,
						       lb->pXLabel);

      old_dw->get(px,           lb->pXLabel,              pset);
      new_dw->get(pvol,         lb->pVolumeDeformedLabel, pset);
      if(d_8or27==27){
        old_dw->get(psize,      lb->pSizeLabel,           pset);
      }

      // Get the particle erosion information
      constParticleVariable<double> pErosion;
      if (d_doErosion) {
        old_dw->get(pErosion, lb->pErosionLabel, pset);
      } else {
        setParticleDefaultWithTemp(pErosion, pset, new_dw, 1.0);
      }

      new_dw->get(gTemperature, lb->gTemperatureLabel,   dwi, patch, gac,2*NGN);
      new_dw->allocateAndPut(internalHeatRate, lb->gInternalHeatRateLabel,
			     dwi, patch);
      new_dw->allocateTemporary(pTemperatureGradient, pset);
  
      internalHeatRate.initialize(0.);

      // for Fracture
      constParticleVariable<Short27> pgCode;
      constNCVariable<double> GTemperature;
      NCVariable<double> GinternalHeatRate;

      new_dw->get(pgCode, lb->pgCodeLabel, pset);
      new_dw->get(GTemperature, lb->GTemperatureLabel, dwi, patch, gac, 2*NGN);
      new_dw->allocateAndPut(GinternalHeatRate, lb->GInternalHeatRateLabel,
                                                                  dwi, patch);
      GinternalHeatRate.initialize(0.);

      // First compute the temperature gradient at each particle
      IntVector ni[MAX_BASIS];
      Vector d_S[MAX_BASIS];

      for (ParticleSubset::iterator iter = pset->begin();
                                    iter != pset->end(); 
                                    iter++){
	particleIndex idx = *iter;

	// Get the node indices that surround the cell
	if(d_8or27==8){
	  patch->findCellAndShapeDerivatives(px[idx], ni, d_S);
	}
	else if(d_8or27==27){
	  patch->findCellAndShapeDerivatives27(px[idx], ni, d_S, psize[idx]);
	}

	pTemperatureGradient[idx] = Vector(0.0,0.0,0.0);
	for (int k = 0; k < d_8or27; k++){
	  d_S[k] *= pErosion[idx];
	  for (int j = 0; j<3; j++) {
            if(pgCode[idx][k]==1) {
              pTemperatureGradient[idx][j] +=
                  gTemperature[ni[k]] * d_S[k][j] * oodx[j];
            }
            else if(pgCode[idx][k]==2) {
              pTemperatureGradient[idx][j] +=
                  GTemperature[ni[k]] * d_S[k][j] * oodx[j];
            }
	  }
	}
      }


      for(ParticleSubset::iterator iter = pset->begin();
	  iter != pset->end(); iter++){
	particleIndex idx = *iter;
  
	// Get the node indices that surround the cell
	if(d_8or27==8){
	  patch->findCellAndShapeDerivatives(px[idx], ni, d_S);
	}
	else if(d_8or27==27){
	  patch->findCellAndShapeDerivatives27(px[idx], ni, d_S, psize[idx]);
	}

	for (int k = 0; k < d_8or27; k++){
	  if(patch->containsNode(ni[k])){
	    Vector div(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],
		       d_S[k].z()*oodx[2]);
            if(pgCode[idx][k]==1) {
               internalHeatRate[ni[k]] -= Dot( div, pTemperatureGradient[idx]) *
                                          pvol[idx] * thermalConductivity;
            }
            else if(pgCode[idx][k]==2) {
              GinternalHeatRate[ni[k]] -= Dot( div, pTemperatureGradient[idx]) *
                                          pvol[idx] * thermalConductivity;
            }
	  }
	}
      } // End of loop over particles
    }  // End of loop over materials
  }  // End of loop over patches
}


void FractureMPM::solveEquationsMotion(const ProcessorGroup*,
				     const PatchSubset* patches,
				     const MaterialSubset*,
				     DataWarehouse* old_dw,
				     DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing solveEquationsMotion on patch " << patch->getID()
	       <<"\t\t\t MPM"<< endl;

    Vector gravity = d_sharedState->getGravity();
    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label() );

    Ghost::GhostType  gnone = Ghost::None;
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      // Get required variables for this patch
      constNCVariable<Vector> internalforce;
      constNCVariable<Vector> externalforce;
      constNCVariable<Vector> gradPAccNC;  // for MPMICE
      constNCVariable<Vector> AccArchesNC; // for MPMArches
      constNCVariable<double> mass;
 
      new_dw->get(internalforce, lb->gInternalForceLabel, dwi, patch, gnone, 0);
      new_dw->get(externalforce, lb->gExternalForceLabel, dwi, patch, gnone, 0);
      new_dw->get(mass,          lb->gMassLabel,          dwi, patch, gnone, 0);
      if(d_with_ice){
         new_dw->get(gradPAccNC, lb->gradPAccNCLabel,     dwi, patch, gnone, 0);
      }
      else{
  	 NCVariable<Vector> gradPAccNC_create;
	 new_dw->allocateTemporary(gradPAccNC_create,  patch);
	 gradPAccNC_create.initialize(Vector(0.,0.,0.));
	 gradPAccNC = gradPAccNC_create; // reference created data
      }
      if(d_with_arches){
         new_dw->get(AccArchesNC,lb->AccArchesNCLabel,    dwi, patch, gnone, 0);
      }
      else{
  	 NCVariable<Vector> AccArchesNC_create;
	 new_dw->allocateTemporary(AccArchesNC_create,  patch);
	 AccArchesNC_create.initialize(Vector(0.,0.,0.));
	 AccArchesNC = AccArchesNC_create; // reference created data	 
      }

//    Uncomment to use damping
//    constNCVariable<Vector> velocity;
//    new_dw->get(velocity,      lb->gVelocityLabel,      dwi, patch, gnone, 0);

      // Create variables for the results
      NCVariable<Vector> acceleration;
      new_dw->allocateAndPut(acceleration, lb->gAccelerationLabel, dwi, patch);
      acceleration.initialize(Vector(0.,0.,0.));

      // for Fracture
      constNCVariable<double> Gmass;
      constNCVariable<Vector> Ginternalforce;
      constNCVariable<Vector> Gexternalforce;
      new_dw->get(Gmass,         lb->GMassLabel,         dwi, patch, gnone, 0);
      new_dw->get(Ginternalforce,lb->GInternalForceLabel,dwi, patch, gnone, 0);
      new_dw->get(Gexternalforce,lb->GExternalForceLabel,dwi, patch, gnone, 0);

      NCVariable<Vector> Gacceleration;
      new_dw->allocateAndPut(Gacceleration,lb->GAccelerationLabel, dwi, patch);
      Gacceleration.initialize(Vector(0.,0.,0.));

       for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
         IntVector c = *iter;
         // for primary field
         acceleration[c] =
                 (internalforce[c] + externalforce[c])/mass[c] +
                 gravity + gradPAccNC[c] + AccArchesNC[c];
         // for additional field
         Gacceleration[c] =
                 (Ginternalforce[c] + Gexternalforce[c])/Gmass[c] +
                 gravity + gradPAccNC[c] + AccArchesNC[c];
//         acceleration[c] =
//            (internalforce[c] + externalforce[c]
//                                        -1000.*velocity[c]*mass[c])/mass[c]
//                                + gravity + gradPAccNC[c] + AccArchesNC[c];
       }
    }
  }
}

void FractureMPM::solveHeatEquations(const ProcessorGroup*,
				   const PatchSubset* patches,
				   const MaterialSubset* ,
				   DataWarehouse* /*old_dw*/,
				   DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing solveHeatEquations on patch " << patch->getID()
	       <<"\t\t\t MPM"<< endl;

    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      double specificHeat = mpm_matl->getSpecificHeat();
     
      // Get required variables for this patch
      constNCVariable<double> mass,externalHeatRate,gvolume;
      constNCVariable<double> thermalContactHeatExchangeRate;
      NCVariable<double> internalHeatRate;
      constNCVariable<double> htrate_gasNC; // for MPMArches
            
      new_dw->get(mass,    lb->gMassLabel,      dwi, patch, Ghost::None, 0);
      new_dw->get(gvolume, lb->gVolumeLabel,    dwi, patch, Ghost::None, 0);
      new_dw->get(externalHeatRate,           lb->gExternalHeatRateLabel,
                                                dwi, patch, Ghost::None, 0);
      new_dw->getModifiable(internalHeatRate, lb->gInternalHeatRateLabel,
                                                dwi, patch);

      new_dw->get(thermalContactHeatExchangeRate,
                  lb->gThermalContactHeatExchangeRateLabel,
                                                dwi, patch, Ghost::None, 0);

      // for Fracture
      constNCVariable<double> Gmass,GexternalHeatRate,Gvolume;
      constNCVariable<double> GthermalContactHeatExchangeRate;
      NCVariable<double> GinternalHeatRate;

      new_dw->get(Gmass,   lb->GMassLabel,      dwi, patch, Ghost::None, 0);
      new_dw->get(Gvolume, lb->GVolumeLabel,    dwi, patch, Ghost::None, 0);
      new_dw->get(GexternalHeatRate,          lb->GExternalHeatRateLabel,
                                                dwi, patch, Ghost::None, 0);
      new_dw->getModifiable(GinternalHeatRate,lb->GInternalHeatRateLabel,
                                                dwi, patch);

      new_dw->get(GthermalContactHeatExchangeRate,
                  lb->GThermalContactHeatExchangeRateLabel,
                                                dwi, patch, Ghost::None, 0);

      if (d_with_arches) {
	new_dw->get(htrate_gasNC,lb->heaTranSolid_NCLabel,    
		    dwi, patch, Ghost::None, 0);
      }
      else{
	NCVariable<double> htrate_gasNC_create;
	new_dw->allocateTemporary(htrate_gasNC_create, patch);				  

	htrate_gasNC_create.initialize(0.0);
	htrate_gasNC = htrate_gasNC_create; // reference created data
      }

      MPMBoundCond bc;
      bc.setBoundaryCondition(patch,dwi,"Temperature",internalHeatRate,
			      gvolume);
      bc.setBoundaryCondition(patch,dwi,"Temperature",GinternalHeatRate,
			      gvolume);


      // Create variables for the results
      // for primary field
      NCVariable<double> tempRate;
      new_dw->allocateAndPut(tempRate, lb->gTemperatureRateLabel, dwi, patch);
      tempRate.initialize(0.0);

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
          IntVector c = *iter;
	  tempRate[c] = (internalHeatRate[c]
		      +  externalHeatRate[c]
	              +  htrate_gasNC[c]) /
   		        (mass[c] * specificHeat) + 
                         thermalContactHeatExchangeRate[c];
      }

      // for additional field
      NCVariable<double> GtempRate;
      new_dw->allocateAndPut(GtempRate,lb->GTemperatureRateLabel, dwi, patch);
      GtempRate.initialize(0.0);

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
          IntVector c = *iter;
          GtempRate[c] = (GinternalHeatRate[c]
                          +  GexternalHeatRate[c])/(Gmass[c] * specificHeat) +
                             GthermalContactHeatExchangeRate[c];
      }

    }
  }
}


void FractureMPM::integrateAcceleration(const ProcessorGroup*,
				      const PatchSubset* patches,
				      const MaterialSubset*,
				      DataWarehouse* old_dw,
				      DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing integrateAcceleration on patch " << patch->getID()
	       <<"\t\t\t MPM"<< endl;

    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      constNCVariable<Vector>  acceleration, velocity;
      delt_vartype delT;

      new_dw->get(acceleration,lb->gAccelerationLabel,dwi, patch,Ghost::None,0);
      new_dw->get(velocity,    lb->gVelocityLabel,    dwi, patch,Ghost::None,0);

      old_dw->get(delT, d_sharedState->get_delt_label() );

      // Create variables for the results
      NCVariable<Vector> velocity_star;
      new_dw->allocateAndPut(velocity_star, lb->gVelocityStarLabel, dwi, patch);
      velocity_star.initialize(Vector(0.0));

      // for Fracture
      constNCVariable<Vector>  Gacceleration, Gvelocity;
      new_dw->get(Gacceleration,lb->GAccelerationLabel,dwi, patch,Ghost::None,0);
      new_dw->get(Gvelocity,    lb->GVelocityLabel,    dwi, patch,Ghost::None,0);

      NCVariable<Vector> Gvelocity_star;
      new_dw->allocateAndPut(Gvelocity_star,lb->GVelocityStarLabel, dwi, patch);
      Gvelocity_star.initialize(Vector(0.0));

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
        IntVector c = *iter;
        // primary field
	velocity_star[c] = velocity[c] + acceleration[c] * delT;
        // additional field
        Gvelocity_star[c]=Gvelocity[c] +Gacceleration[c] * delT;
      }
    }
  }
}

void FractureMPM::integrateTemperatureRate(const ProcessorGroup*,
					 const PatchSubset* patches,
					 const MaterialSubset*,
					 DataWarehouse* old_dw,
					 DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing integrateTemperatureRate on patch " << patch->getID()
	       << "\t\t MPM"<< endl;

    Ghost::GhostType  gnone = Ghost::None;
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      constNCVariable<double> temp_old,temp_oldNoBC;
      NCVariable<double> temp_rate,tempStar;
      delt_vartype delT;
 
      new_dw->get(temp_old,    lb->gTemperatureLabel,     dwi,patch,gnone,0);
      new_dw->get(temp_oldNoBC,lb->gTemperatureNoBCLabel, dwi,patch,gnone,0);
      new_dw->getModifiable(temp_rate, lb->gTemperatureRateLabel,dwi,patch);

      old_dw->get(delT, d_sharedState->get_delt_label() );

      new_dw->allocateAndPut(tempStar, lb->gTemperatureStarLabel, dwi,patch);
      tempStar.initialize(0.0);

      // for Fracture
      constNCVariable<double> Gtemp_old,Gtemp_oldNoBC;
      NCVariable<double> Gtemp_rate,GtempStar;
      new_dw->get(Gtemp_old,    lb->GTemperatureLabel,    dwi,patch,gnone,0);
      new_dw->get(Gtemp_oldNoBC,lb->GTemperatureRateLabel,dwi,patch,gnone,0);
      new_dw->getModifiable(Gtemp_rate, lb->GTemperatureRateLabel,dwi,patch);
      new_dw->allocateAndPut(GtempStar, lb->GTemperatureStarLabel,dwi,patch);
      GtempStar.initialize(0.0);

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
        IntVector c = *iter;
        // primary field
        tempStar[c] = temp_old[c] + temp_rate[c] * delT;
        // additional field
        GtempStar[c]=Gtemp_old[c] +Gtemp_rate[c] * delT;
      }

      // Apply grid boundary conditions to the temperature 

      MPMBoundCond bc;
      bc.setBoundaryCondition(patch,dwi,"Temperature",tempStar);
      bc.setBoundaryCondition(patch,dwi,"Temperature",GtempStar);


      // Now recompute temp_rate as the difference between the temperature
      // interpolated to the grid (no bcs applied) and the new tempStar
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
        IntVector c = *iter;
        // primary field
        temp_rate[c] = (tempStar[c] - temp_oldNoBC[c]) / delT;
        // additional field
        Gtemp_rate[c]= (GtempStar[c] -Gtemp_oldNoBC[c]) / delT;
      }
    }
  }
}

void FractureMPM::setGridBoundaryConditions(const ProcessorGroup*,
						const PatchSubset* patches,
						const MaterialSubset* ,
						DataWarehouse* old_dw,
						DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing setGridBoundaryConditions on patch " << patch->getID()
	       <<"\t\t MPM"<< endl;

    int numMPMMatls=d_sharedState->getNumMPMMatls();
    
    delt_vartype delT;            
    old_dw->get(delT, d_sharedState->get_delt_label() );
                      
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      NCVariable<Vector> gvelocity_star, gacceleration;
      new_dw->getModifiable(gacceleration, lb->gAccelerationLabel,   dwi,patch);
      new_dw->getModifiable(gvelocity_star,lb->gVelocityStarLabel,   dwi,patch);

      // for Fracture
      NCVariable<Vector> Gvelocity_star, Gacceleration;
      new_dw->getModifiable(Gacceleration, lb->GAccelerationLabel,   dwi,patch);
      new_dw->getModifiable(Gvelocity_star,lb->GVelocityStarLabel,   dwi,patch);

      // Apply grid boundary conditions to the velocity_star and
      // acceleration before interpolating back to the particles
      MPMBoundCond bc;
      bc.setBoundaryCondition(patch,dwi,"Velocity",gvelocity_star);
      bc.setBoundaryCondition(patch,dwi,"Velocity",Gvelocity_star);
      bc.setBoundaryCondition(patch,dwi,"Acceleration",gacceleration);
      bc.setBoundaryCondition(patch,dwi,"Acceleration",Gacceleration);
      bc.setBoundaryCondition(patch,dwi,"Symmetric",gvelocity_star);
      bc.setBoundaryCondition(patch,dwi,"Symmetric",Gvelocity_star);
      bc.setBoundaryCondition(patch,dwi,"Symmetric",gacceleration);
      bc.setBoundaryCondition(patch,dwi,"Symmetric",Gacceleration);

    } // matl loop
  }  // patch loop
}

void FractureMPM::applyExternalLoads(const ProcessorGroup* ,
                                   const PatchSubset* patches,
                                   const MaterialSubset*,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw)
{
  // Get the current time
  double time = d_sharedState->getElapsedTime();
  cout_doing << "Current Time (applyExternalLoads) = " << time << endl;

  // Calculate the force vector at each particle for each pressure bc
  std::vector<double> forcePerPart;
  std::vector<PressureBC*> pbcP;
  if (d_useLoadCurves) {
    for (int ii = 0; 
             ii < (int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); ii++) {
      string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
      if (bcs_type == "Pressure") {

	// Get the material points per load curve
	PressureBC* pbc = 
          dynamic_cast<PressureBC*>(MPMPhysicalBCFactory::mpmPhysicalBCs[ii]);
	pbcP.push_back(pbc);

	// Calculate the force per particle at current time
	forcePerPart.push_back(pbc->forcePerParticle(time));
      } 
    }
  }

  // Loop thru patches to update external force vector
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing applyExternalLoads on patch " 
	       << patch->getID() << "\t MPM"<< endl;

    // Place for user defined loading scenarios to be defined,
    // otherwise pExternalForce is just carried forward.

    int numMPMMatls=d_sharedState->getNumMPMMatls();

    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      if (d_useLoadCurves) {
	// Get the particle position data
	constParticleVariable<Point>  px;
	old_dw->get(px, lb->pXLabel, pset);

	// Get the load curve data
	constParticleVariable<int> pLoadCurveID;
	old_dw->get(pLoadCurveID, lb->pLoadCurveIDLabel, pset);

	// Get the external force data and allocate new space for
	// external force
	ParticleVariable<Vector> pExternalForce;
	ParticleVariable<Vector> pExternalForce_new;
	old_dw->getModifiable(pExternalForce, lb->pExternalForceLabel, pset);
	new_dw->allocateAndPut(pExternalForce_new, 
			       lb->pExtForceLabel_preReloc,  pset);

	// Iterate over the particles
	ParticleSubset::iterator iter = pset->begin();
	for(;iter != pset->end(); iter++){
	  particleIndex idx = *iter;
	  int loadCurveID = pLoadCurveID[idx]-1;
	  if (loadCurveID < 0) {
	    pExternalForce_new[idx] = pExternalForce[idx];
	  } else {
	    PressureBC* pbc = pbcP[loadCurveID];
	    double force = forcePerPart[loadCurveID];
	    pExternalForce_new[idx] = pbc->getForceVector(px[idx], force);
	  }
	}

	// Recycle the loadCurveIDs
	ParticleVariable<int> pLoadCurveID_new;
	new_dw->allocateAndPut(pLoadCurveID_new, 
			       lb->pLoadCurveIDLabel_preReloc, pset);
	pLoadCurveID_new.copyData(pLoadCurveID);
      } else {

	// Get the external force data and allocate new space for
	// external force and copy the data
	ParticleVariable<Vector> pExternalForce;
	ParticleVariable<Vector> pExternalForce_new;
	old_dw->getModifiable(pExternalForce, lb->pExternalForceLabel, pset);
	new_dw->allocateAndPut(pExternalForce_new, 
			       lb->pExtForceLabel_preReloc,  pset);
	pExternalForce_new.copyData(pExternalForce);
      }
    } // matl loop
  }  // patch loop
}

void FractureMPM::calculateDampingRate(const ProcessorGroup*,
				     const PatchSubset* patches,
				     const MaterialSubset* ,
				     DataWarehouse* old_dw,
				     DataWarehouse* new_dw)
{
  if (d_artificialDampCoeff > 0.0) {
    for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);

      cout_doing <<"Doing calculateDampingRate on patch " 
		 << patch->getID() << "\t MPM"<< endl;

      double alphaDot = 0.0;
      int numMPMMatls=d_sharedState->getNumMPMMatls();
      for(int m = 0; m < numMPMMatls; m++){
	MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
	int dwi = mpm_matl->getDWIndex();

	// Get the arrays of particle values to be changed
	constParticleVariable<Point> px;
	constParticleVariable<Vector> psize;

	// Get the arrays of grid data on which the new part. values depend
	constNCVariable<Vector> gvelocity_star;

	ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
	old_dw->get(px, lb->pXLabel, pset);
	if(d_8or27==27) old_dw->get(psize, lb->pSizeLabel, pset);
	Ghost::GhostType  gac = Ghost::AroundCells;
	new_dw->get(gvelocity_star,   lb->gVelocityStarLabel,   dwi,patch,gac,NGP);

        // for Fracture
        constParticleVariable<Short27> pgCode;
        constNCVariable<Vector> Gvelocity_star;
        new_dw->get(pgCode,lb->pgCodeLabel,pset);
        new_dw->get(Gvelocity_star,lb->GVelocityStarLabel,dwi,patch,gac,NGP);

	IntVector ni[MAX_BASIS];
	double S[MAX_BASIS];
	Vector d_S[MAX_BASIS];

	// Calculate artificial dampening rate based on the interpolated particle
	// velocities (ref. Ayton et al., 2002, Biophysical Journal, 1026-1038)
	// d(alpha)/dt = 1/Q Sum(vp*^2)
	ParticleSubset::iterator iter = pset->begin();
	for(;iter != pset->end(); iter++){
	  particleIndex idx = *iter;
	  if (d_8or27 == 27) 
	    patch->findCellAndWeightsAndShapeDerivatives27(px[idx],ni,S,d_S,psize[idx]);
	  else
	    patch->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S);
	  Vector vel(0.0,0.0,0.0);
	  for (int k = 0; k < d_8or27; k++) {
            if(pgCode[idx][k]==1) vel += gvelocity_star[ni[k]]*S[k];
            if(pgCode[idx][k]==2) vel += Gvelocity_star[ni[k]]*S[k];
          }
	  alphaDot += Dot(vel,vel);
	}
	alphaDot /= d_artificialDampCoeff;
      } 
      new_dw->put(sum_vartype(alphaDot), lb->pDampingRateLabel);
    }
  }
}


void FractureMPM::interpolateToParticlesAndUpdate(const ProcessorGroup*,
						const PatchSubset* patches,
						const MaterialSubset* ,
						DataWarehouse* old_dw,
						DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing interpolateToParticlesAndUpdate on patch " 
	       << patch->getID() << "\t MPM"<< endl;

    // Performs the interpolation from the cell vertices of the grid
    // acceleration and velocity to the particles to update their
    // velocity and position respectively
 
    // DON'T MOVE THESE!!!
    double thermal_energy = 0.0;
    Vector CMX(0.0,0.0,0.0);
    Vector CMV(0.0,0.0,0.0);
    double ke=0;
    int numMPMMatls=d_sharedState->getNumMPMMatls();
    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label() );
    bool combustion_problem=false;

    // Artificial Damping 
    double alphaDot = 0.0;
    double alpha = 0.0;
    if (d_artificialDampCoeff > 0.0) {
      max_vartype dampingCoeff; 
      sum_vartype dampingRate;
      old_dw->get(dampingCoeff, lb->pDampingCoeffLabel);
      new_dw->get(dampingRate, lb->pDampingRateLabel);
      alpha = (double) dampingCoeff;
      alphaDot = (double) dampingRate;
      alpha += alphaDot*delT; // Calculate damping coefficient from damping rate
      new_dw->put(max_vartype(alpha), lb->pDampingCoeffLabel);
    }

    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      // Get the arrays of particle values to be changed
      constParticleVariable<Point> px;
      ParticleVariable<Point> pxnew;
      constParticleVariable<Vector> pvelocity, psize;
      ParticleVariable<Vector> pvelocitynew, psizeNew;
      constParticleVariable<double> pmass, pvolume, pTemperature, pSp_vol;
      ParticleVariable<double> pmassNew,pvolumeNew,pTempNew, pSp_volNew;
      constParticleVariable<long64> pids;
      ParticleVariable<long64> pids_new;
      constParticleVariable<Vector> pdisp;
      ParticleVariable<Vector> pdispnew;
      ParticleVariable<double> pkineticEnergyDensity;

      // Get the arrays of grid data on which the new part. values depend
      constNCVariable<Vector> gvelocity_star, gacceleration;
      constNCVariable<double> gTemperatureRate, gTemperature, gTemperatureNoBC;
      constNCVariable<double> dTdt, massBurnFraction, frictionTempRate;
      constNCVariable<double> gSp_vol_src;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      old_dw->get(px,                    lb->pXLabel,                     pset);
      old_dw->get(pdisp,                 lb->pDispLabel,                  pset);
      old_dw->get(pmass,                 lb->pMassLabel,                  pset);
      old_dw->get(pids,                  lb->pParticleIDLabel,            pset);
      old_dw->get(pSp_vol,               lb->pSp_volLabel,                pset);
      new_dw->get(pvolume,               lb->pVolumeDeformedLabel,        pset);
      old_dw->get(pvelocity,             lb->pVelocityLabel,              pset);
      old_dw->get(pTemperature,          lb->pTemperatureLabel,           pset);

      new_dw->allocateAndPut(pvelocitynew, lb->pVelocityLabel_preReloc,   pset);
      new_dw->allocateAndPut(pxnew,        lb->pXLabel_preReloc,          pset);
      new_dw->allocateAndPut(pdispnew,     lb->pDispLabel_preReloc,       pset);
      new_dw->allocateAndPut(pmassNew,     lb->pMassLabel_preReloc,       pset);
      new_dw->allocateAndPut(pvolumeNew,   lb->pVolumeLabel_preReloc,     pset);
      new_dw->allocateAndPut(pids_new,     lb->pParticleIDLabel_preReloc, pset);
      new_dw->allocateAndPut(pTempNew,     lb->pTemperatureLabel_preReloc,pset);
      new_dw->allocateAndPut(pSp_volNew,   lb->pSp_volLabel_preReloc,     pset);
      new_dw->allocateAndPut(pkineticEnergyDensity,
                                          lb->pKineticEnergyDensityLabel, pset);

      ParticleSubset* delset = scinew ParticleSubset
	(pset->getParticleSet(),false,dwi,patch, 0);

      pids_new.copyData(pids);
      if(d_8or27==27){
	old_dw->get(psize,               lb->pSizeLabel,                 pset);
	new_dw->allocateAndPut(psizeNew, lb->pSizeLabel_preReloc,        pset);
	psizeNew.copyData(psize);
      }

      Ghost::GhostType  gac = Ghost::AroundCells;
      new_dw->get(gvelocity_star,   lb->gVelocityStarLabel,   dwi,patch,gac,NGP);
      new_dw->get(gacceleration,    lb->gAccelerationLabel,   dwi,patch,gac,NGP);
      new_dw->get(gTemperatureRate, lb->gTemperatureRateLabel,dwi,patch,gac,NGP);
      new_dw->get(gTemperature,     lb->gTemperatureLabel,    dwi,patch,gac,NGP);
      new_dw->get(gTemperatureNoBC, lb->gTemperatureNoBCLabel,dwi,patch,gac,NGP);
      new_dw->get(frictionTempRate, lb->frictionalWorkLabel,  dwi,patch,gac,NGP);    

      // for Fracture
      constParticleVariable<Short27> pgCode;
      new_dw->get(pgCode, lb->pgCodeLabel, pset);
      constNCVariable<Vector> Gvelocity_star, Gacceleration;
      constNCVariable<double> GTemperatureRate, GTemperature, GTemperatureNoBC;
      constNCVariable<double> GSp_vol_src;

      new_dw->get(Gvelocity_star,   lb->GVelocityStarLabel,   dwi,patch,gac,NGP);
      new_dw->get(Gacceleration,    lb->GAccelerationLabel,   dwi,patch,gac,NGP);
      new_dw->get(GTemperatureRate, lb->GTemperatureRateLabel,dwi,patch,gac,NGP);
      new_dw->get(GTemperature,     lb->GTemperatureLabel,    dwi,patch,gac,NGP);
      new_dw->get(GTemperatureNoBC, lb->GTemperatureNoBCLabel,dwi,patch,gac,NGP);

      if(d_with_ice){
	new_dw->get(dTdt,            lb->dTdt_NCLabel,         dwi,patch,gac,NGP);
	new_dw->get(gSp_vol_src,     lb->gSp_vol_srcLabel,     dwi,patch,gac,NGP);
        new_dw->get(GSp_vol_src,     lb->GSp_vol_srcLabel,     dwi,patch,gac,NGP);
	new_dw->get(massBurnFraction,lb->massBurnFractionLabel,dwi,patch,gac,NGP);
      }
      else{
	NCVariable<double> dTdt_create,massBurnFraction_create,gSp_vol_src_create;
        NCVariable<double> GSp_vol_src_create;
	new_dw->allocateTemporary(dTdt_create,                     patch,gac,NGP);
	new_dw->allocateTemporary(gSp_vol_src_create,              patch,gac,NGP);
        new_dw->allocateTemporary(GSp_vol_src_create,              patch,gac,NGP);
	new_dw->allocateTemporary(massBurnFraction_create,         patch,gac,NGP);
	dTdt_create.initialize(0.);
	gSp_vol_src_create.initialize(0.);
	massBurnFraction_create.initialize(0.);
        GSp_vol_src_create.initialize(0.);
	dTdt = dTdt_create;                         // reference created data
	gSp_vol_src = gSp_vol_src_create;           // reference created data
        GSp_vol_src = GSp_vol_src_create;           // reference created data
 	massBurnFraction = massBurnFraction_create; // reference created data
      }

      // Get the particle erosion information
      constParticleVariable<double> pErosion;
      if (d_doErosion) {
        old_dw->get(pErosion, lb->pErosionLabel, pset);
      } else {
        setParticleDefaultWithTemp(pErosion, pset, new_dw, 1.0);
      }

      double Cp=mpm_matl->getSpecificHeat();
      double rho_init=mpm_matl->getInitialDensity();

      IntVector ni[MAX_BASIS];
      double S[MAX_BASIS];
      Vector d_S[MAX_BASIS];

      double rho_frac_min = 0.;
      if(mpm_matl->getRxProduct() == Material::reactant){
	combustion_problem=true;
	rho_frac_min = .1;
      }

      // Loop over particles
      for(ParticleSubset::iterator iter = pset->begin();
	                           iter != pset->end(); 
	                           iter++){
	particleIndex idx = *iter;

	// Get the node indices that surround the cell
	if(d_8or27==8){
	  patch->findCellAndWeightsAndShapeDerivatives(px[idx], ni, S, d_S);
	}
	else if(d_8or27==27){
	  patch->findCellAndWeightsAndShapeDerivatives27(px[idx], ni, S, d_S,
							 psize[idx]);
	}

	Vector vel(0.0,0.0,0.0);
	Vector acc(0.0,0.0,0.0);
	double tempRate = 0;
	double burnFraction = 0;
	double sp_vol_dt = 0.0;

	// Accumulate the contribution from each surrounding vertex
	for (int k = 0; k < d_8or27; k++) {
	  S[k] *= pErosion[idx];
          if(pgCode[idx][k]==1) {
             vel      += gvelocity_star[ni[k]]  * S[k];
             acc      += gacceleration[ni[k]]   * S[k];
             tempRate += (gTemperatureRate[ni[k]] + dTdt[ni[k]] +
                          frictionTempRate[ni[k]])   * S[k];
             burnFraction += massBurnFraction[ni[k]] * S[k];
             sp_vol_dt += gSp_vol_src[ni[k]]   * S[k];
          }
          else if(pgCode[idx][k]==2) {
             vel      += Gvelocity_star[ni[k]]  * S[k];
             acc      += Gacceleration[ni[k]]   * S[k];
             tempRate += (GTemperatureRate[ni[k]] + dTdt[ni[k]] +
                          frictionTempRate[ni[k]])   * S[k];
             burnFraction += massBurnFraction[ni[k]] * S[k];
             sp_vol_dt += GSp_vol_src[ni[k]]   * S[k];
          }
	}

	// Update the particle's position and velocity
	pxnew[idx]           = px[idx] + vel * delT;
        pdispnew[idx]        = pdisp[idx] + vel * delT;
	pvelocitynew[idx]    = pvelocity[idx] + (acc - alpha*vel)*delT;
	pTempNew[idx]        = pTemperature[idx] + tempRate * delT;
	pSp_volNew[idx]      = pSp_vol[idx] + sp_vol_dt * delT;

	double rho;
	if(pvolume[idx] > 0.){
	  rho = max(pmass[idx]/pvolume[idx],rho_frac_min*rho_init);
	}
	else{
	  rho = rho_init;
	}

        pkineticEnergyDensity[idx]=0.5*rho*pvelocitynew[idx].length2();

	pmassNew[idx]        = Max(pmass[idx]*(1.    - burnFraction),0.);
	pvolumeNew[idx]      = pmassNew[idx]/rho;
	if(pmassNew[idx] <= d_min_part_mass ||
	   (rho_frac_min < 1.0 && pvelocitynew[idx].length() > d_max_vel)){
	  delset->addParticle(idx);
	}
	    
	thermal_energy += pTemperature[idx] * pmass[idx] * Cp;
	ke += .5*pmass[idx]*pvelocitynew[idx].length2();
	CMX = CMX + (pxnew[idx]*pmass[idx]).asVector();
	CMV += pvelocitynew[idx]*pmass[idx];
      }

      new_dw->deleteParticles(delset);      
    }

    if(combustion_problem){
      if(delT < 5.e-9){
	if(delT < 1.e-10){
	  d_min_part_mass = min(d_min_part_mass*2.0,5.e-9);
	  cout << "New d_min_part_mass = " << d_min_part_mass << endl;
	}
	else{
	  d_min_part_mass = min(d_min_part_mass*2.0,5.e-12);
	  cout << "New d_min_part_mass = " << d_min_part_mass << endl;
	}
      }
      else if(delT > 2.e-8){
	d_min_part_mass = max(d_min_part_mass/2.0,3.e-15);
      }
    }

    // DON'T MOVE THESE!!!
    new_dw->put(sum_vartype(ke),     lb->KineticEnergyLabel);
    new_dw->put(sumvec_vartype(CMX), lb->CenterOfMassPositionLabel);
    new_dw->put(sumvec_vartype(CMV), lb->CenterOfMassVelocityLabel);

    // cout << "Solid mass lost this timestep = " << massLost << endl;
    // cout << "Solid momentum after advection = " << CMV << endl;

    // cout << "THERMAL ENERGY " << thermal_energy << endl;
  }
}

void 
FractureMPM::setParticleDefaultWithTemp(constParticleVariable<double>& pvar,
				      ParticleSubset* pset,
				      DataWarehouse* new_dw,
				      double val)
{
  ParticleVariable<double>  temp;
  new_dw->allocateTemporary(temp,  pset);
  ParticleSubset::iterator iter = pset->begin();
  for(;iter != pset->end();iter++){
    temp[*iter]=val;
  }
  pvar = temp; 
}

void 
FractureMPM::setParticleDefault(ParticleVariable<double>& pvar,
			      const VarLabel* label, 
			      ParticleSubset* pset,
			      DataWarehouse* new_dw,
                              double val)
{
  new_dw->allocateAndPut(pvar, label, pset);
  ParticleSubset::iterator iter = pset->begin();
  for (; iter != pset->end(); iter++) {
    pvar[*iter] = val;
  }
}

void FractureMPM::setSharedState(SimulationStateP& ssp)
{
  d_sharedState = ssp;
}

