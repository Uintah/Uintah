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
#include <Packages/Uintah/Core/Grid/BoundCond.h>
#include <Packages/Uintah/Core/Grid/VelocityBoundCond.h>
#include <Packages/Uintah/Core/Grid/SymmetryBoundCond.h>
#include <Packages/Uintah/Core/Grid/TemperatureBoundCond.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Grid/fillFace.h>


#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/MinMax.h>
#include <Core/Util/NotFinished.h>
#include <Core/Util/DebugStream.h>

#include <iostream>
#include <fstream>

using namespace Uintah;
using namespace SCIRun;

using namespace std;

#define MAX_BASIS 27

static DebugStream cout_doing("FRACTURE", false);

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
  NGP     = 1;
  NGN     = 1;

  d_artificialDampCoeff = 0.0;
  d_dampingRateLabel = 
    VarLabel::create("dampingRate", sum_vartype::getTypeDescription() );
  d_dampingCoeffLabel = 
    VarLabel::create("dampingCoeff", max_vartype::getTypeDescription() );

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
     mpm_soln_ps->get("minimum_particle_mass", d_min_part_mass);
     mpm_soln_ps->get("artificial_damping_coeff", d_artificialDampCoeff);
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
   }
   
   MPMPhysicalBCFactory::create(prob_spec);

   contactModel = ContactFactory::create(prob_spec,sharedState, lb, d_8or27);
   thermalContactModel =
		 ThermalContactFactory::create(prob_spec, sharedState, lb);

   // for Fracture (read in crack parameters) ---------------------- 
   crackMethod = scinew Crack(prob_spec,sharedState,lb,d_8or27);
   // --------------------------------------------------------------

   ProblemSpecP p = prob_spec->findBlock("DataArchiver");
   if(!p->get("outputInterval", d_outputInterval))
      d_outputInterval = 1.0;

   //Search for the MaterialProperties block and then get the MPM section

   ProblemSpecP mat_ps =  prob_spec->findBlock("MaterialProperties");

   ProblemSpecP mpm_mat_ps = mat_ps->findBlock("MPM");

   for (ProblemSpecP ps = mpm_mat_ps->findBlock("material"); ps != 0;
       ps = ps->findNextBlock("material") ) {
     MPMMaterial *mat = scinew MPMMaterial(ps, lb, d_8or27,integrator_type,
					   d_useLoadCurves);
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
  t->computes(lb->pMassLabel);
  t->computes(lb->pVolumeLabel);
  t->computes(lb->pTemperatureLabel);
  t->computes(lb->pVelocityLabel);
  t->computes(lb->pExternalForceLabel);
  t->computes(lb->pParticleIDLabel);
  t->computes(lb->pDeformationMeasureLabel);
  t->computes(lb->pStressLabel);
  t->computes(lb->pSizeLabel);
  t->computes(lb->doMechLabel);
  t->computes(d_sharedState->get_delt_label());
  t->computes(lb->pCellNAPIDLabel,zeroth_matl);

  int numMPM = d_sharedState->getNumMPMMatls();
  const PatchSet* patches = level->eachPatch();
  for(int m = 0; m < numMPM; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->addInitialComputesAndRequires(t, mpm_matl, patches);
  }

  sched->addTask(t, level->eachPatch(), d_sharedState->allMPMMaterials());

  t = scinew Task("FractureMPM::printParticleCount",
		  this, &FractureMPM::printParticleCount);
  t->requires(Task::NewDW, lb->partCountLabel);
  sched->addTask(t, level->eachPatch(), d_sharedState->allMPMMaterials());

  // for Farcture ------------------------------------------------------
  // descritize crack plane into triangular elements
  t = scinew Task("Crack:CrackDiscretization",
                   crackMethod, &Crack::CrackDiscretization);
  crackMethod->addComputesAndRequiresCrackDiscretization(t,
                  level->eachPatch(),d_sharedState->allMPMMaterials());
  sched->addTask(t,level->eachPatch(),d_sharedState->allMPMMaterials());
  //--------------------------------------------------------------------

  // The task will have a reference to zeroth_matl
  if (zeroth_matl->removeReference())
    delete zeroth_matl; // shouln't happen, but...
}

void FractureMPM::scheduleComputeStableTimestep(const LevelP&,
					      SchedulerP&)
{
   // Nothing to do here - delt is computed as a by-product of the
   // consitutive model
}

void FractureMPM::scheduleTimeAdvance(const LevelP& level,
				    SchedulerP& sched,
				      int, int)
{
  const PatchSet* patches = level->eachPatch();
  const MaterialSet* matls = d_sharedState->allMPMMaterials();
  //LoadBalancer* loadbal = sched->getLoadBalancer();
  //const PatchSet* allpatches=loadbal->createPerProcessorPatchSet(level,d_myworld);

  scheduleParticleVelocityField(          sched, patches, matls); //for Fracture
  scheduleInterpolateParticlesToGrid(     sched, patches, matls);
  scheduleComputeHeatExchange(            sched, patches, matls);
  scheduleCrackAdjustInterpolated(        sched, patches, matls); //for Fracture
  scheduleExMomInterpolated(              sched, patches, matls);
  scheduleComputeStressTensor(            sched, patches, matls);
  scheduleComputeInternalForce(           sched, patches, matls);
  scheduleComputeInternalHeatRate(        sched, patches, matls);
  scheduleSolveEquationsMotion(           sched, patches, matls);
  scheduleSolveHeatEquations(             sched, patches, matls);
  scheduleIntegrateAcceleration(          sched, patches, matls);
  // scheduleIntegrateTemperatureRate(    sched, patches, matls);
  scheduleCrackAdjustIntegrated(          sched, patches, matls); //for Fracture
  scheduleExMomIntegrated(                sched, patches, matls);
  scheduleSetGridBoundaryConditions(      sched, patches, matls);
  scheduleInterpolateToParticlesAndUpdate(sched, patches, matls);
  scheduleMoveCrack(                      sched, patches, matls); //for Fracture

  sched->scheduleParticleRelocation(level, lb->pXLabel_preReloc,
				    lb->d_particleState_preReloc,
				    lb->pXLabel, lb->d_particleState,
				    lb->pParticleIDLabel, matls);
}

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
  if(d_8or27==27){
   t->requires(Task::OldDW,lb->pSizeLabel,             gan,NGP);
  }
//t->requires(Task::OldDW, lb->pExternalHeatRateLabel, gan,NGP);

  t->computes(lb->gMassLabel);
  t->computes(lb->gMassLabel,        d_sharedState->getAllInOneMatl(),
	      Task::OutOfDomain);
  t->computes(lb->gTemperatureLabel, d_sharedState->getAllInOneMatl(),
	      Task::OutOfDomain);
  t->computes(lb->gVolumeLabel);
  t->computes(lb->gVelocityLabel);
  t->computes(lb->gExternalForceLabel);
  t->computes(lb->gTemperatureLabel);
  t->computes(lb->gTemperatureNoBCLabel);
  t->computes(lb->gExternalHeatRateLabel);
  t->computes(lb->gNumNearParticlesLabel);

  // for Fracture ----------------------------------------------
  t->requires(Task::NewDW, lb->pgCodeLabel,            gan,NGP);
  t->computes(lb->GMassLabel);
  t->computes(lb->GVolumeLabel);
  t->computes(lb->GVelocityLabel);
  t->computes(lb->GExternalForceLabel);
  t->computes(lb->GTemperatureLabel);
  t->computes(lb->GTemperatureNoBCLabel);
  t->computes(lb->GExternalHeatRateLabel);

  t->requires(Task::OldDW, lb->pX0Label,               gan,NGP);
  t->computes(lb->gDisplacementLabel);
  t->computes(lb->GDisplacementLabel);
  //------------------------------------------------------------

  t->computes(lb->TotalMassLabel);
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

// for Fracture ------------------------------------------------------
void FractureMPM::scheduleCrackAdjustInterpolated(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)
{
  Task* t = scinew Task("Crack::CrackContactAdjustInterpolated",
                    crackMethod,&Crack::CrackContactAdjustInterpolated);

  crackMethod->addComputesAndRequiresCrackAdjustInterpolated(t, 
                                                     patches, matls);

  sched->addTask(t, patches, matls);
}
//--------------------------------------------------------------------

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
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->addComputesAndRequires(t, mpm_matl, patches);
  }
	 
  t->computes(d_sharedState->get_delt_label());
  t->computes(lb->StrainEnergyLabel);

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

  if(d_with_ice){
    t->requires(Task::NewDW, lb->pPressureLabel,          gan,NGP);
  }

  // for Fracture ------------------------------------------------------
   t->requires(Task::NewDW,lb->pgCodeLabel,               gan,NGP);
   t->requires(Task::NewDW,lb->GMassLabel, gnone);
   t->computes(lb->GInternalForceLabel);
  // -------------------------------------------------------------------

  t->computes(lb->gInternalForceLabel);
  t->computes(lb->NTractionZMinusLabel);
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

  // for Fracture --------------------------------------------------
  t->requires(Task::NewDW, lb->pgCodeLabel,          gan, NGP);
  t->requires(Task::NewDW, lb->GTemperatureLabel,    gac, 2*NGP);
  t->computes(lb->GInternalHeatRateLabel);
  // ---------------------------------------------------------------

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
  t->requires(Task::OldDW, lb->doMechLabel);
//Uncomment  the next line to use damping
//t->requires(Task::NewDW, lb->gVelocityLabel,      Ghost::None);     
  t->computes(lb->doMechLabel);

  if(d_with_ice){
    t->requires(Task::NewDW, lb->gradPAccNCLabel,   Ghost::None);
  }
  if(d_with_arches){
    t->requires(Task::NewDW, lb->AccArchesNCLabel,  Ghost::None);
  }

  t->computes(lb->gAccelerationLabel);

  // for Fracture -----------------------------------------------
  t->requires(Task::NewDW, lb->GMassLabel,          Ghost::None);
  t->requires(Task::NewDW, lb->GInternalForceLabel, Ghost::None);
  t->requires(Task::NewDW, lb->GExternalForceLabel, Ghost::None);
  t->computes(lb->GAccelerationLabel);
  // ------------------------------------------------------------

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

  Ghost::GhostType  gnone = Ghost::None;
  t->requires(Task::NewDW, lb->gMassLabel,                           gnone);
  t->requires(Task::NewDW, lb->gVolumeLabel,                         gnone);
  t->requires(Task::NewDW, lb->gExternalHeatRateLabel,               gnone);
  t->requires(Task::NewDW, lb->gInternalHeatRateLabel,               gnone);
  t->requires(Task::NewDW, lb->gThermalContactHeatExchangeRateLabel, gnone);
		
  t->computes(lb->gTemperatureRateLabel);

  // for Fracture ------------------------------------------------------------
  t->requires(Task::NewDW, lb->GMassLabel,                           gnone);
  t->requires(Task::NewDW, lb->GVolumeLabel,                         gnone);
  t->requires(Task::NewDW, lb->GExternalHeatRateLabel,               gnone);
  t->requires(Task::NewDW, lb->GInternalHeatRateLabel,               gnone);
  t->requires(Task::NewDW, lb->GThermalContactHeatExchangeRateLabel, gnone);

  t->computes(lb->GTemperatureRateLabel);
  //--------------------------------------------------------------------------

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

  // for Fracture --------------------------------------------------------
  t->requires(Task::NewDW, lb->GAccelerationLabel,      Ghost::None);
  t->requires(Task::NewDW, lb->GVelocityLabel,          Ghost::None);
  t->computes(lb->GVelocityStarLabel);
  // ---------------------------------------------------------------------

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

  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  t->requires(Task::NewDW, lb->gTemperatureLabel,     Ghost::None);
  t->requires(Task::NewDW, lb->gTemperatureRateLabel, Ghost::None);
		     
  t->computes(lb->gTemperatureStarLabel);

  // for Fracture --------------------------------------------------
  t->requires(Task::NewDW, lb->GTemperatureLabel,     Ghost::None);
  t->requires(Task::NewDW, lb->GTemperatureRateLabel, Ghost::None);
  t->computes(lb->GTemperatureStarLabel);
  // ---------------------------------------------------------------
		     
  sched->addTask(t, patches, matls);
}

// for Fracture ------------------------------------------------------
void FractureMPM::scheduleCrackAdjustIntegrated(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)
{
  Task* t = scinew Task("Crack::CrackContactAdjustIntegrated",
                    crackMethod,&Crack::CrackContactAdjustIntegrated);

  crackMethod->addComputesAndRequiresCrackAdjustIntegrated(t,
                                                      patches, matls);

  sched->addTask(t, patches, matls);
}
//--------------------------------------------------------------------

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
  t->modifies(             lb->gTemperatureRateLabel,  mss);
  t->requires(Task::NewDW, lb->gTemperatureNoBCLabel,  Ghost::None,0);

  //for Fracture ------------------------------------------------------
  t->modifies(             lb->GAccelerationLabel,     mss);
  t->modifies(             lb->GVelocityStarLabel,     mss);
  t->modifies(             lb->GTemperatureRateLabel,  mss);
  t->requires(Task::NewDW, lb->GTemperatureNoBCLabel,  Ghost::None,0);
  // ------------------------------------------------------------------

  sched->addTask(t, patches, matls);
}

void FractureMPM::scheduleInterpolateToParticlesAndUpdate(SchedulerP& sched,
						       const PatchSet* patches,
						       const MaterialSet* matls)

{
 /*
  * interpolateToParticlesAndUpdate
  in(G.ACCELERATION, G.VELOCITY_STAR, P.NAT_X)
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
  t->requires(Task::NewDW, lb->frictionalWorkLabel,    gac,NGN);
  t->requires(Task::OldDW, lb->pXLabel,                Ghost::None);
  t->requires(Task::OldDW, lb->pExternalForceLabel,    Ghost::None);
  t->requires(Task::OldDW, lb->pMassLabel,             Ghost::None);
  t->requires(Task::OldDW, lb->pParticleIDLabel,       Ghost::None);
  t->requires(Task::OldDW, lb->pTemperatureLabel,      Ghost::None);
  t->requires(Task::OldDW, lb->pVelocityLabel,         Ghost::None);
  t->requires(Task::OldDW, lb->pMassLabel,             Ghost::None);
 
  // for Fracture ---------------------------------------------------
  t->requires(Task::NewDW, lb->GAccelerationLabel,     gac,NGN);
  t->requires(Task::NewDW, lb->GVelocityStarLabel,     gac,NGN);
  t->requires(Task::NewDW, lb->GTemperatureRateLabel,  gac,NGN);
  t->requires(Task::NewDW, lb->GTemperatureLabel,      gac,NGN);
  t->requires(Task::NewDW, lb->GTemperatureNoBCLabel,  gac,NGN);
  t->requires(Task::NewDW, lb->pgCodeLabel,            Ghost::None);
  t->requires(Task::OldDW, lb->pX0Label,               Ghost::None);

  t->computes(lb->pX0Label_preReloc);

  // ----------------------------------------------------------------

  if(d_8or27==27){
   t->requires(Task::OldDW, lb->pSizeLabel,            Ghost::None);
  }
  t->requires(Task::NewDW, lb->pVolumeDeformedLabel,   Ghost::None);

  if(d_with_ice){
    t->requires(Task::NewDW, lb->dTdt_NCLabel,         gac,NGN);
    t->requires(Task::NewDW, lb->massBurnFractionLabel,gac,NGN);
  }

  t->computes(lb->pVelocityLabel_preReloc);
  t->computes(lb->pXLabel_preReloc);
  t->computes(lb->pExtForceLabel_preReloc);
  t->computes(lb->pParticleIDLabel_preReloc);
  t->computes(lb->pTemperatureLabel_preReloc);
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

void FractureMPM::scheduleMoveCrack(SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSet* matls)
{

  Task* t = scinew Task("Crack::MoveCrack", crackMethod,
                        &Crack::MoveCrack);

  crackMethod->addComputesAndRequiresMoveCrack(t, patches, matls);
  sched->addTask(t, patches, matls);
}
/*********************************************************************
void FractureMPM::scheduleUpdateCrackData(SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSet* matls)
{

  Task* t = scinew Task("Crack::UpdateCrackData", crackMethod,
                        &Crack::UpdateCrackData);

  crackMethod->addComputesAndRequiresUpdateCrackData(t, patches, matls);
  sched->addTask(t, patches, matls);
}
*************************************************************************/

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
      cerr << "Created " << pcount << " total particles\n";
      printed=true;
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
						mpm_matl, new_dw);
    }
    // allocateAndPut instead:
    /* new_dw->put(cellNAPID, lb->pCellNAPIDLabel, 0, patch); */;

    double doMech = -999.9;
    new_dw->put(delt_vartype(doMech), lb->doMechLabel);
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
  //double time0, time1;
  //time0 = clock();

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing <<"Doing interpolateParticlesToGrid on patch " << patch->getID()
	       <<"\t\t MPM"<< endl;

    int numMatls = d_sharedState->getNumMPMMatls();

    NCVariable<double> gmassglobal,gtempglobal;
    new_dw->allocateAndPut(gmassglobal, lb->gMassLabel,
		     d_sharedState->getAllInOneMatl()->get(0), patch);
    new_dw->allocateAndPut(gtempglobal, lb->gTemperatureLabel,
		     d_sharedState->getAllInOneMatl()->get(0), patch);
    gmassglobal.initialize(d_SMALL_NUM_MPM);
    gtempglobal.initialize(0.0);

    Ghost::GhostType  gan = Ghost::AroundNodes;
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      // Create arrays for the particle data
      constParticleVariable<Point>  px,px0;
      constParticleVariable<double> pmass, pvolume, pTemperature;
      constParticleVariable<Vector> pvelocity, pexternalforce,psize;
      ParticleVariable<double> pexternalheatrate;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       gan, NGP, lb->pXLabel);
      old_dw->get(px0,            lb->pX0Label,            pset);//for Fracture
      old_dw->get(px,             lb->pXLabel,             pset);
      old_dw->get(pmass,          lb->pMassLabel,          pset);
      old_dw->get(pvolume,        lb->pVolumeLabel,        pset);
      old_dw->get(pvelocity,      lb->pVelocityLabel,      pset);
      old_dw->get(pTemperature,   lb->pTemperatureLabel,   pset);
      old_dw->get(pexternalforce, lb->pExternalForceLabel, pset);
      if(d_8or27==27){
        old_dw->get(psize,        lb->pSizeLabel,          pset);
      }

      new_dw->allocateTemporary(pexternalheatrate,  pset);
 
      // for Fracture ----------------------------------------------------------
      constParticleVariable<Short27> pgCode;
      new_dw->get(pgCode,lb->pgCodeLabel,pset);

#if 0 // output particle velocity field code
      cout << "\n*** Particle velocity field information retrieved from"
           << " dataWarehouse (new_dw)" << endl; 
      cout << "    in FractureMPM::interpolateParticlesToGrid\n" << endl;
      for(ParticleSubset::iterator iter=pset->begin();
                        iter!=pset->end();iter++) {
        particleIndex idx=*iter;
        cout << "p["<< idx << "]: " << px[idx]<< ", mass=" << pmass[idx]
            << ", force=" << pexternalforce[idx] << endl;
        for(int k=0; k<d_8or27; k++) {
          if(pgCode[idx][k]==1)
             cout << setw(10) << "Node: " << k 
                  << ",\tvfld: " << pgCode[idx][k] << endl;
          else if(pgCode[idx][k]==2)
             cout << setw(10) << "Node: " << k 
                  << ",\tvfld: " << pgCode[idx][k] << " ***" << endl;
          else {
             cout << "Unknown particle velocity code in "
                  << "FractureMPM::interpolateParticlesToGrid" << endl;
             exit(1);
          }
        }  // End loop over nodes (k)
      }  // End loop over particles
#endif
      // -----------------------------------------------------------------------

      // Create arrays for the grid data of primary velocity field
      NCVariable<double> gmass;
      NCVariable<double> gvolume;
      NCVariable<Vector> gvelocity;
      NCVariable<Vector> gexternalforce;
      NCVariable<double> gexternalheatrate;
      NCVariable<double> gTemperature;
      NCVariable<double> gTemperatureNoBC;
      NCVariable<double> gnumnearparticles;

      new_dw->allocateAndPut(gmass,            lb->gMassLabel,       dwi,patch);
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

      // for Fracture ---------------------------------------------------------
      // Create arrays for additional velocity field (for Farcture)
      NCVariable<double> Gmass;
      NCVariable<double> Gvolume;
      NCVariable<Vector> Gvelocity;
      NCVariable<double> GTemperature;
      NCVariable<double> GTemperatureNoBC;
      NCVariable<Vector> Gexternalforce;
      NCVariable<double> Gexternalheatrate;
      NCVariable<Vector> gdisplacement;
      NCVariable<Vector> Gdisplacement;

      new_dw->allocateAndPut(Gmass,            lb->GMassLabel,       dwi,patch);
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

      //-------------------------------------------------------------------------

      // Interpolate particle data to Grid data.
      // This currently consists of the particle velocity and mass
      // Need to compute the lumped global mass matrix and velocity
      // Vector from the individual mass matrix and velocity vector
      // GridMass * GridVelocity =  S^T*M_D*ParticleVelocity
      
      double totalmass = 0;
      Vector total_mom(0.0,0.0,0.0);

#if 0
      for(ParticleSubset::iterator iter = pset->begin();
                iter != pset->end(); iter++){
        particleIndex idx = *iter;

        if(px[idx].z() < -0.0475 && fabs(px[idx].x()) < .03
                                 && fabs(px[idx].y()) < .03){
          pexternalheatrate[idx]=20.0;
        }
        else{
          pexternalheatrate[idx]=0.0;
        }
      }
#endif
      IntVector ni[MAX_BASIS];
      double S[MAX_BASIS];

      for(ParticleSubset::iterator iter = pset->begin();
						iter != pset->end(); iter++){
	  particleIndex idx = *iter;

          //cout << "p:" << idx << ", x0=" << px0[idx] << ", x=" << px[idx] << endl;
	  // Get the node indices that surround the cell
          if(d_8or27==8){
  	    patch->findCellAndWeights(px[idx], ni, S);
          }
          else if(d_8or27==27){
  	    patch->findCellAndWeights27(px[idx], ni, S, psize[idx]);
          }

          total_mom += pvelocity[idx]*pmass[idx];

	  // Add each particles contribution to the local mass & velocity 
	  // Must use the node indices
	  for(int k = 0; k < d_8or27; k++) {
            if(patch->containsNode(ni[k])) {
              gmassglobal[ni[k]]      += pmass[idx]                     * S[k];
              gtempglobal[ni[k]]      += pTemperature[idx] * pmass[idx] * S[k];
              if(pgCode[idx][k]==1) {   // for primary field
                gdisplacement[ni[k]]  += (px[idx]-px0[idx])*pmass[idx]  * S[k];
	        gmass[ni[k]]          += pmass[idx]                     * S[k];
	        gvolume[ni[k]]        += pvolume[idx]                   * S[k];
	        gexternalforce[ni[k]] += pexternalforce[idx]            * S[k];
	        gvelocity[ni[k]]      += pvelocity[idx]    * pmass[idx] * S[k];
	        gTemperature[ni[k]]   += pTemperature[idx] * pmass[idx] * S[k];
                gexternalheatrate[ni[k]] += pexternalheatrate[idx]      * S[k];
                gnumnearparticles[ni[k]] += 1.0;
              }
              else if(pgCode[idx][k]==2) {  // for additional field
                Gdisplacement[ni[k]]  += (px[idx]-px0[idx])*pmass[idx]  * S[k];
                Gmass[ni[k]]          += pmass[idx]                     * S[k];
                Gvolume[ni[k]]        += pvolume[idx]                   * S[k];
                Gexternalforce[ni[k]] += pexternalforce[idx]            * S[k];
                Gvelocity[ni[k]]      += pvelocity[idx]    * pmass[idx] * S[k];
                GTemperature[ni[k]]   += pTemperature[idx] * pmass[idx] * S[k];
                Gexternalheatrate[ni[k]] += pexternalheatrate[idx]      * S[k];
              }
              else {  // wrong velocity field
                cout << "Unknown velocity field in "
                     << "Fracture::interpolateParticleToGrid: pgCode=" 
                     << pgCode[idx][k]
                     << " for particle: " << idx << px[idx]
                     << " and node " << ni[k] << endl;
                exit(1);
              }
	    }
	  } // End loop over nodes
        }

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
        IntVector c = *iter; 
        totalmass       += gmassglobal[c];
        //gmassglobal[c]  += gmass[c]; 
        
        // for primary field
	gvelocity[c]    /= gmass[c];
        gTemperature[c] /= gmass[c];
        gTemperatureNoBC[c] = gTemperature[c];
        gdisplacement[c]/=gmass[c];

        // for Fracture (additional field) ----------
        Gvelocity[c]    /= Gmass[c];
        GTemperature[c] /= Gmass[c];
        GTemperatureNoBC[c] = GTemperature[c];
        Gdisplacement[c]/=Gmass[c];
        // ------------------------------------------
      }

      // Apply grid boundary conditions to the velocity before storing the data
      IntVector offset =  IntVector(0,0,0);

      for(Patch::FaceType face = Patch::startFace;
	face <= Patch::endFace; face=Patch::nextFace(face)){
        const BoundCondBase *vel_bcs, *temp_bcs, *sym_bcs;
        if (patch->getBCType(face) == Patch::None) {
	   vel_bcs  = patch->getBCValues(dwi,"Velocity",face);
	   temp_bcs = patch->getBCValues(dwi,"Temperature",face);
	   sym_bcs  = patch->getBCValues(dwi,"Symmetric",face);
        } else
          continue;

	  if (vel_bcs != 0) {
	    const VelocityBoundCond* bc =
	      dynamic_cast<const VelocityBoundCond*>(vel_bcs);
	    if (bc->getKind() == "Dirichlet") {
	      //cout << "Velocity bc value = " << bc->getValue() << endl;
	      fillFace(gvelocity,patch, face,bc->getValue(),offset);
              fillFace(Gvelocity,patch, face,bc->getValue(),offset);//for Fracture
	    }
	  }
	  if (sym_bcs != 0) {
	     fillFaceNormal(gvelocity,patch, face,offset);
             fillFaceNormal(Gvelocity,patch, face,offset); // for Fracture
	  }
	  if (temp_bcs != 0) {
            const TemperatureBoundCond* bc =
	      dynamic_cast<const TemperatureBoundCond*>(temp_bcs);
            if (bc->getKind() == "Dirichlet") {
              fillFace(gTemperature,patch, face,bc->getValue(),offset);
              fillFace(GTemperature,patch, face,bc->getValue(),offset);//for Fracture
	    }
	  }
      }

      new_dw->put(sum_vartype(totalmass), lb->TotalMassLabel);

    }  // End loop over materials

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
        IntVector c = *iter;
        gtempglobal[c] /= gmassglobal[c];
    }
  }  // End loop over patches
  //time1=clock()-time0;
  //time1/=CLOCKS_PER_SEC;
  //cout << "***time for interpolateParticlesToGrid = " << time1 << endl;

}

void FractureMPM::computeStressTensor(const ProcessorGroup*,
				    const PatchSubset* patches,
				    const MaterialSubset* ,
				    DataWarehouse* old_dw,
				    DataWarehouse* new_dw)
{

  cout_doing <<"Doint computeStressTensor " <<"\t\t\t\t MPM"<< endl;

   for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
      ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
      cm->computeStressTensor(patches, mpm_matl, old_dw, new_dw);
   }
}

void FractureMPM::computeInternalForce(const ProcessorGroup*,
				     const PatchSubset* patches,
				     const MaterialSubset* ,
				     DataWarehouse* old_dw,
				     DataWarehouse* new_dw)
{
  //double time0, time1;
  //time0 = clock();

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

    double integralTraction = 0.;
    double integralArea = 0.;

    NCVariable<Matrix3>       gstressglobal;
    constNCVariable<double>   gmassglobal;
    new_dw->get(gmassglobal,  lb->gMassLabel,
		d_sharedState->getAllInOneMatl()->get(0), patch, Ghost::None,0);
    new_dw->allocateAndPut(gstressglobal, lb->gStressForSavingLabel, 
		     d_sharedState->getAllInOneMatl()->get(0), patch);

    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      // Create arrays for the particle position, volume
      // and the constitutive model
      constParticleVariable<Point>   px;
      constParticleVariable<double>  pvol, pmass;
      constParticleVariable<double>  p_pressure;
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
 
      // for Fracture ----------------------------------------------------------
      constParticleVariable<Short27> pgCode;
      new_dw->get(pgCode,  lb->pgCodeLabel,                  pset);

      constNCVariable<double> Gmass;
      new_dw->get(Gmass,   lb->GMassLabel, dwi, patch, Ghost::None, 0);
     
      NCVariable<Vector> Ginternalforce;
      new_dw->allocateAndPut(Ginternalforce,lb->GInternalForceLabel, dwi,patch);
      Ginternalforce.initialize(Vector(0,0,0)); 
      // -----------------------------------------------------------------------

      if(d_with_ice){
        new_dw->get(p_pressure,lb->pPressureLabel, pset);
      }
      else {
	ParticleVariable<double>  p_pressure_create;
	new_dw->allocateTemporary(p_pressure_create,  pset);
	for(ParticleSubset::iterator iter = pset->begin();
                                     iter != pset->end(); iter++){
	   p_pressure_create[*iter]=0.0;
	}
	p_pressure = p_pressure_create; // reference created data
      }

      internalforce.initialize(Vector(0,0,0));
      IntVector ni[MAX_BASIS];
      double S[MAX_BASIS];
      Vector d_S[MAX_BASIS];

      for(ParticleSubset::iterator iter = pset->begin();
						iter != pset->end(); iter++){
          particleIndex idx = *iter;
  
          // Get the node indices that surround the cell
           if(d_8or27==8){
             patch->findCellAndWeightsAndShapeDerivatives(px[idx], ni, S,  d_S);
           }
           else if(d_8or27==27){
             patch->findCellAndWeightsAndShapeDerivatives27(px[idx], ni, S,d_S,
                                                            psize[idx]);
           }

          for (int k = 0; k < d_8or27; k++){
	    if(patch->containsNode(ni[k])){
	       Vector div(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],
						d_S[k].z()*oodx[2]);
               // for Fracture ------------------------------------------------
               if(pgCode[idx][k]==1) 
	          internalforce[ni[k]] -=
                      (div * (pstress[idx] + Id*p_pressure[idx]) * pvol[idx]);
               else if(pgCode[idx][k]==2) 
                  Ginternalforce[ni[k]] -=
                      (div * (pstress[idx] + Id*p_pressure[idx]) * pvol[idx]);
               else {
                  cout << "Unknown particle velocity field in "
                       << "FractureMPM::ComputeInternalForce:" 
                       << pgCode[idx][k] << endl;
                  exit(1);
               }
               // -------------------------------------------------------------
               gstress[ni[k]]       += pstress[idx] * pmass[idx] * S[k];
               gstressglobal[ni[k]] += pstress[idx] * pmass[idx] * S[k];
	     }
          }
      }

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
        IntVector c = *iter;
        gstress[c] /= (gmass[c]+Gmass[c]);//add in additional mass, for Fracture
    }

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
    //__________________________________
    // Set internal force = 0 on symmetric boundaries
    for(Patch::FaceType face = Patch::startFace;
	  face <= Patch::endFace; face=Patch::nextFace(face)){
      const BoundCondBase *sym_bcs;
      if (patch->getBCType(face) == Patch::None) {
        sym_bcs  = patch->getBCValues(dwi,"Symmetric",face);
      } else
        continue;
      if (sym_bcs != 0) {
        IntVector offset(0,0,0);
        fillFaceNormal( internalforce,patch,face,offset);
        fillFaceNormal(Ginternalforce,patch,face,offset);//for Fracture
      }
    }
  }
  if(integralArea > 0.){
    integralTraction=integralTraction/integralArea;
  }
  else{
    integralTraction=0.;
  }
  new_dw->put(sum_vartype(integralTraction), lb->NTractionZMinusLabel);

  for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
    IntVector c = *iter;
    gstressglobal[c] /= gmassglobal[c];
  }
  }
  //time1=clock()-time0;
  //time1/=CLOCKS_PER_SEC;
  //cout << "***time for computeInternalForce = " << time1 << endl;

}

void FractureMPM::computeInternalHeatRate(const ProcessorGroup*,
				        const PatchSubset* patches,
					const MaterialSubset* ,
				        DataWarehouse* old_dw,
				        DataWarehouse* new_dw)
{
  //double time0, time1;
  //time0 = clock();

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

      new_dw->get(gTemperature, lb->gTemperatureLabel, dwi, patch, gac,2*NGN);
      new_dw->allocateAndPut(internalHeatRate, lb->gInternalHeatRateLabel,
                                                                    dwi, patch);
      new_dw->allocateTemporary(pTemperatureGradient, pset);
  
      internalHeatRate.initialize(0.);

      // for Fracture ----------------------------------------------------------
      constParticleVariable<Short27> pgCode;
      new_dw->get(pgCode, lb->pgCodeLabel, pset);
      
      constNCVariable<double> GTemperature;
      new_dw->get(GTemperature, lb->GTemperatureLabel, dwi, patch, gac, 2*NGN);

      NCVariable<double> GinternalHeatRate;
      new_dw->allocateAndPut(GinternalHeatRate, lb->GInternalHeatRateLabel,
                                                                  dwi, patch);
      GinternalHeatRate.initialize(0.);
      // -----------------------------------------------------------------------

      // First compute the temperature gradient at each particle
      IntVector ni[MAX_BASIS];
      Vector d_S[MAX_BASIS];

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

	 pTemperatureGradient[idx] = Vector(0.0,0.0,0.0);
         for(int k = 0; k < d_8or27; k++){
           for(int j = 0; j<3; j++) {
             // for Fracture -------------------------------------
             if(pgCode[idx][k]==1) 
               pTemperatureGradient[idx][j] += 
                   gTemperature[ni[k]] * d_S[k][j] * oodx[j];
             else if(pgCode[idx][k]==2)
               pTemperatureGradient[idx][j] +=
                   GTemperature[ni[k]] * d_S[k][j] * oodx[j];
             else {
               cout << "Unknown velocity field in "
                    << "FractureMPM::computeInternalHeatRate: "
                    << pgCode[idx][k] << endl;
               exit(1);
             }
             // --------------------------------------------------
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
             // for Fracture ----------------------------------------------------
             if(pgCode[idx][k]==1)
	        internalHeatRate[ni[k]] -= Dot( div, pTemperatureGradient[idx]) * 
	                                   pvol[idx] * thermalConductivity;
             else if(pgCode[idx][k]==2) 
               GinternalHeatRate[ni[k]] -= Dot( div, pTemperatureGradient[idx]) *
                                           pvol[idx] * thermalConductivity;
             else {
               cout << "Unknown velocity field in "
                    << "FractureMPM::computeInternalHeatRate: "
                    << pgCode[idx][k] << endl;
               exit(1);
             }
             // -----------------------------------------------------------------
	   }
         }
      }
    }  // End of loop over materials
  }  // End of loop over patches
  //time1=clock()-time0;
  //time1/=CLOCKS_PER_SEC;
  //cout << "***time for computeInternalHeatRate = " << time1 << endl;
}


void FractureMPM::solveEquationsMotion(const ProcessorGroup*,
				     const PatchSubset* patches,
				     const MaterialSubset*,
				     DataWarehouse* old_dw,
				     DataWarehouse* new_dw)
{
  //double time0, time1;
  //time0 = clock();

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing solveEquationsMotion on patch " << patch->getID()
	       <<"\t\t\t MPM"<< endl;

    Vector gravity = d_sharedState->getGravity();
    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label() );
    delt_vartype doMechOld;
    old_dw->get(doMechOld, lb->doMechLabel);

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

      // for Fracture ----------------------------------------------------------
      constNCVariable<double> Gmass;
      constNCVariable<Vector> Ginternalforce;
      constNCVariable<Vector> Gexternalforce;
      new_dw->get(Gmass,         lb->GMassLabel,         dwi, patch, gnone, 0);
      new_dw->get(Ginternalforce,lb->GInternalForceLabel,dwi, patch, gnone, 0);
      new_dw->get(Gexternalforce,lb->GExternalForceLabel,dwi, patch, gnone, 0);

      NCVariable<Vector> Gacceleration;
      new_dw->allocateAndPut(Gacceleration,lb->GAccelerationLabel, dwi, patch);
      Gacceleration.initialize(Vector(0.,0.,0.));
      // ------------------------------------------------------------------------
      
      if(doMechOld < -1.5){
       for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
         IntVector c = *iter;
         // for primary field
         acceleration[c] =
                 (internalforce[c] + externalforce[c])/mass[c] +
                 gravity + gradPAccNC[c] + AccArchesNC[c];
         // for Fracture ----------------------------------------------
         Gacceleration[c] =
                 (Ginternalforce[c] + Gexternalforce[c])/Gmass[c] +
                 gravity + gradPAccNC[c] + AccArchesNC[c];
         // ----------------------------------------------------------
//         acceleration[c] =
//            (internalforce[c] + externalforce[c]
//                                        -1000.*velocity[c]*mass[c])/mass[c]
//                                + gravity + gradPAccNC[c] + AccArchesNC[c];
       }
      }
    }
    new_dw->put(doMechOld, lb->doMechLabel);
  }
  //time1=clock()-time0;
  //time1/=CLOCKS_PER_SEC;
  //cout << "***time for solveEquationsMotion  = " << time1 << endl;

}

void FractureMPM::solveHeatEquations(const ProcessorGroup*,
				   const PatchSubset* patches,
				   const MaterialSubset* ,
				   DataWarehouse* /*old_dw*/,
				   DataWarehouse* new_dw)
{
  //double time0, time1;
  //time0 = clock();

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing solveHeatEquations on patch " << patch->getID()
	       <<"\t\t\t MPM"<< endl;

    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      double specificHeat = mpm_matl->getSpecificHeat();
     
      // Get required variables for this patch
      // data of primary field
      constNCVariable<double> mass,externalHeatRate,gvolume;
      constNCVariable<double> thermalContactHeatExchangeRate;
      NCVariable<double> internalHeatRate;
      
      new_dw->get(mass,    lb->gMassLabel,      dwi, patch, Ghost::None, 0);
      new_dw->get(gvolume, lb->gVolumeLabel,    dwi, patch, Ghost::None, 0);
      new_dw->getCopy(internalHeatRate, lb->gInternalHeatRateLabel,
                                                dwi, patch, Ghost::None, 0);
      new_dw->get(externalHeatRate,     lb->gExternalHeatRateLabel,
                                                dwi, patch, Ghost::None, 0);

      new_dw->get(thermalContactHeatExchangeRate,
                  lb->gThermalContactHeatExchangeRateLabel,
                                                dwi, patch, Ghost::None, 0);
      
      // for Fracture --------------------------------------------------------
      constNCVariable<double> Gmass,GexternalHeatRate,Gvolume;
      constNCVariable<double> GthermalContactHeatExchangeRate;
      NCVariable<double> GinternalHeatRate;

      new_dw->get(Gmass,   lb->GMassLabel,      dwi, patch, Ghost::None, 0);
      new_dw->get(Gvolume, lb->GVolumeLabel,    dwi, patch, Ghost::None, 0);
      new_dw->getCopy(GinternalHeatRate, lb->GInternalHeatRateLabel,
                                                dwi, patch, Ghost::None, 0);
      new_dw->get(GexternalHeatRate,     lb->GExternalHeatRateLabel,
                                                dwi, patch, Ghost::None, 0);

      new_dw->get(GthermalContactHeatExchangeRate,
                  lb->GThermalContactHeatExchangeRateLabel,
                                                dwi, patch, Ghost::None, 0);
      // --------------------------------------------------------------------

      Vector dx = patch->dCell();
      for(Patch::FaceType face = Patch::startFace;
	  face <= Patch::endFace; face=Patch::nextFace(face)){
	const BoundCondBase* temp_bcs;
        if (patch->getBCType(face) == Patch::None) {
	   temp_bcs = patch->getBCValues(dwi,"Temperature",face);
        } else
          continue;

	if (temp_bcs != 0) {
            const TemperatureBoundCond* bc =
	      dynamic_cast<const TemperatureBoundCond*>(temp_bcs);
            if (bc->getKind() == "Neumann"){
	      double value = bc->getValue();

             IntVector low = patch->getInteriorNodeLowIndex();
             IntVector hi  = patch->getInteriorNodeHighIndex();     
              if(face==Patch::xplus || face==Patch::xminus){
                int I;
                if(face==Patch::xminus){ I=low.x(); }
                if(face==Patch::xplus){ I=hi.x()-1; }
                for (int j = low.y(); j<hi.y(); j++) {
                  for (int k = low.z(); k<hi.z(); k++) {
                    internalHeatRate[IntVector(I,j,k)] +=
				value*(2.0*gvolume[IntVector(I,j,k)]/dx.x());
                   // for Fracture -------------------------------------- 
                   GinternalHeatRate[IntVector(I,j,k)] +=
                                value*(2.0*Gvolume[IntVector(I,j,k)]/dx.x());
                  }
                }
              }
              if(face==Patch::yplus || face==Patch::yminus){
                int J;
                if(face==Patch::yminus){ J=low.y(); }
                if(face==Patch::yplus){ J=hi.y()-1; }
                for (int i = low.x(); i<hi.x(); i++) {
                  for (int k = low.z(); k<hi.z(); k++) {
                    internalHeatRate[IntVector(i,J,k)] +=
				value*(2.0*gvolume[IntVector(i,J,k)]/dx.y());
                    // for Fracture -------------------------------------
                    GinternalHeatRate[IntVector(i,J,k)] +=
                                value*(2.0*Gvolume[IntVector(i,J,k)]/dx.y()); 
                  }
                }
              }
              if(face==Patch::zplus || face==Patch::zminus){
                int K;
                if(face==Patch::zminus){ K=low.z(); }
                if(face==Patch::zplus){ K=hi.z()-1; }
                for (int i = low.x(); i<hi.x(); i++) {
                  for (int j = low.y(); j<hi.y(); j++) {
                    internalHeatRate[IntVector(i,j,K)] +=
				value*(2.0*gvolume[IntVector(i,j,K)]/dx.z());
                   // for Fracture -------------------------------------
                    GinternalHeatRate[IntVector(i,j,K)] +=
                                value*(2.0*Gvolume[IntVector(i,j,K)]/dx.z());
                  }
                }
              }
            }
          }
        
      }

      // Create variables for the results
      // for primary field
      NCVariable<double> tempRate;
      new_dw->allocateAndPut(tempRate, lb->gTemperatureRateLabel, dwi, patch);
      tempRate.initialize(0.0);

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
          IntVector c = *iter;
	  tempRate[c] = (internalHeatRate[c]
		          +  externalHeatRate[c])/(mass[c] * specificHeat) + 
                             thermalContactHeatExchangeRate[c];
      } 

      // for Fracture -------------------------------------------------------
      NCVariable<double> GtempRate;
      new_dw->allocateAndPut(GtempRate,lb->GTemperatureRateLabel, dwi, patch);
      GtempRate.initialize(0.0);

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
          IntVector c = *iter;
          GtempRate[c] = (GinternalHeatRate[c]
                          +  GexternalHeatRate[c])/(Gmass[c] * specificHeat) +
                             GthermalContactHeatExchangeRate[c];
      }
      // ---------------------------------------------------------------------

    }
  }
  //time1=clock()-time0;
  //time1/=CLOCKS_PER_SEC;
  //cout << "***time for solveHeatEquations  = " << time1 << endl;

}


void FractureMPM::integrateAcceleration(const ProcessorGroup*,
				      const PatchSubset* patches,
				      const MaterialSubset*,
				      DataWarehouse* old_dw,
				      DataWarehouse* new_dw)
{
  //double time0, time1;
  //time0=clock();

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
      velocity_star.initialize(0.0);

      // for Fracture -----------------------------------------------------------
      constNCVariable<Vector>  Gacceleration, Gvelocity;
      new_dw->get(Gacceleration,lb->GAccelerationLabel,dwi, patch,Ghost::None,0); 
      new_dw->get(Gvelocity,    lb->GVelocityLabel,    dwi, patch,Ghost::None,0);

      NCVariable<Vector> Gvelocity_star;
      new_dw->allocateAndPut(Gvelocity_star,lb->GVelocityStarLabel, dwi, patch);
      Gvelocity_star.initialize(0.0);
      // ------------------------------------------------------------------------

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
        IntVector c = *iter;
	velocity_star[c] = velocity[c] + acceleration[c] * delT;
        Gvelocity_star[c]=Gvelocity[c] +Gacceleration[c] * delT; // for Fracture
      }
    }
  }
  //time1=clock()-time0;
  //time1/=CLOCKS_PER_SEC;
  //cout << "***time for integrateAcceleration  = " << time1 << endl;

}

void FractureMPM::integrateTemperatureRate(const ProcessorGroup*,
					 const PatchSubset* patches,
					 const MaterialSubset*,
					 DataWarehouse* old_dw,
					 DataWarehouse* new_dw)
{
  //double time0,time1;
  //time0=clock();

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing integrateTemperatureRate on patch " << patch->getID()
	       << "\t\t MPM"<< endl;

    Ghost::GhostType  gnone = Ghost::None;
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      constNCVariable<double> temperature;
      constNCVariable<double> temperatureRate;
      NCVariable<double> tempStar;
      delt_vartype delT;
 
      new_dw->get(temperature,     lb->gTemperatureLabel,    dwi,patch,gnone,0);
      new_dw->get(temperatureRate, lb->gTemperatureRateLabel,dwi,patch,gnone,0);

      old_dw->get(delT, d_sharedState->get_delt_label() );

      new_dw->allocateAndPut(tempStar, lb->gTemperatureStarLabel, dwi,patch);
      tempStar.initialize(0.0);

      // for Fracture ---------------------------------------------------------
      constNCVariable<double> Gtemperature;
      constNCVariable<double> GtemperatureRate;
      NCVariable<double> GtempStar;

      new_dw->get(Gtemperature,    lb->GTemperatureLabel,    dwi,patch,gnone,0);
      new_dw->get(GtemperatureRate,lb->GTemperatureRateLabel,dwi,patch,gnone,0);
    
      new_dw->allocateAndPut(GtempStar, lb->GTemperatureStarLabel, dwi,patch);
      GtempStar.initialize(0.0);
      // ----------------------------------------------------------------------

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
        IntVector c = *iter;
        tempStar[c] = temperature[c] + temperatureRate[c] * delT;
        GtempStar[c]=Gtemperature[c] +GtemperatureRate[c] * delT; //for Fracture
      }
    }
  }
  //time1=clock()-time0;
  //time1/=CLOCKS_PER_SEC;
  //cout << "***time for integrateTemperatureRate  = " << time1 << endl;
}

void FractureMPM::setGridBoundaryConditions(const ProcessorGroup*,
						const PatchSubset* patches,
						const MaterialSubset* ,
						DataWarehouse* old_dw,
						DataWarehouse* new_dw)
{
  //double time0,time1;
  //time0=clock();

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
      NCVariable<double> gTempRate;
      constNCVariable<double> gTemperatureNoBC;
      
      new_dw->getModifiable(gacceleration, lb->gAccelerationLabel,   dwi,patch);
      new_dw->getModifiable(gvelocity_star,lb->gVelocityStarLabel,   dwi,patch);
      new_dw->getModifiable(gTempRate,     lb->gTemperatureRateLabel,dwi,patch);
      new_dw->get(gTemperatureNoBC, lb->gTemperatureNoBCLabel,
                                                    dwi, patch, Ghost::None, 0);

      // for Fracture ----------------------------------------------------------
      NCVariable<Vector> Gvelocity_star, Gacceleration;
      NCVariable<double> GTempRate;
      constNCVariable<double> GTemperatureNoBC;

      new_dw->getModifiable(Gacceleration, lb->GAccelerationLabel,   dwi,patch);
      new_dw->getModifiable(Gvelocity_star,lb->GVelocityStarLabel,   dwi,patch);
      new_dw->getModifiable(GTempRate,     lb->GTemperatureRateLabel,dwi,patch);
      new_dw->get(GTemperatureNoBC, lb->GTemperatureNoBCLabel,
                                                    dwi, patch, Ghost::None, 0);
      // -----------------------------------------------------------------------

      // Apply grid boundary conditions to the velocity_star and
      // acceleration before interpolating back to the particles
      IntVector offset(0,0,0);
      for(Patch::FaceType face = Patch::startFace;
	  face <= Patch::endFace; face=Patch::nextFace(face)){
        const BoundCondBase *vel_bcs, *temp_bcs, *sym_bcs;
        if (patch->getBCType(face) == Patch::None) {
	   vel_bcs  = patch->getBCValues(dwi,"Velocity",   face);
	   temp_bcs = patch->getBCValues(dwi,"Temperature",face);
	   sym_bcs  = patch->getBCValues(dwi,"Symmetric",  face);
        } else
          continue;
         //__________________________________
         // Velocity and Acceleration
	  if (vel_bcs != 0) {
	    const VelocityBoundCond* bc = 
	      dynamic_cast<const VelocityBoundCond*>(vel_bcs);
	    //cout << "Velocity bc value = " << bc->getValue() << endl;
	    if (bc->getKind() == "Dirichlet") {
	      fillFace(gvelocity_star,patch, face,bc->getValue(),     offset);
	      fillFace(gacceleration, patch, face,Vector(0.0,0.0,0.0),offset);
              // for Fracture ------------------------------------------------
              fillFace(Gvelocity_star,patch, face,bc->getValue(),     offset);
              fillFace(Gacceleration, patch, face,Vector(0.0,0.0,0.0),offset);
              // -------------------------------------------------------------

	    }
	  }
	  if (sym_bcs != 0) {
	     fillFaceNormal(gvelocity_star,patch, face,offset);
	     fillFaceNormal(gacceleration, patch, face,offset);
             // for Fracture ------------------------------------------------
             fillFaceNormal(Gvelocity_star,patch, face,offset);
             fillFaceNormal(Gacceleration, patch, face,offset);
             // -------------------------------------------------------------

	  }
         //__________________________________
         // Temperature BC
	  if (temp_bcs != 0) {
	    const TemperatureBoundCond* bc = 
	      dynamic_cast<const TemperatureBoundCond*>(temp_bcs);
	    if (bc->getKind() == "Dirichlet") {
	      //cout << "Temperature bc value = " << bc->getValue() << endl;
              
            IntVector low = patch->getInteriorNodeLowIndex();
            IntVector hi  = patch->getInteriorNodeHighIndex();
	     double boundTemp = bc->getValue();
	     if(face==Patch::xplus || face==Patch::xminus){
		int I;
		if(face==Patch::xminus){ I=low.x(); }
		if(face==Patch::xplus){  I=hi.x()-1; }
		for (int j = low.y(); j<hi.y(); j++) { 
		  for (int k = low.z(); k<hi.z(); k++) {
		    gTempRate[IntVector(I,j,k)] +=
		      (boundTemp - gTemperatureNoBC[IntVector(I,j,k)])/delT;
                    // for Fracture ----------------------------------------
                    GTempRate[IntVector(I,j,k)] +=
                      (boundTemp - GTemperatureNoBC[IntVector(I,j,k)])/delT;
		  }
		}
	     }
	     if(face==Patch::yplus || face==Patch::yminus){
	       int J;
	       if(face==Patch::yminus){ J=low.y(); }
	       if(face==Patch::yplus){  J=hi.y()-1; }
	       for (int i = low.x(); i<hi.x(); i++) {
		  for (int k = low.z(); k<hi.z(); k++) {
		    gTempRate[IntVector(i,J,k)] +=
		      (boundTemp - gTemperatureNoBC[IntVector(i,J,k)])/delT;
                    // for Fracture ----------------------------------------
                    GTempRate[IntVector(i,J,k)] +=
                      (boundTemp - GTemperatureNoBC[IntVector(i,J,k)])/delT;
		  }
	       }
	     }
	     if(face==Patch::zplus || face==Patch::zminus){
	       int K;
	       if(face==Patch::zminus){ K=low.z(); }
	       if(face==Patch::zplus){  K=hi.z()-1; }
	       for (int i = low.x(); i<hi.x(); i++) {
		  for (int j = low.y(); j<hi.y(); j++) {
		    gTempRate[IntVector(i,j,K)] +=
		      (boundTemp - gTemperatureNoBC[IntVector(i,j,K)])/delT;
                    // for Fracture ----------------------------------------
                    GTempRate[IntVector(i,j,K)] +=
                      (boundTemp - GTemperatureNoBC[IntVector(i,j,K)])/delT;
		  }
	       }
	     }
	   }  // if(dirichlet)
	   if (bc->getKind() == "Neumann") {
	      //cout << "bc value = " << bc->getValue() << endl;
	   }
	 }  //if(temp_bc}
      }  // End loop over patch faces 

    } // End loop over matls
  }  // End loop over patches
  //time1=clock()-time0;
  //time1/=CLOCKS_PER_SEC;
  //cout << "***time for setGridBoundaryConditions  = " << time1 << endl;
}


void FractureMPM::interpolateToParticlesAndUpdate(const ProcessorGroup*,
						const PatchSubset* patches,
						const MaterialSubset* ,
						DataWarehouse* old_dw,
						DataWarehouse* new_dw)
{
 //double time0,time1;
 //time0=clock();

 for(int p=0;p<patches->size();p++){
   const Patch* patch = patches->get(p);
   cout_doing <<"Doing interpolateToParticlesAndUpdate on patch " 
       << patch->getID() << "\t MPM"<< endl;

   // Performs the interpolation from the cell vertices of the grid
   // acceleration and velocity to the particles to update their
   // velocity and position respectively
   Vector vel(0.0,0.0,0.0);
   Vector acc(0.0,0.0,0.0);
 
   // DON'T MOVE THESE!!!
   double thermal_energy = 0.0;
   Vector CMX(0.0,0.0,0.0);
   Vector CMV(0.0,0.0,0.0);
   double ke=0;
   double massLost=0;
   int numMPMMatls=d_sharedState->getNumMPMMatls();

   for(int m = 0; m < numMPMMatls; m++){
     MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
     int dwi = mpm_matl->getDWIndex();
     // Get the arrays of particle values to be changed
     constParticleVariable<Point> px,px0;
     ParticleVariable<Point> pxnew,px0new;
     constParticleVariable<Vector> pvelocity, pexternalForce,psize;
     ParticleVariable<Vector> pvelocitynew, pextForceNew,psizeNew;
     constParticleVariable<double> pmass, pvolume, pTemperature;
     ParticleVariable<double> pmassNew,pvolumeNew,pTempNew;
     constParticleVariable<long64> pids;
     ParticleVariable<long64> pids_new;
     ParticleVariable<int> keep_delete;

     // Get the arrays of grid data on which the new part. values depend
     constNCVariable<Vector> gvelocity_star, gacceleration;
     constNCVariable<double> gTemperatureRate, gTemperature, gTemperatureNoBC;
     constNCVariable<double> dTdt, massBurnFraction, frictionTempRate;

     delt_vartype delT;

     ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

     old_dw->get(px,                    lb->pXLabel,                     pset);
     old_dw->get(px0,                   lb->pX0Label,                    pset);//for Fracture
     old_dw->get(pmass,                 lb->pMassLabel,                  pset);
     new_dw->get(pvolume,               lb->pVolumeDeformedLabel,        pset);
     old_dw->get(pexternalForce,        lb->pExternalForceLabel,         pset);
     old_dw->get(pTemperature,          lb->pTemperatureLabel,           pset);
     old_dw->get(pvelocity,             lb->pVelocityLabel,              pset);
     old_dw->get(pids,                  lb->pParticleIDLabel,            pset);
     
     new_dw->allocateAndPut(pvelocitynew, lb->pVelocityLabel_preReloc,   pset);
     new_dw->allocateAndPut(pxnew,        lb->pXLabel_preReloc,          pset);
     new_dw->allocateAndPut(px0new,       lb->pX0Label_preReloc,         pset);//for Fracture
     new_dw->allocateAndPut(pmassNew,     lb->pMassLabel_preReloc,       pset);
     new_dw->allocateAndPut(pvolumeNew,   lb->pVolumeLabel_preReloc,     pset);
     new_dw->allocateAndPut(pids_new,     lb->pParticleIDLabel_preReloc, pset);
     new_dw->allocateAndPut(pTempNew,     lb->pTemperatureLabel_preReloc,pset);
     new_dw->allocateAndPut(pextForceNew, lb->pExtForceLabel_preReloc,
                                                                         pset);
     px0new.copyData(px0);//for Fracture
     pids_new.copyData(pids);
     pextForceNew.copyData(pexternalForce);
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

     // for Fracture -----------------------------------------------------------
     constParticleVariable<Short27> pgCode;
     new_dw->get(pgCode, lb->pgCodeLabel, pset);

     constNCVariable<Vector> Gvelocity_star, Gacceleration;
     constNCVariable<double> GTemperatureRate, GTemperature, GTemperatureNoBC;
     new_dw->get(Gvelocity_star,   lb->GVelocityStarLabel,   dwi,patch,gac,NGP);
     new_dw->get(Gacceleration,    lb->GAccelerationLabel,   dwi,patch,gac,NGP);
     new_dw->get(GTemperatureRate, lb->GTemperatureRateLabel,dwi,patch,gac,NGP);
     new_dw->get(GTemperature,     lb->GTemperatureLabel,    dwi,patch,gac,NGP);
     new_dw->get(GTemperatureNoBC, lb->GTemperatureNoBCLabel,dwi,patch,gac,NGP);
     // -------------------------------------------------------------------------

     if(d_with_ice){
      new_dw->get(dTdt,            lb->dTdt_NCLabel,         dwi,patch,gac,NGP);
      new_dw->get(massBurnFraction,lb->massBurnFractionLabel,dwi,patch,gac,NGP);
     }
     else{
      NCVariable<double> dTdt_create, massBurnFraction_create;	
      new_dw->allocateTemporary(dTdt_create,                     patch,gac,NGP);
      new_dw->allocateTemporary(massBurnFraction_create,         patch,gac,NGP);
      dTdt_create.initialize(0.);
      massBurnFraction_create.initialize(0.);
      dTdt = dTdt_create;                         // reference created data
      massBurnFraction = massBurnFraction_create; // reference created data
     }

     old_dw->get(delT, d_sharedState->get_delt_label() );

     double Cp=mpm_matl->getSpecificHeat();
     double rho_init=mpm_matl->getInitialDensity();

     IntVector ni[MAX_BASIS];
     double S[MAX_BASIS];
     Vector d_S[MAX_BASIS];

     for(ParticleSubset::iterator iter = pset->begin();
                                                  iter != pset->end(); iter++){
	  particleIndex idx = *iter;
            
          // Get the node indices that surround the cell
          if(d_8or27==8){
            patch->findCellAndWeightsAndShapeDerivatives(px[idx], ni, S, d_S);
          }
          else if(d_8or27==27){
            patch->findCellAndWeightsAndShapeDerivatives27(px[idx], ni, S, d_S,
                                                           psize[idx]);
          }
                
          vel = Vector(0.0,0.0,0.0);
          acc = Vector(0.0,0.0,0.0);
          double tempRate = 0;
          double burnFraction = 0;
                  
          // Accumulate the contribution from each surrounding vertex
          for (int k = 0; k < d_8or27; k++) {
              //for Fracture -----------------------------------------------------
              if(pgCode[idx][k]==1) {
                 vel      += gvelocity_star[ni[k]]  * S[k];
                 acc      += gacceleration[ni[k]]   * S[k];
                 tempRate += (gTemperatureRate[ni[k]] + dTdt[ni[k]] +
                              frictionTempRate[ni[k]])   * S[k];
                 burnFraction += massBurnFraction[ni[k]] * S[k];
              }
              else if(pgCode[idx][k]==2) {
                 vel      += Gvelocity_star[ni[k]]  * S[k];
                 acc      += Gacceleration[ni[k]]   * S[k];
                 tempRate += (GTemperatureRate[ni[k]] + dTdt[ni[k]] +
                              frictionTempRate[ni[k]])   * S[k];
                 burnFraction += massBurnFraction[ni[k]] * S[k];
              }
              else {
                 cout << "Unknown particle velocity field in "
                      << "FractureMPM::interpolateGridToParticleAndUpdate" << endl;
                 exit(1);
              }
              // ----------------------------------------------------------------
          }
            
          // Accumulate the contribution from each surrounding vertex
          // Update the particle's position and velocity
          pxnew[idx]           = px[idx] + vel * delT;
          pvelocitynew[idx]    = pvelocity[idx] + acc * delT;
          pTempNew[idx]        = pTemperature[idx] + tempRate * delT;
                    
          double rho;
	  if(pvolume[idx] > 0.){
	    rho = pmass[idx]/pvolume[idx];
	  }
	  else{
	    rho = rho_init;
	  }
          pmassNew[idx]        = Max(pmass[idx]*(1.    - burnFraction),0.);
          pvolumeNew[idx]      = pmassNew[idx]/rho;
          keep_delete[idx]     = 1;
#if 1                
	  if(pmassNew[idx] <= 3.e-15){
            keep_delete[idx]     = 0;
	  }
#endif          
                 
          thermal_energy += pTemperature[idx] * pmass[idx] * Cp;
          ke += .5*pmass[idx]*pvelocitynew[idx].length2();
          CMX = CMX + (pxnew[idx]*pmass[idx]).asVector();
          CMV += pvelocitynew[idx]*pmass[idx];
          massLost += (pmass[idx] - pmassNew[idx]);
     } // End loop over particles
      
   } // End loop over matls
   // DON'T MOVE THESE!!!
   new_dw->put(sum_vartype(ke),     lb->KineticEnergyLabel);
   new_dw->put(sumvec_vartype(CMX), lb->CenterOfMassPositionLabel);
   new_dw->put(sumvec_vartype(CMV), lb->CenterOfMassVelocityLabel);

// cout << "Solid mass lost this timestep = " << massLost << endl;
// cout << "Solid momentum after advection = " << CMV << endl;

// cout << "THERMAL ENERGY " << thermal_energy << endl;
 } // End loop over patches
  //time1=clock()-time0;
  //time1/=CLOCKS_PER_SEC;
  //cout << "***time for interpolateToParticlesAndUpdate  = " << time1 << endl;
}

void FractureMPM::setSharedState(SimulationStateP& ssp)
{
  d_sharedState = ssp;
}

