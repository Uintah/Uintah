// MPMArches.cc

#include <Packages/Uintah/CCA/Components/MPMArches/MPMArches.h>
#include <Core/Containers/StaticArray.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesMaterial.h>
#include <Packages/Uintah/CCA/Components/Arches/BoundaryCondition.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformation.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformationP.h>
#include <Packages/Uintah/CCA/Components/Arches/PicardNonlinearSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/TurbulenceModel.h>
#include <Packages/Uintah/CCA/Components/HETransformation/Burn.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/BoundaryCond.h>
#include <Packages/Uintah/CCA/Components/MPM/ThermalContact/ThermalContact.h>
#include <Packages/Uintah/CCA/Components/MPMArches/MPMArchesLabel.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/TemperatureBoundCond.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Grid/VelocityBoundCond.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

#include <Packages/Uintah/CCA/Components/MPMArches/fortran/collect_drag_cc_fort.h>
#include <Packages/Uintah/CCA/Components/MPMArches/fortran/interp_centertoface_fort.h>
#include <Packages/Uintah/CCA/Components/MPMArches/fortran/momentum_exchange_term_continuous_cc_fort.h>
#include <Packages/Uintah/CCA/Components/MPMArches/fortran/pressure_force_fort.h>

MPMArches::MPMArches(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld)
{
  Mlb  = scinew MPMLabel();
  d_fracture = false;
  d_MAlb = scinew MPMArchesLabel();
  d_mpm      = scinew SerialMPM(myworld);
  d_arches      = scinew Arches(myworld);
  d_SMALL_NUM = 1.e-100;
}

MPMArches::~MPMArches()
{
  delete d_MAlb;
  delete d_mpm;
  delete d_arches;
}

//______________________________________________________________________
//

void MPMArches::problemSetup(const ProblemSpecP& prob_spec, GridP& grid,
			  SimulationStateP& sharedState)
{
   d_sharedState = sharedState;

   // add for MPMArches, if any parameters are reqd
   d_mpm->setMPMLabel(Mlb);
   d_mpm->setWithArches();
   d_mpm->problemSetup(prob_spec, grid, d_sharedState);
   // set multimaterial label in Arches to access interface variables
   d_arches->setMPMArchesLabel(d_MAlb);
   d_Alab = d_arches->getArchesLabel();
   d_arches->problemSetup(prob_spec, grid, d_sharedState);
   
   cerr << "Done with problemSetup \t\t\t MPMArches" <<endl;
   cerr << "--------------------------------\n"<<endl;
}

//______________________________________________________________________
//

void MPMArches::scheduleInitialize(const LevelP& level,
				SchedulerP& sched)
{
  d_mpm->scheduleInitialize(      level, sched);

  const PatchSet* patches = level->eachPatch();
  const MaterialSet* arches_matls = d_sharedState->allArchesMaterials();
  const MaterialSet* mpm_matls = d_sharedState->allMPMMaterials();
  const MaterialSet* all_matls = d_sharedState->allMaterials();

  scheduleInterpolateParticlesToGrid(sched, patches, mpm_matls);
  scheduleInterpolateNCToCC(sched, patches, mpm_matls);
  scheduleComputeVoidFrac(sched, patches, arches_matls, mpm_matls, all_matls);

  d_arches->scheduleInitialize(      level, sched);

  cerr << "Doing Initialization \t\t\t MPMArches" <<endl;
  cerr << "--------------------------------\n"<<endl; 
}

//______________________________________________________________________
//

void MPMArches::scheduleInterpolateParticlesToGrid(SchedulerP& sched,
						   const PatchSet* patches,
						   const MaterialSet* matls)
{
  /* interpolateParticlesToGrid
   *   in(P.MASS, P.VELOCITY, P.NAT_X)
   *   operation(interpolate the P.MASS and P.VEL to the grid
   *             using P.NAT_X and some shape function evaluations)
   *   out(G.MASS, G.VELOCITY) */

  Task* t = scinew Task("MPMArches::interpolateParticlesToGrid",
			this,&MPMArches::interpolateParticlesToGrid);

  t->requires(Task::NewDW, Mlb->pMassLabel,          Ghost::AroundNodes,1);
  t->requires(Task::NewDW, Mlb->pVolumeLabel,        Ghost::AroundNodes,1);
  t->requires(Task::NewDW, Mlb->pXLabel,             Ghost::AroundNodes,1);
  t->requires(Task::NewDW, Mlb->pVelocityLabel,      Ghost::AroundNodes,1);
  //  t->requires(Task::NewDW, Mlb->pTemperatureLabel,   Ghost::AroundNodes,1);

  t->computes(Mlb->gMassLabel);
  t->computes(Mlb->gMassLabel, d_sharedState->getAllInOneMatl(),
	      Task::OutOfDomain);
  t->computes(Mlb->gVolumeLabel);
  t->computes(Mlb->gVelocityLabel);
  //  t->computes(Mlb->gTemperatureLabel);
  t->computes(Mlb->TotalMassLabel);

  sched->addTask(t, patches, matls);
}

//______________________________________________________________________
//

void MPMArches::scheduleComputeStableTimestep(const LevelP& level,
					   SchedulerP& sched)
{
  // Schedule computing the Arches stable timestep
  d_arches->scheduleComputeStableTimestep(level, sched);
  // MPM stable timestep is a by product of the CM
}

//______________________________________________________________________
//

void MPMArches::scheduleTimeAdvance(const LevelP&   level,
				    SchedulerP&     sched)
{
  const PatchSet* patches = level->eachPatch();
  const MaterialSet* arches_matls = d_sharedState->allArchesMaterials();
  const MaterialSet* mpm_matls = d_sharedState->allMPMMaterials();
  const MaterialSet* all_matls = d_sharedState->allMaterials();

  if(d_fracture) {
    d_mpm->scheduleSetPositions(sched, patches, mpm_matls);
    d_mpm->scheduleComputeBoundaryContact(sched, patches, mpm_matls);
    d_mpm->scheduleComputeConnectivity(sched, patches, mpm_matls);
  }
  d_mpm->scheduleInterpolateParticlesToGrid(sched, patches, mpm_matls);

  d_mpm->scheduleComputeHeatExchange(             sched, patches, mpm_matls);
  // interpolate mpm properties from node center to cell center
  // and subsequently to face center
  // these computed variables are used by void fraction and mom exchange

  scheduleInterpolateNCToCC(sched, patches, mpm_matls);
  scheduleInterpolateCCToFC(sched, patches, mpm_matls);

  scheduleComputeVoidFrac(sched, patches, arches_matls, mpm_matls, all_matls);

  // compute celltypeinit

  d_arches->getBoundaryCondition()->sched_mmWallCellTypeInit(sched,
						     patches, arches_matls);

  // for explicit calculation, exchange will be at the beginning

  scheduleMomExchange(sched, patches, arches_matls, mpm_matls, all_matls);

  schedulePutAllForcesOnCC(sched, patches, mpm_matls);
  schedulePutAllForcesOnNC(sched, patches, mpm_matls);

  // we also need mass and energy exchanges here.  These are
  // not implemented yet.  They will be of the form
  // scheduleEnergyExchange(level, sched, old_dw, new_dw)
  // scheduleMassExchange(level, sched, old_dw, new_dw)
  // (though, compare with mpm's scheduleComputeHeatExchange)

  // both Arches and MPM now have all the information they need
  // to proceed independently with their solution
  // Arches steps are identical with those in single-material code
  // once exchange terms are determined

  d_arches->scheduleTimeAdvance(level, sched);

  // remaining MPM steps are explicitly shown here.
  d_mpm->scheduleExMomInterpolated(sched, patches, mpm_matls);
  d_mpm->scheduleComputeStressTensor(sched, patches, mpm_matls);

  d_mpm->scheduleComputeInternalForce(sched, patches, mpm_matls);
  d_mpm->scheduleComputeInternalHeatRate(sched, patches, mpm_matls);
  d_mpm->scheduleSolveEquationsMotion(sched, patches, mpm_matls);
  d_mpm->scheduleSolveHeatEquations(sched, patches, mpm_matls);
  d_mpm->scheduleIntegrateAcceleration(sched, patches, mpm_matls);
  d_mpm->scheduleIntegrateTemperatureRate(sched, patches, mpm_matls);
  d_mpm->scheduleExMomIntegrated(sched, patches, mpm_matls);
  d_mpm->scheduleInterpolateToParticlesAndUpdate(sched, patches, mpm_matls);

  if(d_fracture) {
    d_mpm->scheduleComputeFracture(sched, patches, mpm_matls);
    d_mpm->scheduleComputeCrackExtension(sched, patches, mpm_matls);
  }
  d_mpm->scheduleCarryForwardVariables(sched, patches, mpm_matls);

  //int numMPMMatls = d_sharedState->getNumMPMMatls();

  sched->scheduleParticleRelocation(level, 
                                    Mlb->pXLabel_preReloc,
                                    Mlb->d_particleState_preReloc,
                                    Mlb->pXLabel, Mlb->d_particleState,
				    Mlb->pParticleIDLabel,
				    mpm_matls);


}

//______________________________________________________________________
//

void MPMArches::scheduleInterpolateNCToCC(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls)
{
   /* interpolateNCToCC */

    // primitive variable initialization
  Task* t=scinew Task("MPMArches::interpolateNCToCC",
		      this, &MPMArches::interpolateNCToCC);
  //int numMPMMatls = d_sharedState->getNumMPMMatls();
  int numGhostCells = 1;
  t->requires(Task::NewDW, Mlb->gMassLabel, 
	      Ghost::AroundCells, numGhostCells);
  t->requires(Task::NewDW, Mlb->gVolumeLabel,
	      Ghost::AroundCells, numGhostCells);
  t->requires(Task::NewDW, Mlb->gVelocityLabel,
	      Ghost::AroundCells, numGhostCells);

  t->computes(d_MAlb->cMassLabel);
  t->computes(d_MAlb->cVolumeLabel);
  t->computes(d_MAlb->vel_CCLabel);
  sched->addTask(t, patches, matls);
}

//______________________________________________________________________
//

void MPMArches::scheduleInterpolateCCToFC(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls)
{
  Task* t=scinew Task("MPMArches::interpolateCCToFC",
		      this, &MPMArches::interpolateCCToFC);
  int numGhostCells = 1;
  t->requires(Task::NewDW, d_MAlb->cMassLabel,
	      Ghost::AroundCells, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->cVolumeLabel,
	      Ghost::AroundCells, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->vel_CCLabel,
	      Ghost::AroundCells, numGhostCells);
  t->computes(d_MAlb->xvel_CCLabel);
  t->computes(d_MAlb->yvel_CCLabel);
  t->computes(d_MAlb->zvel_CCLabel);
  	      			  
  t->computes(d_MAlb->xvel_FCXLabel);
  t->computes(d_MAlb->xvel_FCYLabel);
  t->computes(d_MAlb->xvel_FCZLabel);
  	      			  
  t->computes(d_MAlb->yvel_FCXLabel);
  t->computes(d_MAlb->yvel_FCYLabel);
  t->computes(d_MAlb->yvel_FCZLabel);
  	      			  
  t->computes(d_MAlb->zvel_FCXLabel);
  t->computes(d_MAlb->zvel_FCYLabel);
  t->computes(d_MAlb->zvel_FCZLabel);
  
  sched->addTask(t, patches, matls);
}

//______________________________________________________________________
//

void MPMArches::scheduleComputeVoidFrac(SchedulerP& sched,
					const PatchSet* patches,
					const MaterialSet* arches_matls,
					const MaterialSet* mpm_matls,
					const MaterialSet* all_matls)
{
  // primitive variable initialization
  
  Task* t=scinew Task("MPMArches::computeVoidFrac",
		      this, &MPMArches::computeVoidFrac);

  int zeroGhostCells = 0;

  t->requires(Task::NewDW, d_MAlb->cVolumeLabel,   
	      mpm_matls->getUnion(), Ghost::None, zeroGhostCells);
  t->computes(d_MAlb->solid_fraction_CCLabel, mpm_matls->getUnion());
  t->computes(d_MAlb->void_frac_CCLabel, arches_matls->getUnion());
  sched->addTask(t, patches, all_matls);
}

//______________________________________________________________________
//

void MPMArches::scheduleMomExchange(SchedulerP& sched,
				    const PatchSet* patches,
				    const MaterialSet* arches_matls,
				    const MaterialSet* mpm_matls,
				    const MaterialSet* all_matls)

  // first step: su_drag and sp_drag for arches are calculated
  // at face centers and cell centers, using face-centered
  // solid velocities and cell-centered solid velocities, along
  // with cell-centered gas velocities.  In this step, 
  // pressure forces at face centers are also calculated.

{ 
    // primitive variable initialization
  Task* t=scinew Task("MPMArches::doMomExchange",
		      this, &MPMArches::doMomExchange);
  int numGhostCells = 1;
  int zeroGhostCells = 0;
  // requires, from mpm, solid velocities at cc, fcx, fcy, and fcz
  t->requires(Task::NewDW, d_MAlb->xvel_CCLabel, mpm_matls->getUnion(),
	      Ghost::None, zeroGhostCells);
  t->requires(Task::NewDW, d_MAlb->yvel_CCLabel, mpm_matls->getUnion(),
	      Ghost::None, zeroGhostCells);
  t->requires(Task::NewDW, d_MAlb->zvel_CCLabel, mpm_matls->getUnion(),
	      Ghost::None, zeroGhostCells);

  t->requires(Task::NewDW, d_MAlb->xvel_FCXLabel, mpm_matls->getUnion(),
	      Ghost::None, zeroGhostCells);
  t->requires(Task::NewDW, d_MAlb->yvel_FCYLabel, mpm_matls->getUnion(),
	      Ghost::None, zeroGhostCells);
  t->requires(Task::NewDW, d_MAlb->zvel_FCZLabel, mpm_matls->getUnion(),
	      Ghost::None, zeroGhostCells);

  t->requires(Task::NewDW, d_MAlb->xvel_FCYLabel, mpm_matls->getUnion(),
	      Ghost::None, zeroGhostCells);
  t->requires(Task::NewDW, d_MAlb->xvel_FCZLabel, mpm_matls->getUnion(),
	      Ghost::None, zeroGhostCells);

  t->requires(Task::NewDW, d_MAlb->yvel_FCZLabel, mpm_matls->getUnion(),
	      Ghost::None, zeroGhostCells);
  t->requires(Task::NewDW, d_MAlb->yvel_FCXLabel, mpm_matls->getUnion(),
	      Ghost::None, zeroGhostCells);

  t->requires(Task::NewDW, d_MAlb->zvel_FCXLabel, mpm_matls->getUnion(),
	      Ghost::None, zeroGhostCells);
  t->requires(Task::NewDW, d_MAlb->zvel_FCYLabel,    mpm_matls->getUnion(),
	      Ghost::None, zeroGhostCells);

  t->requires(Task::NewDW, d_MAlb->solid_fraction_CCLabel, mpm_matls->getUnion(),
	      Ghost::None, zeroGhostCells);

  // computes, for mpm, pressure forces and drag forces
  // at all face centers

  t->computes(d_MAlb->DragForceX_CCLabel, mpm_matls->getUnion());
  t->computes(d_MAlb->DragForceY_CCLabel, mpm_matls->getUnion());
  t->computes(d_MAlb->DragForceZ_CCLabel, mpm_matls->getUnion());
	      
  t->computes(d_MAlb->DragForceX_FCYLabel, mpm_matls->getUnion());
  t->computes(d_MAlb->DragForceX_FCZLabel, mpm_matls->getUnion());
  	      
  t->computes(d_MAlb->DragForceY_FCZLabel, mpm_matls->getUnion());
  t->computes(d_MAlb->DragForceY_FCXLabel, mpm_matls->getUnion());
  	      
  t->computes(d_MAlb->DragForceZ_FCXLabel, mpm_matls->getUnion());
  t->computes(d_MAlb->DragForceZ_FCYLabel, mpm_matls->getUnion());
  	      
  t->computes(d_MAlb->PressureForce_FCXLabel, mpm_matls->getUnion());
  t->computes(d_MAlb->PressureForce_FCYLabel, mpm_matls->getUnion());
  t->computes(d_MAlb->PressureForce_FCZLabel, mpm_matls->getUnion());


  // requires from Arches: celltype, pressure, velocity at cc.
  // also, from mpmarches, void fraction
  // use old_dw since using at the beginning of the time advance loop

  // use modified celltype
  t->requires(Task::NewDW, d_Alab->d_mmcellTypeLabel,      arches_matls->getUnion(), 
	      Ghost::AroundCells, numGhostCells);
  //  t->requires(Task::OldDW, d_Alab->d_pressureSPBCLabel,  arches_matls->getUnion(), 
  //      Ghost::AroundCells, numGhostCells);
  t->requires(Task::OldDW, d_Alab->d_pressPlusHydroLabel,  arches_matls->getUnion(), 
	      Ghost::AroundCells, numGhostCells);
  
  t->requires(Task::OldDW, d_Alab->d_newCCUVelocityLabel, arches_matls->getUnion(),
	      Ghost::AroundCells, numGhostCells);
  t->requires(Task::OldDW, d_Alab->d_newCCVVelocityLabel, arches_matls->getUnion(),
	      Ghost::AroundCells, numGhostCells);
  t->requires(Task::OldDW, d_Alab->d_newCCWVelocityLabel, arches_matls->getUnion(),
	      Ghost::AroundCells, numGhostCells);
  
  t->requires(Task::NewDW,  d_Alab->d_mmgasVolFracLabel,   arches_matls->getUnion(),
	      Ghost::AroundCells, numGhostCells);
  
  // computes, for arches, su_drag[x,y,z], sp_drag[x,y,z] at the
  // face centers and cell centers
  
  t->computes(d_MAlb->d_uVel_mmLinSrc_CCLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_uVel_mmLinSrc_FCYLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_uVel_mmLinSrc_FCZLabel, arches_matls->getUnion());
  	      
  t->computes(d_MAlb->d_uVel_mmNonlinSrc_CCLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_uVel_mmNonlinSrc_FCYLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_uVel_mmNonlinSrc_FCZLabel, arches_matls->getUnion());
  	      
  t->computes(d_MAlb->d_vVel_mmLinSrc_CCLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_vVel_mmLinSrc_FCZLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_vVel_mmLinSrc_FCXLabel, arches_matls->getUnion());
  	      
  t->computes(d_MAlb->d_vVel_mmNonlinSrc_CCLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_vVel_mmNonlinSrc_FCZLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_vVel_mmNonlinSrc_FCXLabel, arches_matls->getUnion());
  	      
  t->computes(d_MAlb->d_wVel_mmLinSrc_CCLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_wVel_mmLinSrc_FCXLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_wVel_mmLinSrc_FCYLabel, arches_matls->getUnion());
  	      
  t->computes(d_MAlb->d_wVel_mmNonlinSrc_CCLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_wVel_mmNonlinSrc_FCXLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_wVel_mmNonlinSrc_FCYLabel, arches_matls->getUnion());
  
  sched->addTask(t, patches, all_matls);


  // second step: interpolate/collect sources from previous step to 
  // cell-centered sources

  // primitive variable initialization
  t=scinew Task("MPMArches::collectToCCGasMomExchSrcs",
		this, &MPMArches::collectToCCGasMomExchSrcs);
  numGhostCells = 1;

  t->requires(Task::NewDW, d_MAlb->d_uVel_mmLinSrc_CCLabel, arches_matls->getUnion(),
	      Ghost::AroundCells, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->d_uVel_mmLinSrc_FCYLabel, arches_matls->getUnion(),
	      Ghost::AroundCells, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->d_uVel_mmLinSrc_FCZLabel, arches_matls->getUnion(),
	      Ghost::AroundCells, numGhostCells);
  
  t->requires(Task::NewDW, d_MAlb->d_uVel_mmNonlinSrc_CCLabel, arches_matls->getUnion(),
	      Ghost::AroundCells, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->d_uVel_mmNonlinSrc_FCYLabel, arches_matls->getUnion(),
	      Ghost::AroundCells, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->d_uVel_mmNonlinSrc_FCZLabel, arches_matls->getUnion(),
	      Ghost::AroundCells, numGhostCells);
  
  t->requires(Task::NewDW, d_MAlb->d_vVel_mmLinSrc_CCLabel, arches_matls->getUnion(),
	      Ghost::AroundCells, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->d_vVel_mmLinSrc_FCZLabel, arches_matls->getUnion(),
	      Ghost::AroundCells, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->d_vVel_mmLinSrc_FCXLabel, arches_matls->getUnion(),
	      Ghost::AroundCells, numGhostCells);
  
  t->requires(Task::NewDW, d_MAlb->d_vVel_mmNonlinSrc_CCLabel, arches_matls->getUnion(),
	      Ghost::AroundCells, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->d_vVel_mmNonlinSrc_FCZLabel, arches_matls->getUnion(),
	      Ghost::AroundCells, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->d_vVel_mmNonlinSrc_FCXLabel, arches_matls->getUnion(),
	      Ghost::AroundCells, numGhostCells);
  
  t->requires(Task::NewDW, d_MAlb->d_wVel_mmLinSrc_CCLabel, arches_matls->getUnion(),
	      Ghost::AroundCells, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->d_wVel_mmLinSrc_FCXLabel, arches_matls->getUnion(),
	      Ghost::AroundCells, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->d_wVel_mmLinSrc_FCYLabel, arches_matls->getUnion(),
	      Ghost::AroundCells, numGhostCells);
  
  t->requires(Task::NewDW, d_MAlb->d_wVel_mmNonlinSrc_CCLabel, arches_matls->getUnion(),
	      Ghost::AroundCells, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->d_wVel_mmNonlinSrc_FCXLabel, arches_matls->getUnion(),
	      Ghost::AroundCells, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->d_wVel_mmNonlinSrc_FCYLabel, arches_matls->getUnion(),
	      Ghost::AroundCells, numGhostCells);

   // computes 

  t->computes(d_MAlb->d_uVel_mmLinSrc_CC_CollectLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_uVel_mmNonlinSrc_CC_CollectLabel, arches_matls->getUnion());
  	      
  t->computes(d_MAlb->d_vVel_mmLinSrc_CC_CollectLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_vVel_mmNonlinSrc_CC_CollectLabel, arches_matls->getUnion());
  	      
  t->computes(d_MAlb->d_wVel_mmLinSrc_CC_CollectLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_wVel_mmNonlinSrc_CC_CollectLabel, arches_matls->getUnion());
  
  sched->addTask(t, patches, arches_matls);


 // third step: interpolates sources from previous step to 
 // SFC(X,Y,Z) source arrays that arches actually uses in the 
 // momentum equations

  // primitive variable initialization
  
  t=scinew Task("MPMArches::interpolateCCToFCGasMomExchSrcs",
		      this, &MPMArches::interpolateCCToFCGasMomExchSrcs);
  numGhostCells = 1;
  // requires

  t->requires(Task::NewDW, d_MAlb->d_uVel_mmLinSrc_CC_CollectLabel, 
	      arches_matls->getUnion(),
	      Ghost::AroundCells, numGhostCells);

  t->requires(Task::NewDW, d_MAlb->d_uVel_mmNonlinSrc_CC_CollectLabel, 
	      arches_matls->getUnion(),
	      Ghost::AroundCells, numGhostCells);

  t->requires(Task::NewDW, d_MAlb->d_vVel_mmLinSrc_CC_CollectLabel, arches_matls->getUnion(),
	      Ghost::AroundCells, numGhostCells);

  t->requires(Task::NewDW, d_MAlb->d_vVel_mmNonlinSrc_CC_CollectLabel,
	      arches_matls->getUnion(),
	      Ghost::AroundCells, numGhostCells);

  t->requires(Task::NewDW, d_MAlb->d_wVel_mmLinSrc_CC_CollectLabel, arches_matls->getUnion(),
	      Ghost::AroundCells, numGhostCells);
   
  t->requires(Task::NewDW, d_MAlb->d_wVel_mmNonlinSrc_CC_CollectLabel, 
	      arches_matls->getUnion(),
	      Ghost::AroundCells, numGhostCells);

  // computes 
  
  t->computes(d_MAlb->d_uVel_mmLinSrcLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_uVel_mmNonlinSrcLabel, arches_matls->getUnion());
  
  t->computes(d_MAlb->d_vVel_mmLinSrcLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_vVel_mmNonlinSrcLabel, arches_matls->getUnion());
  
  t->computes(d_MAlb->d_wVel_mmLinSrcLabel, arches_matls->getUnion());
  t->computes(d_MAlb->d_wVel_mmNonlinSrcLabel, arches_matls->getUnion());
  
  sched->addTask(t, patches, arches_matls);

#if 0

  // fourth step: redistributes the drag force calculated at the
  // cell-center (due to partially filled cells) to face centers
  // to supply to mpm
  
  // primitive variable initialization
  t=scinew Task("MPMArches::redistributeDragForceFromCCtoFC",
		      this, &MPMArches::redistributeDragForceFromCCtoFC);
  numGhostCells = 1;

  // redistributes the drag forces calculated at cell center to 
  // staggered face centers in the direction of flow
  t->requires(Task::NewDW, d_MAlb->DragForceX_CCLabel, mpm_matls->getUnion(),
	      Ghost::AroundCells, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->DragForceY_CCLabel, mpm_matls->getUnion(),
	      Ghost::AroundCells, numGhostCells);
  t->requires(Task::NewDW, d_MAlb->DragForceZ_CCLabel, mpm_matls->getUnion(),
	      Ghost::AroundCells, numGhostCells);

  //computes 

  t->computes(d_MAlb->DragForceX_FCXLabel, mpm_matls->getUnion());
  t->computes(d_MAlb->DragForceY_FCYLabel, mpm_matls->getUnion());
  t->computes(d_MAlb->DragForceZ_FCZLabel, mpm_matls->getUnion());
  sched->addTask(t, patches, mpm_matls);

#endif

}

//______________________________________________________________________
//

void MPMArches::schedulePutAllForcesOnCC(SchedulerP& sched,
				         const PatchSet* patches,
				         const MaterialSet* mpm_matls)
{
  // Grab all of the forces which Arches wants to give to MPM and
  // accumulate them on the cell centers
  Task* t=scinew Task("MPMArches::putAllForcesOnCC",
		      this, &MPMArches::putAllForcesOnCC);

  t->requires(Task::NewDW, d_MAlb->cMassLabel, Ghost::None, 0);

  t->requires(Task::NewDW, d_MAlb->DragForceX_CCLabel, mpm_matls->getUnion(),
								Ghost::None, 0);
  t->requires(Task::NewDW, d_MAlb->DragForceY_CCLabel, mpm_matls->getUnion(),
								Ghost::None, 0);
  t->requires(Task::NewDW, d_MAlb->DragForceZ_CCLabel, mpm_matls->getUnion(),
								Ghost::None, 0);
	      
  t->requires(Task::NewDW, d_MAlb->PressureForce_FCXLabel,mpm_matls->getUnion(),
						Ghost::AroundCells, 1);
  t->requires(Task::NewDW, d_MAlb->PressureForce_FCYLabel,mpm_matls->getUnion(),
						Ghost::AroundCells, 1);
  t->requires(Task::NewDW, d_MAlb->PressureForce_FCZLabel,mpm_matls->getUnion(),
						Ghost::AroundCells, 1);

  t->computes(d_MAlb->SumAllForcesCCLabel, mpm_matls->getUnion());
  t->computes(d_MAlb->AccArchesCCLabel,    mpm_matls->getUnion());

  sched->addTask(t, patches, mpm_matls);
}

//______________________________________________________________________
//

void MPMArches::schedulePutAllForcesOnNC(SchedulerP& sched,
				         const PatchSet* patches,
				         const MaterialSet* mpm_matls)
{
  // Take the cell centered forces from Arches and put them on the
  // nodes where SerialMPM can grab and use them
  Task* t=scinew Task("MPMArches::putAllForcesOnNC",
		      this, &MPMArches::putAllForcesOnNC);

  t->requires(Task::NewDW,d_MAlb->AccArchesCCLabel, mpm_matls->getUnion(),
					Ghost::AroundCells, 1);

  t->computes(d_MAlb->AccArchesNCLabel,             mpm_matls->getUnion());

  sched->addTask(t, patches, mpm_matls);
}

//______________________________________________________________________
//

void MPMArches::interpolateParticlesToGrid(const ProcessorGroup*,
					   const PatchSubset* patches,
					   const MaterialSubset* ,
					   DataWarehouse* old_dw,
					   DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    int numMatls = d_sharedState->getNumMPMMatls();

    NCVariable<double> gmassglobal;

    new_dw->allocate(gmassglobal,Mlb->gMassLabel,
		     d_sharedState->getAllInOneMatl()->get(0), patch);

    gmassglobal.initialize(d_mpm->d_SMALL_NUM_MPM);

    for(int m = 0; m < numMatls; m++){

      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();

      // Create arrays for the particle data

      constParticleVariable<Point>  px;
      constParticleVariable<double> pmass;
      constParticleVariable<double> pvolume;
      constParticleVariable<Vector> pvelocity;
      constParticleVariable<double> pTemperature;

      ParticleSubset* pset = new_dw->getParticleSubset(matlindex, patch,
						       Ghost::AroundNodes, 1,
						       Mlb->pXLabel);

      new_dw->get(px,             Mlb->pXLabel,             pset);
      new_dw->get(pmass,          Mlb->pMassLabel,          pset);
      new_dw->get(pvolume,        Mlb->pVolumeLabel,        pset);
      new_dw->get(pvelocity,      Mlb->pVelocityLabel,      pset);
      new_dw->get(pTemperature,   Mlb->pTemperatureLabel,   pset);

      // Create arrays for the grid data

      NCVariable<double> gmass;
      NCVariable<double> gvolume;
      NCVariable<Vector> gvelocity;
      NCVariable<double> gTemperature;

      new_dw->allocate(gmass,            Mlb->gMassLabel,        matlindex, patch);
      new_dw->allocate(gvolume,          Mlb->gVolumeLabel,      matlindex, patch);
      new_dw->allocate(gvelocity,        Mlb->gVelocityLabel,    matlindex, patch);
      new_dw->allocate(gTemperature,     Mlb->gTemperatureLabel, matlindex, patch);

      gmass.initialize(d_mpm->d_SMALL_NUM_MPM);
      gvolume.initialize(0);
      gvelocity.initialize(Vector(0,0,0));
      gTemperature.initialize(0);

      // Interpolate particle data to Grid data.
      // This currently consists of the particle velocity and mass
      // Need to compute the lumped global mass matrix and velocity
      // Vector from the individual mass matrix and velocity vector
      // GridMass * GridVelocity =  S^T*M_D*ParticleVelocity

      double totalmass = 0;
      Vector total_mom(0.0,0.0,0.0);

      for(ParticleSubset::iterator iter = pset->begin();
	  iter != pset->end(); iter++){

	particleIndex idx = *iter;

	// Get the node indices that surround the cell
	IntVector ni[8];
	double S[8];

	patch->findCellAndWeights(px[idx], ni, S);

	total_mom += pvelocity[idx]*pmass[idx];

	// Add each particles contribution to the local mass & velocity 
	// Must use the node indices

	for(int k = 0; k < 8; k++) {

	  if(patch->containsNode(ni[k])) {

	    gmassglobal[ni[k]]    += pmass[idx]                     * S[k];
	    gmass[ni[k]]          += pmass[idx]                     * S[k];
	    gvolume[ni[k]]        += pvolume[idx]                   * S[k];
	    gvelocity[ni[k]]      += pvelocity[idx]    * pmass[idx] * S[k];
	    gTemperature[ni[k]]   += pTemperature[idx] * pmass[idx] * S[k];
	    totalmass             += pmass[idx]                     * S[k];

	  }
        }
      }

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){

	gvelocity[*iter] /= gmass[*iter];
	gTemperature[*iter] /= gmass[*iter];

      }

      // Apply grid boundary conditions to the velocity before storing the data

      IntVector offset = 
	patch->getInteriorCellLowIndex() - patch->getCellLowIndex();
      // cout << "offset = " << offset << endl;

      for(Patch::FaceType face = Patch::startFace;
	  face <= Patch::endFace; face=Patch::nextFace(face)){

        BoundCondBase *vel_bcs, *temp_bcs, *sym_bcs;
        if (patch->getBCType(face) == Patch::None) {

	  vel_bcs  = patch->getBCValues(matlindex,"Velocity",face);
	  temp_bcs = patch->getBCValues(matlindex,"Temperature",face);
	  sym_bcs  = patch->getBCValues(matlindex,"Symmetric",face);

        } else

          continue;

	  if (vel_bcs != 0) {

	    VelocityBoundCond* bc = dynamic_cast<VelocityBoundCond*>(vel_bcs);
	    if (bc->getKind() == "Dirichlet") {

	      //cout << "Velocity bc value = " << bc->getValue() << endl;
	      fillFace(gvelocity,patch, face, bc->getValue(),offset);

	    }
	  }

	  if (sym_bcs != 0) {

	     fillFaceNormal(gvelocity,patch, face, offset);

	  }

	  if (temp_bcs != 0) {

	    TemperatureBoundCond* bc =
	      dynamic_cast<TemperatureBoundCond*>(temp_bcs);
	    if (bc->getKind() == "Dirichlet") {

	      fillFace(gTemperature,patch, face, bc->getValue(),offset);

	    }

	  }

      }

      new_dw->put(sum_vartype(totalmass), Mlb->TotalMassLabel);
      new_dw->put(gmass,         Mlb->gMassLabel,          matlindex, patch);
      new_dw->put(gvolume,       Mlb->gVolumeLabel,        matlindex, patch);
      new_dw->put(gvelocity,     Mlb->gVelocityLabel,      matlindex, patch);
      new_dw->put(gTemperature,  Mlb->gTemperatureLabel,   matlindex, patch);

    }  // End loop over materials
    new_dw->put(gmassglobal, Mlb->gMassLabel,
			d_sharedState->getAllInOneMatl()->get(0), patch);
    // End loop over patches
  }
}

//______________________________________________________________________
//

void MPMArches::interpolateNCToCC(const ProcessorGroup*,
				  const PatchSubset* patches,
				  const MaterialSubset* matls,
				  DataWarehouse* /*old_dw*/,
				  DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m=0;m<matls->size();m++){
      Vector zero(0.0,0.0,0.);
      int numGhostCells = 1;
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      // Create arrays for the grid data
      constNCVariable<double > gmass, gvolume;
      constNCVariable<Vector > gvelocity;
      CCVariable<double > cmass, cvolume;
      CCVariable<Vector > vel_CC;
      
      new_dw->allocate(cmass,     d_MAlb->cMassLabel,         
		       matlindex, patch);
      new_dw->allocate(cvolume,   d_MAlb->cVolumeLabel,       
		       matlindex, patch);
      new_dw->allocate(vel_CC,    d_MAlb->vel_CCLabel,        
		       matlindex, patch);
      
      cmass.initialize(0.);
      cvolume.initialize(0.);
      vel_CC.initialize(zero); 

      new_dw->get(gmass,     Mlb->gMassLabel,        matlindex, patch, 
		  Ghost::AroundCells, numGhostCells);
      new_dw->get(gvolume,   Mlb->gVolumeLabel,      matlindex, patch,
		  Ghost::AroundCells, numGhostCells);
      new_dw->get(gvelocity, Mlb->gVelocityLabel,    matlindex, patch,
		  Ghost::AroundCells, numGhostCells);
      
      IntVector nodeIdx[8];
      
      for (CellIterator iter =patch->getExtraCellIterator();
	   !iter.done();iter++){

	patch->findNodesFromCell(*iter,nodeIdx);
	for (int in=0;in<8;in++){
	  cmass[*iter]    += .125*gmass[nodeIdx[in]];
	  cvolume[*iter]  += .125*gvolume[nodeIdx[in]];
	  vel_CC[*iter]   += gvelocity[nodeIdx[in]]*.125*
	    gmass[nodeIdx[in]];
	}
	vel_CC[*iter]      /= (cmass[*iter]     + d_SMALL_NUM);
      }
      
      new_dw->put(cmass,     d_MAlb->cMassLabel,    matlindex, patch);
      new_dw->put(cvolume,   d_MAlb->cVolumeLabel,  matlindex, patch);
      new_dw->put(vel_CC,    d_MAlb->vel_CCLabel,   matlindex, patch);
    }
  }
}

//______________________________________________________________________
//

void MPMArches::interpolateCCToFC(const ProcessorGroup*,
				  const PatchSubset* patches,
				  const MaterialSubset* matls,
				  DataWarehouse* /*old_dw*/,
				  DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m=0;m<matls->size();m++){
      //Vector zero(0.0,0.0,0.0);
      int numGhostCells = 1;
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      
      constCCVariable<double > cmass;
      constCCVariable<Vector > vel_CC;
      
      CCVariable<double> xvel_CC;
      CCVariable<double> yvel_CC;
      CCVariable<double> zvel_CC;
      
      SFCXVariable<double> xvelFCX;
      SFCYVariable<double> xvelFCY;
      SFCZVariable<double> xvelFCZ;
      
      SFCXVariable<double> yvelFCX;
      SFCYVariable<double> yvelFCY;
      SFCZVariable<double> yvelFCZ;
      
      SFCXVariable<double> zvelFCX;
      SFCYVariable<double> zvelFCY;
      SFCZVariable<double> zvelFCZ;
      
      new_dw->get(cmass,    d_MAlb->cMassLabel,         matlindex, patch,
					Ghost::AroundCells, numGhostCells);
      new_dw->get(vel_CC,   d_MAlb->vel_CCLabel,        matlindex, patch,
		  Ghost::AroundCells, numGhostCells);
      
      new_dw->allocate(xvel_CC, d_MAlb->xvel_CCLabel,
		       matlindex, patch);
      new_dw->allocate(yvel_CC, d_MAlb->yvel_CCLabel,
		       matlindex, patch);
      new_dw->allocate(zvel_CC, d_MAlb->zvel_CCLabel,
		       matlindex, patch);
      
      new_dw->allocate(xvelFCX, d_MAlb->xvel_FCXLabel, 
		       matlindex, patch);
      new_dw->allocate(xvelFCY, d_MAlb->xvel_FCYLabel, 
		       matlindex, patch);
      new_dw->allocate(xvelFCZ, d_MAlb->xvel_FCZLabel, 
		       matlindex, patch);
      
      new_dw->allocate(yvelFCX, d_MAlb->yvel_FCXLabel, 
		       matlindex, patch);
      new_dw->allocate(yvelFCY, d_MAlb->yvel_FCYLabel, 
		       matlindex, patch);
      new_dw->allocate(yvelFCZ, d_MAlb->yvel_FCZLabel, 
		       matlindex, patch);
      
      new_dw->allocate(zvelFCX, d_MAlb->zvel_FCXLabel, 
		       matlindex, patch);
      new_dw->allocate(zvelFCY, d_MAlb->zvel_FCYLabel, 
		       matlindex, patch);
      new_dw->allocate(zvelFCZ, d_MAlb->zvel_FCZLabel, 
		       matlindex, patch);
      
      xvel_CC.initialize(0.);
      yvel_CC.initialize(0.);
      zvel_CC.initialize(0.);
      
      xvelFCX.initialize(0.);
      xvelFCY.initialize(0.);
      xvelFCZ.initialize(0.);
      
      yvelFCX.initialize(0.);
      yvelFCY.initialize(0.);
      yvelFCZ.initialize(0.);
      
      zvelFCX.initialize(0.);
      zvelFCY.initialize(0.);
      zvelFCZ.initialize(0.);
      
      double mass;
      
      for(CellIterator iter = patch->getExtraCellIterator();
	  !iter.done(); iter++){
	
	IntVector curcell = *iter;
	
	xvel_CC[curcell] = vel_CC[curcell].x();
	yvel_CC[curcell] = vel_CC[curcell].y();
	zvel_CC[curcell] = vel_CC[curcell].z();
	
	//___________________________________
	//   L E F T   F A C E S (FCX Values)
	
	if (curcell.x() >= (patch->getInteriorCellLowIndex()).x()) {
	  
	  IntVector adjcell(curcell.x()-1,curcell.y(),curcell.z());
	  mass = cmass[curcell] + cmass[adjcell];
	  
	  xvelFCX[curcell] = (vel_CC[curcell].x() * cmass[curcell] +
			      vel_CC[adjcell].x() * cmass[adjcell])/mass;
	  yvelFCX[curcell] = (vel_CC[curcell].y() * cmass[curcell] +
			      vel_CC[adjcell].y() * cmass[adjcell])/mass;
	  zvelFCX[curcell] = (vel_CC[curcell].z() * cmass[curcell] +
			      vel_CC[adjcell].z() * cmass[adjcell])/mass;
	  
	}
	//_____________________________________
	//   S O U T H   F A C E S (FCY Values)
	
	if (curcell.y() >= (patch->getInteriorCellLowIndex()).y()) {
	  
	  IntVector adjcell(curcell.x(),curcell.y()-1,curcell.z());
	  mass = cmass[curcell] + cmass[adjcell];
	  
	  xvelFCY[curcell] = (vel_CC[curcell].x() * cmass[curcell] +
			      vel_CC[adjcell].x() * cmass[adjcell])/mass;
	  yvelFCY[curcell] = (vel_CC[curcell].y() * cmass[curcell] +
			      vel_CC[adjcell].y() * cmass[adjcell])/mass;
	  zvelFCY[curcell] = (vel_CC[curcell].z() * cmass[curcell] +
			      vel_CC[adjcell].z() * cmass[adjcell])/mass;
	  
	}
	//_______________________________________
	//   B O T T O M   F A C E S (FCZ Values)
	
	if (curcell.z() >= (patch->getInteriorCellLowIndex()).z()) {
	  
	  IntVector adjcell(curcell.x(),curcell.y(),curcell.z()-1);
	  mass = cmass[curcell] + cmass[adjcell];
	  
	  xvelFCZ[curcell] = (vel_CC[curcell].x() * cmass[curcell] +
			      vel_CC[adjcell].x() * cmass[adjcell])/mass;
	  yvelFCZ[curcell] = (vel_CC[curcell].y() * cmass[curcell] +
			      vel_CC[adjcell].y() * cmass[adjcell])/mass;
	  zvelFCZ[curcell] = (vel_CC[curcell].z() * cmass[curcell] +
			      vel_CC[adjcell].z() * cmass[adjcell])/mass;
	  
	}
      }
      
      new_dw->put(xvel_CC, d_MAlb->xvel_CCLabel, matlindex, patch);
      new_dw->put(yvel_CC, d_MAlb->yvel_CCLabel, matlindex, patch);
      new_dw->put(zvel_CC, d_MAlb->zvel_CCLabel, matlindex, patch);
      
      new_dw->put(xvelFCX, d_MAlb->xvel_FCXLabel, matlindex, patch);
      new_dw->put(yvelFCX, d_MAlb->yvel_FCXLabel, matlindex, patch);
      new_dw->put(zvelFCX, d_MAlb->zvel_FCXLabel, matlindex, patch);
      
      new_dw->put(xvelFCY, d_MAlb->xvel_FCYLabel, matlindex, patch);
      new_dw->put(yvelFCY, d_MAlb->yvel_FCYLabel, matlindex, patch);
      new_dw->put(zvelFCY, d_MAlb->zvel_FCYLabel, matlindex, patch);
      
      new_dw->put(xvelFCZ, d_MAlb->xvel_FCZLabel, matlindex, patch);
      new_dw->put(yvelFCZ, d_MAlb->yvel_FCZLabel, matlindex, patch);
      new_dw->put(zvelFCZ, d_MAlb->zvel_FCZLabel, matlindex, patch);
      
    }
  }
}

//______________________________________________________________________
//

void MPMArches::computeVoidFrac(const ProcessorGroup*,
				const PatchSubset* patches,
				const MaterialSubset*,
				DataWarehouse* /*old_dw*/,
				DataWarehouse* new_dw) 

{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlindex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    int numMPMMatls = d_sharedState->getNumMPMMatls();
    StaticArray<constCCVariable<double> > mat_vol(numMPMMatls);
    StaticArray<CCVariable<double> > solid_fraction_cc(numMPMMatls);
    
    int zeroGhostCells = 0;

  // get and allocate

    for (int m = 0; m < numMPMMatls; m++) {

      Material* matl = d_sharedState->getMPMMaterial( m );
      int dwindex = matl->getDWIndex();

      new_dw->get(mat_vol[m], d_MAlb->cVolumeLabel,
		  dwindex, patch, Ghost::None, zeroGhostCells);		

      new_dw->allocate(solid_fraction_cc[m], d_MAlb->solid_fraction_CCLabel,
		       dwindex, patch);

      solid_fraction_cc[m].initialize(0);
    }

    CCVariable<double> void_frac;
    new_dw->allocate(void_frac, d_MAlb->void_frac_CCLabel, 
		     matlindex, patch); 

    // actual computation

    void_frac.initialize(0);
    
    for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++) {

      double total_vol = patch->dCell().x()*patch->dCell().y()*patch->dCell().z();
      double solid_frac_sum = 0.0;
      for (int m = 0; m < numMPMMatls; m++) {
	
	solid_fraction_cc[m][*iter] = mat_vol[m][*iter]/total_vol;
	solid_frac_sum += solid_fraction_cc[m][*iter];
	
      }
      void_frac[*iter] = 1.0 - solid_frac_sum;
      //      void_frac[*iter] = 1.0;
    }

    // Computation done; now put back in dw

    for (int m = 0; m < numMPMMatls; m++) {
      Material* matl = d_sharedState->getMPMMaterial( m );
      int dwindex = matl->getDWIndex();
      
      new_dw->put(solid_fraction_cc[m], d_MAlb->solid_fraction_CCLabel,
		  dwindex, patch);
    }
    
    new_dw->put(void_frac, d_MAlb->void_frac_CCLabel, matlindex, patch);

  }
}
  
//______________________________________________________________________
//

void MPMArches::doMomExchange(const ProcessorGroup*,
			      const PatchSubset* patches,
			      const MaterialSubset*,
			      DataWarehouse* old_dw,
			      DataWarehouse* new_dw)

{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    int numMPMMatls  = d_sharedState->getNumMPMMatls();

  // MPM stuff

    StaticArray<constCCVariable<double> > solid_fraction_cc(numMPMMatls);

    StaticArray<constCCVariable<double> > xvelCC_solid(numMPMMatls);
    StaticArray<constCCVariable<double> > yvelCC_solid(numMPMMatls);
    StaticArray<constCCVariable<double> > zvelCC_solid(numMPMMatls);
    
    StaticArray<constSFCXVariable<double> > xvelFCX_solid(numMPMMatls);
    StaticArray<constSFCXVariable<double> > yvelFCX_solid(numMPMMatls);
    StaticArray<constSFCXVariable<double> > zvelFCX_solid(numMPMMatls);
    
    StaticArray<constSFCYVariable<double> > xvelFCY_solid(numMPMMatls);
    StaticArray<constSFCYVariable<double> > yvelFCY_solid(numMPMMatls);
    StaticArray<constSFCYVariable<double> > zvelFCY_solid(numMPMMatls);
    
    StaticArray<constSFCZVariable<double> > xvelFCZ_solid(numMPMMatls);
    StaticArray<constSFCZVariable<double> > yvelFCZ_solid(numMPMMatls);
    StaticArray<constSFCZVariable<double> > zvelFCZ_solid(numMPMMatls);
    
    StaticArray<CCVariable<double> >   dragForceX_cc(numMPMMatls);
    StaticArray<CCVariable<double> >   dragForceY_cc(numMPMMatls);
    StaticArray<CCVariable<double> >   dragForceZ_cc(numMPMMatls);
    
    StaticArray<SFCYVariable<double> > dragForceX_fcy(numMPMMatls);
    StaticArray<SFCZVariable<double> > dragForceX_fcz(numMPMMatls);
    
    StaticArray<SFCZVariable<double> > dragForceY_fcz(numMPMMatls);
    StaticArray<SFCXVariable<double> > dragForceY_fcx(numMPMMatls);
    
    StaticArray<SFCXVariable<double> > dragForceZ_fcx(numMPMMatls);
    StaticArray<SFCYVariable<double> > dragForceZ_fcy(numMPMMatls);
    
    StaticArray<SFCXVariable<double> > pressForceX(numMPMMatls);
    StaticArray<SFCYVariable<double> > pressForceY(numMPMMatls);
    StaticArray<SFCZVariable<double> > pressForceZ(numMPMMatls);
    
    // Arches stuff
    
    constCCVariable<int> cellType;
    constCCVariable<double> pressure;
    
    constCCVariable<double> xvelCC_gas;
    constCCVariable<double> yvelCC_gas;
    constCCVariable<double> zvelCC_gas;
    
    constCCVariable<double> gas_fraction_cc;
    
    // multimaterial contribution to SP and SU terms in Arches momentum eqns
    // currently at cc and fcs.  Later we will interpolate to where
    // Arches wants them.
    
    CCVariable<double> uVelLinearSrc_cc; 
    SFCYVariable<double> uVelLinearSrc_fcy; 
    SFCZVariable<double> uVelLinearSrc_fcz; 
    
    CCVariable<double> uVelNonlinearSrc_cc;
    SFCYVariable<double> uVelNonlinearSrc_fcy;
    SFCZVariable<double> uVelNonlinearSrc_fcz;
    
    CCVariable<double> vVelLinearSrc_cc; 
    SFCZVariable<double> vVelLinearSrc_fcz; 
    SFCXVariable<double> vVelLinearSrc_fcx; 
    
    CCVariable<double> vVelNonlinearSrc_cc;
    SFCZVariable<double> vVelNonlinearSrc_fcz;
    SFCXVariable<double> vVelNonlinearSrc_fcx;
    
    CCVariable<double> wVelLinearSrc_cc; 
    SFCXVariable<double> wVelLinearSrc_fcx; 
    SFCYVariable<double> wVelLinearSrc_fcy; 
    
    CCVariable<double> wVelNonlinearSrc_cc; 
    SFCXVariable<double> wVelNonlinearSrc_fcx; 
    SFCYVariable<double> wVelNonlinearSrc_fcy; 
    
    int numGhostCells = 0;
    
    int numGhostCellsG = 1;

    new_dw->get(cellType, d_Alab->d_mmcellTypeLabel,          matlIndex, 
		patch, Ghost::AroundCells, numGhostCellsG);
    //    old_dw->get(pressure, d_Alab->d_pressureSPBCLabel,        matlIndex, 
    //		patch, Ghost::AroundCells, numGhostCellsG);
    old_dw->get(pressure, d_Alab->d_pressPlusHydroLabel,        matlIndex, 
		patch, Ghost::AroundCells, numGhostCellsG);

    old_dw->get(xvelCC_gas, d_Alab->d_newCCUVelocityLabel,   matlIndex, 
		patch, Ghost::AroundCells, numGhostCellsG);
    old_dw->get(yvelCC_gas, d_Alab->d_newCCVVelocityLabel,   matlIndex, 
		patch, Ghost::AroundCells, numGhostCellsG);
    old_dw->get(zvelCC_gas, d_Alab->d_newCCWVelocityLabel,   matlIndex, 
		patch, Ghost::AroundCells, numGhostCellsG);

    new_dw->get(gas_fraction_cc, d_Alab->d_mmgasVolFracLabel, matlIndex, 
		patch, Ghost::AroundCells, numGhostCellsG);

    // patch geometry information
    
    PerPatch<CellInformationP> cellInfoP;
    if (new_dw->exists(d_Alab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_Alab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_Alab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    // computes su_drag[x,y,z], sp_drag[x,y,z] for arches at cell centers
    // and face centers
    
    new_dw->allocate(uVelLinearSrc_cc, d_MAlb->d_uVel_mmLinSrc_CCLabel, 
		     matlIndex, patch);
    new_dw->allocate(uVelLinearSrc_fcy, d_MAlb->d_uVel_mmLinSrc_FCYLabel, 
		     matlIndex, patch);
    new_dw->allocate(uVelLinearSrc_fcz, d_MAlb->d_uVel_mmLinSrc_FCZLabel, 
		     matlIndex, patch);

    new_dw->allocate(uVelNonlinearSrc_cc, d_MAlb->d_uVel_mmNonlinSrc_CCLabel,
		     matlIndex, patch);
    new_dw->allocate(uVelNonlinearSrc_fcy, d_MAlb->d_uVel_mmNonlinSrc_FCYLabel,
		     matlIndex, patch);
    new_dw->allocate(uVelNonlinearSrc_fcz, d_MAlb->d_uVel_mmNonlinSrc_FCZLabel,
		     matlIndex, patch);
    
    new_dw->allocate(vVelLinearSrc_cc, d_MAlb->d_vVel_mmLinSrc_CCLabel, 
		     matlIndex, patch);
    new_dw->allocate(vVelLinearSrc_fcz, d_MAlb->d_vVel_mmLinSrc_FCZLabel, 
		     matlIndex, patch);
    new_dw->allocate(vVelLinearSrc_fcx, d_MAlb->d_vVel_mmLinSrc_FCXLabel, 
		     matlIndex, patch);
    
    new_dw->allocate(vVelNonlinearSrc_cc, d_MAlb->d_vVel_mmNonlinSrc_CCLabel,
		     matlIndex, patch);
    new_dw->allocate(vVelNonlinearSrc_fcz, d_MAlb->d_vVel_mmNonlinSrc_FCZLabel,
		     matlIndex, patch);
    new_dw->allocate(vVelNonlinearSrc_fcx, d_MAlb->d_vVel_mmNonlinSrc_FCXLabel,
		     matlIndex, patch);
    
    new_dw->allocate(wVelLinearSrc_cc, d_MAlb->d_wVel_mmLinSrc_CCLabel, 
		     matlIndex, patch);
    new_dw->allocate(wVelLinearSrc_fcx, d_MAlb->d_wVel_mmLinSrc_FCXLabel, 
		     matlIndex, patch);
    new_dw->allocate(wVelLinearSrc_fcy, d_MAlb->d_wVel_mmLinSrc_FCYLabel, 
		     matlIndex, patch);
    
    new_dw->allocate(wVelNonlinearSrc_cc, d_MAlb->d_wVel_mmNonlinSrc_CCLabel,
		     matlIndex, patch);
    new_dw->allocate(wVelNonlinearSrc_fcx, d_MAlb->d_wVel_mmNonlinSrc_FCXLabel,
		     matlIndex, patch);
    new_dw->allocate(wVelNonlinearSrc_fcy, d_MAlb->d_wVel_mmNonlinSrc_FCYLabel,
		     matlIndex, patch);
    
    uVelLinearSrc_cc.initialize(0.);
    uVelLinearSrc_fcy.initialize(0.);
    uVelLinearSrc_fcz.initialize(0.);
    
    uVelNonlinearSrc_cc.initialize(0.);
    uVelNonlinearSrc_fcy.initialize(0.);
    uVelNonlinearSrc_fcz.initialize(0.);
    
    vVelLinearSrc_cc.initialize(0.);
    vVelLinearSrc_fcz.initialize(0.);
    vVelLinearSrc_fcx.initialize(0.);
    
    vVelNonlinearSrc_cc.initialize(0.);
    vVelNonlinearSrc_fcz.initialize(0.);
    vVelNonlinearSrc_fcx.initialize(0.);
    
    wVelLinearSrc_cc.initialize(0.);
    wVelLinearSrc_fcx.initialize(0.);
    wVelLinearSrc_fcy.initialize(0.);
    
    wVelNonlinearSrc_cc.initialize(0.);
    wVelNonlinearSrc_fcx.initialize(0.);
    wVelNonlinearSrc_fcy.initialize(0.);
    
    for (int m = 0; m < numMPMMatls; m++) {

      Material* matl = d_sharedState->getMPMMaterial( m );
      int idx = matl->getDWIndex();

      new_dw->get(solid_fraction_cc[m], d_MAlb->solid_fraction_CCLabel,
		  idx, patch, Ghost::None, numGhostCells);
      
      new_dw->get(xvelCC_solid[m], d_MAlb->xvel_CCLabel, 
		  idx, patch, Ghost::None, numGhostCells);
      
      new_dw->get(yvelCC_solid[m], d_MAlb->yvel_CCLabel, 
		  idx, patch, Ghost::None, numGhostCells);
      
      new_dw->get(zvelCC_solid[m], d_MAlb->zvel_CCLabel, 
		  idx, patch, Ghost::None, numGhostCells);
      
      new_dw->get(xvelFCX_solid[m], d_MAlb->xvel_FCXLabel, 
		  idx, patch, Ghost::None, numGhostCells);
      
      new_dw->get(yvelFCX_solid[m], d_MAlb->yvel_FCXLabel, 
		  idx, patch, Ghost::None, numGhostCells);
      
      new_dw->get(zvelFCX_solid[m], d_MAlb->zvel_FCXLabel, 
		  idx, patch, Ghost::None, numGhostCells);
      
      new_dw->get(xvelFCY_solid[m], d_MAlb->xvel_FCYLabel, 
		  idx, patch, Ghost::None, numGhostCells);
      
      new_dw->get(yvelFCY_solid[m], d_MAlb->yvel_FCYLabel, 
		  idx, patch, Ghost::None, numGhostCells);
      
      new_dw->get(zvelFCY_solid[m], d_MAlb->zvel_FCYLabel, 
		  idx, patch, Ghost::None, numGhostCells);
      
      new_dw->get(xvelFCZ_solid[m], d_MAlb->xvel_FCZLabel, 
		  idx, patch, Ghost::None, numGhostCells);
      
      new_dw->get(yvelFCZ_solid[m], d_MAlb->yvel_FCZLabel, 
		  idx, patch, Ghost::None, numGhostCells);
      
      new_dw->get(zvelFCZ_solid[m], d_MAlb->zvel_FCZLabel, 
		  idx, patch, Ghost::None, numGhostCells);
      
      new_dw->allocate(dragForceX_cc[m], d_MAlb->DragForceX_CCLabel,
		       idx, patch);
      dragForceX_cc[m].initialize(0.);
      
      new_dw->allocate(dragForceY_cc[m], d_MAlb->DragForceY_CCLabel,
		       idx, patch);
      dragForceY_cc[m].initialize(0.);
      
      new_dw->allocate(dragForceZ_cc[m], d_MAlb->DragForceZ_CCLabel,
		       idx, patch);
      dragForceZ_cc[m].initialize(0.);
      
      new_dw->allocate(dragForceX_fcy[m], d_MAlb->DragForceX_FCYLabel, 
		       idx, patch);
      dragForceX_fcy[m].initialize(0.);
      
      new_dw->allocate(dragForceX_fcz[m], d_MAlb->DragForceX_FCZLabel, 
		       idx, patch);
      dragForceX_fcz[m].initialize(0.);
      
      new_dw->allocate(dragForceY_fcz[m], d_MAlb->DragForceY_FCZLabel, 
		       idx, patch);
      dragForceY_fcz[m].initialize(0.);
      
      new_dw->allocate(dragForceY_fcx[m], d_MAlb->DragForceY_FCXLabel, 
		       idx, patch);
      dragForceY_fcx[m].initialize(0.);
      
      new_dw->allocate(dragForceZ_fcx[m], d_MAlb->DragForceZ_FCXLabel, 
		       idx, patch);
      dragForceZ_fcx[m].initialize(0.);
      
      new_dw->allocate(dragForceZ_fcy[m], d_MAlb->DragForceZ_FCYLabel, 
		       idx, patch);
      dragForceZ_fcy[m].initialize(0.);
      
      new_dw->allocate(pressForceX[m], d_MAlb->PressureForce_FCXLabel,
		       idx, patch);
      pressForceX[m].initialize(0.);
      
      new_dw->allocate(pressForceY[m], d_MAlb->PressureForce_FCYLabel,
		       idx, patch);
      pressForceY[m].initialize(0.);
      
      new_dw->allocate(pressForceZ[m], d_MAlb->PressureForce_FCZLabel,
		       idx, patch);
      pressForceZ[m].initialize(0.);

    }

    // Begin loop to calculate gas-solid exchange terms for each
    // solid material with the gas phase
    
    int ffieldid = d_arches->getBoundaryCondition()->getFlowId();
    int mmwallid = d_arches->getBoundaryCondition()->getMMWallId();
    
    double viscos = d_arches->getTurbulenceModel()->getMolecularViscosity();
    double csmag = d_arches->getTurbulenceModel()->getSmagorinskyConst();

//    IntVector dim_lo = cellType.getFortLowIndex();
//    IntVector dim_hi = cellType.getFortHighIndex();
    
//    IntVector dim_lo_eps = gas_fraction_cc.getFortLowIndex();
//    IntVector dim_hi_eps = gas_fraction_cc.getFortHighIndex();
    
    IntVector dim_lo_ugc;
    IntVector dim_hi_ugc;
    
    IntVector dim_lo_dx_cc;
    IntVector dim_hi_dx_cc;
    
    IntVector valid_lo;
    IntVector valid_hi;
    
    for (int m = 0; m < numMPMMatls; m++) {
      
//      IntVector dim_lo_upc = xvelCC_solid[m].getFortLowIndex();
//      IntVector dim_hi_upc = xvelCC_solid[m].getFortHighIndex();
      
//      IntVector dim_lo_epss = solid_fraction_cc[m].getFortLowIndex();
//      IntVector dim_hi_epss = solid_fraction_cc[m].getFortHighIndex();
      
      // code for x-direction momentum exchange
      
      dim_lo_ugc = xvelCC_gas.getFortLowIndex();
      dim_hi_ugc = xvelCC_gas.getFortHighIndex();
      
      dim_lo_dx_cc = dragForceX_cc[m].getFortLowIndex();
      dim_hi_dx_cc = dragForceX_cc[m].getFortHighIndex();
      
      valid_lo = patch->getSFCXFORTLowIndex();
      valid_hi = patch->getSFCXFORTHighIndex();
      
      // gas variables
      
//      IntVector xdim_lo_su_fcy = uVelNonlinearSrc_fcy.getFortLowIndex();
//      IntVector xdim_hi_su_fcy = uVelNonlinearSrc_fcy.getFortHighIndex();
      
//      IntVector xdim_lo_sp_fcy = uVelLinearSrc_fcy.getFortLowIndex();
//      IntVector xdim_hi_sp_fcy = uVelLinearSrc_fcy.getFortHighIndex();
      
//      IntVector xdim_lo_su_fcz = uVelNonlinearSrc_fcz.getFortLowIndex();
//      IntVector xdim_hi_su_fcz = uVelNonlinearSrc_fcz.getFortHighIndex();
      
//      IntVector xdim_lo_sp_fcz = uVelLinearSrc_fcz.getFortLowIndex();
//      IntVector xdim_hi_sp_fcz = uVelLinearSrc_fcz.getFortHighIndex();
      
//      IntVector xdim_lo_su_cc = uVelNonlinearSrc_cc.getFortLowIndex();
//      IntVector xdim_hi_su_cc = uVelNonlinearSrc_cc.getFortHighIndex();
      
//      IntVector xdim_lo_sp_cc = uVelLinearSrc_cc.getFortLowIndex();
//      IntVector xdim_hi_sp_cc = uVelLinearSrc_cc.getFortHighIndex();
      
      int ioff = 1;
      int joff = 0;
      int koff = 0;
      
      int indexflo = 1;
      int indext1 =  2;
      int indext2 =  3;
      
      // solid material variables
      
//      IntVector xdim_lo_dx_fcy = dragForceX_fcy[m].getFortLowIndex();
//      IntVector xdim_hi_dx_fcy = dragForceX_fcy[m].getFortHighIndex();
      
//      IntVector xdim_lo_dx_fcz = dragForceX_fcz[m].getFortLowIndex();
//      IntVector xdim_hi_dx_fcz = dragForceX_fcz[m].getFortHighIndex();
      
//      IntVector xdim_lo_upy = xvelFCY_solid[m].getFortLowIndex();
//      IntVector xdim_hi_upy = xvelFCY_solid[m].getFortHighIndex();
      
//      IntVector xdim_lo_upz = xvelFCZ_solid[m].getFortLowIndex();
//      IntVector xdim_hi_upz = xvelFCZ_solid[m].getFortHighIndex();
      
      fort_momentum_exchange_term_continuous_cc(uVelNonlinearSrc_fcy,
						uVelLinearSrc_fcy,
						uVelNonlinearSrc_fcz,
						uVelLinearSrc_fcz,
						uVelNonlinearSrc_cc,
						uVelLinearSrc_cc,
						dragForceX_fcy[m],
						dragForceX_fcz[m],
						dragForceX_cc[m],
						xvelCC_gas,
						xvelCC_solid[m],
						xvelFCY_solid[m],
						xvelFCZ_solid[m],
						gas_fraction_cc,
						solid_fraction_cc[m],
						viscos, csmag,
						cellinfo->sew, cellinfo->sns,
						cellinfo->stb, cellinfo->yy,
						cellinfo->zz, cellinfo->yv,
						cellinfo->zw,
						valid_lo, valid_hi,
						ioff, joff, koff,
						indexflo, indext1, indext2,
						cellType, mmwallid, ffieldid);
      
      // code for y-direction momentum exchange
      
      dim_lo_ugc = yvelCC_gas.getFortLowIndex();
      dim_hi_ugc = yvelCC_gas.getFortHighIndex();
      
      dim_lo_dx_cc = dragForceY_cc[m].getFortLowIndex();
      dim_hi_dx_cc = dragForceY_cc[m].getFortHighIndex();
      
      valid_lo = patch->getSFCYFORTLowIndex();
      valid_hi = patch->getSFCYFORTHighIndex();
      
      // gas variables
      
//      IntVector ydim_lo_su_fcy = vVelNonlinearSrc_fcz.getFortLowIndex();
//      IntVector ydim_hi_su_fcy = vVelNonlinearSrc_fcz.getFortHighIndex();
      
//      IntVector ydim_lo_sp_fcy = vVelLinearSrc_fcz.getFortLowIndex();
//      IntVector ydim_hi_sp_fcy = vVelLinearSrc_fcz.getFortHighIndex();
      
//      IntVector ydim_lo_su_fcz = vVelNonlinearSrc_fcx.getFortLowIndex();
//      IntVector ydim_hi_su_fcz = vVelNonlinearSrc_fcx.getFortHighIndex();
      
//      IntVector ydim_lo_sp_fcz = vVelLinearSrc_fcx.getFortLowIndex();
//      IntVector ydim_hi_sp_fcz = vVelLinearSrc_fcx.getFortHighIndex();
      
//      IntVector ydim_lo_su_cc = vVelNonlinearSrc_cc.getFortLowIndex();
//      IntVector ydim_hi_su_cc = vVelNonlinearSrc_cc.getFortHighIndex();
      
//      IntVector ydim_lo_sp_cc = vVelLinearSrc_cc.getFortLowIndex();
//      IntVector ydim_hi_sp_cc = vVelLinearSrc_cc.getFortHighIndex();
      
      ioff = 0;
      joff = 1;
      koff = 0;
      
      indexflo = 2;
      indext1 =  3;
      indext2 =  1;
      
      // solid material variables
      
//      IntVector ydim_lo_dx_fcy = dragForceY_fcz[m].getFortLowIndex();
//      IntVector ydim_hi_dx_fcy = dragForceY_fcz[m].getFortHighIndex();
      
//      IntVector ydim_lo_dx_fcz = dragForceY_fcx[m].getFortLowIndex();
//      IntVector ydim_hi_dx_fcz = dragForceY_fcx[m].getFortHighIndex();
      
//      IntVector ydim_lo_upy = yvelFCZ_solid[m].getFortLowIndex();
//      IntVector ydim_hi_upy = yvelFCZ_solid[m].getFortHighIndex();
      
//      IntVector ydim_lo_upz = yvelFCX_solid[m].getFortLowIndex();
//      IntVector ydim_hi_upz = yvelFCX_solid[m].getFortHighIndex();

      fort_momentum_exchange_term_continuous_cc(vVelNonlinearSrc_fcz,
						vVelLinearSrc_fcz,
						vVelNonlinearSrc_fcx,
						vVelLinearSrc_fcx,
						vVelNonlinearSrc_cc,
						vVelLinearSrc_cc,
						dragForceY_fcz[m],
						dragForceY_fcx[m],
						dragForceY_cc[m],
						yvelCC_gas,
						yvelCC_solid[m],
						yvelFCZ_solid[m],
						yvelFCX_solid[m],
						gas_fraction_cc,
						solid_fraction_cc[m],
						viscos, csmag,
						cellinfo->sew, cellinfo->sns,
						cellinfo->stb, cellinfo->zz,
						cellinfo->xx, cellinfo->zw,
						cellinfo->xu,
						valid_lo, valid_hi,
						ioff, joff, koff,
						indexflo, indext1, indext2,
						cellType, mmwallid, ffieldid);

    // code for z-direction momentum exchange
			  
      dim_lo_ugc = zvelCC_gas.getFortLowIndex();
      dim_hi_ugc = zvelCC_gas.getFortHighIndex();
      
      dim_lo_dx_cc = dragForceZ_cc[m].getFortLowIndex();
      dim_hi_dx_cc = dragForceZ_cc[m].getFortHighIndex();
      
      valid_lo = patch->getSFCZFORTLowIndex();
      valid_hi = patch->getSFCZFORTHighIndex();
      
      // gas variables
      
//      IntVector zdim_lo_su_fcy = wVelNonlinearSrc_fcx.getFortLowIndex();
//      IntVector zdim_hi_su_fcy = wVelNonlinearSrc_fcx.getFortHighIndex();
      
//      IntVector zdim_lo_sp_fcy = wVelLinearSrc_fcx.getFortLowIndex();
//      IntVector zdim_hi_sp_fcy = wVelLinearSrc_fcx.getFortHighIndex();
      
//      IntVector zdim_lo_su_fcz = wVelNonlinearSrc_fcy.getFortLowIndex();
//      IntVector zdim_hi_su_fcz = wVelNonlinearSrc_fcy.getFortHighIndex();
      
//      IntVector zdim_lo_sp_fcz = wVelLinearSrc_fcy.getFortLowIndex();
//      IntVector zdim_hi_sp_fcz = wVelLinearSrc_fcy.getFortHighIndex();
      
//      IntVector zdim_lo_su_cc = wVelNonlinearSrc_cc.getFortLowIndex();
//      IntVector zdim_hi_su_cc = wVelNonlinearSrc_cc.getFortHighIndex();
      
//      IntVector zdim_lo_sp_cc = wVelLinearSrc_cc.getFortLowIndex();
//      IntVector zdim_hi_sp_cc = wVelLinearSrc_cc.getFortHighIndex();
      
      ioff = 0;
      joff = 0;
      koff = 1;
      
      indexflo = 3;
      indext1 =  1;
      indext2 =  2;
      
      // solid material variables
      
//      IntVector zdim_lo_dx_fcy = dragForceZ_fcx[m].getFortLowIndex();
//      IntVector zdim_hi_dx_fcy = dragForceZ_fcx[m].getFortHighIndex();
      
//      IntVector zdim_lo_dx_fcz = dragForceZ_fcy[m].getFortLowIndex();
//      IntVector zdim_hi_dx_fcz = dragForceZ_fcy[m].getFortHighIndex();
      
//      IntVector zdim_lo_upy = zvelFCX_solid[m].getFortLowIndex();
//      IntVector zdim_hi_upy = zvelFCX_solid[m].getFortHighIndex();
      
//      IntVector zdim_lo_upz = zvelFCY_solid[m].getFortLowIndex();
//      IntVector zdim_hi_upz = zvelFCY_solid[m].getFortHighIndex();

      fort_momentum_exchange_term_continuous_cc(wVelNonlinearSrc_fcx,
						wVelLinearSrc_fcx,
						wVelNonlinearSrc_fcy,
						wVelLinearSrc_fcy,
						wVelNonlinearSrc_cc,
						wVelLinearSrc_cc,
						dragForceZ_fcx[m],
						dragForceZ_fcy[m],
						dragForceZ_cc[m],
						zvelCC_gas,
						zvelCC_solid[m],
						zvelFCX_solid[m],
						zvelFCY_solid[m],
						gas_fraction_cc,
						solid_fraction_cc[m],
						viscos, csmag,
						cellinfo->sew, cellinfo->sns,
						cellinfo->stb, cellinfo->xx,
						cellinfo->yy, cellinfo->xu,
						cellinfo->yv,
						valid_lo, valid_hi,
						ioff, joff, koff,
						indexflo, indext1, indext2,
						cellType, mmwallid, ffieldid);
      
      // code for pressure forces (direction-independent)
      
//      IntVector dim_lo_fcx = pressForceX[m].getFortLowIndex();
//      IntVector dim_hi_fcx = pressForceX[m].getFortHighIndex();
      
//      IntVector dim_lo_fcy = pressForceY[m].getFortLowIndex();
//      IntVector dim_hi_fcy = pressForceY[m].getFortHighIndex();
      
//      IntVector dim_lo_fcz = pressForceZ[m].getFortLowIndex();
//      IntVector dim_hi_fcz = pressForceZ[m].getFortHighIndex();
      
//      IntVector dim_lo_pres = pressure.getFortLowIndex();
//      IntVector dim_hi_pres = pressure.getFortHighIndex();
      
      valid_lo = patch->getCellFORTLowIndex();
      valid_hi = patch->getCellFORTHighIndex();
      
      fort_pressure_force(pressForceX[m], pressForceY[m], pressForceZ[m],
			  gas_fraction_cc, solid_fraction_cc[m],
			  pressure, cellinfo->sew, cellinfo->sns,
			  cellinfo->stb, valid_lo, valid_hi, cellType,
			  mmwallid, ffieldid);
    }
    
    // Calculation done: now put things back in data warehouse
    
    // Solid variables
    
    for (int m = 0; m < numMPMMatls; m++) {
      
      Material* matl = d_sharedState->getMPMMaterial( m );
      int idx = matl->getDWIndex();
      
      new_dw->put(dragForceX_cc[m], d_MAlb->DragForceX_CCLabel,
		  idx, patch);
      new_dw->put(dragForceY_cc[m], d_MAlb->DragForceY_CCLabel,
		  idx, patch);
      new_dw->put(dragForceZ_cc[m], d_MAlb->DragForceZ_CCLabel,
		  idx, patch);
      
      new_dw->put(dragForceX_fcy[m], d_MAlb->DragForceX_FCYLabel,
		  idx, patch);
      new_dw->put(dragForceX_fcz[m], d_MAlb->DragForceX_FCZLabel,
		  idx, patch);
      
      new_dw->put(dragForceY_fcz[m], d_MAlb->DragForceY_FCZLabel,
		  idx, patch);
      new_dw->put(dragForceY_fcx[m], d_MAlb->DragForceY_FCXLabel,
		  idx, patch);
      
      new_dw->put(dragForceZ_fcx[m], d_MAlb->DragForceZ_FCXLabel,
		  idx, patch);
      new_dw->put(dragForceZ_fcy[m], d_MAlb->DragForceZ_FCYLabel,
		  idx, patch);
      
      new_dw->put(pressForceX[m], d_MAlb->PressureForce_FCXLabel,
		  idx, patch);
      new_dw->put(pressForceY[m], d_MAlb->PressureForce_FCYLabel,
		  idx, patch);
      new_dw->put(pressForceZ[m], d_MAlb->PressureForce_FCZLabel,
		  idx, patch);

    }
    
    // Gas variables
    
    new_dw->put(uVelLinearSrc_cc, d_MAlb->d_uVel_mmLinSrc_CCLabel,
		matlIndex, patch);
    new_dw->put(uVelLinearSrc_fcy, d_MAlb->d_uVel_mmLinSrc_FCYLabel,
		matlIndex, patch);
    new_dw->put(uVelLinearSrc_fcz, d_MAlb->d_uVel_mmLinSrc_FCZLabel,
		matlIndex, patch);
    
    new_dw->put(uVelNonlinearSrc_cc, d_MAlb->d_uVel_mmNonlinSrc_CCLabel,
		matlIndex, patch);
    new_dw->put(uVelNonlinearSrc_fcy, d_MAlb->d_uVel_mmNonlinSrc_FCYLabel,
		matlIndex, patch);
    new_dw->put(uVelNonlinearSrc_fcz, d_MAlb->d_uVel_mmNonlinSrc_FCZLabel,
		matlIndex, patch);
    
    new_dw->put(vVelLinearSrc_cc, d_MAlb->d_vVel_mmLinSrc_CCLabel,
		matlIndex, patch);
    new_dw->put(vVelLinearSrc_fcz, d_MAlb->d_vVel_mmLinSrc_FCZLabel,
		matlIndex, patch);
    new_dw->put(vVelLinearSrc_fcx, d_MAlb->d_vVel_mmLinSrc_FCXLabel,
		matlIndex, patch);
    
    new_dw->put(vVelNonlinearSrc_cc, d_MAlb->d_vVel_mmNonlinSrc_CCLabel,
		matlIndex, patch);
    new_dw->put(vVelNonlinearSrc_fcz, d_MAlb->d_vVel_mmNonlinSrc_FCZLabel,
		matlIndex, patch);
    new_dw->put(vVelNonlinearSrc_fcx, d_MAlb->d_vVel_mmNonlinSrc_FCXLabel,
		matlIndex, patch);
    
    new_dw->put(wVelLinearSrc_cc, d_MAlb->d_wVel_mmLinSrc_CCLabel,
		matlIndex, patch);
    new_dw->put(wVelLinearSrc_fcx, d_MAlb->d_wVel_mmLinSrc_FCXLabel,
		matlIndex, patch);
    new_dw->put(wVelLinearSrc_fcy, d_MAlb->d_wVel_mmLinSrc_FCYLabel,
		matlIndex, patch);

    new_dw->put(wVelNonlinearSrc_cc, d_MAlb->d_wVel_mmNonlinSrc_CCLabel,
		matlIndex, patch);
    new_dw->put(wVelNonlinearSrc_fcx, d_MAlb->d_wVel_mmNonlinSrc_FCXLabel,
		matlIndex, patch);
    new_dw->put(wVelNonlinearSrc_fcy, d_MAlb->d_wVel_mmNonlinSrc_FCYLabel,
		matlIndex, patch);
  }
}

//______________________________________________________________________
//

void MPMArches::collectToCCGasMomExchSrcs(const ProcessorGroup*,
					  const PatchSubset* patches,
					  const MaterialSubset*,
					  DataWarehouse* /*old_dw*/,
					  DataWarehouse* new_dw)

{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    CCVariable<double> su_dragx_cc;
    constSFCYVariable<double> su_dragx_fcy;
    constSFCZVariable<double> su_dragx_fcz;
    
    CCVariable<double> sp_dragx_cc;
    constSFCYVariable<double> sp_dragx_fcy;
    constSFCZVariable<double> sp_dragx_fcz;
    
    CCVariable<double> su_dragy_cc;
    constSFCZVariable<double> su_dragy_fcz;
    constSFCXVariable<double> su_dragy_fcx;
    
    CCVariable<double> sp_dragy_cc;
    constSFCZVariable<double> sp_dragy_fcz;
    constSFCXVariable<double> sp_dragy_fcx;
    
    CCVariable<double> su_dragz_cc;
    constSFCXVariable<double> su_dragz_fcx;
    constSFCYVariable<double> su_dragz_fcy;
    
    CCVariable<double> sp_dragz_cc;
    constSFCXVariable<double> sp_dragz_fcx;
    constSFCYVariable<double> sp_dragz_fcy;

    int numGhostCells = 1;

    new_dw->allocate(su_dragx_cc, d_MAlb->d_uVel_mmNonlinSrc_CC_CollectLabel,
		     matlIndex, patch);
    new_dw->allocate(sp_dragx_cc, d_MAlb->d_uVel_mmLinSrc_CC_CollectLabel,
		     matlIndex, patch);
    
    new_dw->allocate(su_dragy_cc, d_MAlb->d_vVel_mmNonlinSrc_CC_CollectLabel,
		     matlIndex, patch);
    new_dw->allocate(sp_dragy_cc, d_MAlb->d_vVel_mmLinSrc_CC_CollectLabel,
		     matlIndex, patch);
    
    new_dw->allocate(su_dragz_cc, d_MAlb->d_wVel_mmNonlinSrc_CC_CollectLabel,
		     matlIndex, patch);
    new_dw->allocate(sp_dragz_cc, d_MAlb->d_wVel_mmLinSrc_CC_CollectLabel,
		     matlIndex, patch);
    
    
    new_dw->copyOut(su_dragx_cc, d_MAlb->d_uVel_mmNonlinSrc_CCLabel,
		    matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(su_dragx_fcy, d_MAlb->d_uVel_mmNonlinSrc_FCYLabel,
		    matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(su_dragx_fcz, d_MAlb->d_uVel_mmNonlinSrc_FCZLabel,
		    matlIndex, patch, Ghost::AroundCells, numGhostCells);
    
    new_dw->copyOut(sp_dragx_cc, d_MAlb->d_uVel_mmLinSrc_CCLabel,
		    matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(sp_dragx_fcy, d_MAlb->d_uVel_mmLinSrc_FCYLabel,
		    matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(sp_dragx_fcz, d_MAlb->d_uVel_mmLinSrc_FCZLabel,
		    matlIndex, patch, Ghost::AroundCells, numGhostCells);
    
    new_dw->copyOut(su_dragy_cc, d_MAlb->d_vVel_mmNonlinSrc_CCLabel,
		    matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(su_dragy_fcz, d_MAlb->d_vVel_mmNonlinSrc_FCZLabel,
		    matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(su_dragy_fcx, d_MAlb->d_vVel_mmNonlinSrc_FCXLabel,
		    matlIndex, patch, Ghost::AroundCells, numGhostCells);
    
    new_dw->copyOut(sp_dragy_cc, d_MAlb->d_vVel_mmLinSrc_CCLabel,
		    matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(sp_dragy_fcz, d_MAlb->d_vVel_mmLinSrc_FCZLabel,
		    matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(sp_dragy_fcx, d_MAlb->d_vVel_mmLinSrc_FCXLabel,
		    matlIndex, patch, Ghost::AroundCells, numGhostCells);
    
    new_dw->copyOut(su_dragz_cc, d_MAlb->d_wVel_mmNonlinSrc_CCLabel,
		    matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(su_dragz_fcx, d_MAlb->d_wVel_mmNonlinSrc_FCXLabel,
		    matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(su_dragz_fcy, d_MAlb->d_wVel_mmNonlinSrc_FCYLabel,
		    matlIndex, patch, Ghost::AroundCells, numGhostCells);
    
    new_dw->copyOut(sp_dragz_cc, d_MAlb->d_wVel_mmLinSrc_CCLabel,
		    matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(sp_dragz_fcx, d_MAlb->d_wVel_mmLinSrc_FCXLabel,
		    matlIndex, patch, Ghost::AroundCells, numGhostCells);
    new_dw->get(sp_dragz_fcy, d_MAlb->d_wVel_mmLinSrc_FCYLabel,
		    matlIndex, patch, Ghost::AroundCells, numGhostCells);
    
    IntVector valid_lo;
    IntVector valid_hi;
    
    int ioff;
    int joff;
    int koff;
    
    IntVector dim_lo_su_cc;
    IntVector dim_hi_su_cc;
    IntVector dim_lo_sp_cc;
    IntVector dim_hi_sp_cc;
    IntVector dim_lo_su_fc;
    IntVector dim_hi_su_fc;
    IntVector dim_lo_sp_fc;
    IntVector dim_hi_sp_fc;
    
    // collect x-direction sources from face centers to cell center
    
    valid_lo = patch->getSFCXFORTLowIndex();
    valid_hi = patch->getSFCXFORTHighIndex();
    
    ioff = 1;
    joff = 0;
    koff = 0;
    
    dim_lo_su_cc = su_dragx_cc.getFortLowIndex();
    dim_hi_su_cc = su_dragx_cc.getFortHighIndex();
    
    dim_lo_sp_cc = sp_dragx_cc.getFortLowIndex();
    dim_hi_sp_cc = sp_dragx_cc.getFortHighIndex();
    
    // for first transverse direction, i.e., y
    
    dim_lo_su_fc = su_dragx_fcy.getFortLowIndex();
    dim_hi_su_fc = su_dragx_fcy.getFortHighIndex();
    
    dim_lo_sp_fc = sp_dragx_fcy.getFortLowIndex();
    dim_hi_sp_fc = sp_dragx_fcy.getFortHighIndex();

    fort_collect_drag_cc(su_dragx_cc, sp_dragx_cc,
			 su_dragx_fcy, sp_dragx_fcy,
			 koff, ioff, joff,
			 valid_lo, valid_hi);

    // for second transverse direction, i.e., z
    
    dim_lo_su_fc = su_dragx_fcz.getFortLowIndex();
    dim_hi_su_fc = su_dragx_fcz.getFortHighIndex();
    
    dim_lo_sp_fc = sp_dragx_fcz.getFortLowIndex();
    dim_hi_sp_fc = sp_dragx_fcz.getFortHighIndex();

    fort_collect_drag_cc(su_dragx_cc, sp_dragx_cc,
			 su_dragx_fcz, sp_dragx_fcz,
			 joff, joff, ioff,
			 valid_lo, valid_hi);
    
    // collect y-direction sources from face centers to cell center
    
    valid_lo = patch->getSFCYFORTLowIndex();
    valid_hi = patch->getSFCYFORTHighIndex();
    
    ioff = 0;
    joff = 1;
    koff = 0;
    
    dim_lo_su_cc = su_dragy_cc.getFortLowIndex();
    dim_hi_su_cc = su_dragy_cc.getFortHighIndex();
    
    dim_lo_sp_cc = sp_dragy_cc.getFortLowIndex();
    dim_hi_sp_cc = sp_dragy_cc.getFortHighIndex();
    
  // for first transverse direction, i.e., z
    
    dim_lo_su_fc = su_dragy_fcz.getFortLowIndex();
    dim_hi_su_fc = su_dragy_fcz.getFortHighIndex();
    
    dim_lo_sp_fc = sp_dragy_fcz.getFortLowIndex();
    dim_hi_sp_fc = sp_dragy_fcz.getFortHighIndex();
    
    fort_collect_drag_cc(su_dragy_cc, sp_dragy_cc,
			 su_dragy_fcz, sp_dragy_fcz,
			 koff, ioff, joff,
			 valid_lo, valid_hi);

    
    // for second transverse direction, i.e., x
    
    dim_lo_su_fc = su_dragy_fcx.getFortLowIndex();
    dim_hi_su_fc = su_dragy_fcx.getFortHighIndex();
    
    dim_lo_sp_fc = sp_dragy_fcx.getFortLowIndex();
    dim_hi_sp_fc = sp_dragy_fcx.getFortHighIndex();
    
    fort_collect_drag_cc(su_dragy_cc, sp_dragy_cc,
			 su_dragy_fcx, sp_dragy_fcx,
			 joff, koff, ioff,
			 valid_lo, valid_hi);
    
    // collect z-direction sources from face centers to cell center
    
    valid_lo = patch->getSFCZFORTLowIndex();
    valid_hi = patch->getSFCZFORTHighIndex();
    
    ioff = 0;
    joff = 0;
    koff = 1;
    
    dim_lo_su_cc = su_dragz_cc.getFortLowIndex();
    dim_hi_su_cc = su_dragz_cc.getFortHighIndex();
    
    dim_lo_sp_cc = sp_dragz_cc.getFortLowIndex();
    dim_hi_sp_cc = sp_dragz_cc.getFortHighIndex();
    
    // for first transverse direction, i.e., x
    
    dim_lo_su_fc = su_dragz_fcx.getFortLowIndex();
    dim_hi_su_fc = su_dragz_fcx.getFortHighIndex();
    
    dim_lo_sp_fc = sp_dragz_fcx.getFortLowIndex();
    dim_hi_sp_fc = sp_dragz_fcx.getFortHighIndex();
    
    fort_collect_drag_cc(su_dragz_cc, sp_dragz_cc,
			 su_dragz_fcx, sp_dragz_fcx,
			 koff, ioff, joff,
			 valid_lo, valid_hi);

    
    // for second transverse direction, i.e., y
    
    dim_lo_su_fc = su_dragz_fcy.getFortLowIndex();
    dim_hi_su_fc = su_dragz_fcy.getFortHighIndex();
    
    dim_lo_sp_fc = sp_dragz_fcy.getFortLowIndex();
    dim_hi_sp_fc = sp_dragz_fcy.getFortHighIndex();
    
    fort_collect_drag_cc(su_dragz_cc, sp_dragz_cc,
			 su_dragz_fcy, sp_dragz_fcy,
			 joff, koff, ioff,
			 valid_lo, valid_hi);
    
    // Calculation done: now put things in DW
    
    new_dw->put(su_dragx_cc, d_MAlb->d_uVel_mmNonlinSrc_CC_CollectLabel,
		matlIndex, patch);
    new_dw->put(sp_dragx_cc, d_MAlb->d_uVel_mmLinSrc_CC_CollectLabel,
		matlIndex, patch);
    
    new_dw->put(su_dragy_cc, d_MAlb->d_vVel_mmNonlinSrc_CC_CollectLabel,
		matlIndex, patch);
    new_dw->put(sp_dragy_cc, d_MAlb->d_vVel_mmLinSrc_CC_CollectLabel,
		matlIndex, patch);
    
    new_dw->put(su_dragz_cc, d_MAlb->d_wVel_mmNonlinSrc_CC_CollectLabel,
		matlIndex, patch);
    new_dw->put(sp_dragz_cc, d_MAlb->d_wVel_mmLinSrc_CC_CollectLabel,
		matlIndex, patch);
  }
}

//______________________________________________________________________
//

void MPMArches::interpolateCCToFCGasMomExchSrcs(const ProcessorGroup*,
						const PatchSubset* patches,
						const MaterialSubset* ,
						DataWarehouse* /*old_dw*/,
						DataWarehouse* new_dw)

  // This function interpolates the source terms that are calculated 
  // and collected at the cell center to the staggered face centers
  // for each momentum equation of the gas phase.  At the end of this
  // function execution, the gas phase has all the momentum exchange
  // source terms it needs for its calculations.

{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    constCCVariable<double> su_dragx_cc;
    constCCVariable<double> sp_dragx_cc;
    constCCVariable<double> su_dragy_cc;
    constCCVariable<double> sp_dragy_cc;
    constCCVariable<double> su_dragz_cc;
    constCCVariable<double> sp_dragz_cc;
    
    SFCXVariable<double> su_dragx_fcx;
    SFCXVariable<double> sp_dragx_fcx;
    SFCYVariable<double> su_dragy_fcy;
    SFCYVariable<double> sp_dragy_fcy;
    SFCZVariable<double> su_dragz_fcz;
    SFCZVariable<double> sp_dragz_fcz;
    
    int numGhostCells = 1;
    
    // gets CC variables

    new_dw->get(su_dragx_cc, d_MAlb->d_uVel_mmNonlinSrc_CC_CollectLabel,
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    
    new_dw->get(sp_dragx_cc, d_MAlb->d_uVel_mmLinSrc_CC_CollectLabel,
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    
    new_dw->get(su_dragy_cc, d_MAlb->d_vVel_mmNonlinSrc_CC_CollectLabel,
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    
    new_dw->get(sp_dragy_cc, d_MAlb->d_vVel_mmLinSrc_CC_CollectLabel,
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    
    new_dw->get(su_dragz_cc, d_MAlb->d_wVel_mmNonlinSrc_CC_CollectLabel,
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    
    new_dw->get(sp_dragz_cc, d_MAlb->d_wVel_mmLinSrc_CC_CollectLabel,
		matlIndex, patch, Ghost::AroundCells, numGhostCells);
    
    // computes FC interpolants
    
    new_dw->allocate(su_dragx_fcx, d_MAlb->d_uVel_mmNonlinSrcLabel,
		     matlIndex, patch);
    
    new_dw->allocate(sp_dragx_fcx, d_MAlb->d_uVel_mmLinSrcLabel,
		     matlIndex, patch);
    
    new_dw->allocate(su_dragy_fcy, d_MAlb->d_vVel_mmNonlinSrcLabel,
		     matlIndex, patch);
    
    new_dw->allocate(sp_dragy_fcy, d_MAlb->d_vVel_mmLinSrcLabel,
		     matlIndex, patch);
    
    new_dw->allocate(su_dragz_fcz, d_MAlb->d_wVel_mmNonlinSrcLabel,
		     matlIndex, patch);
    
    new_dw->allocate(sp_dragz_fcz, d_MAlb->d_wVel_mmLinSrcLabel,
		     matlIndex, patch);
    
    IntVector dim_lo_fc;
    IntVector dim_hi_fc;
    
    IntVector dim_lo_cc;
    IntVector dim_hi_cc;
    
    IntVector valid_lo;
    IntVector valid_hi;
    
    int ioff;
    int joff;
    int koff;
    
    // Interpolate x-momentum source terms
    
    ioff = 1;
    joff = 0;
    koff = 0;
    
    valid_lo = patch->getSFCXFORTLowIndex();
    valid_hi = patch->getSFCXFORTHighIndex();
    
    // nonlinear source
    
    dim_lo_fc = su_dragx_fcx.getFortLowIndex();
    dim_hi_fc = su_dragx_fcx.getFortHighIndex();
    
    dim_lo_cc = su_dragx_cc.getFortLowIndex();
    dim_hi_cc = su_dragx_cc.getFortHighIndex();
    
    fort_interp_centertoface(su_dragx_fcx,
			     su_dragx_cc, ioff, joff, koff,
			     valid_lo, valid_hi);
    
    // linear source
    
    dim_lo_fc = sp_dragx_fcx.getFortLowIndex();
    dim_hi_fc = sp_dragx_fcx.getFortHighIndex();
    
    dim_lo_cc = sp_dragx_cc.getFortLowIndex();
    dim_hi_cc = sp_dragx_cc.getFortHighIndex();
    
    fort_interp_centertoface(sp_dragx_fcx,
			     sp_dragx_cc, ioff, joff, koff,
			     valid_lo, valid_hi);
    
    // Interpolate y-momentum source terms
    
    ioff = 0;
    joff = 1;
    koff = 0;
    
    valid_lo = patch->getSFCYFORTLowIndex();
    valid_hi = patch->getSFCYFORTHighIndex();
    
    // nonlinear source
    
    dim_lo_fc = su_dragy_fcy.getFortLowIndex();
    dim_hi_fc = su_dragy_fcy.getFortHighIndex();
    
    dim_lo_cc = su_dragy_cc.getFortLowIndex();
    dim_hi_cc = su_dragy_cc.getFortHighIndex();
    
    fort_interp_centertoface(su_dragy_fcy,
			     su_dragy_cc, ioff, joff, koff,
			     valid_lo, valid_hi);
    
    // linear source
    
    dim_lo_fc = sp_dragy_fcy.getFortLowIndex();
    dim_hi_fc = sp_dragy_fcy.getFortHighIndex();
    
    dim_lo_cc = sp_dragy_cc.getFortLowIndex();
    dim_hi_cc = sp_dragy_cc.getFortHighIndex();
    
    fort_interp_centertoface(sp_dragy_fcy,
			     sp_dragy_cc, ioff, joff, koff,
			     valid_lo, valid_hi);
    
    // Interpolate z-momentum source terms
    
    ioff = 0;
    joff = 0;
    koff = 1;
    
    valid_lo = patch->getSFCZFORTLowIndex();
    valid_hi = patch->getSFCZFORTHighIndex();
    
    // nonlinear source
    
    dim_lo_fc = su_dragz_fcz.getFortLowIndex();
    dim_hi_fc = su_dragz_fcz.getFortHighIndex();
    
    dim_lo_cc = su_dragz_cc.getFortLowIndex();
    dim_hi_cc = su_dragz_cc.getFortHighIndex();
    
    fort_interp_centertoface(su_dragz_fcz,
			     su_dragz_cc, ioff, joff, koff,
			     valid_lo, valid_hi);
    
    // linear source
    
    dim_lo_fc = sp_dragz_fcz.getFortLowIndex();
    dim_hi_fc = sp_dragz_fcz.getFortHighIndex();
    
    dim_lo_cc = sp_dragz_cc.getFortLowIndex();
    dim_hi_cc = sp_dragz_cc.getFortHighIndex();
    
    fort_interp_centertoface(sp_dragz_fcz,
			     sp_dragz_cc, ioff, joff, koff,
			     valid_lo, valid_hi);
    // Calculation done: now put things in DW
    
    new_dw->put(su_dragx_fcx, d_MAlb->d_uVel_mmNonlinSrcLabel,
		matlIndex, patch);
    
    new_dw->put(sp_dragx_fcx, d_MAlb->d_uVel_mmLinSrcLabel,
		matlIndex, patch);
    
    new_dw->put(su_dragy_fcy, d_MAlb->d_vVel_mmNonlinSrcLabel,
		matlIndex, patch);
    
    new_dw->put(sp_dragy_fcy, d_MAlb->d_vVel_mmLinSrcLabel,
		matlIndex, patch);
    
    new_dw->put(su_dragz_fcz, d_MAlb->d_wVel_mmNonlinSrcLabel,
		matlIndex, patch);
    
    new_dw->put(sp_dragz_fcz, d_MAlb->d_wVel_mmLinSrcLabel,
		matlIndex, patch);
  }
}

//______________________________________________________________________
//

#if 0
void MPMArches::redistributeDragForceFromCCtoFC(const ProcessorGroup*,
						const PatchSubset* patches,
						const MaterialSubset* ,
						DataWarehouse* old_dw,
						DataWarehouse* new_dw)

  //
  // redistributes the drag forces experienced by the solid materials,
  // which are calculated at cell centers for partially filled 
  // cells in the previous step, to face centers
  //

{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int numMPMMatls  = d_sharedState->getNumMPMMatls();
    // MPM stuff

    CCVariable<double> dragForceX_cc;
    CCVariable<double> dragForceY_cc;
    CCVariable<double> dragForceZ_cc;

    SFCXVariable<double> dragForceX_fcx;
    SFCYVariable<double> dragForceY_fcy;
    SFCZVariable<double> dragForceZ_fcz;

    int numGhostCells = 1;
    
    for (int m = 0; m < numMPMMatls; m++) {
      
      Material* matl = d_sharedState->getMPMMaterial( m );
      int idx = matl->getDWIndex();

      new_dw->get(dragForceX_cc, d_MAlb->DragForceX_CCLabel,
		  idx, patch, Ghost::AroundCells, numGhostCells);

      new_dw->get(dragForceY_cc, d_MAlb->DragForceY_CCLabel,
		  idx, patch, Ghost::AroundCells, numGhostCells);

      new_dw->get(dragForceZ_cc, d_MAlb->DragForceZ_CCLabel,
		  idx, patch, Ghost::AroundCells, numGhostCells);

      new_dw->allocate(dragForceX_fcx, d_MAlb->DragForceX_FCXLabel,
		       idx, patch);
      
      new_dw->allocate(dragForceY_fcy, d_MAlb->DragForceY_FCYLabel,
		       idx, patch);
      
      new_dw->allocate(dragForceZ_fcz, d_MAlb->DragForceZ_FCZLabel,
		       idx, patch);

      IntVector dim_lo_cc;
      IntVector dim_hi_cc;
      
      IntVector dim_lo_fcx;
      IntVector dim_hi_fcx;
      
      IntVector valid_lo;
      IntVector valid_hi;

      // redistribute x-direction drag forces

      dim_lo_cc = dragForceX_cc.getFortLowIndex();
      dim_hi_cc = dragForceX_cc.getFortHighIndex();
      
      dim_lo_fcx = dragForceX_fcx.getFortLowIndex();
      dim_hi_fcx = dragForceX_fcx.getFortHighIndex();

      valid_lo = patch->getSFCXFORTLowIndex();
      valid_hi = patch->getSFCXFORTHighIndex();

      int ioff = 1;
      int joff = 0;
      int koff = 0;

      fort_mm_redistribute_drag(dragForceX_fcx, dragForceX_cc,
				ioff, joff, koff, valid_lo, valid_hi);
      
      // redistribute y-direction drag forces
      
      dim_lo_cc = dragForceY_cc.getFortLowIndex();
      dim_hi_cc = dragForceY_cc.getFortHighIndex();
      
      dim_lo_fcx = dragForceY_fcy.getFortLowIndex();
      dim_hi_fcx = dragForceY_fcy.getFortHighIndex();
      
      valid_lo = patch->getSFCYFORTLowIndex();
      valid_hi = patch->getSFCYFORTHighIndex();
      
      ioff = 0;
      joff = 1;
      koff = 0;
      
      fort_mm_redistribute_drag(dragForceY_fcx, dragForceY_cc,
				ioff, joff, koff, valid_lo, valid_hi);
      
      // redistribute z-direction drag forces
      
      dim_lo_cc = dragForceZ_cc.getFortLowIndex();
      dim_hi_cc = dragForceZ_cc.getFortHighIndex();
      
      dim_lo_fcx = dragForceZ_fcz.getFortLowIndex();
      dim_hi_fcx = dragForceZ_fcz.getFortHighIndex();
      
      valid_lo = patch->getSFCZFORTLowIndex();
      valid_hi = patch->getSFCZFORTHighIndex();
      
      ioff = 0;
      joff = 0;
      koff = 1;
      
      fort_mm_redistribute_drag(dragForceZ_fcx, dragForceZ_cc,
				iof,f joff, koff, valid_lo, valid_hi);
      
      // Calculation done; now put things back in DW
      
      new_dw->put(dragForceX_fcx, d_MAlb->DragForceX_FCXLabel, 
		  idx, patch);
      new_dw->put(dragForceY_fcy, d_MAlb->DragForceY_FCYLabel, 
		  idx, patch);
      new_dw->put(dragForceZ_fcz, d_MAlb->DragForceZ_FCZLabel, 
		  idx, patch);

    }  
  }
    
}
#endif

void MPMArches::putAllForcesOnCC(const ProcessorGroup*,
				const PatchSubset* patches,
				const MaterialSubset* matls,
				DataWarehouse* /*old_dw*/,
				DataWarehouse* new_dw)

{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m=0;m<matls->size();m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      CCVariable<Vector> totalforce;
      CCVariable<Vector> acc_arches;
      constCCVariable<double> DFX_CC, DFY_CC, DFZ_CC, cmass;
      constSFCXVariable<double> PRX_FC;
      constSFCYVariable<double> PRY_FC;
      constSFCZVariable<double> PRZ_FC;

      new_dw->allocate(totalforce, d_MAlb->SumAllForcesCCLabel,matlindex,patch);
      new_dw->allocate(acc_arches, d_MAlb->AccArchesCCLabel,   matlindex,patch);

      new_dw->get(cmass,  d_MAlb->cMassLabel,         matlindex, patch,
							Ghost::None, 0);
      new_dw->get(DFX_CC, d_MAlb->DragForceX_CCLabel, matlindex, patch,
							Ghost::None,0);
      new_dw->get(DFY_CC, d_MAlb->DragForceY_CCLabel, matlindex, patch,
							Ghost::None,0);
      new_dw->get(DFZ_CC, d_MAlb->DragForceZ_CCLabel, matlindex, patch,
							Ghost::None,0);

      new_dw->get(PRX_FC, d_MAlb->PressureForce_FCXLabel, matlindex, patch,
							Ghost::AroundCells, 1);
      new_dw->get(PRY_FC, d_MAlb->PressureForce_FCYLabel, matlindex, patch,
							Ghost::AroundCells, 1);
      new_dw->get(PRZ_FC, d_MAlb->PressureForce_FCZLabel, matlindex, patch,
							Ghost::AroundCells, 1);

      acc_arches.initialize(Vector(0.,0.,0.));

      for(CellIterator iter =patch->getExtraCellIterator();!iter.done();iter++){
	totalforce[*iter] = Vector(DFX_CC[*iter], DFY_CC[*iter], DFZ_CC[*iter]);
	IntVector curcell = *iter;
	double XCPF, YCPF, ZCPF;
	if (curcell.x() >= (patch->getInteriorCellLowIndex()).x()) {
	  IntVector adjcell(curcell.x()-1,curcell.y(),curcell.z());
	  XCPF = .5*(PRX_FC[curcell] + PRX_FC[adjcell]);
	}
	if (curcell.y() >= (patch->getInteriorCellLowIndex()).y()) {
	  IntVector adjcell(curcell.x(),curcell.y()-1,curcell.z());
	  YCPF = .5*(PRY_FC[curcell] + PRY_FC[adjcell]);
	}
	if (curcell.z() >= (patch->getInteriorCellLowIndex()).z()) {
	  IntVector adjcell(curcell.x(),curcell.y(),curcell.z()-1);
	  ZCPF = .5*(PRZ_FC[curcell] + PRZ_FC[adjcell]);
	}
	totalforce[*iter] += Vector(XCPF, YCPF, ZCPF);
	if(cmass[*iter] > d_SMALL_NUM){
	  acc_arches[*iter] = totalforce[*iter]/cmass[*iter];
        }
      }
      new_dw->put(totalforce, d_MAlb->SumAllForcesCCLabel, matlindex, patch);
      new_dw->put(acc_arches, d_MAlb->AccArchesCCLabel,    matlindex, patch);
    }
  }
}

void MPMArches::putAllForcesOnNC(const ProcessorGroup*,
				const PatchSubset* patches,
				const MaterialSubset* matls,
				DataWarehouse* /*old_dw*/,
				DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    IntVector cIdx[8];

    for(int m=0;m<matls->size();m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      constCCVariable<Vector> acc_archesCC;
      NCVariable<Vector> acc_archesNC;
      new_dw->get(acc_archesCC, d_MAlb->AccArchesCCLabel,   matlindex,patch,
						Ghost::AroundCells, 1);
      new_dw->allocate(acc_archesNC, d_MAlb->AccArchesNCLabel, matlindex,patch);

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
        patch->findCellsFromNode(*iter,cIdx);
        for (int in=0;in<8;in++){
          acc_archesNC[*iter]  += acc_archesCC[cIdx[in]]*.125;
        }
      }
      new_dw->put(acc_archesNC, d_MAlb->AccArchesNCLabel,  matlindex, patch);
    }
  }
}
