// MPMArches.cc

#include <Packages/Uintah/CCA/Components/MPMArches/MPMArches.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformationP.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformation.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Core/Util/NotFinished.h>


using namespace Uintah;
using namespace SCIRun;
using namespace std;
MPMArches::MPMArches(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld)
{
  Mlb  = scinew MPMLabel();
  d_MAlb = scinew MPMArchesLabel();
  d_mpm      = scinew SerialMPM(myworld);
  d_fracture = false;
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
  d_arches->scheduleInitialize(      level, sched);
  // add compute void fraction to be used by gas density and viscosity
  cerr << "Doing Initialization \t\t\t MPMArches" <<endl;
  cerr << "--------------------------------\n"<<endl; 
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
void MPMArches::scheduleTimeAdvance(double time, double delt,
				    const LevelP&   level,
				    SchedulerP&     sched)
{
  const PatchSet* patches = level->eachPatch();
  NOT_FINISHED("probably wrong material set for MPMArches");
  const MaterialSet* matls = d_sharedState->allMPMMaterials();

  // Rajesh/Kumar  Because interpolation from particles to nodes
  // is the fist thing done in our algorithm, it is, unfortunately,
  // going to be necessary to put that step of the algorithm in
  // prior to the NCToCC stuff, and so of course the remaining
  // MPM steps will follow.  Without doing a heap of other work,
  // I don't see any way around this, and the only real cost is
  // to ugly this up a bit.

  if(d_fracture) {
    d_mpm->scheduleSetPositions(sched, patches, matls);
    d_mpm->scheduleComputeBoundaryContact(sched, patches, matls);
    d_mpm->scheduleComputeConnectivity(sched, patches, matls);
  }
  d_mpm->scheduleInterpolateParticlesToGrid(sched, patches, matls);
  d_mpm->scheduleComputeHeatExchange(sched, patches, matls);

  // interpolate mpm properties from node center to cell center/ face center
  // these computed variables are used by void fraction and mom exchange
  scheduleInterpolateNCToCC(sched, patches, matls);
  // Velocity and temperature are needed at the faces, right?
  scheduleInterpolateCCToFC(sched, patches, matls);

  // for explicit calculation, exchange will be at the beginning
  scheduleComputeVoidFrac(sched, patches, matls);

  // for heat transfer HeatExchangeCoeffs will be called
  scheduleMomExchange(sched, patches, matls);

  d_arches->scheduleTimeAdvance(time, delt, level, sched);

  d_mpm->scheduleExMomInterpolated(               sched, patches, matls);
  d_mpm->scheduleComputeStressTensor(             sched, patches, matls);
  d_mpm->scheduleComputeInternalForce(            sched, patches, matls);
  d_mpm->scheduleComputeInternalHeatRate(         sched, patches, matls);
  d_mpm->scheduleSolveEquationsMotion(            sched, patches, matls);
  d_mpm->scheduleSolveHeatEquations(              sched, patches, matls);
  d_mpm->scheduleIntegrateAcceleration(           sched, patches, matls);
  d_mpm->scheduleIntegrateTemperatureRate(        sched, patches, matls);
  d_mpm->scheduleExMomIntegrated(                 sched, patches, matls);
  d_mpm->scheduleInterpolateToParticlesAndUpdate( sched, patches, matls);
  d_mpm->scheduleComputeMassRate(                 sched, patches, matls);
  if(d_fracture) {
    d_mpm->scheduleComputeFracture(               sched, patches, matls);
  }
  d_mpm->scheduleCarryForwardVariables(           sched, patches, matls);

  sched->scheduleParticleRelocation(level,
                                    Mlb->pXLabel_preReloc,
                                    Mlb->d_particleState_preReloc,
                                    Mlb->pXLabel, Mlb->d_particleState,
                                    matls);
}
//
//
void MPMArches::scheduleInterpolateNCToCC(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls)
{
#if 0
   /* interpolateNCToCC */
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    // primitive variable initialization

   Task* t=scinew Task("MPMArches::interpolateNCToCC",
		        patch, old_dw, new_dw,
		        this, &MPMArches::interpolateNCToCC);
   int numMPMMatls = d_sharedState->getNumMPMMatls();
   int numGhostCells = 1;
   for(int m = 0; m < numMPMMatls; m++){
     MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
     int idx = mpm_matl->getDWIndex();
     t->requires(new_dw, Mlb->gMassLabel,         idx, patch,
		Ghost::AroundCells, numGhostCells);
     t->requires(new_dw, Mlb->gVolumeLabel,       idx, patch,
		Ghost::AroundCells, numGhostCells);
     t->requires(new_dw, Mlb->gVelocityLabel,     idx, patch,
		Ghost::AroundCells, numGhostCells);

     t->computes(new_dw, d_MAlb->cMassLabel,         idx, patch);
     t->computes(new_dw, d_MAlb->cVolumeLabel,       idx, patch);
     t->computes(new_dw, d_MAlb->vel_CCLabel,        idx, patch);
     
   }

   sched->addTask(t);
  }
#else
  NOT_FINISHED("new task stuff");
#endif
}


void MPMArches::scheduleInterpolateCCToFC(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* matls)
{
#if 0
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;

    Task* t=scinew Task("MPMArches::interpolateNCToCC",
		        patch, old_dw, new_dw,
		        this, &MPMArches::interpolateNCToCC);
    int numMPMMatls = d_sharedState->getNumMPMMatls();
    int numGhostCells = 1;
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
      int idx = mpm_matl->getDWIndex();

      t->requires(new_dw, d_MAlb->cMassLabel,         idx, patch,
			Ghost::AroundCells, numGhostCells);
      t->requires(new_dw, d_MAlb->cVolumeLabel,         idx, patch,
			Ghost::AroundCells, numGhostCells);
      t->requires(new_dw, d_MAlb->vel_CCLabel,         idx, patch,
			Ghost::AroundCells, numGhostCells);

      // Rajesh, feel free to change the names of these FC variables
      // to something more in line with Arches
      t->computes(new_dw, d_MAlb->d_xVelFCLabel,  idx, patch);
      t->computes(new_dw, d_MAlb->d_yVelFCLabel,  idx, patch);
      t->computes(new_dw, d_MAlb->d_zVelFCLabel,  idx, patch);
   }

   sched->addTask(t);
  }
#else
  NOT_FINISHED("new task stuff");
#endif
}

void MPMArches::scheduleComputeVoidFrac(SchedulerP& sched,
					const PatchSet* patches,
					const MaterialSet* matls)
{
#if 0
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    // primitive variable initialization
    Task* t=scinew Task("MPMArches::computeVoidFrac",
		        patch, old_dw, new_dw,
		        this, &MPMArches::computeVoidFrac);
    // this will ony give mpm materials...will change it to add arches material
    // later. This should work either ways
    int numMPMMatls  = d_sharedState->getNumMPMMatls();                      
    for (int m = 0; m < numMPMMatls; m++) {
      Material* matl = d_sharedState->getMPMMaterial( m );
      int idx = matl->getDWIndex();
      t->requires(new_dw, d_MAlb->cVolumeLabel,   idx,patch,Ghost::None,0);
    }
    int matlIndex = 0; // for Arches and MPMArches variables
    // for computing cell volume
    t->computes(new_dw, d_MAlb->void_frac_CCLabel, matlIndex, patch);
    sched->addTask(t);

  }
#else
  NOT_FINISHED("new task stuff");
#endif
}


void MPMArches::scheduleMomExchange(SchedulerP& sched,
				    const PatchSet* patches,
				    const MaterialSet* matls)
{
#if 0
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    // primitive variable initialization
    Task* t=scinew Task("MPMArches::doMomExchange",
		        patch, old_dw, new_dw,
		        this, &MPMArches::doMomExchange);
    int numGhostCells = 0;
    // since arches materials are not registered it'll only get
    // mpm materials
    int numMPMMatls  = d_sharedState->getNumMPMMatls();                      
    for (int m = 0; m < numMPMMatls; m++) {
      Material* matl = d_sharedState->getMPMMaterial( m );
      int idx = matl->getDWIndex();
      // Rajesh:
      // I created the CCToFC task to do get velocity on the faces
      // I can also put a mass there what is needed exactly.  Jim
      t->requires(new_dw, d_MAlb->d_xVelFCLabel,  idx, patch, Ghost::None, 0);
      t->requires(new_dw, d_MAlb->d_yVelFCLabel,  idx, patch, Ghost::None, 0);
      t->requires(new_dw, d_MAlb->d_zVelFCLabel,  idx, patch, Ghost::None, 0);
      // will compute vector of pressure forces and drag forces
      // computed at face center
      t->computes(new_dw, d_MAlb->momExDragForceFCXLabel, idx, patch);
      t->computes(new_dw, d_MAlb->momExPressureForceFCXLabel, idx, patch);
      t->computes(new_dw, d_MAlb->momExDragForceFCYLabel, idx, patch);
      t->computes(new_dw, d_MAlb->momExPressureForceFCYLabel, idx, patch);
      t->computes(new_dw, d_MAlb->momExDragForceFCZLabel, idx, patch);
      t->computes(new_dw, d_MAlb->momExPressureForceFCZLabel, idx, patch);
    }
    // from Arches get u, v, and w components of velocity
    // use old_dw since using at the beginning of the time advance loop
    int matlIndex = 0;
    t->requires(old_dw, d_Alab->d_cellTypeLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
    t->requires(old_dw, d_Alab->d_pressureSPBCLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
    t->requires(old_dw, d_Alab->d_uVelocitySPBCLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
    t->requires(old_dw, d_Alab->d_vVelocitySPBCLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
    t->requires(old_dw, d_Alab->d_wVelocitySPBCLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
    t->requires(old_dw, d_Alab->d_densityCPLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
    t->requires(old_dw, d_Alab->d_viscosityCTSLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
    // requires from mpmarches interface
    t->requires(new_dw,  d_MAlb->void_frac_CCLabel, matlIndex,patch,
		Ghost::None, numGhostCells);
    // computes su_drag[x,y,z], sp_drag[x,y,z] for arches
    t->computes(new_dw, d_MAlb->d_uVel_mmLinSrcLabel, matlIndex, patch);
    t->computes(new_dw, d_MAlb->d_uVel_mmNonlinSrcLabel, matlIndex, patch);
    t->computes(new_dw, d_MAlb->d_vVel_mmLinSrcLabel, matlIndex, patch);
    t->computes(new_dw, d_MAlb->d_vVel_mmNonlinSrcLabel, matlIndex, patch);
    t->computes(new_dw, d_MAlb->d_wVel_mmLinSrcLabel, matlIndex, patch);
    t->computes(new_dw, d_MAlb->d_wVel_mmNonlinSrcLabel, matlIndex, patch);
    sched->addTask(t);
  }
#else
  NOT_FINISHED("new task stuff");
#endif
}


//______________________________________________________________________
//
void MPMArches::interpolateNCToCC(const ProcessorGroup*,
				  const PatchSubset* patches,
				  const MaterialSubset* matls,
				  DataWarehouse* old_dw,
				  DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

  int numMatls = d_sharedState->getNumMPMMatls();
  Vector zero(0.0,0.0,0.);
  int numGhostCells = 1;
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
    int matlindex = mpm_matl->getDWIndex();

     // Create arrays for the grid data
     NCVariable<double > gmass, gvolume;
     NCVariable<Vector > gvelocity;
     CCVariable<double > cmass, cvolume;
     CCVariable<Vector > vel_CC;

     new_dw->allocate(cmass,     d_MAlb->cMassLabel,         matlindex, patch);
     new_dw->allocate(cvolume,   d_MAlb->cVolumeLabel,       matlindex, patch);
     new_dw->allocate(vel_CC,    d_MAlb->vel_CCLabel,        matlindex, patch);
      
     cmass.initialize(0.);
     cvolume.initialize(0.);
     vel_CC.initialize(zero); 

     new_dw->get(gmass,     Mlb->gMassLabel,           matlindex, patch, 
                        		 Ghost::AroundCells, numGhostCells);
     new_dw->get(gvolume,   Mlb->gVolumeLabel,         matlindex, patch,
					 Ghost::AroundCells, numGhostCells);
     new_dw->get(gvelocity, Mlb->gVelocityLabel,       matlindex, patch,
					 Ghost::AroundCells, numGhostCells);

     IntVector nodeIdx[8];

     for (CellIterator iter =patch->getExtraCellIterator();!iter.done();iter++){
       patch->findNodesFromCell(*iter,nodeIdx);
       for (int in=0;in<8;in++){
	 cmass[*iter]    += .125*gmass[nodeIdx[in]];
	 cvolume[*iter]  += .125*gvolume[nodeIdx[in]];
	 vel_CC[*iter]   +=      gvelocity[nodeIdx[in]]*.125*gmass[nodeIdx[in]];
       }
       vel_CC[*iter]      /= (cmass[*iter]     + d_SMALL_NUM);
     }

     new_dw->put(cmass,     d_MAlb->cMassLabel,         matlindex, patch);
     new_dw->put(cvolume,   d_MAlb->cVolumeLabel,       matlindex, patch);
     new_dw->put(vel_CC,    d_MAlb->vel_CCLabel,        matlindex, patch);
  }
  }
}

void MPMArches::interpolateCCToFC(const ProcessorGroup*,
				  const PatchSubset* patches,
				  const MaterialSubset* matls,
				  DataWarehouse* old_dw,
				  DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

  int numMatls = d_sharedState->getNumMPMMatls();
  Vector zero(0.0,0.0,0.);
  int numGhostCells = 1;
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
    int matlindex = mpm_matl->getDWIndex();

    CCVariable<double > cmass;
    CCVariable<Vector > vel_CC;
    SFCXVariable<double> xvelFC;
    SFCYVariable<double> yvelFC;
    SFCZVariable<double> zvelFC;

    new_dw->get(cmass,    d_MAlb->cMassLabel,         matlindex, patch,
					Ghost::AroundCells, numGhostCells);
    new_dw->get(vel_CC,   d_MAlb->vel_CCLabel,        matlindex, patch,
					Ghost::AroundCells, numGhostCells);

    new_dw->allocate(xvelFC,    d_MAlb->d_xVelFCLabel,   matlindex, patch);
    new_dw->allocate(yvelFC,    d_MAlb->d_yVelFCLabel,   matlindex, patch);
    new_dw->allocate(zvelFC,    d_MAlb->d_zVelFCLabel,   matlindex, patch);

    xvelFC.initialize(0.);
    yvelFC.initialize(0.);
    zvelFC.initialize(0.);
    double mass;

    for(CellIterator iter = patch->getExtraCellIterator();!iter.done(); iter++){
      IntVector curcell = *iter;
      //__________________________________
      //   B O T T O M   F A C E S
      //   Extend the computations into the left
      //   and right ghost cells
      if (curcell.y() >= (patch->getInteriorCellLowIndex()).y()) {
        IntVector adjcell(curcell.x(),curcell.y()-1,curcell.z());

	mass = cmass[curcell] + cmass[adjcell];
        yvelFC[curcell] = (vel_CC[curcell].y() * cmass[curcell] +
			   vel_CC[adjcell].y() * cmass[adjcell])/mass;
      }
      //__________________________________
      //   L E F T   F A C E S
      //   Extend the computations into the left
      //   and right ghost cells
      if (curcell.x() >= (patch->getInteriorCellLowIndex()).x()) {
	IntVector adjcell(curcell.x()-1,curcell.y(),curcell.z());

	mass = cmass[curcell] + cmass[adjcell];
        xvelFC[curcell] = (vel_CC[curcell].x() * cmass[curcell] +
			   vel_CC[adjcell].x() * cmass[adjcell])/mass;
      }
      //__________________________________
      //   L E F T   F A C E S
      //   Extend the computations into the left
      //   and right ghost cells
      if (curcell.z() >= (patch->getInteriorCellLowIndex()).z()) {
        IntVector adjcell(curcell.x(),curcell.y(),curcell.z()-1);

	mass = cmass[curcell] + cmass[adjcell];
        zvelFC[curcell] = (vel_CC[curcell].z() * cmass[curcell] +
			   vel_CC[adjcell].z() * cmass[adjcell])/mass;
      }
    }

    new_dw->put(xvelFC,    d_MAlb->d_xVelFCLabel,   matlindex, patch);
    new_dw->put(yvelFC,    d_MAlb->d_yVelFCLabel,   matlindex, patch);
    new_dw->put(zvelFC,    d_MAlb->d_zVelFCLabel,   matlindex, patch);
  }
  }
}
//
//
void 
MPMArches::computeVoidFrac(const ProcessorGroup*,
			   const PatchSubset* patches,
			   const MaterialSubset* matls,
			   DataWarehouse* old_dw,
			   DataWarehouse* new_dw) 
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

  int numMPMMatls = d_sharedState->getNumMPMMatls();
  vector<CCVariable<double> > mat_vol(numMPMMatls);
  int zeroGhostCells = 0;
  int matlindex = 0;
  for (int m = 0; m < numMPMMatls; m++) {
    Material* matl = d_sharedState->getMPMMaterial( m );
    int dwindex = matl->getDWIndex();
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl)
      new_dw->get(mat_vol[m], d_MAlb->cVolumeLabel,dwindex, patch, 
		  Ghost::None, zeroGhostCells);
  }
  CCVariable<double> void_frac;
  new_dw->allocate(void_frac, d_MAlb->void_frac_CCLabel, matlindex, patch); 
  void_frac.initialize(0);
  for (CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++) {
    double total_vol = patch->dCell().x()*patch->dCell().y()*patch->dCell().z();
    for (int m = 0; m < numMPMMatls; m++) 
      void_frac[*iter] += mat_vol[m][*iter]/total_vol;
  }
  new_dw->put(void_frac, d_MAlb->void_frac_CCLabel, matlindex, patch);
  }
}
    

    

//______________________________________________________________________
//
void MPMArches::doMomExchange(const ProcessorGroup*,
			      const PatchSubset* patches,
			      const MaterialSubset* matls,
			      DataWarehouse* old_dw,
			      DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    // since arches materials are not registered it'll only get
    // mpm materials
    int numMPMMatls  = d_sharedState->getNumMPMMatls();                      
    vector<NCVariable<Vector> > gvelocity(numMPMMatls);
    vector<CCVariable<Vector> > cc_pvelocity(numMPMMatls);
    vector<SFCXVariable<Vector> > dragForceX(numMPMMatls);
    vector<SFCXVariable<Vector> > pressForceX(numMPMMatls);
    vector<SFCYVariable<Vector> > dragForceY(numMPMMatls);
    vector<SFCYVariable<Vector> > pressForceY(numMPMMatls);
    vector<SFCZVariable<Vector> > dragForceZ(numMPMMatls);
    vector<SFCZVariable<Vector> > pressForceZ(numMPMMatls);
    CCVariable<int> cellType; // bc type info
    CCVariable<double> pressure;
    CCVariable<double> density;
    CCVariable<double> viscosity;
    SFCXVariable<double> uVelocity;
    SFCYVariable<double> vVelocity;
    SFCZVariable<double> wVelocity;
//multimaterial contribution to SP and SU terms in Arches momentum eqns
    SFCXVariable<double> uVelLinearSrc; 
    SFCXVariable<double> uVelNonlinearSrc;
    SFCYVariable<double> vVelLinearSrc; 
    SFCYVariable<double> vVelNonlinearSrc;
    SFCZVariable<double> wVelLinearSrc; 
    SFCZVariable<double> wVelNonlinearSrc; 


    int numGhostCells = 0;
    int matlIndex = 0;
    for (int m = 0; m < numMPMMatls; m++) {
      Material* matl = d_sharedState->getMPMMaterial( m );
      int idx = matl->getDWIndex();
      new_dw->get(gvelocity[m], Mlb->gVelocityLabel, idx, patch,
		  Ghost::None, numGhostCells);
      new_dw->get(cc_pvelocity[m], Mlb->gVelocityLabel, idx, patch,
		  Ghost::None, numGhostCells);
      new_dw->allocate(dragForceX[m], d_MAlb->momExDragForceFCXLabel, idx, patch);
      new_dw->allocate(pressForceX[m],d_MAlb->momExPressureForceFCXLabel,idx, patch);
      new_dw->allocate(dragForceY[m], d_MAlb->momExDragForceFCYLabel, idx, patch);
      new_dw->allocate(pressForceY[m],d_MAlb->momExPressureForceFCYLabel,idx, patch);
      new_dw->allocate(dragForceZ[m], d_MAlb->momExDragForceFCZLabel, idx, patch);
      new_dw->allocate(pressForceZ[m],d_MAlb->momExPressureForceFCZLabel,idx, patch);
    }

    old_dw->get(cellType, d_Alab->d_cellTypeLabel, matlIndex, patch,
		Ghost::None, numGhostCells);
    old_dw->get(pressure, d_Alab->d_pressureSPBCLabel, matlIndex, patch, 
		Ghost::None, numGhostCells);
    old_dw->get(uVelocity, d_Alab->d_uVelocitySPBCLabel, matlIndex, patch, 
		Ghost::None, numGhostCells);
    old_dw->get(vVelocity, d_Alab->d_vVelocitySPBCLabel, matlIndex, patch, 
		Ghost::None, numGhostCells);
    old_dw->get(wVelocity, d_Alab->d_wVelocitySPBCLabel, matlIndex, patch, 
		Ghost::None, numGhostCells);
    old_dw->get(density, d_Alab->d_densityCPLabel, matlIndex, patch, 
		Ghost::None, numGhostCells);
    old_dw->get(viscosity, d_Alab->d_viscosityCTSLabel, matlIndex, patch, 
		Ghost::None, numGhostCells);
    // patch geometry information
    PerPatch<CellInformationP> cellInfoP;
    if (new_dw->exists(d_Alab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_Alab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_Alab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    // computes su_drag[x,y,z], sp_drag[x,y,z] for arches
    new_dw->allocate(uVelLinearSrc, d_MAlb->d_uVel_mmLinSrcLabel, matlIndex, patch);
    new_dw->allocate(uVelNonlinearSrc, d_MAlb->d_uVel_mmNonlinSrcLabel,
		     matlIndex, patch);
    new_dw->allocate(vVelLinearSrc, d_MAlb->d_vVel_mmLinSrcLabel, matlIndex, patch);
    new_dw->allocate(vVelNonlinearSrc, d_MAlb->d_vVel_mmNonlinSrcLabel,
		     matlIndex, patch);
    new_dw->allocate(wVelLinearSrc, d_MAlb->d_wVel_mmLinSrcLabel, matlIndex, patch);
    new_dw->allocate(wVelNonlinearSrc, d_MAlb->d_wVel_mmNonlinSrcLabel,
		     matlIndex, patch);
  // need Jim's and Kumar's help to complete it
  }
}

