// MPMArches.cc

#include <Packages/Uintah/CCA/Components/MPMArches/MPMArches.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMPhysicalModules.h>
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
				SchedulerP& sched,
				DataWarehouseP& dw)
{
  d_mpm->scheduleInitialize(      level, sched, dw);
  d_arches->scheduleInitialize(      level, sched, dw);
  // add compute void fraction to be used by gas density and viscosity
  cerr << "Doing Initialization \t\t\t MPMArches" <<endl;
  cerr << "--------------------------------\n"<<endl; 
}
//______________________________________________________________________
//
void MPMArches::scheduleComputeStableTimestep(const LevelP& level,
					   SchedulerP& sched,
					   DataWarehouseP& dw)
{
  // Schedule computing the Arches stable timestep
  d_arches->scheduleComputeStableTimestep(level, sched, dw);
  // MPM stable timestep is a by product of the CM
}

//______________________________________________________________________
//
void MPMArches::scheduleTimeAdvance(double time, double delt,
				 const LevelP&   level,
				 SchedulerP&     sched,
				 DataWarehouseP& old_dw, 
				 DataWarehouseP& new_dw)
{
  // Rajesh/Kumar  Because interpolation from particles to nodes
  // is the fist thing done in our algorithm, it is, unfortunately,
  // going to be necessary to put that step of the algorithm in
  // prior to the NCToCC stuff, and so of course the remaining
  // MPM steps will follow.  Without doing a heap of other work,
  // I don't see any way around this, and the only real cost is
  // to ugly this up a bit.

  for(Level::const_patchIterator iter=level->patchesBegin();
			       iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;

    if(d_fracture) {
      d_mpm->scheduleSetPositions(patch,sched,old_dw,new_dw);
      d_mpm->scheduleComputeBoundaryContact(patch,sched,old_dw,new_dw);
      d_mpm->scheduleComputeConnectivity(patch,sched,old_dw,new_dw);
    }
    d_mpm->scheduleInterpolateParticlesToGrid(patch,sched,old_dw,new_dw);
    if (MPMPhysicalModules::thermalContactModel) {
       d_mpm->scheduleComputeHeatExchange(patch,sched,old_dw,new_dw);
    }
  }

  // interpolate mpm properties from node center to cell center/ face center
  // these computed variables are used by void fraction and mom exchange
  scheduleInterpolateNCToCC(level, sched, old_dw, new_dw);
  // Velocity and temperature are needed at the faces, right?
  scheduleInterpolateCCToFC(level, sched, old_dw, new_dw);

  // for explicit calculation, exchange will be at the beginning
  scheduleComputeVoidFrac(level, sched, old_dw, new_dw);

  // for heat transfer HeatExchangeCoeffs will be called
  scheduleMomExchange(level, sched, old_dw, new_dw);

  d_arches->scheduleTimeAdvance(time, delt, level, sched, old_dw, new_dw);

  for(Level::const_patchIterator iter=level->patchesBegin();
			       iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;

    d_mpm->scheduleExMomInterpolated(               patch,sched,old_dw,new_dw);
    d_mpm->scheduleComputeStressTensor(             patch,sched,old_dw,new_dw);
    d_mpm->scheduleComputeInternalForce(            patch,sched,old_dw,new_dw);
    d_mpm->scheduleComputeInternalHeatRate(         patch,sched,old_dw,new_dw);
    d_mpm->scheduleSolveEquationsMotion(            patch,sched,old_dw,new_dw);
    d_mpm->scheduleSolveHeatEquations(              patch,sched,old_dw,new_dw);
    d_mpm->scheduleIntegrateAcceleration(           patch,sched,old_dw,new_dw);
    d_mpm->scheduleIntegrateTemperatureRate(        patch,sched,old_dw,new_dw);
    d_mpm->scheduleExMomIntegrated(                 patch,sched,old_dw,new_dw);
    d_mpm->scheduleInterpolateToParticlesAndUpdate( patch,sched,old_dw,new_dw);
    d_mpm->scheduleComputeMassRate(                 patch,sched,old_dw,new_dw);
    if(d_fracture) {
      d_mpm->scheduleComputeFracture(               patch,sched,old_dw,new_dw);
    }
    d_mpm->scheduleCarryForwardVariables(           patch,sched,old_dw,new_dw);
  }

  int numMPMMatls = d_sharedState->getNumMPMMatls();
  sched->scheduleParticleRelocation(level, old_dw, new_dw,
                                    Mlb->pXLabel_preReloc,
                                    Mlb->d_particleState_preReloc,
                                    Mlb->pXLabel, Mlb->d_particleState,
                                    numMPMMatls);
}
//
//
void MPMArches::scheduleInterpolateNCToCC(const LevelP& level,
                                       SchedulerP& sched,
                                       DataWarehouseP& old_dw,
                                       DataWarehouseP& new_dw)
{
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
}


void MPMArches::scheduleInterpolateCCToFC(const LevelP& level,
                                       SchedulerP& sched,
                                       DataWarehouseP& old_dw,
                                       DataWarehouseP& new_dw)
{
   /* interpolateNCToCC */
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
      t->computes(new_dw, d_MAlb->d_xMomFCLabel,  idx, patch);
      t->computes(new_dw, d_MAlb->d_yMomFCLabel,  idx, patch);
      t->computes(new_dw, d_MAlb->d_zMomFCLabel,  idx, patch);
   }

   sched->addTask(t);
  }
}

void MPMArches::scheduleComputeVoidFrac(const LevelP& level,
				       SchedulerP& sched,
				       DataWarehouseP& old_dw,
				       DataWarehouseP& new_dw)
{
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
}


void MPMArches::scheduleMomExchange(const LevelP& level,
                                   SchedulerP& sched,
                                   DataWarehouseP& old_dw,
                                   DataWarehouseP& new_dw)
{
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
      // I created the CCToFC task to do get momentum on the faces
      // I can also put a mass there to get a velocity, let me know
      // what is needed exactly.  Jim
      t->requires(new_dw, d_MAlb->d_xMomFCLabel,  idx, patch, Ghost::None, 0);
      t->requires(new_dw, d_MAlb->d_yMomFCLabel,  idx, patch, Ghost::None, 0);
      t->requires(new_dw, d_MAlb->d_zMomFCLabel,  idx, patch, Ghost::None, 0);
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

}


//______________________________________________________________________
//
void MPMArches::interpolateNCToCC(const ProcessorGroup*,
			       const Patch* patch,
			       DataWarehouseP& old_dw,
			       DataWarehouseP& new_dw)
{
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

void MPMArches::interpolateCCToFC(const ProcessorGroup*,
			       const Patch* patch,
			       DataWarehouseP& old_dw,
			       DataWarehouseP& new_dw)
{
  int numMatls = d_sharedState->getNumMPMMatls();
  Vector zero(0.0,0.0,0.);
  int numGhostCells = 1;
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
    int matlindex = mpm_matl->getDWIndex();

    CCVariable<double > cmass;
    CCVariable<Vector > vel_CC;
    SFCXVariable<double> xmomFC;
    SFCYVariable<double> ymomFC;
    SFCZVariable<double> zmomFC;

    new_dw->get(cmass,    d_MAlb->cMassLabel,         matlindex, patch,
					Ghost::AroundCells, numGhostCells);
    new_dw->get(vel_CC,   d_MAlb->vel_CCLabel,        matlindex, patch,
					Ghost::AroundCells, numGhostCells);

    new_dw->allocate(xmomFC,    d_MAlb->d_xMomFCLabel,   matlindex, patch);
    new_dw->allocate(ymomFC,    d_MAlb->d_yMomFCLabel,   matlindex, patch);
    new_dw->allocate(zmomFC,    d_MAlb->d_zMomFCLabel,   matlindex, patch);

    xmomFC.initialize(0.);
    ymomFC.initialize(0.);
    zmomFC.initialize(0.);

    for(CellIterator iter = patch->getExtraCellIterator();!iter.done(); iter++){
      IntVector curcell = *iter;
      //__________________________________
      //   B O T T O M   F A C E S
      //   Extend the computations into the left
      //   and right ghost cells
      if (curcell.y() >= (patch->getInteriorCellLowIndex()).y()) {
        IntVector adjcell(curcell.x(),curcell.y()-1,curcell.z());

        ymomFC[curcell] = 0.5*(vel_CC[curcell].y() * cmass[curcell] +
			       vel_CC[adjcell].y() * cmass[adjcell]);
      }
      //__________________________________
      //   L E F T   F A C E S
      //   Extend the computations into the left
      //   and right ghost cells
      if (curcell.x() >= (patch->getInteriorCellLowIndex()).x()) {
	IntVector adjcell(curcell.x()-1,curcell.y(),curcell.z());

        xmomFC[curcell] = 0.5*(vel_CC[curcell].x() * cmass[curcell] +
			       vel_CC[adjcell].x() * cmass[adjcell]);
      }
      //__________________________________
      //   L E F T   F A C E S
      //   Extend the computations into the left
      //   and right ghost cells
      if (curcell.z() >= (patch->getInteriorCellLowIndex()).z()) {
        IntVector adjcell(curcell.x(),curcell.y(),curcell.z()-1);

        zmomFC[curcell] = 0.5*(vel_CC[curcell].z() * cmass[curcell] +
			       vel_CC[adjcell].z() * cmass[adjcell]);
      }
    }

    new_dw->put(xmomFC,    d_MAlb->d_xMomFCLabel,   matlindex, patch);
    new_dw->put(ymomFC,    d_MAlb->d_yMomFCLabel,   matlindex, patch);
    new_dw->put(zmomFC,    d_MAlb->d_zMomFCLabel,   matlindex, patch);
  }
}
//
//
void 
MPMArches::computeVoidFrac(const ProcessorGroup*,
			   const Patch* patch,
			   DataWarehouseP& old_dw,
			   DataWarehouseP& new_dw) 
{
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
    

    

//______________________________________________________________________
//
void MPMArches::doMomExchange(const ProcessorGroup*,
                             const Patch* patch,
                             DataWarehouseP& old_dw,
                             DataWarehouseP& new_dw)
{
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

