#include <Packages/Uintah/CCA/Components/MPM/RigidMPM.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMFlags.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/ForceBC.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/PressureBC.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleVariable.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Variables/SoleVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMBoundCond.h>

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/MinMax.h>
#include <Core/Util/DebugStream.h>
#include <Core/Thread/Mutex.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <fstream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;
using namespace SCIRun;

using namespace std;

#undef INTEGRAL_TRACTION

static DebugStream cout_doing("RIGID_MPM", false);

// From ThreadPool.cc:  Used for syncing cerr'ing so it is easier to read.
extern Mutex cerrLock;

RigidMPM::RigidMPM(const ProcessorGroup* myworld) :
  SerialMPM(myworld)
{
}

RigidMPM::~RigidMPM()
{
}

void RigidMPM::problemSetup(const ProblemSpecP& prob_spec, GridP& grid,
                       SimulationStateP& sharedState)
{

  SerialMPM::problemSetup(prob_spec, grid, sharedState);
  ProblemSpecP cfd_ps = prob_spec->findBlock("CFD");
  if(cfd_ps){
    cout << "\n__________________________________"<< endl;
    cout << "  W A R N I N G :  " << endl;
    cout << "  You must use stiff MPM material properties" << endl;
    cout << "  to get the correct pressure solution in rmpmice" << endl;
    cout << "__________________________________\n"<< endl;
  }
}

void RigidMPM::computeStressTensor(const ProcessorGroup*,
				    const PatchSubset* patches,
				    const MaterialSubset* ,
				    DataWarehouse* old_dw,
				    DataWarehouse* new_dw)
{
  if (cout_doing.active())
    cout_doing <<"Doing computeStressTensor " <<"\t\t\t\t RigidMPM"<< endl;

  for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->carryForward(patches, mpm_matl, old_dw, new_dw);
  }

  new_dw->put(delt_vartype(getLevel(patches)->adjustDelt(999.0)), 
              lb->delTLabel);

}

void RigidMPM::scheduleComputeInternalForce(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex()))
    return;

  Task* t = scinew Task("MPM::computeInternalForce",
                    this, &RigidMPM::computeInternalForce);
  sched->addTask(t, patches, matls);
}

void RigidMPM::computeInternalForce(const ProcessorGroup*,
				    const PatchSubset* patches,
				    const MaterialSubset* ,
				    DataWarehouse*,
				    DataWarehouse*)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    if (cout_doing.active()) {
      cout_doing <<"Doing computeInternalForce on patch " << patch->getID()
		 <<"\t\t\t RigidMPM"<< endl;
    }

  }
}

void RigidMPM::scheduleSolveEquationsMotion(SchedulerP& sched,
                                             const PatchSet* patches,
                                             const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex()))
    return;

  Task* t = scinew Task("MPM::solveEquationsMotion",
                    this, &RigidMPM::solveEquationsMotion);

  t->computes(lb->gAccelerationLabel);
  sched->addTask(t, patches, matls);
}

void RigidMPM::solveEquationsMotion(const ProcessorGroup*,
				     const PatchSubset* patches,
				     const MaterialSubset*,
				     DataWarehouse* old_dw,
				     DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    if (cout_doing.active()) {
      cout_doing <<"Doing solveEquationsMotion on patch " << patch->getID()
		 <<"\t\t\t RigidMPM"<< endl;
    }

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );

    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
 
      // Create variables for the results
      NCVariable<Vector> acceleration;
      new_dw->allocateAndPut(acceleration, lb->gAccelerationLabel, dwi, patch);
      acceleration.initialize(Vector(0.,0.,0.));

    }
  }
}

void RigidMPM::scheduleInterpolateToParticlesAndUpdate(SchedulerP& sched,
                                                       const PatchSet* patches,
                                                       const MaterialSet* matls)

{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex()))
    return;

  Task* t=scinew Task("MPM::interpolateToParticlesAndUpdate",
                      this, &RigidMPM::interpolateToParticlesAndUpdate);


  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  Ghost::GhostType  gac = Ghost::AroundCells;
  t->requires(Task::NewDW, lb->gTemperatureRateLabel,  gac,NGN);
  t->requires(Task::NewDW, lb->gTemperatureLabel,      gac,NGN);
  t->requires(Task::NewDW, lb->gTemperatureNoBCLabel,  gac,NGN);
  t->requires(Task::NewDW, lb->gAccelerationLabel,     gac,NGN);
  t->requires(Task::OldDW, lb->pXLabel,                Ghost::None);
  t->requires(Task::OldDW, lb->pMassLabel,             Ghost::None);
  t->requires(Task::OldDW, lb->pParticleIDLabel,       Ghost::None);
  t->requires(Task::OldDW, lb->pTemperatureLabel,      Ghost::None);
  t->requires(Task::OldDW, lb->pVelocityLabel,         Ghost::None);
  t->requires(Task::OldDW, lb->pDispLabel,             Ghost::None);
  t->requires(Task::OldDW, lb->pSizeLabel,             Ghost::None);
  t->requires(Task::NewDW, lb->pVolumeDeformedLabel,   Ghost::None);
  // for thermal stress analysis
  t->requires(Task::NewDW, lb->pTempCurrentLabel,      Ghost::None);  

  // The dampingCoeff (alpha) is 0.0 for standard usage, otherwise
  // it is determined by the damping rate if the artificial damping
  // coefficient Q is greater than 0.0
  if (flags->d_artificialDampCoeff > 0.0) {
    t->requires(Task::OldDW, lb->pDampingCoeffLabel);
    t->requires(Task::NewDW, lb->pDampingRateLabel);
    t->computes(lb->pDampingCoeffLabel);
  }

  if(d_with_ice){
    t->requires(Task::NewDW, lb->dTdt_NCLabel,         gac,NGN);
  }

  t->computes(lb->pDispLabel_preReloc);
  t->computes(lb->pVelocityLabel_preReloc);
  t->computes(lb->pXLabel_preReloc);
  t->computes(lb->pParticleIDLabel_preReloc);
  t->computes(lb->pTemperatureLabel_preReloc);
  t->computes(lb->pTempPreviousLabel_preReloc); // for thermal stress 
  t->computes(lb->pMassLabel_preReloc);
  t->computes(lb->pVolumeLabel_preReloc);
  t->computes(lb->pSizeLabel_preReloc);
  t->computes(lb->pXXLabel);

  t->computes(lb->KineticEnergyLabel);
  t->computes(lb->ThermalEnergyLabel);
  t->computes(lb->CenterOfMassPositionLabel);
  t->computes(lb->CenterOfMassVelocityLabel);
  
  // debugging scalar
  if(flags->d_with_color) {
    t->requires(Task::OldDW, lb->pColorLabel,  Ghost::None);
    t->computes(lb->pColorLabel_preReloc);
  }
  
  sched->addTask(t, patches, matls);
}

void RigidMPM::interpolateToParticlesAndUpdate(const ProcessorGroup*,
                                                const PatchSubset* patches,
                                                const MaterialSubset* ,
                                                DataWarehouse* old_dw,
                                                DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    if (cout_doing.active()) {
      cout_doing <<"Doing interpolateToParticlesAndUpdate on patch " 
		 << patch->getID() << "\t MPM"<< endl;
    }

    ParticleInterpolator* interpolator = flags->d_interpolator->clone(patch);
    vector<IntVector> ni;
    ni.reserve(interpolator->size());
    vector<double> S;
    S.reserve(interpolator->size());
    vector<Vector> d_S;
    d_S.reserve(interpolator->size());


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
    old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );

    // Artificial Damping 
    if (flags->d_artificialDampCoeff > 0.0) {
      double alphaDot = 0.0;
      double alpha = 0.0;
      max_vartype dampingCoeff; 
      sum_vartype dampingRate;
      old_dw->get(dampingCoeff, lb->pDampingCoeffLabel);
      new_dw->get(dampingRate, lb->pDampingRateLabel);
      alpha = (double) dampingCoeff;
      alphaDot = (double) dampingRate;
      alpha += alphaDot*delT; // Calculate damping coefficient from damping rate
      new_dw->put(max_vartype(alpha), lb->pDampingCoeffLabel);
    }

    double move_particles=1.;
    if(!d_doGridReset){
      move_particles=0.;
    }
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      // Get the arrays of particle values to be changed
      constParticleVariable<Point> px;
      ParticleVariable<Point> pxnew,pxx;
      constParticleVariable<Vector> pvelocity, psize;
      ParticleVariable<Vector> pvelocitynew, psizeNew;
      constParticleVariable<double> pmass, pvolume, pTemperature;
      ParticleVariable<double> pmassNew,pvolumeNew,pTempNew;
      constParticleVariable<long64> pids;
      ParticleVariable<long64> pids_new;
      constParticleVariable<Vector> pdisp;
      ParticleVariable<Vector> pdispnew;

      // for thermal stress analysis
      constParticleVariable<double> pTempCurrent;
      ParticleVariable<double> pTempPreNew;       

      // Get the arrays of grid data on which the new part. values depend
      constNCVariable<Vector> gvelocity_star, gacceleration;
      constNCVariable<double> gTemperatureRate, gTemperature, gTemperatureNoBC;
      constNCVariable<double> dTdt;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      old_dw->get(px,                    lb->pXLabel,                     pset);
      old_dw->get(pdisp,                 lb->pDispLabel,                  pset);
      old_dw->get(pmass,                 lb->pMassLabel,                  pset);
      old_dw->get(pids,                  lb->pParticleIDLabel,            pset);
      new_dw->get(pvolume,               lb->pVolumeDeformedLabel,        pset);
      old_dw->get(pvelocity,             lb->pVelocityLabel,              pset);
      old_dw->get(pTemperature,          lb->pTemperatureLabel,           pset);
      // for thermal stress analysis
      new_dw->get(pTempCurrent,          lb->pTempCurrentLabel,           pset);      
      new_dw->allocateAndPut(pvelocitynew, lb->pVelocityLabel_preReloc,   pset);
      new_dw->allocateAndPut(pxnew,        lb->pXLabel_preReloc,          pset);
      new_dw->allocateAndPut(pxx,          lb->pXXLabel,                  pset);
      new_dw->allocateAndPut(pdispnew,     lb->pDispLabel_preReloc,       pset);
      new_dw->allocateAndPut(pmassNew,     lb->pMassLabel_preReloc,       pset);
      new_dw->allocateAndPut(pvolumeNew,   lb->pVolumeLabel_preReloc,     pset);
      new_dw->allocateAndPut(pids_new,     lb->pParticleIDLabel_preReloc, pset);
      new_dw->allocateAndPut(pTempNew,     lb->pTemperatureLabel_preReloc,pset);
      // for thermal stress analysis
      new_dw->allocateAndPut(pTempPreNew, lb->pTempPreviousLabel_preReloc, pset);
     
      ParticleSubset* delset = scinew ParticleSubset(pset->getParticleSet(),
                                                     false,dwi,patch, 0);

      pids_new.copyData(pids);
      old_dw->get(psize,               lb->pSizeLabel,                 pset);
      new_dw->allocateAndPut(psizeNew, lb->pSizeLabel_preReloc,        pset);
      psizeNew.copyData(psize);

      Ghost::GhostType  gac = Ghost::AroundCells;
      new_dw->get(gTemperatureRate,lb->gTemperatureRateLabel,dwi,patch,gac,NGP);
      new_dw->get(gTemperature,    lb->gTemperatureLabel,    dwi,patch,gac,NGP);
      new_dw->get(gTemperatureNoBC,lb->gTemperatureNoBCLabel,dwi,patch,gac,NGP);
      new_dw->get(gacceleration,   lb->gAccelerationLabel,   dwi,patch,gac,NGP);
      if(d_with_ice){
        new_dw->get(dTdt,          lb->dTdt_NCLabel,         dwi,patch,gac,NGP);
      }
      else{
        NCVariable<double> dTdt_create;
        new_dw->allocateTemporary(dTdt_create,                   patch,gac,NGP);
        dTdt_create.initialize(0.);
        dTdt = dTdt_create;                         // reference created data
      }

      // Get the constitutive model (needed for plastic temperature
      // update) and get the plastic temperature from the plasticity
      // model
      // For RigidMPM, this isn't done

      const Level* lvl = patch->getLevel();
      double Cp=mpm_matl->getSpecificHeat();

      // Loop over particles
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // Get the node indices that surround the cell
	interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,
							    psize[idx]);

        double tempRate = 0.0;
        Vector acc(0.0,0.0,0.0);

        // Accumulate the contribution from each surrounding vertex
        // All we care about is the temperature field, everything else
        // should be zero.
        for (int k = 0; k < flags->d_8or27; k++) {
          tempRate += (gTemperatureRate[ni[k]] + dTdt[ni[k]])   * S[k];
          acc      += gacceleration[ni[k]]   * S[k];
        }
        pTempNew[idx]        = pTemperature[idx] + tempRate  * delT;
        pvelocitynew[idx]    = pvelocity[idx]+acc*delT;
        // If there is no adiabatic heating, add the plastic temperature
        // to the particle temperature

        // Update the particle's position and velocity
        pxnew[idx]           = px[idx] + pvelocity[idx]*delT;
        pdispnew[idx]        = pvelocity[idx]*delT;
        pmassNew[idx]        = pmass[idx];
        pvolumeNew[idx]      = pvolume[idx];
        // pxx is only useful if we're not in normal grid resetting mode.
        pxx[idx]             = px[idx];
        pTempPreNew[idx]     = pTempCurrent[idx]; // for thermal stress

        thermal_energy += pTemperature[idx] * pmass[idx] * Cp;
        ke += .5*pmass[idx]*pvelocitynew[idx].length2();
        CMX = CMX + (pxnew[idx]*pmass[idx]).asVector();
        CMV += pvelocitynew[idx]*pmass[idx];
      }
      
      
      //__________________________________
      //  hardwiring for Northrup Grumman nozzle
      #define RigidMPM_1
      #include "../MPMICE/NGC_nozzle.i"
      #undef RigidMPM_1
      
      
      //__________________________________
      //  particle debugging label-- carry forward
      if (flags->d_with_color) {
        constParticleVariable<double> pColor;
        ParticleVariable<double>pColor_new;
        old_dw->get(pColor, lb->pColorLabel, pset);
        new_dw->allocateAndPut(pColor_new, lb->pColorLabel_preReloc, pset);
        pColor_new.copyData(pColor);
      }    

      // Delete particles that have left the domain
      // This is only needed if extra cells are being used.
      if(flags->d_8or27==27){
        for(ParticleSubset::iterator iter  = pset->begin();
                                     iter != pset->end(); iter++){
          particleIndex idx = *iter;
          bool pointInReal = lvl->containsPointInRealCells(pxnew[idx]);
          bool pointInAny = lvl->containsPoint(pxnew[idx]);
          if((!pointInReal && pointInAny)){
            delset->addParticle(idx);
          }
        }
      }
      new_dw->deleteParticles(delset);      
    }

    // DON'T MOVE THESE!!!
    new_dw->put(sum_vartype(ke),     lb->KineticEnergyLabel);
    new_dw->put(sumvec_vartype(CMX), lb->CenterOfMassPositionLabel);
    new_dw->put(sumvec_vartype(CMV), lb->CenterOfMassVelocityLabel);
    new_dw->put(sum_vartype(thermal_energy), lb->ThermalEnergyLabel);

    delete interpolator;
  }
}
