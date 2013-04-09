/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/MPM/RigidMPM.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/MPMFlags.h>
#include <CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <CCA/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>
#include <CCA/Components/MPM/PhysicalBC/ForceBC.h>
#include <CCA/Components/MPM/PhysicalBC/PressureBC.h>
#include <Core/Math/Matrix3.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <CCA/Components/MPM/MPMBoundCond.h>

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/MinMax.h>
#include <Core/Util/DebugStream.h>
#include <Core/Thread/Mutex.h>

#include <iostream>
#include <fstream>

using namespace Uintah;

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

void RigidMPM::problemSetup(const ProblemSpecP& prob_spec, 
                            const ProblemSpecP& restart_prob_spec, 
                            GridP& grid, SimulationStateP& sharedState)
{

  SerialMPM::problemSetup(prob_spec, restart_prob_spec,grid, sharedState);
  ProblemSpecP cfd_ps = prob_spec->findBlock("CFD");
  if(cfd_ps && UintahParallelComponent::d_myworld->myrank() == 0){
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

  new_dw->put(delt_vartype(999.0), lb->delTLabel, getLevel(patches));

}

void RigidMPM::scheduleComputeInternalForce(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  Task* t = scinew Task("MPM::computeInternalForce",
                    this, &RigidMPM::computeInternalForce);

  // require pStress so it will be saved in a checkpoint, 
  // allowing the user to restart using mpmice
  t->requires(Task::OldDW,lb->pStressLabel, Ghost::None);                
  
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

void RigidMPM::scheduleComputeAndIntegrateAcceleration(SchedulerP& sched,
                                                       const PatchSet* patches,
                                                       const MaterialSet* matls)
{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPM::scheduleComputeAndIntegrateAcceleration");

  Task* t = scinew Task("MPM::computeAndIntegrateAcceleration",
                        this, &RigidMPM::computeAndIntegrateAcceleration);

  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  t->requires(Task::NewDW, lb->gVelocityLabel,          Ghost::None);

  t->computes(lb->gVelocityStarLabel);
  t->computes(lb->gAccelerationLabel);

  sched->addTask(t, patches, matls);
}

void RigidMPM::computeAndIntegrateAcceleration(const ProcessorGroup*,
                                                const PatchSubset* patches,
                                                const MaterialSubset*,
                                                DataWarehouse* old_dw,
                                                DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing computeAndIntegrateAcceleration");

    Ghost::GhostType  gnone = Ghost::None;
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      // Get required variables for this patch
      constNCVariable<Vector> velocity;
      new_dw->get(velocity,     lb->gVelocityLabel,      dwi, patch, gnone, 0);

      // Create variables for the results
      NCVariable<Vector> velocity_star,acceleration;
      new_dw->allocateAndPut(velocity_star, lb->gVelocityStarLabel, dwi, patch);
      new_dw->allocateAndPut(acceleration,  lb->gAccelerationLabel, dwi, patch);

      acceleration.initialize(Vector(0.,0.,0.));

      for(NodeIterator iter=patch->getExtraNodeIterator();
                        !iter.done();iter++){
        IntVector c = *iter;
        velocity_star[c] = velocity[c];
      }
    }    // matls
  }
}

void RigidMPM::scheduleInterpolateToParticlesAndUpdate(SchedulerP& sched,
                                                       const PatchSet* patches,
                                                       const MaterialSet* matls)

{
  if (!flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
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
  t->requires(Task::OldDW, lb->pVolumeLabel,           Ghost::None);
  t->requires(Task::OldDW, lb->pParticleIDLabel,       Ghost::None);
  t->requires(Task::OldDW, lb->pTemperatureLabel,      Ghost::None);
  t->requires(Task::OldDW, lb->pVelocityLabel,         Ghost::None);
  t->requires(Task::OldDW, lb->pDispLabel,             Ghost::None);
  t->requires(Task::OldDW, lb->pSizeLabel,             Ghost::None);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel, Ghost::None);

  if(flags->d_with_ice){
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
  t->computes(lb->pVelGradLabel_preReloc);
  t->computes(lb->pDeformationMeasureLabel_preReloc);
  t->computes(lb->pXXLabel);

  t->computes(lb->KineticEnergyLabel);
  t->computes(lb->ThermalEnergyLabel);
  t->computes(lb->CenterOfMassPositionLabel);
  t->computes(lb->TotalMomentumLabel);
  
  // debugging scalar
  if(flags->d_with_color) {
    t->requires(Task::OldDW, lb->pColorLabel,  Ghost::None);
    t->computes(lb->pColorLabel_preReloc);
  }

  MaterialSubset* z_matl = scinew MaterialSubset();
  z_matl->add(0);
  z_matl->addReference();
  t->requires(Task::OldDW, lb->NC_CCweightLabel, z_matl, Ghost::None);
  t->computes(             lb->NC_CCweightLabel, z_matl);

  sched->addTask(t, patches, matls);

  // The task will have a reference to z_matl
  if (z_matl->removeReference())
    delete z_matl; // shouln't happen, but...
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
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());
    vector<Vector> d_S(interpolator->size());
   

    // Performs the interpolation from the cell vertices of the grid
    // acceleration and velocity to the particles to update their
    // velocity and position respectively
 
    // DON'T MOVE THESE!!!
    double thermal_energy = 0.0;
    Vector CMX(0.0,0.0,0.0);
    Vector total_mom(0.0,0.0,0.0);
    double ke=0;
    int numMPMMatls=d_sharedState->getNumMPMMatls();
    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );

    double move_particles=1.;
    if(!flags->d_doGridReset){
      move_particles=0.;
    }
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      // Get the arrays of particle values to be changed
      constParticleVariable<Point> px;
      ParticleVariable<Point> pxnew,pxx;
      constParticleVariable<Vector> pvelocity;
      constParticleVariable<Matrix3> psize;
      ParticleVariable<Vector> pvelocitynew;
      ParticleVariable<Matrix3> psizeNew;
      constParticleVariable<double> pmass, pTemperature,pvolume;
      ParticleVariable<double> pmassNew,pTempNew,pVolNew;
      constParticleVariable<long64> pids;
      ParticleVariable<long64> pids_new;
      constParticleVariable<Vector> pdisp;
      ParticleVariable<Vector> pdispnew;
      constParticleVariable<Matrix3> pFOld;
      ParticleVariable<Matrix3> pFNew,pVelGrad;

      // for thermal stress analysis
      ParticleVariable<double> pTempPreNew;       

      // Get the arrays of grid data on which the new part. values depend
      constNCVariable<Vector> gacceleration;
      constNCVariable<double> gTemperatureRate, gTemperature, gTemperatureNoBC;
      constNCVariable<double> dTdt;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      old_dw->get(px,                    lb->pXLabel,                     pset);
      old_dw->get(pdisp,                 lb->pDispLabel,                  pset);
      old_dw->get(pmass,                 lb->pMassLabel,                  pset);
      old_dw->get(pids,                  lb->pParticleIDLabel,            pset);
      old_dw->get(pvolume,               lb->pVolumeLabel,                pset);
      old_dw->get(pvelocity,             lb->pVelocityLabel,              pset);
      old_dw->get(pTemperature,          lb->pTemperatureLabel,           pset);
      old_dw->get(pFOld,                 lb->pDeformationMeasureLabel,    pset);

      new_dw->allocateAndPut(pvelocitynew, lb->pVelocityLabel_preReloc,   pset);
      new_dw->allocateAndPut(pxnew,        lb->pXLabel_preReloc,          pset);
      new_dw->allocateAndPut(pxx,          lb->pXXLabel,                  pset);
      new_dw->allocateAndPut(pdispnew,     lb->pDispLabel_preReloc,       pset);
      new_dw->allocateAndPut(pmassNew,     lb->pMassLabel_preReloc,       pset);
      new_dw->allocateAndPut(pVolNew,      lb->pVolumeLabel_preReloc,     pset);
      new_dw->allocateAndPut(pids_new,     lb->pParticleIDLabel_preReloc, pset);
      new_dw->allocateAndPut(pTempNew,     lb->pTemperatureLabel_preReloc,pset);
      new_dw->allocateAndPut(pTempPreNew, lb->pTempPreviousLabel_preReloc,pset);
      new_dw->allocateAndPut(pVelGrad,    lb->pVelGradLabel_preReloc,     pset);
      new_dw->allocateAndPut(pFNew,       lb->pDeformationMeasureLabel_preReloc,
                                                                          pset);
     
      pids_new.copyData(pids);
      old_dw->get(psize,               lb->pSizeLabel,                 pset);
      new_dw->allocateAndPut(psizeNew, lb->pSizeLabel_preReloc,        pset);
      psizeNew.copyData(psize);
      pVolNew.copyData(pvolume);

      //Carry forward NC_CCweight
      constNCVariable<double> NC_CCweight;
      NCVariable<double> NC_CCweight_new;
      Ghost::GhostType  gnone = Ghost::None;
      old_dw->get(NC_CCweight,      lb->NC_CCweightLabel,  0, patch, gnone, 0);
      new_dw->allocateAndPut(NC_CCweight_new, lb->NC_CCweightLabel,0,patch);
      NC_CCweight_new.copyData(NC_CCweight);

      Ghost::GhostType  gac = Ghost::AroundCells;
      new_dw->get(gTemperatureRate,lb->gTemperatureRateLabel,dwi,patch,gac,NGP);
      new_dw->get(gTemperature,    lb->gTemperatureLabel,    dwi,patch,gac,NGP);
      new_dw->get(gTemperatureNoBC,lb->gTemperatureNoBCLabel,dwi,patch,gac,NGP);
      new_dw->get(gacceleration,   lb->gAccelerationLabel,   dwi,patch,gac,NGP);
      if(flags->d_with_ice){
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

      double Cp=mpm_matl->getSpecificHeat();
      Matrix3 Identity; Identity.Identity();

      // Loop over particles
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // Get the node indices that surround the cell
        interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,
                                                         psize[idx],pFOld[idx]);

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
        // pxx is only useful if we're not in normal grid resetting mode.
        pxx[idx]             = px[idx];
        pTempPreNew[idx]     = pTemperature[idx]; // for thermal stress
        pVelGrad[idx]        = Matrix3(0.);
        pFNew[idx]           = Identity;

        thermal_energy += pTempNew[idx] * pmass[idx] * Cp;
        ke += .5*pmass[idx]*pvelocitynew[idx].length2();
        CMX = CMX + (pxnew[idx]*pmass[idx]).asVector();
        total_mom += pvelocitynew[idx]*pmass[idx];
      }

      //__________________________________
      //  particle debugging label-- carry forward
      if (flags->d_with_color) {
        constParticleVariable<double> pColor;
        ParticleVariable<double>pColor_new;
        old_dw->get(pColor, lb->pColorLabel, pset);
        new_dw->allocateAndPut(pColor_new, lb->pColorLabel_preReloc, pset);
        pColor_new.copyData(pColor);
      }    

      ParticleSubset* delset = scinew ParticleSubset(0, dwi, patch);
      new_dw->deleteParticles(delset);      
    }

    // DON'T MOVE THESE!!!
    new_dw->put(sum_vartype(ke),     lb->KineticEnergyLabel);
    new_dw->put(sumvec_vartype(CMX), lb->CenterOfMassPositionLabel);
    new_dw->put(sumvec_vartype(total_mom),   lb->TotalMomentumLabel);
    new_dw->put(sum_vartype(thermal_energy), lb->ThermalEnergyLabel);

    delete interpolator;
  }
}
