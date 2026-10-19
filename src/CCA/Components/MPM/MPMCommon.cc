/*
 * The MIT License
 *
 * Copyright (c) 1997-2026 The University of Utah
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

#include <CCA/Components/MPM/MPMCommon.h> 
#include <CCA/Components/MPM/Core/MPMFlags.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/DamageModel.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/ErosionModel.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/ProblemSpec/ProblemSpec.h>

using namespace std;
using namespace Uintah;

static DebugStream cout_doing("MPM", false);


MaterialManagerP MPMCommon::d_matlManager;  // needed since it is static;

//______________________________________________________________________
//
MPMCommon::MPMCommon( const MaterialManagerP materialManager)
{
  lb = scinew MPMLabel();
  d_matlManager = materialManager;

}

MPMCommon::~MPMCommon()
{
  delete lb;
}

//______________________________________________________________________
//
void MPMCommon::materialProblemSetup(const ProblemSpecP& prob_spec, 
                                     MPMFlags* flags, 
                                     bool isRestart)
{
  d_flags = flags;

  //! so all components can know how many particle ghost cells to ask for
  d_flags->d_particle_ghost_type  = particle_ghost_type;
  d_flags->d_particle_ghost_layer = particle_ghost_layer;
  
  //Search for the MaterialProperties block and then get the MPM section
  ProblemSpecP mat_ps     = prob_spec->findBlockWithOutAttribute( "MaterialProperties" );
  ProblemSpecP mpm_mat_ps = mat_ps->findBlock( "MPM" );
  for( ProblemSpecP ps = mpm_mat_ps->findBlock( "material" ); ps != nullptr; ps = ps->findNextBlock( "material" ) ) {
    string index( "" );
    ps->getAttribute( "index",index );
    stringstream id(index);
    const int DEFAULT_VALUE = -1;
    int index_val = DEFAULT_VALUE;

    id >> index_val;

    if( !id ) {
      // stringstream parsing failed... on many (most) systems, the
      // original value assigned to index_val would be left
      // intact... but on some systems it inserts garbage,
      // so we have to manually restore the value.
      index_val = DEFAULT_VALUE;
    }
    // cout << "Material attribute = " << index_val << ", " << index << ", " << id << "\n";

    //Create and register as an MPM material
    MPMMaterial *mat = scinew MPMMaterial(ps, d_matlManager, flags,isRestart);

    mat->registerParticleState( d_particleState,
                                d_particleState_preReloc );
    
    // When doing restart, we need to make sure that we load the materials
    // in the same order that they were initially created.  Restarts will
    // ALWAYS have an index number as in <material index = "0">.
    // Index_val = -1 means that we don't register the material by its 
    // index number.
    if (index_val > -1){
      d_matlManager->registerMaterial( "MPM", mat,index_val);
    }
    else{
      d_matlManager->registerMaterial( "MPM", mat);
    }
  }
}
//______________________________________________________________________
//
void MPMCommon::scheduleUpdateStress_DamageErosionModels(SchedulerP   & sched,
                                                     const PatchSet * patches,
                                                     const MaterialSet * matls )
{
  printSchedule(patches,cout_doing,"MPMCommon::scheduleUpdateStress_DamageErosionModels");
  
  Task* t = scinew Task("MPM::updateStress_DamageErosionModels", this, 
                        &MPMCommon::updateStress_DamageErosionModels);

  t->requiresVar(Task::OldDW, lb->simulationTimeLabel);
  
  int numMatls = d_matlManager->getNumMatls( "MPM" );
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = (MPMMaterial*) d_matlManager->getMaterial( "MPM", m);
    
    DamageModel* dm = mpm_matl->getDamageModel();
    dm->addComputesAndRequires(t, mpm_matl);
    
    ErosionModel* em = mpm_matl->getErosionModel();
    em->addComputesAndRequires(t, mpm_matl);
  }
  
  sched->addTask(t, patches, matls);
}
//______________________________________________________________________
//
void MPMCommon::updateStress_DamageErosionModels(const ProcessorGroup *,
                                                 const PatchSubset    * patches,
                                                 const MaterialSubset * ,
                                                 DataWarehouse        * old_dw,
                                                 DataWarehouse        * new_dw)
{
  for (int p = 0; p<patches->size(); p++) {
    const Patch* patch = patches->get(p);
   
    printTask(patches, patch,cout_doing,
              "Doing updateStress_DamageModel");

    int numMPMMatls = d_matlManager->getNumMatls( "MPM" );
    for(int m = 0; m < numMPMMatls; m++){
    
      MPMMaterial* mpm_matl = (MPMMaterial*) d_matlManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
      
      DamageModel* dm = mpm_matl->getDamageModel();
      dm->computeSomething( pset, mpm_matl, patch, old_dw, new_dw );
      
      ErosionModel* em = mpm_matl->getErosionModel();
      em->updateStress_Erosion( pset, old_dw, new_dw );
    }
  }
}

////______________________________________________________________________
//
void MPMCommon::scheduleConservedQtys(SchedulerP        & sched,
                                      const PatchSet    * patches,
                                      const MaterialSet * matls,
                                      const MaterialSubset * reduction_mss )
{
  if (!d_flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"MPMCommon::scheduleConservedQtys");
  
  Task* t = scinew Task("MPM::scheduleConservedQtys", this, 
                        &MPMCommon::conservedQtys);

  const MaterialSubset* mpm_mss = matls->getUnion();
  Ghost::GhostType gn = Ghost::None;
  t->requiresVar(Task::NewDW, lb->pXLabel,          mpm_mss,   gn );
  t->requiresVar(Task::NewDW, lb->pMassLabel,       mpm_mss,  gn );
  t->requiresVar(Task::NewDW, lb->pTemperatureLabel,mpm_mss,  gn );
  
  if( d_flags->d_reductionVars->mass ){
    t->computesVar(lb->TotalMassLabel,     reduction_mss, Task::OutOfDomain);
  }
  if( d_flags->d_reductionVars->momentum ){
    t->computesVar(lb->TotalMomentumLabel, reduction_mss, Task::OutOfDomain);
  }
  if( d_flags->d_reductionVars->KE ){
    t->computesVar(lb->KineticEnergyLabel, reduction_mss, Task::OutOfDomain);
  }
  if( d_flags->d_reductionVars->thermalEnergy ){
    t->computesVar(lb->ThermalEnergyLabel, reduction_mss, Task::OutOfDomain);
  }
  if( d_flags->d_reductionVars->centerOfMass ){
    t->computesVar(lb->CenterOfMassPositionLabel, reduction_mss,Task::OutOfDomain);
  }
  
  sched->addTask(t, patches, matls);
}

//______________________________________________________________________
//
void MPMCommon::conservedQtys(const ProcessorGroup *,
                               const PatchSubset    * patches,
                               const MaterialSubset * matls,
                               DataWarehouse        * ,
                               DataWarehouse        * new_dw)
{
  for (int p = 0; p<patches->size(); p++) {
    const Patch* patch = patches->get(p);
   
    printTask(patches, patch,cout_doing,"Doing ConservedQtys");

    map<int,double> zeroD = initializeMap(0.0);
    map<int,Vector> zeroV = initializeMap(Vector(0.));

    map<int,double> kineticEng = zeroD;
    map<int,double> thermalEng = zeroD;
    map<int,Vector> totalMom   = zeroV;
    map<int,Vector> CMX        = zeroV;
    map<int,double> totalMass  = zeroD;

    double allMatls_totalMass  = 0.0;
    double allMatls_kineticEng = 0.0;
    double allMatls_thermalEng = 0.0;
    Vector allMatls_totalMom(0.0,0.0,0.0);
    Vector allMatls_CMX(0.0,0.0,0.0);

    for(int m = 0; m<matls->size(); m++){
      int dwi = matls->get(m);
      MPMMaterial* mpm_matl = (MPMMaterial*) d_matlManager->getMaterial( dwi );

      ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
      constParticleVariable<Point> pX;
      constParticleVariable<Vector> pVelocity;
      constParticleVariable<double> pMass;
      constParticleVariable<double> pTemperature;
      
      new_dw->get(pX,           lb->pXLabel,           pset);
      new_dw->get(pMass,        lb->pMassLabel,        pset);
      new_dw->get(pTemperature, lb->pTemperatureLabel, pset);
      new_dw->get(pVelocity,    lb->pVelocityLabel,    pset);
      
      double Cv=mpm_matl->getSpecificHeat();
      
      //__________________________________        
      // Loop over particles
      ParticleSubset::iterator iter;
      for(iter = pset->begin();iter != pset->end(); iter++){
        particleIndex idx = *iter;
 
        double mass = pMass[idx];
        allMatls_totalMass += mass;
        totalMass[dwi]     += mass;

        Vector centerOfMass = (pX[idx] * mass).asVector();
        CMX[dwi]      += centerOfMass;
        allMatls_CMX  += centerOfMass;

        double thermalEngy   = pTemperature[idx] * mass * Cv;
        thermalEng[dwi]     += thermalEngy;
        allMatls_thermalEng += thermalEngy;

        double ke            = .5* mass * pVelocity[idx].length2();
        kineticEng[dwi]     += ke;
        allMatls_kineticEng += ke;

        Vector mom = pVelocity[idx] * mass;
        allMatls_totalMom += mom;
        totalMom[dwi]     += mom;
      }
    }

    //__________________________________
    //  reduction variables
    if( d_flags->d_reductionVars->mass ){
      new_dw->put( sum_vartype(allMatls_totalMass), 
                   lb->TotalMassLabel, nullptr, -1);

      new_dw->put_sum_vartype( totalMass,   lb->TotalMassLabel, matls );
    }
    if( d_flags->d_reductionVars->momentum ){
      new_dw->put( sumvec_vartype(allMatls_totalMom), 
                   lb->TotalMomentumLabel, nullptr, -1);

      new_dw->put_sum_vartype( totalMom,   lb->TotalMomentumLabel, matls );
    }

    if( d_flags->d_reductionVars->KE ){
      new_dw->put( sum_vartype(allMatls_kineticEng),  
                   lb->KineticEnergyLabel, nullptr, -1);

      new_dw->put_sum_vartype( kineticEng, lb->KineticEnergyLabel, matls );
    }

    if( d_flags->d_reductionVars->thermalEnergy ){
      new_dw->put( sum_vartype(allMatls_thermalEng),  
                   lb->ThermalEnergyLabel, nullptr, -1);

      new_dw->put_sum_vartype( thermalEng, lb->ThermalEnergyLabel, matls );

    }

    if(d_flags->d_reductionVars->centerOfMass){
      new_dw->put(sumvec_vartype(allMatls_CMX), 
                  lb->CenterOfMassPositionLabel, nullptr, -1);

      new_dw->put_sum_vartype( CMX,  lb->CenterOfMassPositionLabel, matls );
    }
  }
}


//______________________________________________________________________
//  Utilities
//    initialize the map used in reduction variables

template<class T>
std::map<int,T> MPMCommon::initializeMap(T val)
{
  std::map<int,T>  myMap;
  int numMPMMatls = d_matlManager->getNumMatls( "MPM" );
  
  for(int m = 0; m < numMPMMatls; m++){
    MPMMaterial* mpm_matl = (MPMMaterial*) d_matlManager->getMaterial( "MPM", m);
    int dwi = mpm_matl->getDWIndex();
    myMap[dwi] = val;
  }
  return myMap;

}

template std::map<int,double> MPMCommon::initializeMap(double);
template std::map<int,Vector> MPMCommon::initializeMap(Vector);

