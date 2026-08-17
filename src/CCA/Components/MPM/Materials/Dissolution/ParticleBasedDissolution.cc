/*
 * Copyright © 2026 by Geocosm LLC                                   
 */

// ParticleBasedDissolution.cc
// One of the derived Dissolution classes.
//
// This dissolution model computes a change in particle mass at the
// particle level, based on a "rate of dissolution"
// where "rate of dissolution" is the velocity with which a surface is removed.
// In this model, dissolution occurs if the following criteria are met:
// 1.
// 2.
// The output of this model is pDeltaMassLabel.  This is applied to surface
// particles and comes from the outer surface.

// The dissolution rate is converted to a rate of mass decrease which is
// then applied to identified surface particles in 
// interpolateToParticlesAndUpdate

#include <CCA/Components/MPM/Materials/Dissolution/ParticleBasedDissolution.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <vector>

using namespace std;
using namespace Uintah;

ParticleBasedDissolution::ParticleBasedDissolution(const ProcessorGroup* myworld,
                                 ProblemSpecP& ps, MaterialManagerP& d_sS, 
                                 MPMLabel* Mlb)
  : Dissolution(myworld, Mlb, ps)
{
  // Constructor
  d_materialManager = d_sS;
  lb = Mlb;
/*
  ps->require("masterModalID",        d_masterModalID);
  ps->require("InContactWithModalID", d_inContactWithModalID);
  ps->require("Ao_mol_cm2-us",        d_Ao);
  ps->require("Ea_ug-cm2_us2-mol",    d_Ea);
  ps->require("R_ug-cm2_us2-mol-K",   d_R);
  ps->require("Vm_cm3_mol",           d_Vm);
  ps->require("StressThreshold",      d_StressThresh);
  ps->getWithDefault("Temperature",   d_temperature, 300.0);
  ps->getWithDefault("MaxCementThickness_cm", d_maxCemThickness, 9.e99);
  ps->getWithDefault("Ao_clay_mol_cm2-us",        d_Ao_clay, d_Ao);
  ps->getWithDefault("Ea_clay_ug-cm2_us2-mol",    d_Ea_clay, d_Ea);
*/
}

ParticleBasedDissolution::~ParticleBasedDissolution()
{
}

void ParticleBasedDissolution::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP dissolution_ps = ps->appendChild("dissolution");
  dissolution_ps->appendElement("type",         "particleBasedDissolution");
/*
  dissolution_ps->appendElement("masterModalID",        d_masterModalID);
  dissolution_ps->appendElement("InContactWithModalID", d_inContactWithModalID);
  dissolution_ps->appendElement("Ao_mol_cm2-us",        d_Ao);
  dissolution_ps->appendElement("Ea_ug-cm2_us2-mol",    d_Ea);
  dissolution_ps->appendElement("R_ug-cm2_us2-mol-K",   d_R);
  dissolution_ps->appendElement("Vm_cm3_mol",           d_Vm);
  dissolution_ps->appendElement("StressThreshold",      d_StressThresh);
  dissolution_ps->appendElement("Temperature",          d_temperature);
  dissolution_ps->appendElement("MaxCementThickness_cm",d_maxCemThickness);
  dissolution_ps->appendElement("Ao_clay_mol_cm2-us",     d_Ao_clay);
  dissolution_ps->appendElement("Ea_clay_ug-cm2_us2-mol", d_Ea_clay);
*/
}

void ParticleBasedDissolution::computeMassBurnFraction(const ProcessorGroup*,
                                              const PatchSubset* patches,
                                              const MaterialSubset* matls,
                                              DataWarehouse* old_dw,
                                              DataWarehouse* new_dw)
{
   int numMatls = d_materialManager->getNumMatls("MPM");
   ASSERTEQ(numMatls, matls->size());

   for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    double area = dx.x()*dx.y();

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    // Retrieve necessary data from DataWarehouse
    constParticleVariable<double> pmass, psurf, pvolume;
    ParticleVariable<double> pdeltaMass;
//    std::vector<NCVariable<double> >  massBurnRate(numMatls);
//    std::vector<NCVariable<double> >  dLdt(numMatls);
//    constNCVariable<double> NC_CCweight;
//    std::vector<bool> masterMatls(numMatls);
//    std::vector<bool> inContactWithMatls(numMatls);

    for(int m=0;m<matls->size();m++){
      MPMMaterial* mat=(MPMMaterial *) d_materialManager->getMaterial("MPM", m);
      ParticleSubset* pset = old_dw->getParticleSubset(m, patch);

      old_dw->get(pmass,              lb->pMassLabel,               pset);
      old_dw->get(pvolume,            lb->pVolumeLabel,             pset);
      new_dw->get(psurf,              lb->pSurfLabel_preReloc,      pset);
      new_dw->allocateAndPut(pdeltaMass,
                                      lb->pDeltaMassLabel,          pset);
      //new_dw->getModifiable(massBurnRate[m],
      //                          lb->massBurnFractionLabel,dwi, patch);
      //new_dw->getModifiable(dLdt[m],
      //                          lb->dLdtDissolutionLabel, dwi, patch);

      double dL_dt = 0.1;
      // dMdt = dL_dt*area*density
      //        0.1*(0.001*0.001)*1000. = 1.e-4

      for(ParticleSubset::iterator iter = pset->begin();
                                        iter != pset->end(); iter++){
        particleIndex idx = *iter;

        if(psurf[idx] > 0){
          // Need some work here
          double pEdgeLength = dx.y(); //cbrt(pvolume[idx]);
          double density = pmass[idx]/pvolume[idx];
          pdeltaMass[idx] = density*pEdgeLength*pEdgeLength*dL_dt*delT;
        } else {
          pdeltaMass[idx] = 0.0;
        }
      } // loop over particles
    } // materials
  } // patches
}

void ParticleBasedDissolution::addComputesAndRequiresMassBurnFrac(SchedulerP & sched,
                                                      const PatchSet* patches,
                                                      const MaterialSet* ms)
{
#if 1
  Task * t = scinew Task("ParticleBasedDissolution::computeMassBurnFraction", 
                      this, &ParticleBasedDissolution::computeMassBurnFraction);
  
  const MaterialSubset* mss = ms->getUnion();
  Ghost::GhostType gnone = Ghost::None;

  t->requiresVar(Task::OldDW, lb->delTLabel);
  t->requiresVar(Task::OldDW, lb->pXLabel,                  gnone);
  t->requiresVar(Task::OldDW, lb->pMassLabel,               gnone);

  t->computesVar(lb->pDeltaMassLabel);
//  t->requiresVar(Task::NewDW, lb->gMassLabel,               Ghost::None);
//  t->requiresVar(Task::NewDW, lb->gVolumeLabel,             Ghost::None);
//  t->requiresVar(Task::NewDW, lb->gSurfaceAreaLabel,        Ghost::None);
//  t->requiresVar(Task::NewDW, lb->gSurfaceClayLabel,        Ghost::None);
//  t->requiresVar(Task::NewDW, lb->gSurfaceCementLabel,      Ghost::None);
//  t->requiresVar(Task::NewDW, lb->gLSContactForceLabel,     Ghost::None);
//  t->requiresVar(Task::OldDW, lb->NC_CCweightLabel,z_matl,  Ghost::None);

//  t->modifiesVar(lb->massBurnFractionLabel, mss);
//  t->modifiesVar(lb->dLdtDissolutionLabel,  mss);

  sched->addTask(t, patches, ms);

#endif
}
