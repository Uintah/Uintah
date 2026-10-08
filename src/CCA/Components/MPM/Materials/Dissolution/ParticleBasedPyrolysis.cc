/*
 * Copyright © 2026 by Geocosm LLC                                   
 */

// ParticleBasedPyrolysis.cc
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

#include <CCA/Components/MPM/Materials/Dissolution/ParticleBasedPyrolysis.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Components/MPM/Core/MPMFlags.h>
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

ParticleBasedPyrolysis::ParticleBasedPyrolysis(const ProcessorGroup* myworld,
                                 ProblemSpecP& ps, MaterialManagerP& d_sS, 
                                 MPMLabel* Mlb, MPMFlags* flag)
  : Dissolution(myworld, Mlb, ps, flag)
{
  // Constructor
  d_materialManager = d_sS;
  lb = Mlb;

  ps->require("rate",        d_rate);
}

ParticleBasedPyrolysis::~ParticleBasedPyrolysis()
{
}

void ParticleBasedPyrolysis::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP dissolution_ps = ps->appendChild("dissolution");
  dissolution_ps->appendElement("type",         "particleBasedPyrolysis");
  dissolution_ps->appendElement("rate",          d_rate);
}

void ParticleBasedPyrolysis::computeMassBurnFraction(const ProcessorGroup*,
                                              const PatchSubset* patches,
                                              const MaterialSubset* matls,
                                              DataWarehouse* old_dw,
                                              DataWarehouse* new_dw)
{
//   int numMatls = d_materialManager->getNumMatls("MPM");
//   ASSERTEQ(numMatls, matls->size());

   Ghost::GhostType gac   = Ghost::AroundCells;
   for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    // Retrieve necessary data from DataWarehouse
    ParticleVariable<double> pdeltaMass;

    for(int m=0;m<matls->size();m++){
      MPMMaterial* mpm_matl =
                     (MPMMaterial*) d_materialManager->getMaterial( "MPM", m);
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      new_dw->allocateAndPut(pdeltaMass,
                                      lb->pDeltaMassLabel,          pset);

      for(ParticleSubset::iterator iter = pset->begin();
                                        iter != pset->end(); iter++){
        particleIndex idx = *iter;

          pdeltaMass[idx] = d_rate*delT;
      } // loop over particles
    } // materials
  } // patches
}

void ParticleBasedPyrolysis::addComputesAndRequiresMassBurnFrac(SchedulerP & sched,
                                                      const PatchSet* patches,
                                                      const MaterialSet* ms)
{
  Task * t = scinew Task("ParticleBasedPyrolysis::computeMassBurnFraction", 
                      this, &ParticleBasedPyrolysis::computeMassBurnFraction);
  
  //const MaterialSubset* mss = ms->getUnion();
  Ghost::GhostType gnone = Ghost::None;
  Ghost::GhostType gac   = Ghost::AroundCells;

  t->requiresVar(Task::OldDW, lb->delTLabel);
//  t->requiresVar(Task::OldDW, lb->pXLabel,                  gnone);
//  t->requiresVar(Task::NewDW, lb->pCurSizeLabel,            gnone);
//  t->requiresVar(Task::OldDW, lb->pSizeLabel,               gnone);


  t->computesVar(lb->pDeltaMassLabel);

  sched->addTask(t, patches, ms);
}
