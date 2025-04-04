/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
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


// SingleVel.cc
// One of the derived Contact classes.  This particular
// class contains methods for recapturing single velocity
// field behavior from objects belonging to multiple velocity
// fields.  The main purpose of this type of contact is to
// ensure that one can get the same answer using prescribed
// contact as can be gotten using "automatic" contact.
#include <CCA/Components/MPM/Materials/Contact/SingleVelContact.h>
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
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Ports/DataWarehouse.h>
#include <vector>

using namespace std;
using namespace Uintah;
using std::vector;

SingleVelContact::SingleVelContact(const ProcessorGroup* myworld,
                                   ProblemSpecP& ps, MaterialManagerP& d_sS, 
                                   MPMLabel* Mlb,MPMFlags* MFlag)
  : Contact(myworld, Mlb, MFlag, ps)
{
  // Constructor
  d_materialManager = d_sS;
  lb = Mlb;
  flag = MFlag;
  d_oneOrTwoStep = 2;
  ps->get("OneOrTwoStep",     d_oneOrTwoStep);
  ps->getWithDefault("ExcludeMaterial", d_excludeMatl, -999);
}

SingleVelContact::~SingleVelContact()
{
}

void SingleVelContact::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP contact_ps = ps->appendChild("contact");
  contact_ps->appendElement("type","single_velocity");
  contact_ps->appendElement("OneOrTwoStep",      d_oneOrTwoStep);
  contact_ps->appendElement("ExcludeMaterial",   d_excludeMatl);
  d_matls.outputProblemSpec(contact_ps);
}

void SingleVelContact::exMomInterpolated(const ProcessorGroup*,
                                         const PatchSubset* patches,
                                         const MaterialSubset* matls,
                                         DataWarehouse*,
                                         DataWarehouse* new_dw)
{
  if(d_oneOrTwoStep==2){
   string interp_type = flag->d_interpolator_type;
   int numMatls = d_materialManager->getNumMatls( "MPM" );
   ASSERTEQ(numMatls, matls->size());
   for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    Vector centerOfMassVelocity(0.0,0.0,0.0);

    // Retrieve necessary data from DataWarehouse
    std::vector<constNCVariable<double> > gmass(numMatls);
    std::vector<NCVariable<Vector> > gvelocity(numMatls);
    for(int m=0;m<matls->size();m++){
      MPMMaterial* mpm_matl = 
                        (MPMMaterial*) d_materialManager->getMaterial("MPM", m);
      int dwi = mpm_matl->getDWIndex();
      new_dw->get(gmass[m], lb->gMassLabel,    dwi, patch,Ghost::None,0);
      new_dw->getModifiable(gvelocity[m], lb->gVelocityLabel,dwi, patch);
    }

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
      IntVector c = *iter;

      Vector centerOfMassMom(0,0,0);
      double centerOfMassMass=0.0;

      for(int n = 0; n < numMatls; n++){
        if(d_matls.requested(n)) {
          centerOfMassMom+=gvelocity[n][c] * gmass[n][c];
          centerOfMassMass+=gmass[n][c]; 
        }
      }

      double excludeMass = 0.;
      if(d_excludeMatl >=0){
        excludeMass = gmass[d_excludeMatl][c];
      }

      // Set each field's velocity equal to the center of mass velocity
      centerOfMassVelocity=centerOfMassMom/centerOfMassMass;
      for(int n = 0; n < numMatls; n++) {
        if(d_matls.requested(n) && excludeMass < 1.e-99) {
          gvelocity[n][c] = centerOfMassVelocity;
        }
      }
    }
  }
 }
}

void SingleVelContact::exMomIntegrated(const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw)
{
  int numMatls = d_materialManager->getNumMatls( "MPM" );
  ASSERTEQ(numMatls, matls->size());

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    Vector zero(0.0,0.0,0.0);
    Vector centerOfMassVelocity(0.0,0.0,0.0);
    Vector centerOfMassMom(0.0,0.0,0.0);
    Vector Dvdt;
    double centerOfMassMass;

    // Retrieve necessary data from DataWarehouse
    std::vector<constNCVariable<double> > gmass(numMatls);
    std::vector<NCVariable<Vector> > gvelocity_star(numMatls);

    for(int m=0;m<matls->size();m++){
     MPMMaterial* mpm_matl = 
                        (MPMMaterial*) d_materialManager->getMaterial("MPM", m);
     int dwi = mpm_matl->getDWIndex();
     new_dw->get(gmass[m],lb->gMassLabel, dwi, patch, Ghost::None, 0);
     new_dw->getModifiable(gvelocity_star[m],lb->gVelocityStarLabel, dwi,patch);
    }

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));
    
    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
      IntVector c = *iter;

      centerOfMassMom=zero;
      centerOfMassMass=0.0; 
      for(int  n = 0; n < numMatls; n++){
        if(d_matls.requested(n)) {
          centerOfMassMom+=gvelocity_star[n][c] * gmass[n][c];
          centerOfMassMass+=gmass[n][c]; 
        }
      }

      double excludeMass = 0.;
      if(d_excludeMatl >=0){
        excludeMass = gmass[d_excludeMatl][c];
      }

      // Set each field's velocity equal to the center of mass velocity
      centerOfMassVelocity=centerOfMassMom/centerOfMassMass;
      for(int  n = 0; n < numMatls; n++){
        if(d_matls.requested(n) && excludeMass < 1.e-99) {
          Dvdt = (centerOfMassVelocity - gvelocity_star[n][c])/delT;
          gvelocity_star[n][c] = centerOfMassVelocity;
        }
      }
    }
  }
}

void SingleVelContact::addComputesAndRequiresInterpolated(SchedulerP & sched,
                                                  const PatchSet* patches,
                                                  const MaterialSet* ms)
{
  Task * t = scinew Task("SingleVelContact::exMomInterpolated", 
                      this, &SingleVelContact::exMomInterpolated);
  
  const MaterialSubset* mss = ms->getUnion();
  t->requiresVar( Task::NewDW, lb->gMassLabel,          Ghost::None);

  t->modifiesVar(              lb->gVelocityLabel, mss);

  sched->addTask(t, patches, ms);
}

void SingleVelContact::addComputesAndRequiresIntegrated(SchedulerP & sched,
                                             const PatchSet* patches,
                                             const MaterialSet* ms) 
{
  Task * t = scinew Task("SingleVelContact::exMomIntegrated", 
                      this, &SingleVelContact::exMomIntegrated);
  
  const MaterialSubset* mss = ms->getUnion();
  t->requiresVar(Task::OldDW, lb->delTLabel);    
  t->requiresVar(Task::NewDW, lb->gMassLabel,              Ghost::None);

  t->modifiesVar(             lb->gVelocityStarLabel, mss);

  sched->addTask(t, patches, ms);
}
