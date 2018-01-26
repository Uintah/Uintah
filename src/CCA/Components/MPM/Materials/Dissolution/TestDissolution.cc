/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

// TestDissolution.cc
// One of the derived Dissolution classes.
//
// This dissolution model generates a constant rate of dissolution,
// where "rate of dissolution" is the velocity with which a surface is removed.
// In this model, dissolution occurs if the following criteria are met:
// 1.  The "master_material" (the material being dissolved), and at least
//     one of the materials in the "materials" list is present.
// 2.  The pressure exceeds the "thresholdPressure"
// The dissolution rate is converted to a rate of mass decrease which is
// then applied to identified surface particles in 
// interpolateToParticlesAndUpdate

#include <CCA/Components/MPM/Materials/Dissolution/TestDissolution.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPM/Core/MPMBoundCond.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <vector>

using namespace std;
using namespace Uintah;
using std::vector;

TestDissolution::TestDissolution(const ProcessorGroup* myworld,
                                 ProblemSpecP& ps, SimulationStateP& d_sS, 
                                 MPMLabel* Mlb)
  : Dissolution(myworld, Mlb, ps)
{
  // Constructor
  d_sharedState = d_sS;
  lb = Mlb;
  ps->require("masterModalID",        d_masterModalID);
  ps->require("InContactWithModalID", d_inContactWithModalID);
  ps->require("rate",                 d_rate);
  ps->require("PressureThreshold",    d_PressThresh);
}

TestDissolution::~TestDissolution()
{
}

void TestDissolution::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP dissolution_ps = ps->appendChild("dissolution");
  dissolution_ps->appendElement("type",                 "test");
  dissolution_ps->appendElement("masterModalID",        d_masterModalID);
  dissolution_ps->appendElement("InContactWithModalID", d_inContactWithModalID);
  dissolution_ps->appendElement("rate",                 d_rate);
  dissolution_ps->appendElement("PressureThreshold",    d_PressThresh);

//  d_matls.outputProblemSpec(dissolution_ps);
}

void TestDissolution::computeMassBurnFraction(const ProcessorGroup*,
                                              const PatchSubset* patches,
                                              const MaterialSubset* matls,
                                              DataWarehouse* old_dw,
                                              DataWarehouse* new_dw)
{
   int numMatls = d_sharedState->getNumMPMMatls();
   ASSERTEQ(numMatls, matls->size());

   for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    double area = dx.x()*dx.y();

    Ghost::GhostType  gnone = Ghost::None;

    // Retrieve necessary data from DataWarehouse
    std::vector<constNCVariable<double> > gmass(numMatls),gvolume(numMatls);
    std::vector<constNCVariable<double> > gnormtrac(numMatls);
    std::vector<constNCVariable<Matrix3> > gStress(numMatls);
    std::vector<NCVariable<double> >  massBurnRate(numMatls);
    constNCVariable<double> NC_CCweight;
    std::vector<bool> masterMatls(numMatls);
    std::vector<bool> inContactWithMatls(numMatls);
    old_dw->get(NC_CCweight,  lb->NC_CCweightLabel,0, patch, gnone,0);
    for(int m=0;m<matls->size();m++){
      int dwi = matls->get(m);
      new_dw->get(gmass[m],     lb->gMassLabel,         dwi, patch, gnone, 0);
      new_dw->get(gvolume[m],   lb->gVolumeLabel,       dwi, patch, gnone, 0);
      new_dw->get(gStress[m],   lb->gStressLabel,       dwi, patch, gnone, 0);
      new_dw->get(gnormtrac[m], lb->gNormTractionLabel, dwi, patch, gnone, 0);

      new_dw->getModifiable(massBurnRate[m], lb->massBurnFractionLabel, dwi, patch);
      
      MPMMaterial* mat = d_sharedState->getMPMMaterial(m);
      if(mat->getModalID()==d_masterModalID){
        mat->setNeedSurfaceParticles(true);
        masterMatls[m]=true;
      } else{
        masterMatls[m]=false;
      }

      if(mat->getModalID()==d_inContactWithModalID) {
        inContactWithMatls[m]=true;
      } else{
        inContactWithMatls[m]=false;
      }
    }

    for(int m=0; m < numMatls; m++){
     if(masterMatls[m]){
      int md=m;
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
        IntVector c = *iter;

        double sumMass=0.0;
        for(int n = 0; n < numMatls; n++){
          if(n==md || inContactWithMatls[n]) {
            sumMass+=gmass[n][c]; 
          }
        }

        if(gmass[md][c] >  1.e-100  &&
           gmass[md][c] != sumMass  && 
          -gnormtrac[md][c] > d_PressThresh){ // Compressive stress is negative
//           pressure > d_PressThresh){ // && volFrac > 0.6){
            double rho = gmass[md][c]/gvolume[md][c];
            massBurnRate[md][c] += d_rate*area*rho*2.0*NC_CCweight[c];
        }
      } // nodes
     } // endif a masterMaterial
    } // materials
  } // patches
}

void TestDissolution::addComputesAndRequiresMassBurnFrac(SchedulerP & sched,
                                                      const PatchSet* patches,
                                                      const MaterialSet* ms)
{
  Task * t = scinew Task("TestDissolution::computeMassBurnFraction", 
                      this, &TestDissolution::computeMassBurnFraction);
  
  const MaterialSubset* mss = ms->getUnion();
  MaterialSubset* z_matl = scinew MaterialSubset();
  z_matl->add(0);
  z_matl->addReference();

  t->requires(Task::NewDW, lb->gMassLabel,               Ghost::None);
  t->requires(Task::NewDW, lb->gVolumeLabel,             Ghost::None);
  t->requires(Task::NewDW, lb->gStressLabel,             Ghost::None);
  t->requires(Task::NewDW, lb->gNormTractionLabel,       Ghost::None);
  t->requires(Task::OldDW, lb->NC_CCweightLabel,z_matl,  Ghost::None);

  t->modifies(lb->massBurnFractionLabel, mss);

  sched->addTask(t, patches, ms);

  if (z_matl->removeReference())
    delete z_matl;
}
