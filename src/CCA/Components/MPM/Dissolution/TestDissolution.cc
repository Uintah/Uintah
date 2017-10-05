/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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
#include <CCA/Components/MPM/Dissolution/TestDissolution.h>
#include <CCA/Components/MPM/MPMBoundCond.h>
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
#include <Core/Labels/MPMLabel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Containers/StaticArray.h>
#include <vector>

using namespace std;
using namespace Uintah;
using std::vector;

TestDissolution::TestDissolution(const ProcessorGroup* myworld,
                                   ProblemSpecP& ps, SimulationStateP& d_sS, 
                                   MPMLabel* Mlb,MPMFlags* MFlag)
  : Dissolution(myworld, Mlb, MFlag, ps)
{
  // Constructor
  d_sharedState = d_sS;
  lb = Mlb;
  flag = MFlag;
//  d_oneOrTwoStep = 2;
//  ps->get("OneOrTwoStep",     d_oneOrTwoStep);
}

TestDissolution::~TestDissolution()
{
}

void TestDissolution::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP dissoluton_ps = ps->appendChild("dissoluton");
  dissoluton_ps->appendElement("type","test");
  d_matls.outputProblemSpec(dissoluton_ps);
}

void TestDissolution::computeMassBurnFraction(const ProcessorGroup*,
                                              const PatchSubset* patches,
                                              const MaterialSubset* matls,
                                              DataWarehouse*,
                                              DataWarehouse* new_dw)
{
//   string interp_type = flag->d_interpolator_type;
   int numMatls = d_sharedState->getNumMPMMatls();
   ASSERTEQ(numMatls, matls->size());
   for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    // Retrieve necessary data from DataWarehouse
    StaticArray<constNCVariable<double> > gmass(numMatls);
    StaticArray<constNCVariable<Matrix3> > gStress(numMatls);
//    constNCVariable<double> gmassAll;
    StaticArray<NCVariable<double> > massBurnFrac(numMatls);
    for(int m=0;m<matls->size();m++){
      int dwi = matls->get(m);
      new_dw->get(gmass[m],   lb->gMassLabel,    dwi, patch, Ghost::None,0);
      new_dw->get(gStress[m], lb->gStressForSavingLabel,
                                                 dwi, patch, Ghost::None,0);
      new_dw->allocateAndPut(massBurnFrac[m],
                             lb->massBurnFractionLabel,dwi,patch);
    massBurnFrac[m].initialize(0.);
    }
//    new_dw->get(gMassAll,     lb->gMassLabel,
//                d_sharedState->getAllInOneMatl()->get(0), patch, Ghost::None,0);

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
      IntVector c = *iter;

      double sumMass=0.0;
      for(int m = 0; m < numMatls; m++){
//        if(d_matls.requested(n)) {
          sumMass+=gmass[m][c]; 
//        }
      }

      for(int m = 0; m < numMatls; m++){
       if(d_matls.requested(m)) {
        if(gStress[m][c].Trace()/3.0 < 9.e5 && gmass[m][c] != sumMass){
          massBurnFrac[m][c] = fabs(0.0002*(gStress[m][c].Trace()/3.0)/(9.e5));
        }
       }
      }
    } // nodes
  } // patches
}

void TestDissolution::addComputesAndRequiresMassBurnFrac(SchedulerP & sched,
                                                      const PatchSet* patches,
                                                      const MaterialSet* ms)
{
  Task * t = scinew Task("TestDissolution::computeMassBurnFraction", 
                      this, &TestDissolution::computeMassBurnFraction);
  
//  const MaterialSubset* mss = ms->getUnion();

  t->requires(Task::NewDW, lb->gMassLabel,           Ghost::None);
  t->requires(Task::NewDW, lb->gMassLabel, d_sharedState->getAllInOneMatl(),
              Task::OutOfDomain, Ghost::None);
  t->requires(Task::NewDW, lb->gStressForSavingLabel,Ghost::None);

  t->computes(lb->massBurnFractionLabel);

  sched->addTask(t, patches, ms);
}
