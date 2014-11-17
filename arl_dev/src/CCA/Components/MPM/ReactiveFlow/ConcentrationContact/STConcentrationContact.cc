/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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

#include <CCA/Components/MPM/ReactiveFlow/ConcentrationContact/STConcentrationContact.h>
#include <Core/Malloc/Allocator.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Containers/StaticArray.h>
#include <vector>

using namespace std;
using namespace Uintah;

STConcentrationContact::STConcentrationContact(ProblemSpecP&,SimulationStateP& d_sS,
                                   MPMLabel* Mlb,MPMFlags* MFlag)
{
  d_sharedState = d_sS;
  lb = Mlb;
  flag = MFlag;
}

STConcentrationContact::~STConcentrationContact()
{
}

void STConcentrationContact::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP conc_ps = ps->appendChild("concentration_contact");
}

void STConcentrationContact::computeConcentrationExchange(const ProcessorGroup*,
                                           const PatchSubset* patches,
                                           const MaterialSubset*,
                                           DataWarehouse* old_dw,
                                           DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    int numMatls = d_sharedState->getNumMPMMatls();

    StaticArray<constNCVariable<double> > gmass(numMatls);
    StaticArray<constNCVariable<double> > gConc(numMatls);
    StaticArray<NCVariable<double> > concentrationContactDiffusionRate(numMatls);
    vector<double> Cp(numMatls);
    // for Fracture (additional field)-----------------------------------------
    StaticArray<constNCVariable<double> > Gmass(numMatls);
    StaticArray<constNCVariable<double> > GConc(numMatls);
    StaticArray<NCVariable<double> > GconcentrationContactDiffusionRate(numMatls);

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));
  
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      new_dw->get(gmass[dwi], lb->gMassLabel,        dwi, patch, Ghost::None,0);
      new_dw->get(gConc[dwi], lb->gConcentrationLabel, dwi, patch, Ghost::None,0);
      new_dw->allocateAndPut(concentrationContactDiffusionRate[dwi],
                            lb->gConcentrationContactDiffusionRateLabel,dwi,patch);
      concentrationContactDiffusionRate[dwi].initialize(0.);
      Cp[m]=mpm_matl->getSpecificHeat();
      if (flag->d_fracture) {
        // for Fracture (for additional field)----------------------------------
        new_dw->get(Gmass[dwi],lb->GMassLabel,       dwi, patch, Ghost::None,0);
        new_dw->get(GConc[dwi],lb->GConcentrationLabel,dwi, patch, Ghost::None,0);
        new_dw->allocateAndPut(GconcentrationContactDiffusionRate[dwi],
                           lb->GConcentrationContactDiffusionRateLabel,dwi,patch);
        GconcentrationContactDiffusionRate[dwi].initialize(0);
      }
      // -------------------------------------------------------------------
    }

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
      double numerator=0.0;
      double denominator=0.0;
      IntVector c = *iter;
      for(int m = 0; m < numMatls; m++) {
        MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
        int n = mpm_matl->getDWIndex();
        numerator   += (gConc[n][c] * gmass[n][c]  * Cp[m]);
        denominator += (gmass[n][c]  * Cp[m]);
        if (flag->d_fracture) {
          numerator   += GConc[n][c] * Gmass[n][c]  * Cp[m];
          denominator += Gmass[n][c]  * Cp[m];  // add in second field;
        }
      }
      
      double contactConcentration = numerator/denominator;

//      for(int m = 0; m < numMatls; m++) {
//        concentrationContactConcentrationRate[m][c] =
//                                      (contactConcentration - gConc[m][c])/delT;
//        if (flag->d_fracture){
//          GconcentrationContactConcentrationRate[m][c] =
//                                      (contactConcentration - GConc[m][c])/delT;
//        }
//      }
    }
  }
}

void STConcentrationContact::initializeConcentrationContact(const Patch* /*patch*/,
                                                int /*vfindex*/,
                                                DataWarehouse* /*new_dw*/)
{
}

void STConcentrationContact::addComputesAndRequires(Task* t, const PatchSet*,
                                              const MaterialSet*) const
{
  t->requires(Task::OldDW, lb->delTLabel);  
  t->requires(Task::NewDW, lb->gMassLabel,        Ghost::None);
  t->requires(Task::NewDW, lb->gConcentrationLabel, Ghost::None);
  t->computes(lb->gConcentrationContactDiffusionRateLabel);
  if (flag->d_fracture) {
    // for second field, for Fracture ---------------------------------
    t->requires(Task::NewDW, lb->GMassLabel,        Ghost::None);
    t->requires(Task::NewDW, lb->GConcentrationLabel, Ghost::None);
    t->computes(lb->GConcentrationContactDiffusionRateLabel);
  }
}
