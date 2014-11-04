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

#include <CCA/Components/MPM/ReactiveFlow/ConcentrationContact/NullConcentrationContact.h>
#include <Core/Malloc/Allocator.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Task.h>
#include <Core/Labels/MPMLabel.h>

#include <vector>

using namespace Uintah;

NullConcentrationContact::NullConcentrationContact(ProblemSpecP&,SimulationStateP& d_sS,
                                       MPMLabel* Mlb,MPMFlags* MFlag)
{
  d_sharedState = d_sS;
  lb = Mlb;
  flag = MFlag;
}

NullConcentrationContact::~NullConcentrationContact()
{
}

void NullConcentrationContact::outputProblemSpec(ProblemSpecP& ps)
{
}


void NullConcentrationContact::computeConcentrationExchange(const ProcessorGroup*,
                                             const PatchSubset* patches,
                                             const MaterialSubset*,
                                             DataWarehouse* ,
                                             DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    int numMatls = d_sharedState->getNumMPMMatls();

    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwindex = mpm_matl->getDWIndex();

      NCVariable<double> concentrationContactDiffusionRate;
      new_dw->allocateAndPut(concentrationContactDiffusionRate, 
                             lb->gConcentrationContactDiffusionRateLabel, 
                             dwindex, patch);

      concentrationContactDiffusionRate.initialize(0);
      NCVariable<double> GConcentrationContactDiffusionRate;
      if (flag->d_fracture) {
        new_dw->allocateAndPut(GConcentrationContactDiffusionRate,
                               lb->GConcentrationContactDiffusionRateLabel, 
                               dwindex, patch);
        GConcentrationContactDiffusionRate.initialize(0);
      }

    }
  }
}

void NullConcentrationContact::initializeConcentrationContact(const Patch* /*patch*/,
                                        int /*vfindex*/,
                                        DataWarehouse* /*new_dw*/)
{
}

void NullConcentrationContact::addComputesAndRequires(Task* t,
                                            const PatchSet*,
                                            const MaterialSet*) const
{
  t->computes(lb->gConcentrationContactDiffusionRateLabel);
  if (flag->d_fracture)
    t->computes(lb->GConcentrationContactDiffusionRateLabel); 
}
