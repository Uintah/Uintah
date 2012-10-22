/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#include <CCA/Components/MPM/ThermalContact/NullThermalContact.h>
#include <Core/Malloc/Allocator.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Task.h>
#include <Core/Labels/MPMLabel.h>

#include <vector>

using namespace Uintah;

NullThermalContact::NullThermalContact(ProblemSpecP&,SimulationStateP& d_sS,
                                       MPMLabel* Mlb,MPMFlags* MFlag)
{
  d_sharedState = d_sS;
  lb = Mlb;
  flag = MFlag;
}

NullThermalContact::~NullThermalContact()
{
}

void NullThermalContact::outputProblemSpec(ProblemSpecP& ps)
{
}


void NullThermalContact::computeHeatExchange(const ProcessorGroup*,
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

      NCVariable<double> thermalContactTemperatureRate;
      new_dw->allocateAndPut(thermalContactTemperatureRate, 
                             lb->gThermalContactTemperatureRateLabel, 
                             dwindex, patch);

      thermalContactTemperatureRate.initialize(0);
      NCVariable<double> GthermalContactTemperatureRate;
      if (flag->d_fracture) {
        new_dw->allocateAndPut(GthermalContactTemperatureRate,
                               lb->GThermalContactTemperatureRateLabel, 
                               dwindex, patch);
        GthermalContactTemperatureRate.initialize(0);
      }

    }
  }
}

void NullThermalContact::initializeThermalContact(const Patch* /*patch*/,
                                        int /*vfindex*/,
                                        DataWarehouse* /*new_dw*/)
{
}

void NullThermalContact::addComputesAndRequires(Task* t,
                                            const PatchSet*,
                                            const MaterialSet*) const
{
  t->computes(lb->gThermalContactTemperatureRateLabel);
  if (flag->d_fracture)
    t->computes(lb->GThermalContactTemperatureRateLabel); 
}
