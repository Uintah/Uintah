/*
 * The MIT License
 *
 * Copyright (c) 1997-2023 The University of Utah
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

#include <CCA/Components/MPM/ThermalContact/STThermalContact.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <vector>

using namespace std;
using namespace Uintah;

STThermalContact::STThermalContact(ProblemSpecP&,MaterialManagerP& d_sS,
                                   MPMLabel* Mlb,MPMFlags* MFlag)
{
  d_materialManager = d_sS;
  lb = Mlb;
  flag = MFlag;
}

STThermalContact::~STThermalContact()
{
}

void STThermalContact::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP thermal_ps = ps->appendChild("thermal_contact");
}

void STThermalContact::computeHeatExchange(const ProcessorGroup*,
                                           const PatchSubset* patches,
                                           const MaterialSubset*,
                                           DataWarehouse* old_dw,
                                           DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    int numMatls = d_materialManager->getNumMatls( "MPM" );

    std::vector<constNCVariable<double> > gmass(numMatls);
    std::vector<constNCVariable<double> > gTemp(numMatls);
    std::vector<NCVariable<double> > thermalContactTemperatureRate(numMatls);
    vector<double> Cp(numMatls);

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));
  
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) d_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();
      new_dw->get(gmass[dwi], lb->gMassLabel,        dwi, patch, Ghost::None,0);
      new_dw->get(gTemp[dwi], lb->gTemperatureLabel, dwi, patch, Ghost::None,0);
      new_dw->allocateAndPut(thermalContactTemperatureRate[dwi],
                            lb->gThermalContactTemperatureRateLabel,dwi,patch);
      thermalContactTemperatureRate[dwi].initialize(0.);
      Cp[m]=mpm_matl->getSpecificHeat();
      // -------------------------------------------------------------------
    }

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
      double numerator=0.0;
      double denominator=0.0;
      IntVector c = *iter;
      for(int m = 0; m < numMatls; m++) {
        MPMMaterial* mpm_matl = (MPMMaterial*) d_materialManager->getMaterial( "MPM",  m );
        int n = mpm_matl->getDWIndex();
        numerator   += (gTemp[n][c] * gmass[n][c]  * Cp[m]);
        denominator += (gmass[n][c]  * Cp[m]);
      }
      
      double contactTemperature = numerator/denominator;

      for(int m = 0; m < numMatls; m++) {
        thermalContactTemperatureRate[m][c] =
                                      (contactTemperature - gTemp[m][c])/delT;
      }
    }
  }
}

void STThermalContact::initializeThermalContact(const Patch* /*patch*/,
                                                int /*vfindex*/,
                                                DataWarehouse* /*new_dw*/)
{
}

void STThermalContact::addComputesAndRequires(Task* t, const PatchSet*,
                                              const MaterialSet*) const
{
  t->requires(Task::OldDW, lb->delTLabel);  
  t->requires(Task::NewDW, lb->gMassLabel,        Ghost::None);
  t->requires(Task::NewDW, lb->gTemperatureLabel, Ghost::None);
  t->computes(lb->gThermalContactTemperatureRateLabel);
}
