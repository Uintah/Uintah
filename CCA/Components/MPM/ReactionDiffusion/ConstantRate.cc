/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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

#include <CCA/Components/MPM/ReactionDiffusion/ConstantRate.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/MPMFlags.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/AMR.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Util/DebugStream.h>

#include <iostream>
using namespace Uintah;

static DebugStream cout_doing("AMRMPM", false);


ConstantRate::ConstantRate(ProblemSpecP& ps, SimulationStateP& sS, MPMFlags* Mflag, std::string diff_type)
  : ScalarDiffusionModel(ps, sS, Mflag, diff_type)
{
  ps->require("constant_rate", d_constant_rate);
  std::cout << "rate: " << d_constant_rate << std::endl;
}

ConstantRate::~ConstantRate() {

}

void ConstantRate::scheduleComputeDivergence(Task* task, 
                                                    const MPMMaterial* matl, 
                                                    const PatchSet* patch) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::NewDW, d_lb->gMassLabel, Ghost::None);

  task->computes(d_lb->gConcentrationRateLabel, matlset);
}

void ConstantRate::computeDivergence(const Patch* patch,
                                             const MPMMaterial* matl,
                                             DataWarehouse* old_dw, 
                                             DataWarehouse* new_dw)
{
  int dwi = matl->getDWIndex();
  Ghost::GhostType gn  = Ghost::None;

  //*********Start - Used for testing purposes - CG *******
  int timestep = d_sharedState->getCurrentTopLevelTimeStep();
  //*********End   - Used for testing purposes - CG *******

  constNCVariable<double> gMass;
  NCVariable<double> gConcRate;

  new_dw->get(gMass, d_lb->gMassLabel, dwi, patch, gn, 0);
  new_dw->allocateAndPut(gConcRate,  d_lb->gConcentrationRateLabel,dwi,patch);

  for(NodeIterator iter=patch->getExtraNodeIterator();!iter.done();iter++){
    IntVector n = *iter;
    if(timestep <= 1000){
      gConcRate[n] = d_constant_rate * gMass[n];
    }else{
      gConcRate[n] = 0.0;
    }
  }
}

void ConstantRate::scheduleComputeDivergence_CFI(Task* t,
                                                    const MPMMaterial* matl, 
                                                    const PatchSet* patch) const
{

}

void ConstantRate::computeDivergence_CFI(const PatchSubset* finePatches,
                                                 const MPMMaterial* matl,
                                                 DataWarehouse* old_dw, 
                                                 DataWarehouse* new_dw)
{

}

void ConstantRate::outputProblemSpec(ProblemSpecP& ps, bool output_rdm_tag)
{
  if (output_rdm_tag) {
    ps = ps->appendChild("diffusion_model");
    ps->setAttribute("type","constant_rate");
  }
  ps->appendElement("diffusivity",diffusivity);
  ps->appendElement("max_concentration",max_concentration);
  ps->appendElement("constant_rate", d_constant_rate);
}

