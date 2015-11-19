/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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

#include <CCA/Components/MPM/ReactionDiffusion/SDInterfaceModel.h>
#include <CCA/Components/MPM/ReactionDiffusion/ScalarDiffusionModel.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Grid/Task.h>

using namespace Uintah;

SDInterfaceModel::SDInterfaceModel(ProblemSpecP& ps, SimulationStateP& sS, MPMFlags* Mflag){

  d_Mflag = Mflag;
  d_sharedState = sS;

  d_lb = scinew MPMLabel;

  if(d_Mflag->d_8or27==8){
    NGP=1;
    NGN=1;
  } else {
    NGP=2;
    NGN=2;
  }
}

SDInterfaceModel::~SDInterfaceModel(){
  delete(d_lb);
}

void SDInterfaceModel::addComputesAndRequiresInterpolated(SchedulerP & sched,
                                                   const PatchSet* patches,
                                                   const MaterialSet* matls)
{
}

void SDInterfaceModel::sdInterfaceInterpolated(const ProcessorGroup*,
                                             const PatchSubset* patches,
                                             const MaterialSubset* matls,
                                             DataWarehouse* old_dw,
                                             DataWarehouse* new_dw)
{
}

void SDInterfaceModel::addComputesAndRequiresDivergence(SchedulerP & sched,
                                                 const PatchSet* patches,
                                                 const MaterialSet* matls)
{
}

void SDInterfaceModel::sdInterfaceDivergence(const ProcessorGroup*,
                                           const PatchSubset* patches,
                                           const MaterialSubset* matls,
                                           DataWarehouse* old_dw,
                                           DataWarehouse* new_dw)
{
}

void SDInterfaceModel::outputProblemSpec(ProblemSpecP& ps)
{
  // To be filled out in interface model
}
