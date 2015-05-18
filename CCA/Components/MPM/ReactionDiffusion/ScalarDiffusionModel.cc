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

#include <CCA/Components/MPM/ReactionDiffusion/ScalarDiffusionModel.h>
#include <CCA/Components/MPM/ReactionDiffusion/ReactionDiffusionLabel.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/MPMFlags.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Grid/Task.h>

#include <iostream>
using namespace std;
using namespace Uintah;


ScalarDiffusionModel::ScalarDiffusionModel(ProblemSpecP& ps, SimulationStateP& sS, MPMFlags* Mflag, string diff_type)
{
  d_Mflag = Mflag;
  d_sharedState = sS;

  d_lb = scinew MPMLabel;
  d_rdlb = scinew ReactionDiffusionLabel();

  if(d_Mflag->d_8or27==8){
    NGP=1;
    NGN=1;
  } else {
    NGP=2;
    NGN=2;
  }

  if(d_Mflag->d_scalarDiffusion_type == "explicit"){
    do_explicit = true;
  }else{
    do_explicit = false;
  }

  diffusion_type = diff_type;
}

ScalarDiffusionModel::~ScalarDiffusionModel() {
  delete d_lb;
  delete d_rdlb;
}

string ScalarDiffusionModel::getDiffusionType(){
  return diffusion_type;
}

void ScalarDiffusionModel::addInitialComputesAndRequires(Task* task, const MPMMaterial* matl,
                                              const PatchSet* patch) const
{

}

void ScalarDiffusionModel::initializeSDMData(const Patch* patch, const MPMMaterial* matl,
                                  DataWarehouse* new_dw)
{

}

void ScalarDiffusionModel::addParticleState(std::vector<const VarLabel*>& from,
                                            std::vector<const VarLabel*>& to)
{

}

void ScalarDiffusionModel::scheduleInterpolateParticlesToGrid(Task* task,
                                                         const MPMMaterial* matl,
                                                         const PatchSet* patch) const
{

}

void ScalarDiffusionModel::interpolateParticlesToGrid(const Patch* patch, const MPMMaterial* matl,
                                                      DataWarehouse* old_dw, DataWarehouse* new_dw)
{

}


void ScalarDiffusionModel::scheduleComputeFlux(Task* task, const MPMMaterial* matl, 
		                                                const PatchSet* patch) const
{

}

void ScalarDiffusionModel::computeFlux(const Patch* patch, const MPMMaterial* matl,
                                            DataWarehouse* old_dw, DataWarehouse* new_dw)
{

}

void ScalarDiffusionModel::scheduleComputeDivergence(Task* task, const MPMMaterial* matl, 
		                                                const PatchSet* patch) const
{

}

void ScalarDiffusionModel::computeDivergence(const Patch* patch, const MPMMaterial* matl,
                                            DataWarehouse* old_dw, DataWarehouse* new_dw)
{

}

void ScalarDiffusionModel::scheduleInterpolateToParticlesAndUpdate(Task* task,
                                                                   const MPMMaterial* matl, 
		                                                               const PatchSet* patch) const
{

}

void ScalarDiffusionModel::interpolateToParticlesAndUpdate(const Patch* patch,
                                                           const MPMMaterial* matl,
                                                           DataWarehouse* old_dw,
		      										  							     				 DataWarehouse* new_dw)
{

}

void ScalarDiffusionModel::scheduleFinalParticleUpdate(Task* task, const MPMMaterial* matl, 
		                                                   const PatchSet* patch) const
{

}

void ScalarDiffusionModel::finalParticleUpdate(const Patch* patch, const MPMMaterial* matl,
                                     DataWarehouse* old_dw, DataWarehouse* new_dw)
{

}

