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

#include <CCA/Components/MPM/ReactionDiffusion/CommonIFConcDiff.h>
#include <CCA/Components/MPM/ReactionDiffusion/ScalarDiffusionModel.h>
#include <CCA/Components/MPM/ReactionDiffusion/JGConcentrationDiffusion.h>
#include <CCA/Components/MPM/ReactionDiffusion/RFConcDiffusion1MPM.h>
#include <CCA/Components/MPM/ReactionDiffusion/GaoDiffusion.h>
#include <CCA/Components/MPM/ReactionDiffusion/ReactionDiffusionLabel.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/MPMBoundCond.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Ghost.h>

using namespace Uintah;
using namespace std;

CommonIFConcDiff::CommonIFConcDiff(ProblemSpecP& ps, SimulationStateP& sS, MPMFlags* Mflag)
                 : SDInterfaceModel(ps, sS, Mflag){

  string diffusion_type;

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

  include_hydrostress = false;

  for (ProblemSpecP mat_ps = ps->findBlock("material"); mat_ps != 0;
       mat_ps = mat_ps->findNextBlock("material") ) {
    ProblemSpecP child = mat_ps->findBlock("diffusion_model");
    if(!child)
      throw ProblemSetupException("Cannot find diffusion_model tag", __FILE__, __LINE__);
    string mat_type;
    if(!child->getAttribute("type", mat_type))
      throw ProblemSetupException("No type for diffusion_model", __FILE__, __LINE__);
    if(mat_type == "rf1"){
      include_hydrostress = true;
    }else if(mat_type == "gao_diffusion"){
      include_hydrostress = true;
    }
  }

  if(include_hydrostress){
    int numMPM = d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMPM; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
      ScalarDiffusionModel* sdm = mpm_matl->getScalarDiffusionModel();
      diffusion_type = sdm->getDiffusionType();
      if(diffusion_type == "jg"){
        dynamic_cast<JGConcentrationDiffusion*>(sdm)->setIncludeHydroStress(true);
      }else if( diffusion_type == "rf1"){
        dynamic_cast<RFConcDiffusion1MPM*>(sdm)->setIncludeHydroStress(true);
      }else if( diffusion_type == "gao_diffusion"){
        dynamic_cast<GaoDiffusion*>(sdm)->setIncludeHydroStress(true);
      }
    }
  }
}

CommonIFConcDiff::~CommonIFConcDiff(){
  delete(d_lb);
  delete(d_rdlb);
}

void CommonIFConcDiff::addInitialComputesAndRequires(Task* task,
                                                     const PatchSet* patches) const
{
  int numMPM = d_sharedState->getNumMPMMatls();
  for(int m = 0; m < numMPM; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ScalarDiffusionModel* sdm = mpm_matl->getScalarDiffusionModel();
    sdm->addInitialComputesAndRequires(task, mpm_matl, patches);
    if(include_hydrostress){
      if(mpm_matl->getDWIndex() == 0){
        task->computes(d_rdlb->maxHydroStressLabel0);
        task->computes(d_rdlb->minHydroStressLabel0);
      }else if(mpm_matl->getDWIndex() == 1){
        task->computes(d_rdlb->maxHydroStressLabel1);
        task->computes(d_rdlb->minHydroStressLabel1);
      }else if(mpm_matl->getDWIndex() == 2){
        task->computes(d_rdlb->maxHydroStressLabel2);
        task->computes(d_rdlb->minHydroStressLabel2);
      }else if(mpm_matl->getDWIndex() == 3){
        task->computes(d_rdlb->maxHydroStressLabel3);
        task->computes(d_rdlb->minHydroStressLabel3);
      }else{
        throw ProblemSetupException("Need more HydroStressLabels in",__FILE__, __LINE__);
      }
    }
  }
}

void CommonIFConcDiff::initializeSDMData(const Patch* patch, DataWarehouse* new_dw)
{
  int numMPM = d_sharedState->getNumMPMMatls();
  for(int m = 0; m < numMPM; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ScalarDiffusionModel* sdm = mpm_matl->getScalarDiffusionModel();
    sdm->initializeSDMData(patch, mpm_matl, new_dw);
    if(include_hydrostress){
      if(mpm_matl->getDWIndex() == 0){
        new_dw->put(max_vartype(0), d_rdlb->maxHydroStressLabel0);
        new_dw->put(min_vartype(0), d_rdlb->minHydroStressLabel0);
      }else if(mpm_matl->getDWIndex() == 1){
        new_dw->put(max_vartype(0), d_rdlb->maxHydroStressLabel1);
        new_dw->put(min_vartype(0), d_rdlb->minHydroStressLabel1);
      }else if(mpm_matl->getDWIndex() == 2){
        new_dw->put(max_vartype(0), d_rdlb->maxHydroStressLabel2);
        new_dw->put(min_vartype(0), d_rdlb->minHydroStressLabel2);
      }else if(mpm_matl->getDWIndex() == 3){
        new_dw->put(max_vartype(0), d_rdlb->maxHydroStressLabel3);
        new_dw->put(min_vartype(0), d_rdlb->minHydroStressLabel3);
      }
    }
  }
}

void CommonIFConcDiff::computeDivergence(const Patch* patch,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw)
{
  int numMPM = d_sharedState->getNumMPMMatls();

  for(int m = 0; m < numMPM; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ScalarDiffusionModel* sdm = mpm_matl->getScalarDiffusionModel();
    sdm->computeDivergence(patch, mpm_matl, old_dw, new_dw);
  }

  Ghost::GhostType  gnone = Ghost::None;

  constNCVariable<double> gmass;
  constNCVariable<double> gConcRate;
  NCVariable<double> globalConcRate;
  NCVariable<double> globalmass;
  new_dw->allocateTemporary(globalConcRate, patch, gnone, 0);
  new_dw->allocateTemporary(globalmass,     patch, gnone, 0);
  globalConcRate.initialize(0);
  globalmass.initialize(0);

  for(int m = 0; m < numMPM; m++){
    int dwi = d_sharedState->getMPMMaterial(m)->getDWIndex();

    new_dw->get(gmass,     d_lb->gMassLabel,               dwi, patch, gnone,0);
    new_dw->get(gConcRate, d_rdlb->gConcentrationRateLabel,dwi, patch, gnone,0);

    for(NodeIterator iter=patch->getExtraNodeIterator();
                     !iter.done();iter++){
      IntVector c = *iter; 
      globalConcRate[c] += gmass[c] * gConcRate[c];
      globalmass[c]     += gmass[c];
    }
  }

  for(int m = 0; m < numMPM; m++){
    int dwi = d_sharedState->getMPMMaterial(m)->getDWIndex();
	  NCVariable<double> gConcRate;

    new_dw->getModifiable(gConcRate, d_rdlb->gConcentrationRateLabel, dwi, patch, gnone, 0);
    for(NodeIterator iter=patch->getExtraNodeIterator();
                     !iter.done();iter++){
      IntVector c = *iter; 
      gConcRate[c] = globalConcRate[c]/globalmass[c];
    }
  }
}

void CommonIFConcDiff::outputProblemSpec(ProblemSpecP& ps, bool output_sdim_tag)
{

  ProblemSpecP sdim_ps = ps;
  if (output_sdim_tag) {
    sdim_ps = ps->appendChild("diffusion_interface");
    sdim_ps->appendElement("type","common");
  }
}
