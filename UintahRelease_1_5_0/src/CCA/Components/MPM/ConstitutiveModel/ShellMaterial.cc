/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/MPM/ConstitutiveModel/ShellMaterial.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Grid/Patch.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Math/Matrix3.h>
#include <Core/Math/Short27.h> // for Fracture
#include <Core/Grid/Variables/NodeIterator.h> // just added
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Math/MinMax.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/DebugStream.h>
#include <fstream>
#include <iostream>

using std::cerr;
using namespace Uintah;


static DebugStream debug("ShellMat", false);

////////////////////////////////////////////////////////////////////////////////
//
// Constructor
//
ShellMaterial::ShellMaterial(ProblemSpecP& ps, MPMFlags* Mflag)
    : ConstitutiveModel(Mflag)
{
  // Read Material Constants
  ps->require("bulk_modulus", d_initialData.Bulk);
  ps->require("shear_modulus",d_initialData.Shear);

  d_includeFlowWork = true;
  ps->get("includeFlowWork",d_includeFlowWork);

  // Allocate local VarLabels
  pNormalRotRateLabel = VarLabel::create("p.normalRotRate",
                     ParticleVariable<Vector>::getTypeDescription());
  pDefGradTopLabel = VarLabel::create("p.defGradTop",
                     ParticleVariable<Matrix3>::getTypeDescription());
  pDefGradCenLabel = VarLabel::create("p.defGradCen",
                     ParticleVariable<Matrix3>::getTypeDescription());
  pDefGradBotLabel = VarLabel::create("p.defGradBot",
                     ParticleVariable<Matrix3>::getTypeDescription());
  pStressTopLabel = VarLabel::create("p.stressTop",
                     ParticleVariable<Matrix3>::getTypeDescription());
  pStressCenLabel = VarLabel::create("p.stressCen",
                     ParticleVariable<Matrix3>::getTypeDescription());
  pStressBotLabel = VarLabel::create("p.stressBot",
                     ParticleVariable<Matrix3>::getTypeDescription());

  pNormalRotRateLabel_preReloc = VarLabel::create("p.normalRotRate+",
                     ParticleVariable<Vector>::getTypeDescription());
  pDefGradTopLabel_preReloc = VarLabel::create("p.defGradTop+",
                     ParticleVariable<Matrix3>::getTypeDescription());
  pDefGradCenLabel_preReloc = VarLabel::create("p.defGradCen+",
                     ParticleVariable<Matrix3>::getTypeDescription());
  pDefGradBotLabel_preReloc = VarLabel::create("p.defGradBot+",
                     ParticleVariable<Matrix3>::getTypeDescription());
  pStressTopLabel_preReloc = VarLabel::create("p.stressTop+",
                     ParticleVariable<Matrix3>::getTypeDescription());
  pStressCenLabel_preReloc = VarLabel::create("p.stressCen+",
                     ParticleVariable<Matrix3>::getTypeDescription());
  pStressBotLabel_preReloc = VarLabel::create("p.stressBot+",
                     ParticleVariable<Matrix3>::getTypeDescription());

  pAverageMomentLabel = VarLabel::create("p.averageMoment",
                     ParticleVariable<Matrix3>::getTypeDescription());
  pNormalDotAvStressLabel = VarLabel::create("p.normalDotAvStress",
                     ParticleVariable<Vector>::getTypeDescription());
  pRotMassLabel = VarLabel::create("p.rotMass",
                     ParticleVariable<double>::getTypeDescription());
  pNormalRotAccLabel = VarLabel::create("p.rotAcc",
                     ParticleVariable<Vector>::getTypeDescription());
}

ShellMaterial::ShellMaterial(const ShellMaterial* cm) 
  : ConstitutiveModel(cm)
{
  d_initialData.Bulk = cm->d_initialData.Bulk;
  d_initialData.Shear = cm->d_initialData.Shear;

  // Allocate local VarLabels
  pNormalRotRateLabel = VarLabel::create("p.normalRotRate",
                     ParticleVariable<Vector>::getTypeDescription());
  pDefGradTopLabel = VarLabel::create("p.defGradTop",
                     ParticleVariable<Matrix3>::getTypeDescription());
  pDefGradCenLabel = VarLabel::create("p.defGradCen",
                     ParticleVariable<Matrix3>::getTypeDescription());
  pDefGradBotLabel = VarLabel::create("p.defGradBot",
                     ParticleVariable<Matrix3>::getTypeDescription());
  pStressTopLabel = VarLabel::create("p.stressTop",
                     ParticleVariable<Matrix3>::getTypeDescription());
  pStressCenLabel = VarLabel::create("p.stressCen",
                     ParticleVariable<Matrix3>::getTypeDescription());
  pStressBotLabel = VarLabel::create("p.stressBot",
                     ParticleVariable<Matrix3>::getTypeDescription());

  pNormalRotRateLabel_preReloc = VarLabel::create("p.normalRotRate+",
                     ParticleVariable<Vector>::getTypeDescription());
  pDefGradTopLabel_preReloc = VarLabel::create("p.defGradTop+",
                     ParticleVariable<Matrix3>::getTypeDescription());
  pDefGradCenLabel_preReloc = VarLabel::create("p.defGradCen+",
                     ParticleVariable<Matrix3>::getTypeDescription());
  pDefGradBotLabel_preReloc = VarLabel::create("p.defGradBot+",
                     ParticleVariable<Matrix3>::getTypeDescription());
  pStressTopLabel_preReloc = VarLabel::create("p.stressTop+",
                     ParticleVariable<Matrix3>::getTypeDescription());
  pStressCenLabel_preReloc = VarLabel::create("p.stressCen+",
                     ParticleVariable<Matrix3>::getTypeDescription());
  pStressBotLabel_preReloc = VarLabel::create("p.stressBot+",
                     ParticleVariable<Matrix3>::getTypeDescription());

  pAverageMomentLabel = VarLabel::create("p.averageMoment",
                     ParticleVariable<Matrix3>::getTypeDescription());
  pNormalDotAvStressLabel = VarLabel::create("p.normalDotAvStress",
                     ParticleVariable<Vector>::getTypeDescription());
  pRotMassLabel = VarLabel::create("p.rotMass",
                     ParticleVariable<double>::getTypeDescription());
  pNormalRotAccLabel = VarLabel::create("p.rotAcc",
                     ParticleVariable<Vector>::getTypeDescription());
}

////////////////////////////////////////////////////////////////////////////////
//
// Destructor
//
ShellMaterial::~ShellMaterial()
{
  VarLabel::destroy(pNormalRotRateLabel); 
  VarLabel::destroy(pDefGradTopLabel);
  VarLabel::destroy(pDefGradCenLabel);
  VarLabel::destroy(pDefGradBotLabel);
  VarLabel::destroy(pStressTopLabel);
  VarLabel::destroy(pStressCenLabel);
  VarLabel::destroy(pStressBotLabel);

  VarLabel::destroy(pNormalRotRateLabel_preReloc); 
  VarLabel::destroy(pDefGradTopLabel_preReloc);
  VarLabel::destroy(pDefGradCenLabel_preReloc);
  VarLabel::destroy(pDefGradBotLabel_preReloc);
  VarLabel::destroy(pStressTopLabel_preReloc);
  VarLabel::destroy(pStressCenLabel_preReloc);
  VarLabel::destroy(pStressBotLabel_preReloc);

  VarLabel::destroy(pAverageMomentLabel);
  VarLabel::destroy(pNormalDotAvStressLabel);
  VarLabel::destroy(pRotMassLabel);
  VarLabel::destroy(pNormalRotAccLabel);
}

void ShellMaterial::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","shell_CNH");
  }

  cm_ps->appendElement("bulk_modulus", d_initialData.Bulk);
  cm_ps->appendElement("shear_modulus",d_initialData.Shear);
  cm_ps->appendElement("includeFlowWork",d_includeFlowWork);
}


ShellMaterial* ShellMaterial::clone()
{
  return scinew ShellMaterial(*this);
}

////////////////////////////////////////////////////////////////////////////////
//
// Make sure all labels are correctly relocated
//
void 
ShellMaterial::addParticleState(std::vector<const VarLabel*>& from,
                                std::vector<const VarLabel*>& to)
{
  // Add the local particle state data for this constitutive model.
  from.push_back(lb->pThickTopLabel);
  from.push_back(lb->pInitialThickTopLabel);
  from.push_back(lb->pThickBotLabel);
  from.push_back(lb->pInitialThickBotLabel);
  from.push_back(lb->pNormalLabel);
  from.push_back(lb->pInitialNormalLabel);

  to.push_back(lb->pThickTopLabel_preReloc);
  to.push_back(lb->pInitialThickTopLabel_preReloc);
  to.push_back(lb->pThickBotLabel_preReloc);
  to.push_back(lb->pInitialThickBotLabel_preReloc);
  to.push_back(lb->pNormalLabel_preReloc);
  to.push_back(lb->pInitialNormalLabel_preReloc);

  from.push_back(pNormalRotRateLabel); 
  from.push_back(pDefGradTopLabel);
  from.push_back(pDefGradCenLabel);
  from.push_back(pDefGradBotLabel);
  from.push_back(pStressTopLabel);
  from.push_back(pStressCenLabel);
  from.push_back(pStressBotLabel);

  to.push_back(pNormalRotRateLabel_preReloc); 
  to.push_back(pDefGradTopLabel_preReloc);
  to.push_back(pDefGradCenLabel_preReloc);
  to.push_back(pDefGradBotLabel_preReloc);
  to.push_back(pStressTopLabel_preReloc);
  to.push_back(pStressCenLabel_preReloc);
  to.push_back(pStressBotLabel_preReloc);
}

////////////////////////////////////////////////////////////////////////////////
//
// Create initialization task graph for local variables
//
void 
ShellMaterial::addInitialComputesAndRequires(Task* task,
                                             const MPMMaterial* matl,
                                             const PatchSet*) const
{
  const MaterialSubset* matlset = matl->thisMaterial();

  task->computes(lb->pThickTopLabel,        matlset);
  task->computes(lb->pInitialThickTopLabel, matlset);
  task->computes(lb->pThickBotLabel,        matlset);
  task->computes(lb->pInitialThickBotLabel, matlset);
  task->computes(lb->pNormalLabel,          matlset);
  task->computes(lb->pInitialNormalLabel,   matlset);

  task->computes(pNormalRotRateLabel, matlset);
  task->computes(pDefGradTopLabel,    matlset);
  task->computes(pDefGradCenLabel,    matlset);
  task->computes(pDefGradBotLabel,    matlset);
  task->computes(pStressTopLabel,     matlset);
  task->computes(pStressCenLabel,     matlset);
  task->computes(pStressBotLabel,     matlset);
}

////////////////////////////////////////////////////////////////////////////////
//
// Initialize the data needed for the Shell Material Model
//
void 
ShellMaterial::initializeCMData(const Patch* patch,
                                const MPMMaterial* matl,
                                DataWarehouse* new_dw)
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  initSharedDataForExplicit(patch, matl, new_dw);

  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  Matrix3 One, Zero(0.0); One.Identity();

  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  ParticleVariable<Vector>  pRotRate; 
  ParticleVariable<Matrix3> pDefGradTop, pDefGradCen, pDefGradBot, 
                            pStressTop, pStressCen, pStressBot;
  new_dw->allocateAndPut(pRotRate,    pNormalRotRateLabel, pset);
  new_dw->allocateAndPut(pDefGradTop, pDefGradTopLabel,    pset);
  new_dw->allocateAndPut(pDefGradCen, pDefGradCenLabel,    pset);
  new_dw->allocateAndPut(pDefGradBot, pDefGradBotLabel,    pset);
  new_dw->allocateAndPut(pStressTop,  pStressTopLabel,     pset);
  new_dw->allocateAndPut(pStressCen,  pStressCenLabel,     pset);
  new_dw->allocateAndPut(pStressBot,  pStressBotLabel,     pset);

  ParticleSubset::iterator iter = pset->begin();
  for(; iter != pset->end(); iter++) {
    particleIndex pidx = *iter;
    pRotRate[pidx]    = Vector(0.0,0.0,0.0);
    pDefGradTop[pidx] = One;
    pDefGradCen[pidx] = One;
    pDefGradBot[pidx] = One;
    pStressTop[pidx]  = Zero;
    pStressCen[pidx]  = Zero;
    pStressBot[pidx]  = Zero;
  }

  computeStableTimestep(patch, matl, new_dw);
}

void ShellMaterial::allocateCMDataAddRequires(Task* task,
                                              const MPMMaterial* matl,
                                              const PatchSet* patches,
                                              MPMLabel* ) const
{
  const MaterialSubset* matlset = matl->thisMaterial();

  // Allocate the variables shared by all constitutive models
  // for the particle convert operation
  // This method is defined in the ConstitutiveModel base class.
  addSharedRForConvertExplicit(task, matlset, patches);

  // Add requires local to this model
  Ghost::GhostType  gnone = Ghost::None;
  task->requires(Task::NewDW, pNormalRotRateLabel_preReloc, matlset, gnone);
  task->requires(Task::NewDW, pDefGradTopLabel_preReloc,    matlset, gnone);
  task->requires(Task::NewDW, pDefGradCenLabel_preReloc,    matlset, gnone);
  task->requires(Task::NewDW, pDefGradBotLabel_preReloc,    matlset, gnone);
  task->requires(Task::NewDW, pStressTopLabel_preReloc,     matlset, gnone);
  task->requires(Task::NewDW, pStressCenLabel_preReloc,     matlset, gnone);
  task->requires(Task::NewDW, pStressBotLabel_preReloc,     matlset, gnone);
}


void 
ShellMaterial::allocateCMDataAdd(DataWarehouse* new_dw,
                                 ParticleSubset* addset,
  map<const VarLabel*, ParticleVariableBase*>* newState,
                                 ParticleSubset* delset,
                                 DataWarehouse* )
{
  // Copy the data common to all constitutive models from the particle to be 
  // deleted to the particle to be added. 
  // This method is defined in the ConstitutiveModel base class.
  copyDelToAddSetForConvertExplicit(new_dw, delset, addset, newState);
  
  // Copy the data local to this constitutive model from the particles to 
  // be deleted to the particles to be added
  ParticleVariable<Vector>  pRotRate; 
  ParticleVariable<Matrix3> pDefGradTop, pDefGradCen, pDefGradBot, 
                            pStressTop, pStressCen, pStressBot;

  constParticleVariable<Vector> o_RotRate;
  constParticleVariable<Matrix3> o_DefGradTop, o_DefGradCen, o_DefGradBot,
    o_StressTop, o_StressCen, o_StressBot;

  new_dw->allocateTemporary(pRotRate, addset);
  new_dw->allocateTemporary(pDefGradTop, addset);
  new_dw->allocateTemporary(pDefGradCen, addset);
  new_dw->allocateTemporary(pDefGradBot, addset);
  new_dw->allocateTemporary(pStressTop, addset);
  new_dw->allocateTemporary(pStressCen, addset);
  new_dw->allocateTemporary(pStressBot, addset);

  new_dw->get(o_RotRate,pNormalRotRateLabel_preReloc,delset);
  new_dw->get(o_DefGradTop,pDefGradTopLabel_preReloc,delset);
  new_dw->get(o_DefGradCen,pDefGradCenLabel_preReloc,delset);
  new_dw->get(o_DefGradBot,pDefGradBotLabel_preReloc,delset);
  new_dw->get(o_StressTop,pStressTopLabel_preReloc,delset);
  new_dw->get(o_StressCen,pStressCenLabel_preReloc,delset);
  new_dw->get(o_StressBot,pStressBotLabel_preReloc,delset);

  ParticleSubset::iterator o,n = addset->begin();
  for(o=delset->begin(); o != delset->end(); o++,n++) {
    pRotRate[*n]    = o_RotRate[*o];
    pDefGradTop[*n] = o_DefGradTop[*o];
    pDefGradCen[*n] = o_DefGradCen[*o];
    pDefGradBot[*n] = o_DefGradBot[*o];
    pStressTop[*n]  = o_StressTop[*o];
    pStressCen[*n]  = o_StressCen[*o];
    pStressBot[*n]  = o_StressBot[*o];
  }

  (*newState)[pNormalRotRateLabel]=pRotRate.clone();
  (*newState)[pDefGradTopLabel]=pDefGradTop.clone();
  (*newState)[pDefGradCenLabel]=pDefGradCen.clone();
  (*newState)[pDefGradBotLabel]=pDefGradBot.clone();
  (*newState)[pStressTopLabel]=pStressTop.clone();
  (*newState)[pStressCenLabel]=pStressCen.clone();
  (*newState)[pStressBotLabel]=pStressBot.clone();
}

////////////////////////////////////////////////////////////////////////////////
//
// Compute a stable time step.
// This is only called for the initial timestep - all other timesteps
// are computed as a side-effect of compute Stress Tensor
//
void 
ShellMaterial::computeStableTimestep(const Patch* patch,
                                     const MPMMaterial* matl,
                                     DataWarehouse* new_dw)
{
  int dwi = matl->getDWIndex();
  ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);

  constParticleVariable<double> pmass, pvolume;
  constParticleVariable<Vector> pvelocity;
  new_dw->get(pmass,     lb->pMassLabel, pset);
  new_dw->get(pvolume,   lb->pVolumeLabel, pset);
  new_dw->get(pvelocity, lb->pVelocityLabel, pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double mu = d_initialData.Shear;
  double bulk = d_initialData.Bulk;
  ParticleSubset::iterator iter = pset->begin();
  for(;iter != pset->end();iter++){
    particleIndex idx = *iter;

    // Compute wave speed at each particle, store the maximum
    c_dil = sqrt((bulk + 4.0*mu/3.0)*pvolume[idx]/pmass[idx]);
    WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                     Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                     Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
  }
  Vector dx = patch->dCell();
  WaveSpeed = dx/WaveSpeed;
  double delT_new = WaveSpeed.minComponent();
  new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
}

///////////////////////////////////////////////////////////////////////////
//
// Add computes and requires for interpolation of particle rotation to grid
//
void 
ShellMaterial::addComputesRequiresParticleRotToGrid(Task* task,
                                                    const MPMMaterial* matl,
                                                    const PatchSet* )
{
  Ghost::GhostType  gan = Ghost::AroundNodes;
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW,   lb->pMassLabel,          matlset, gan, NGN);
  task->requires(Task::OldDW,   lb->pXLabel,             matlset, gan, NGN);
  task->requires(Task::OldDW, lb->pSizeLabel,          matlset, gan, NGN);
  task->requires(Task::OldDW,   pNormalRotRateLabel,     matlset, gan, NGN);
  task->requires(Task::NewDW,   lb->gMassLabel,          matlset, gan, NGN);
  task->computes(lb->gNormalRotRateLabel, matlset);
}

///////////////////////////////////////////////////////////////////////////
//
// Actually interpolate normal rotation from particles to the grid
//
void 
ShellMaterial::interpolateParticleRotToGrid(const PatchSubset* patches,
                                            const MPMMaterial* matl,
                                            DataWarehouse* old_dw,
                                            DataWarehouse* new_dw)
{
  // Constants
  Ghost::GhostType  gan = Ghost::AroundNodes;
  int dwi = matl->getDWIndex();

  // Create arrays for the particle data
  constParticleVariable<double> pMass;
  constParticleVariable<Point>  pX;
  constParticleVariable<Vector> pRotRate;
  constParticleVariable<Matrix3> pSize;
  constParticleVariable<Matrix3> deformationGradient;
  constNCVariable<double> gMass;

  // Create arrays for the grid data
  NCVariable<Vector> gRotRate;

  // Loop thru patches
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());

    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch, gan, NGN, 
                                                     lb->pXLabel);

    // Get the required data
    old_dw->get(pMass,          lb->pMassLabel,          pset);
    old_dw->get(pX,             lb->pXLabel,             pset);
    old_dw->get(pSize,        lb->pSizeLabel,          pset);
    old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);
    old_dw->get(pRotRate,       pNormalRotRateLabel,     pset);
    new_dw->get(gMass,          lb->gMassLabel, dwi,     patch, gan, NGN);

    // Allocate arrays for the grid data
    new_dw->allocateAndPut(gRotRate, lb->gNormalRotRateLabel, dwi,patch);
    gRotRate.initialize(Vector(0,0,0));

    // Interpolate particle data to Grid data.  Attempt to conserve 
    // angular momentum (I_grid*omega_grid =  S*I_particle*omega_particle).

    Vector pMom(0.0,0.0,0.0);
    for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); 
         iter++){
      particleIndex idx = *iter;

      // Get the node indices that surround the cell
      interpolator->findCellAndWeights(pX[idx], ni, S,pSize[idx],deformationGradient[idx]);

      // Calculate momentum
      pMom = pRotRate[idx]*pMass[idx];

      // Add each particles contribution to the grid rotation rate
      for(int k = 0; k < flag->d_8or27; k++) {
        if(patch->containsNode(ni[k])) 
          gRotRate[ni[k]] += pMom * S[k];
      }
    } // End of particle loop

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
      gRotRate[*iter] /= gMass[*iter];
    }

    delete interpolator;
  }  // End loop over patches
}

////////////////////////////////////////////////////////////////////////////////
//
// Create task graph for each time step after initialization
//
void 
ShellMaterial::addComputesAndRequires(Task* task,
                                      const MPMMaterial* matl,
                                      const PatchSet* patches) const
{
  // Add the computes and requires that are common to all explicit 
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForHypoExplicit(task, matlset, patches);

  // Other constitutive model and input dependent computes and requires
  Ghost::GhostType  gnone = Ghost::None;
  Ghost::GhostType  gac   = Ghost::AroundCells;

  task->requires(Task::OldDW, lb->pThickTopLabel,           matlset, gnone);
  task->requires(Task::OldDW, lb->pInitialThickTopLabel,    matlset, gnone);
  task->requires(Task::OldDW, lb->pThickBotLabel,           matlset, gnone);
  task->requires(Task::OldDW, lb->pInitialThickBotLabel,    matlset, gnone);
  task->requires(Task::OldDW, lb->pNormalLabel,             matlset, gnone);
  task->requires(Task::OldDW, lb->pVelocityLabel,           matlset, gnone);
  task->requires(Task::OldDW, pNormalRotRateLabel,          matlset, gnone);
  task->requires(Task::OldDW, pDefGradTopLabel,             matlset, gnone);
  task->requires(Task::OldDW, pDefGradCenLabel,             matlset, gnone);
  task->requires(Task::OldDW, pDefGradBotLabel,             matlset, gnone);
  task->requires(Task::OldDW, pStressTopLabel,              matlset, gnone);
  task->requires(Task::OldDW, pStressCenLabel,              matlset, gnone);
  task->requires(Task::OldDW, pStressBotLabel,              matlset, gnone);
  task->requires(Task::NewDW, lb->gNormalRotRateLabel,      matlset, gac, NGN);

  task->computes(lb->pThickTopLabel_preReloc,           matlset);
  task->computes(lb->pInitialThickTopLabel_preReloc,    matlset);
  task->computes(lb->pThickBotLabel_preReloc,           matlset);
  task->computes(lb->pInitialThickBotLabel_preReloc,    matlset);
  task->computes(pDefGradTopLabel_preReloc,             matlset);
  task->computes(pDefGradCenLabel_preReloc,             matlset);
  task->computes(pDefGradBotLabel_preReloc,             matlset);
  task->computes(pStressTopLabel_preReloc,              matlset);
  task->computes(pStressCenLabel_preReloc,              matlset);
  task->computes(pStressBotLabel_preReloc,              matlset);

  task->computes(pAverageMomentLabel,                   matlset);
  task->computes(pNormalDotAvStressLabel,               matlset);
  task->computes(pRotMassLabel,                         matlset);
}

////////////////////////////////////////////////////////////////////////////////
//
// Compute the stress tensor
//
void 
ShellMaterial::computeStressTensor(const PatchSubset* patches,
                                   const MPMMaterial* matl,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw)
{
  // Initialize contants
  //  double onethird = (1.0/3.0);
  Matrix3 One; One.Identity();
  double shear = d_initialData.Shear;
  double bulk  = d_initialData.Bulk;
  double rho_orig = matl->getInitialDensity();
  Ghost::GhostType gac = Ghost::AroundCells;

  // Loop thru patches
  for(int pp=0;pp<patches->size();pp++){

    // Current patch
    const Patch* patch = patches->get(pp);

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<Vector> d_S(interpolator->size());


    // Read the datawarehouse
    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    // Read from datawarehouses 
    constParticleVariable<double>  pMass, pThickTop, pThickBot, pThickTop0,
                                   pThickBot0; 
    constParticleVariable<Point>   pX;
    constParticleVariable<Vector>  pVelocity, pRotRate, pNormal; 
    constParticleVariable<Matrix3> pSize; 
    constParticleVariable<Matrix3> pStressTop, pStressCen, pStressBot, 
                                   pStress, 
                                   pDefGradTop, pDefGradCen, pDefGradBot, 
                                   pDefGrad;
    constNCVariable<Vector>        gVelocity, gRotRate; 
    delt_vartype                   delT; 
    old_dw->get(pMass,       lb->pMassLabel,               pset);
    old_dw->get(pThickTop,   lb->pThickTopLabel,           pset);
    old_dw->get(pThickBot,   lb->pThickBotLabel,           pset);
    old_dw->get(pThickTop0,  lb->pInitialThickTopLabel,    pset);
    old_dw->get(pThickBot0,  lb->pInitialThickBotLabel,    pset);
    old_dw->get(pX,          lb->pXLabel,                  pset);
    old_dw->get(pSize,       lb->pSizeLabel,               pset);
    old_dw->get(pNormal,     lb->pNormalLabel,             pset);
    old_dw->get(pVelocity,   lb->pVelocityLabel,           pset);
    old_dw->get(pRotRate,    pNormalRotRateLabel,          pset);
    old_dw->get(pDefGradTop, pDefGradTopLabel,             pset);
    old_dw->get(pDefGradCen, pDefGradCenLabel,             pset);
    old_dw->get(pDefGradBot, pDefGradBotLabel,             pset);
    old_dw->get(pStressTop,  pStressTopLabel,              pset);
    old_dw->get(pStressCen,  pStressCenLabel,              pset);
    old_dw->get(pStressBot,  pStressBotLabel,              pset);
    old_dw->get(pStress,     lb->pStressLabel,             pset);
    old_dw->get(pDefGrad,    lb->pDeformationMeasureLabel, pset);
    old_dw->get(delT,        lb->delTLabel, getLevel(patches));
    new_dw->get(gVelocity,   lb->gVelocityStarLabel,    dwi, patch, gac, NGN);
    new_dw->get(gRotRate,    lb->gNormalRotRateLabel, dwi, patch, gac, NGN);

    // Allocate for updated variables in new_dw 
    ParticleVariable<double>  pVolume_new, pThickTop_new, pThickBot_new,
                              pThickTop0_new, pThickBot0_new; 
    ParticleVariable<Matrix3> pDefGradTop_new, pDefGradBot_new, pDefGradCen_new,
                              pStressTop_new, pStressCen_new, pStressBot_new, 
                              pStress_new, pDefGrad_new;
    new_dw->allocateAndPut(pVolume_new,    lb->pVolumeLabel_preReloc,     pset);
    new_dw->allocateAndPut(pThickTop_new,  lb->pThickTopLabel_preReloc,   pset);
    new_dw->allocateAndPut(pThickTop0_new, lb->pInitialThickTopLabel_preReloc,
                           pset);
    new_dw->allocateAndPut(pThickBot_new,  lb->pThickBotLabel_preReloc,   pset);
    new_dw->allocateAndPut(pThickBot0_new, lb->pInitialThickBotLabel_preReloc,
                           pset);
    new_dw->allocateAndPut(pDefGradTop_new, pDefGradTopLabel_preReloc,    pset);
    new_dw->allocateAndPut(pDefGradCen_new, pDefGradCenLabel_preReloc,    pset);
    new_dw->allocateAndPut(pDefGradBot_new, pDefGradBotLabel_preReloc,    pset);
    new_dw->allocateAndPut(pStressTop_new,  pStressTopLabel_preReloc,     pset);
    new_dw->allocateAndPut(pStressCen_new,  pStressCenLabel_preReloc,     pset);
    new_dw->allocateAndPut(pStressBot_new,  pStressBotLabel_preReloc,     pset);
    new_dw->allocateAndPut(pStress_new,    lb->pStressLabel_preReloc,     pset);
    new_dw->allocateAndPut(pDefGrad_new,  lb->pDeformationMeasureLabel_preReloc,
                           pset);

    ParticleVariable<double>  pRotMass;
    ParticleVariable<Vector>  pNDotAvSig;
    ParticleVariable<Matrix3> pAvMoment;
    ParticleVariable<double> pdTdt,p_q;
    new_dw->allocateAndPut(pAvMoment,  pAverageMomentLabel,     pset);
    new_dw->allocateAndPut(pNDotAvSig, pNormalDotAvStressLabel, pset);
    new_dw->allocateAndPut(pRotMass,   pRotMassLabel,           pset);
    new_dw->allocateAndPut(pdTdt,      lb->pdTdtLabel_preReloc, pset);
    new_dw->allocateAndPut(p_q,        lb->p_qLabel_preReloc,   pset);

    // Initialize contants
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    // Initialize variables
    double strainEnergy = 0.0;

    // Loop thru particles
    ParticleSubset::iterator iter = pset->begin();
    for(; iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Assign zero internal heating by default - modify if necessary.
      pdTdt[idx] = 0.0;

      // Find the surrounding nodes, interpolation functions and derivatives

      interpolator->findCellAndShapeDerivatives(pX[idx],ni,d_S, pSize[idx],pDefGrad[idx]);

      // Calculate the spatial gradient of the velocity and the 
      // normal rotation rate
      Matrix3 velGrad(0.0), rotGrad(0.0);
      for(int k = 0; k < flag->d_8or27; k++) {
        Vector gvel = gVelocity[ni[k]];
        Vector grot = gRotRate[ni[k]];
        for (int j = 0; j<3; j++){
          double d_SXoodx = d_S[k][j] * oodx[j];
          for (int i = 0; i<3; i++) {
            velGrad(i,j) += gvel[i] * d_SXoodx;
            rotGrad(i,j) += grot[i] * d_SXoodx;
          }
        }
      }
      // Project the velocity gradient and rotation gradient on
      // to surface of the shell
      calcInPlaneGradient(pNormal[idx], velGrad, rotGrad);

      // Calculate the layer-wise velocity gradient for stress
      // calculations
      double zTop = pThickTop[idx];
      double zBot = pThickBot[idx];
      Matrix3 rn(pRotRate[idx], pNormal[idx]);
      Matrix3 velGradTop = (velGrad + rotGrad*zTop) + rn ;
      Matrix3 velGradCen = velGrad + rn ;
      Matrix3 velGradBot = (velGrad - rotGrad*zBot) + rn ;

      // Compute the deformation gradient increment using the time_step
      // velocity gradient (F_n^np1 = dudx * dt + Identity).
      Matrix3 defGradIncTop = velGradTop*delT + One;
      Matrix3 defGradIncCen = velGradCen*delT + One;
      Matrix3 defGradIncBot = velGradBot*delT + One;

      // Update the deformation gradient tensor to its time n+1 value.
      Matrix3 defGradTop_new = defGradIncTop*pDefGradTop[idx];
      Matrix3 defGradCen_new = defGradIncCen*pDefGradCen[idx];
      Matrix3 defGradBot_new = defGradIncBot*pDefGradBot[idx];

      // Compute stress using a constitutive relation
      //Matrix3 sigTop(0.0), sigBot(0.0), sigCen(0.0);
      //computeShellElasticStress(defGradTop_new, sigTop, bulk, shear);
      //computeShellElasticStress(defGradCen_new, sigCen, bulk, shear);
      //computeShellElasticStress(defGradBot_new, sigBot, bulk, shear);

      // Rotate the deformation gradient so that the 33 direction
      // is along the direction of the normal
      Matrix3 R; R.Identity();
      calcTotalRotation(Vector(0,0,1), pNormal[idx], R);
      defGradTop_new = R*defGradTop_new*R.Transpose();
      defGradCen_new = R*defGradCen_new*R.Transpose();
      defGradBot_new = R*defGradBot_new*R.Transpose();
      
      // Enforce the no normal stress condition (Sig33 = 0)
      // (we call this condition, roughly, plane stress)
      Matrix3 sigTop(0.0), sigBot(0.0), sigCen(0.0);
      if (!computePlaneStressAndDefGrad(defGradTop_new, sigTop, bulk, shear)) {
        cerr << "----------------------------------- " << endl;
        cerr << "Particle = " << idx << endl << endl;
        cerr << "Velocity Gradient = " << endl;
        cerr << velGrad << endl;
        cerr << "Rotation Gradient = " << endl;
        cerr << rotGrad << endl;
        cerr << "In-plane Velocity Gradient (top) = " << endl;
        cerr << velGradTop << endl;
        cerr << "In-plane Velocity Gradient (cen) = " << endl;
        cerr << velGradCen << endl;
        cerr << "In-plane Velocity Gradient (bot) = " << endl;
        cerr << velGradBot << endl;
        cerr << "In-plane Def Gradient Inc (top) = " << endl;
        cerr << defGradIncTop << endl;
        cerr << "In-plane Def Gradient Inc (cen) = " << endl;
        cerr << defGradIncCen << endl;
        cerr << "In-plane Def Gradient Inc (bot) = " << endl;
        cerr << defGradIncBot << endl;
        cerr << "New In-plane Def Gradient (top) = " << endl;
        cerr << defGradTop_new << endl;
        cerr << "New In-plane Def Gradient (cen) = " << endl;
        cerr << defGradCen_new << endl;
        cerr << "New In-plane Def Gradient (bot) = " << endl;
        cerr << defGradBot_new << endl;
        cerr << "Normal = " << pNormal[idx] << endl;
        cerr << "R = " << R << endl;
        cerr << "defGradTop = " << defGradTop_new << endl;
        cerr << "SigTop = " << sigTop << endl;
        exit(1);
      }
      if (!computePlaneStressAndDefGrad(defGradCen_new, sigCen, bulk, shear)) {
        cerr << "Normal = " << pNormal[idx] << endl;
        cerr << "R = " << R << endl;
        cerr << "defGradCen = " << defGradCen_new << endl;
        cerr << "SigCen = " << sigCen << endl;
        exit(1);
      }
      if (!computePlaneStressAndDefGrad(defGradBot_new, sigBot, bulk, shear)) {
        if (d_world->myrank() == 16) {
          cerr << "Current Processor = " << d_world->myrank() << endl;
          cerr << "Normal = " << pNormal[idx] << endl;
          cerr << "R = " << R << endl;
          cerr << "defGradBot = " << defGradBot_new << endl;
          cerr << "SigBot = " << sigBot << endl;
        }
        exit(1);
      }

      // Calculate the change in thickness in the direction of
      // the normal
      double zTopInc = 0.5*(defGradTop_new(2,2)+defGradCen_new(2,2));
      double zBotInc = 0.5*(defGradBot_new(2,2)+defGradCen_new(2,2));
      pThickTop_new[idx] = zTopInc*pThickTop0[idx];
      pThickBot_new[idx] = zBotInc*pThickBot0[idx];
      pThickTop0_new[idx] = pThickTop0[idx];
      pThickBot0_new[idx] = pThickBot0[idx];

      // Rotate back to global co-ordinates
      defGradTop_new = R.Transpose()*defGradTop_new*R;
      defGradCen_new = R.Transpose()*defGradCen_new*R;
      defGradBot_new = R.Transpose()*defGradBot_new*R;
      sigTop = R.Transpose()*sigTop*R;
      sigCen = R.Transpose()*sigCen*R;
      sigBot = R.Transpose()*sigBot*R;

      // Update the deformation gradients
      pDefGradTop_new[idx] = defGradTop_new;
      pDefGradCen_new[idx] = defGradCen_new;
      pDefGradBot_new[idx] = defGradBot_new;
      pDefGrad_new[idx] = pDefGradCen_new[idx];

      // Get the volumetric part of the deformation
      double Je = pDefGrad_new[idx].Determinant();
      pVolume_new[idx]=(pMass[idx]/rho_orig)*Je;

      // Calculate the average stress over the thickness of the shell
      // using the trapezoidal rule
      double ht = pThickTop_new[idx]; 
      double hb = pThickBot_new[idx];
      double h = (ht+hb)*2.0;
      ASSERT(h > 0.0);
      Matrix3 avStress = ((sigTop+sigCen)*ht + (sigBot+sigCen)*hb)/h;
      pNDotAvSig[idx] = (pNormal[idx]*avStress)*pVolume_new[idx];

      // Update the stresses
      pStressTop_new[idx] = sigTop;
      pStressCen_new[idx] = sigCen;
      pStressBot_new[idx] = sigBot;
      pStress_new[idx] = avStress;

      // Calculate the average moment over the thickness of the shell
      Matrix3 nn(pNormal[idx], pNormal[idx]);
      Matrix3 Is = One - nn;
      Matrix3 avMoment = (sigTop*(ht*ht) - sigBot*(hb*hb))*(0.5/h);
      pAvMoment[idx] = (Is*avMoment*Is)*pVolume_new[idx];

      // Calculate inertia term
      pRotMass[idx] = pMass[idx]*h*h/12.0;

      // Compute the strain energy for all the particles
      double U = 0.5*bulk*(0.5*(Je*Je - 1.0) - log(Je));
      Matrix3 be = pDefGrad_new[idx]*pDefGrad_new[idx].Transpose();
      Matrix3 bebar = be/pow(Je, 2.0/3.0);
      double W = 0.5*shear*(bebar.Trace() - 3.0);
      double e = (U + W)*pVolume_new[idx]/Je;
      strainEnergy += e;

      Vector pVel = pVelocity[idx];
      double c_dil = sqrt((bulk + 4.*shear/3.)*pVolume_new[idx]/pMass[idx]);
      WaveSpeed=Vector(Max(c_dil+fabs(pVel.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pVel.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pVel.z()),WaveSpeed.z()));

      // Compute artificial viscosity term
      if (flag->d_artificial_viscosity) {
        double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;
        double rho_cur = rho_orig/Je;
        double c_bulk = sqrt(bulk/rho_cur);
        Matrix3 D=(velGrad + velGrad.Transpose())*0.5;
        p_q[idx] = artificialBulkViscosity(D.Trace(), c_bulk, rho_cur, dx_ave);
      } else {
        p_q[idx] = 0.;
      }
    }  // end loop over particles

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
    
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(strainEnergy), lb->StrainEnergyLabel);
    }
    delete interpolator;
  }
}

///////////////////////////////////////////////////////////////////////////
//
// Add computes and requires computation of rotational internal moment
//
void 
ShellMaterial::addComputesRequiresRotInternalMoment(Task* task,
                                                    const MPMMaterial* matl,
                                                    const PatchSet* )
{
  Ghost::GhostType  gan   = Ghost::AroundNodes;
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW,   lb->pXLabel,               matlset, gan, NGN);
  task->requires(Task::OldDW, lb->pSizeLabel,            matlset, gan, NGN);
  task->requires(Task::NewDW,   pAverageMomentLabel,       matlset, gan, NGN);
  task->computes(lb->gNormalRotMomentLabel, matlset);
}

///////////////////////////////////////////////////////////////////////////
//
// Actually compute rotational Internal moment 
//
void 
ShellMaterial::computeRotInternalMoment(const PatchSubset* patches,
                                        const MPMMaterial* matl,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw)
{
  // Initialize constants
  Matrix3 One; One.Identity();
  Ghost::GhostType gan = Ghost::AroundNodes;
  int dwi = matl->getDWIndex();

  // Loop over patches
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch, gan, NGN,
                                                     lb->pXLabel);

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());
    vector<Vector> d_S(interpolator->size());


    // Get stuff from datawarehouse
    constParticleVariable<Point>   pX;
    constParticleVariable<Matrix3> pSize;
    constParticleVariable<Matrix3> pDefGrad;
    constParticleVariable<Matrix3> pAvMoment;
    old_dw->get(pX,         lb->pXLabel,                      pset);
    
    old_dw->get(pSize,    lb->pSizeLabel,                   pset);
    old_dw->get(pDefGrad, lb->pDeformationMeasureLabel,     pset);
    new_dw->get(pAvMoment,  pAverageMomentLabel,              pset);

    // Allocate stuff to be written to datawarehouse
    NCVariable<Vector> gRotMoment;
    new_dw->allocateAndPut(gRotMoment, lb->gNormalRotMomentLabel,  
                           dwi, patch);
    gRotMoment.initialize(Vector(0,0,0));

    // Loop thru particles

    for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); 
         iter++){
      particleIndex idx = *iter;
  
      // Get the node indices that surround the cell and the derivatives
      // of the interpolation functions
      interpolator->findCellAndWeightsAndShapeDerivatives(pX[idx],ni,S,d_S,
                                                          pSize[idx],pDefGrad[idx]);

      // Loop thru nodes
      for (int k = 0; k < flag->d_8or27; k++){
        if(patch->containsNode(ni[k])){
          Vector gradS(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],
                       d_S[k].z()*oodx[2]);
          gRotMoment[ni[k]] -= (gradS*pAvMoment[idx]);
        }
      }
    }
    delete interpolator;
  }
}

///////////////////////////////////////////////////////////////////////////
//
// Add computes and requires computation of rotational acceleration
//
void 
ShellMaterial::addComputesRequiresRotAcceleration(Task* task,
                                                  const MPMMaterial* matl,
                                                  const PatchSet* ) 
{
  Ghost::GhostType  gac   = Ghost::AroundCells;
  //  Ghost::GhostType  gnone = Ghost::None;
  const MaterialSubset* matlset = matl->thisMaterial();

  task->requires(Task::OldDW,   lb->pXLabel,                matlset, gac, NGN);

  task->requires(Task::OldDW, lb->pSizeLabel,             matlset, gac, NGN);
  task->requires(Task::OldDW,   lb->pNormalLabel,           matlset, gac, NGN);
  task->requires(Task::NewDW,   pNormalDotAvStressLabel,    matlset, gac, NGN);
  task->requires(Task::NewDW,   pRotMassLabel,              matlset, gac, NGN);
  task->requires(Task::NewDW,   lb->gNormalRotMomentLabel,  matlset, gac, NGN);
  task->computes(pNormalRotAccLabel, matlset);
}

///////////////////////////////////////////////////////////////////////////
//
// Actually compute rotational accleration
//
void 
ShellMaterial::computeRotAcceleration(const PatchSubset* patches,
                                      const MPMMaterial* matl,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw)
{
  // Constants
  Matrix3 One; One.Identity();
  //Ghost::GhostType  gnone = Ghost::None;
  Ghost::GhostType  gac = Ghost::AroundCells;
  int dwi = matl->getDWIndex();

  // Loop thru patches
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());
    vector<Vector> d_S(interpolator->size());


    // Get stuff from datawarehouse
    constParticleVariable<Point>   pX;
    constParticleVariable<double>  pRotMass;
    constParticleVariable<Vector>  pNormal, pNDotAvSig;
    constParticleVariable<Matrix3> pSize;
    constParticleVariable<Matrix3> pDefGrad;
    constNCVariable<Vector>        gRotMoment;
    old_dw->get(pX,          lb->pXLabel,                      pset);

    old_dw->get(pSize,     lb->pSizeLabel,                   pset);
    old_dw->get(pDefGrad,  lb->pDeformationMeasureLabel,     pset);
    old_dw->get(pNormal,     lb->pNormalLabel,                 pset);
    new_dw->get(pRotMass,    pRotMassLabel,                    pset);
    new_dw->get(pNDotAvSig,  pNormalDotAvStressLabel,          pset);
    new_dw->get(gRotMoment,  lb->gNormalRotMomentLabel, dwi, patch, gac, NGN);

    // Create variables for the results
    ParticleVariable<Vector> pRotAcc;
    new_dw->allocateAndPut(pRotAcc, pNormalRotAccLabel, pset);

    // Loop thru particles

    for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); 
         iter++){
      particleIndex idx = *iter;
  
      // Get the node indices that surround the cell and the derivatives
      // of the interpolation functions
      
      interpolator->findCellAndWeightsAndShapeDerivatives(pX[idx],ni,S,d_S,
                                                          pSize[idx],pDefGrad[idx]);
      // Calculate the in-surface identity tensor
      Matrix3 nn(pNormal[idx], pNormal[idx]);
      Matrix3 Is = One - nn;

      // Loop thru nodes
      pRotAcc[idx] = Vector(0.0,0.0,0.0);
      for (int k = 0; k < flag->d_8or27; k++) {
        //if(patch->containsNode(ni[k])){
          pRotAcc[idx] += gRotMoment[ni[k]]*S[k];
        //}
      }
      pRotAcc[idx] -= pNDotAvSig[idx];
      pRotAcc[idx] /= pRotMass[idx];
      pRotAcc[idx] = pRotAcc[idx]*Is; // project to surface
    }
    delete interpolator;
  }
}

///////////////////////////////////////////////////////////////////////////
//
// Add computes and requires update of rotation rate
//
void 
ShellMaterial::addComputesRequiresRotRateUpdate(Task* task,
                                                const MPMMaterial* matl,
                                                const PatchSet* ) 
{
  Ghost::GhostType gnone = Ghost::None;
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW, lb->delTLabel);
  task->requires(Task::OldDW, lb->pMassLabel,              matlset, gnone);
  task->requires(Task::OldDW, lb->pNormalLabel,            matlset, gnone);
  task->requires(Task::OldDW, lb->pInitialNormalLabel,     matlset, gnone);
  task->requires(Task::OldDW, pNormalRotRateLabel,         matlset, gnone);
  task->requires(Task::OldDW, lb->pVolumeLabel,            matlset, gnone);
  task->requires(Task::NewDW, lb->pThickTopLabel_preReloc, matlset, gnone);
  task->requires(Task::NewDW, lb->pThickBotLabel_preReloc, matlset, gnone);
  task->requires(Task::NewDW, pNormalRotAccLabel,          matlset, gnone);

  task->computes(lb->pNormalLabel_preReloc,             matlset);
  task->computes(lb->pInitialNormalLabel_preReloc,      matlset);
  task->computes(pNormalRotRateLabel_preReloc,          matlset);
}

///////////////////////////////////////////////////////////////////////////
//
// Actually update rotation rate
//
void 
ShellMaterial::particleNormalRotRateUpdate(const PatchSubset* patches,
                                           const MPMMaterial* matl,
                                           DataWarehouse* old_dw,
                                           DataWarehouse* new_dw)
{
  // Constants
  Matrix3 One; One.Identity();
  double K   = d_initialData.Shear;
  double mu  = d_initialData.Bulk;
  double E   = 9.0*K*mu/(3.0*K+mu);
  int    dwi = matl->getDWIndex();
  delt_vartype delT;
  old_dw->get(delT, lb->delTLabel, getLevel(patches));

  // Local storage
  constParticleVariable<double>  pMass, pVol, pThickTop, pThickBot;
  constParticleVariable<Vector>  pNormal, pNormal0, pRotRate, pRotAcc;
  ParticleVariable<Vector>       pRotRate_new, pNormal_new, pNormal0_new;

  // Loop thru patches
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    // Get the needed data
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    old_dw->get(pMass,     lb->pMassLabel,              pset);
    old_dw->get(pNormal,   lb->pNormalLabel,            pset);
    old_dw->get(pNormal0,  lb->pInitialNormalLabel,     pset);
    old_dw->get(pRotRate,  pNormalRotRateLabel,         pset);
    new_dw->get(pThickTop, lb->pThickTopLabel_preReloc, pset);
    new_dw->get(pThickBot, lb->pThickBotLabel_preReloc, pset);
    old_dw->get(pVol,      lb->pVolumeLabel,            pset);
    new_dw->get(pRotAcc,   pNormalRotAccLabel,          pset);

    // Allocate the updated particle variables
    new_dw->allocateAndPut(pNormal_new,  lb->pNormalLabel_preReloc,     pset);
    new_dw->allocateAndPut(pNormal0_new, lb->pInitialNormalLabel_preReloc,
                           pset);
    new_dw->allocateAndPut(pRotRate_new, pNormalRotRateLabel_preReloc, pset);

    // Loop over particles
    ParticleSubset::iterator iter = pset->begin();
    for(; iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Calculate the tilde rot rate
      Vector rotRateTilde = pRotAcc[idx]*delT;

      // Calculate the in-surface identity tensor
      Matrix3 nn(pNormal[idx], pNormal[idx]);
      Matrix3 Is = One - nn;

      // The small value of thickness requires the following
      // implicit correction step (** WARNING ** Taken from cfdlib code)
      double hh = (pThickTop[idx]+pThickBot[idx]); 
      double fac = 6.0*E*(pVol[idx]/pMass[idx])*pow(delT/hh, 2);
      Is = One + Is*fac; 
      Vector corrRotRateTilde(0.0,0.0,0.0);
      Is.solveCramer(rotRateTilde, corrRotRateTilde);

      // Update the particle's rotational velocity
      pRotRate_new[idx] = pRotRate[idx] + corrRotRateTilde;

      // Calculate the incremental rotation matrix and store
      Matrix3 Rinc = calcIncrementalRotation(pRotRate_new[idx], pNormal[idx], 
                                             delT);

      // Update the normal 
      pNormal_new[idx] = Rinc*pNormal[idx];
      double len = pNormal_new[idx].length();
      ASSERT(len > 0.0);
      pNormal_new[idx] = pNormal_new[idx]/len;
      pNormal0_new[idx] = pNormal0[idx];

      // Rotate the rotation rate
      pRotRate_new[idx] = Rinc*pRotRate_new[idx];
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
//
// Functions needed by MPMICE
//
// The "CM" versions use the pressure-volume relationship of the CNH model
double 
ShellMaterial::computeRhoMicroCM(double pressure, 
                                 const double p_ref,
                                 const MPMMaterial* matl,
                                 double temperature,
                                 double rho_guess)
{
  double rho_orig = matl->getInitialDensity();
  double bulk = d_initialData.Bulk;

  double p_gauge = pressure - p_ref;
  double rho_cur;

  rho_cur = rho_orig*(p_gauge/bulk + sqrt((p_gauge/bulk)*(p_gauge/bulk) +1));

  return rho_cur;
}

void 
ShellMaterial::computePressEOSCM(double rho_cur,double& pressure, 
                                 double p_ref,
                                 double& dp_drho, double& tmp,
                                 const MPMMaterial* matl,
                                 double temperature)
{
  double bulk = d_initialData.Bulk;
  double rho_orig = matl->getInitialDensity();

  double p_g = .5*bulk*(rho_cur/rho_orig - rho_orig/rho_cur);
  pressure = p_ref + p_g;
  dp_drho  = .5*bulk*(rho_orig/(rho_cur*rho_cur) + 1./rho_orig);
  tmp = bulk/rho_cur;  // speed of sound squared
}

double 
ShellMaterial::getCompressibility()
{
  return 1.0/d_initialData.Bulk;
}

////////////////////////////////////////////////////////////////////////////////
//
// Calculate the incremental rotation matrix for a shell particle
// (WARNING: Can be optimised considerably .. add to TODO list)
// r == rate of rotation of the shell normal
// n == shell normal
// delT == time increment
//
Matrix3 
ShellMaterial::calcIncrementalRotation(const Vector& r, 
                                       const Vector& n,
                                       double delT)
{
  if (debug.active())
    debug << "r = " << r << " n = " << n << " delT = " << delT << endl;

  Matrix3 I; I.Identity();
  // Calculate the rotation angle
  double phi = r.length()*delT;
  if (phi == 0.0) return I;

  // Create vector a = (n x r)/|(n x r)|
  Vector a = Cross(n,r);
  double len = a.length();

  if (debug.active())
    debug << "ShellMaterial::1198: a = " << a << "len = " << len << endl;

  if (len <= 0.0) return I;
  ASSERT(len > 0.0);  
  a /= len;

  // Return the incremental rotation matrix
  return Matrix3(phi, a);
}

////////////////////////////////////////////////////////////////////////////////
//
// Calculate the total rotation matrix for a shell particle
// (WARNING: Can be optimized considerably .. add to TODO list)
// n0 == initial normal
// n == current normal
//
void 
ShellMaterial::calcTotalRotation(const Vector& n0, 
                                 const Vector& n,
                                 Matrix3& R)
{
  // Calculate the rotation angle (assume n0 and n are unit vectors)
  double phi = acos(Dot(n,n0)/(n.length()*n0.length()));
  if (phi == 0.0) return;

  // Find the rotation axis
  Vector a = Cross(n,n0);
  if (a.length() <= 0.0) {Matrix3 I; I.Identity(); R = I; return;}
  ASSERT(a.length() > 0.0);
  a /= (a.length());

  // Return the rotation matrix
  R = Matrix3(phi, a);
}

////////////////////////////////////////////////////////////////////////////////
//
// Calculate the rotation matrix given an angle of rotation and the 
// axis of rotation
// (WARNING: Can be optimised considerably .. add to TODO list)
// Uses the derivative of the Rodriguez vector.
//
Matrix3 
ShellMaterial::calcRotationMatrix(double angle, const Vector& axis)
{
  // Create matrix A = [[0 -a3 a2];[a3 0 -a1];[-a2 a1 0]]
  Matrix3 A(     0.0, -axis[2],  axis[1], 
             axis[2],      0.0, -axis[0], 
            -axis[1],  axis[0],      0.0);
  
  // Calculate the dyad aa
  Matrix3 aa(axis,axis);

  // Initialize the identity matrix
  Matrix3 I; I.Identity();

  // Calculate the rotation matrix
  Matrix3 R = (I - aa)*cos(angle) + aa - A*sin(angle);
  return R;
}

////////////////////////////////////////////////////////////////////////////////
//
// Calculate the in-plane velocity and rotation gradient.
// 
void
ShellMaterial::calcInPlaneGradient(const Vector& n,
                                   Matrix3& velGrad,
                                   Matrix3& rotGrad)
{
  // Initialize the identity matrix
  Matrix3 I; I.Identity();

  // Calculate the dyad nn
  Matrix3 nn(n,n);

  // Calculate the in-plane identity matrix
  Matrix3 Is = I - nn;

  // Calculate the in-plane velocity and rotation gradients
  velGrad = velGrad*Is;
  rotGrad = rotGrad*Is;
}

////////////////////////////////////////////////////////////////////////////////
//
// Calculate the shell elastic stress
//
void
ShellMaterial::computeShellElasticStress(Matrix3& F, Matrix3& sig,
                                         double bulk, double shear)
{
  // Initialize bulk, shear
  Matrix3 One; One.Identity();

  double J = F.Determinant();
  ASSERT(J > 0.0);
  double p = (0.5*bulk)*(J - 1.0/J);
  Matrix3 b = (F*F.Transpose())/pow(J, 2.0/3.0);
  Matrix3 s = (b - One*(b.Trace()/3.0))*(shear/J);
  sig = One*p + s;
}

////////////////////////////////////////////////////////////////////////////////
//
// Calculate the plane stress deformation gradient corresponding
// to sig33 = 0 (Use an iterative Newton method)
// WARNING : Assume that the shear components of bbar_elastic are not affected
// when sig33 is set to zero.  Can be optimized considerably later.
//
bool
ShellMaterial::computePlaneStressAndDefGrad(Matrix3& F, Matrix3& sig, 
                                            double bulk, double shear)
{
  // Initialize 
  Matrix3 One; One.Identity();

  /*  NO PLANE STRESS */
  Matrix3 b(0.0), Js(0.0), tau(0.0), dTdF(0.0);
  double J = 0.0, Jp = 0.0;

  // Calculate Jacobian
  J = F.Determinant();
  if (!(J > 0.0)) {
    cerr << "** ERROR ** F = " << F << " det F = " << J << endl;
    return false;
  }

  // Calcuate Kirchhoff stress
  Jp = (0.5*bulk)*(J*J - 1.0);
  b = (F*F.Transpose())/pow(J, 2.0/3.0);
  Js = (b - One*(b.Trace()/3.0))*shear;
  tau = One*Jp + Js;

  sig = tau/J;
  return true;
  /**/

  /* PLANE STRESS + ALL COMPONENTS
  while (tau(2,2) > 1.0e-10*bulk) {

    // Calculate dtaudF
    dtau_33_dF(F, J, dTdF);

    // Calculate updated F_ij
    for (int ii = 0; ii < 3; ++ii) {
      for (int jj = 0; jj < 3; ++jj) {
        F(ii,jj) = -tau(2,2)/dTdF(ii,jj) + F(ii,jj);
      }
    }

    // Calculate Jacobian
    J = F.Determinant();
    if (!(J > 0.0)) {
      cerr << "** ERROR ** F(new) = " << F << " det F = " << J << endl;
      cerr << " tau = " << tau << endl;
      return false;
    }

    // Calcuate Kirchhoff stress
    Jp = (0.5*bulk)*(J*J - 1.0);
    b = (F*F.Transpose())/pow(J, 2.0/3.0);
    Js = (b - One*(b.Trace()/3.0))*shear;
    tau = One*Jp + Js;
  }

  sig = tau/J;
  return true;
  */

  /*
  // Other variables
  double J = 1.0;
  double p = 0.0;
  Matrix3 b(0.0);
  Matrix3 s(0.0);
  double Jp = 1.0;
  double pp = 0.0;
  Matrix3 Fp(F);
  Matrix3 bp(0.0);
  Matrix3 sp(0.0);
  double sig33p = 0.0; // Cauchy stress
  double Jm = 1.0;
  double pm = 0.0;
  Matrix3 Fm(F);
  Matrix3 bm(0.0);
  Matrix3 sm(0.0);
  double sig33m = 0.0; // Cauchy stress
  double slope = 0;

  // Initial guess for F(2,2), delta
  double delta = 1.0;
  double epsilon = 1.e-14;
  F(2,2) = 1.0/(F(0,0)*F(1,1));
  do {
    // Central value
    J = F.Determinant();
    if (!(J > 0.0)) {
       cerr << "** ERROR ** F = " << F << " det F = " << J << endl;
       return false;
    }
    //ASSERT(J > 0.0);
    p = (0.5*bulk)*(J - 1.0/J);
    b = (F*F.Transpose())/pow(J, 2.0/3.0);
    s = (b - One*(b.Trace()/3.0))*(shear/J);
    sig = One*p + s;

    // Left value
    Fp(2,2) = 1.00001*F(2,2);
    Jp = Fp.Determinant();
    if (!(Jp > 0.0)) {
       cerr << "** ERROR ** Fp = " << Fp << " det Fp = " << Jp << endl;
       return false;
    }
    //ASSERT(Jp > 0.0);
    pp = (0.5*bulk)*(Jp - 1.0/Jp);
    bp = (Fp*Fp.Transpose())/pow(Jp, 2.0/3.0);
    sp = (bp - One*(bp.Trace()/3.0))*(shear/Jp);
    sig33p = pp + sp(2,2);

    // Right value
    Fm(2,2) = 0.99999*F(2,2);
    Jm = Fm.Determinant();
    if (!(Jm > 0.0)) {
       if (d_world->myrank() == 16) {
         cerr << "Current Processor = " << d_world->myrank() << endl;
         cerr << "** ERROR ** F = " << F << " det F = " << J << endl;
         cerr << "** ERROR ** Fm = " << Fm << " det Fm = " << Jm << endl;
       }
       return false;
    }
    //ASSERT(Jm > 0.0);
    pm = (0.5*bulk)*(Jm - 1.0/Jm);
    bm = (Fm*Fm.Transpose())/pow(Jm, 2.0/3.0);
    sm = (bm - One*(bm.Trace()/3.0))*(shear/Jm);
    sig33m = pm + sm(2,2);

    // Calculate slope and increment F(2,2)
    slope = (Fp(2,2)-Fm(2,2))/(sig33p-sig33m);
    delta = -sig(2,2)*slope;
    F(2,2) += delta;

  } while (fabs(delta) > epsilon);

  // Calculate the stress
  J = F.Determinant();
  if (!(J > 0.0)) {
    cerr << "** ERROR ** F(upd) = " << F << " det F = " << J << endl;
    return false;
  }
  p = (0.5*bulk)*(J - 1.0/J);
  b = (F*F.Transpose())/pow(J, 2.0/3.0);
  s = (b - One*(b.Trace()/3.0))*(shear/J);
  sig = One*p + s;
  sig(2,2) = 0.0;
  return true;
  */
}

///////////////////////////////////////////////////////////////////////////
//
//  Calculate the derivative of the Kirchhoff stress component tau_33
//  with respect to the deformation gradient components F_11 
//
void 
ShellMaterial::dtau_33_dF(const Matrix3& F, double J, Matrix3& dTdF)
{
  // Constants
  double onethird = 1.0/3.0; double twothird = 2.0*onethird;
  double fivethird = 5.0*onethird; double fournine = twothird*twothird; 
  double twonine = 0.5*fournine;

  double J23 = pow(J, twothird); double J53 = pow(J, fivethird);

  double K = d_initialData.Bulk; double mu = d_initialData.Shear;
  double KJ = K*J; double muJ23 = twothird*mu/J23; double muJ53 = mu/J53;

  double F11 = F(0,0); double F12 = F(0,1); double F13 = F(0,2);
  double F21 = F(1,0); double F22 = F(1,1); double F23 = F(1,2);
  double F31 = F(2,0); double F32 = F(2,1); double F33 = F(2,2);

  double F11Sq = F11*F11; double F12Sq = F12*F12; double F13Sq = F13*F13; 
  double F21Sq = F21*F21; double F22Sq = F22*F22; double F23Sq = F23*F23; 
  double F31Sq = F31*F31; double F32Sq = F32*F32; double F33Sq = F33*F33; 

  double SS = twonine*(F23Sq+F21Sq+F22Sq+F13Sq+F11Sq+F12Sq)
              -fournine*(F33Sq+F31Sq+F32Sq);
  double SS_1 = SS - twonine*F11Sq;
  double SS_2 = SS - twonine*F22Sq;
  double SS_3 = SS + fournine*F33Sq;

  double F2332 = F23*F32; double F2233 = F22*F33;
  double F3113 = F31*F13; double F1133 = F11*F33;
  double F1122 = F11*F22; double F1221 = F12*F21;
  double F2133 = F21*F33; double F3123 = F31*F23;
  double F3122 = F31*F22; double F2132 = F21*F32;
  double F1233 = F12*F33; double F1332 = F13*F32;
  double F1132 = F11*F32; double F3112 = F31*F12;
  double F1223 = F12*F23; double F1322 = F13*F22;
  double F1123 = F11*F23; double F2113 = F21*F13;

  // dTdF11
  dTdF(0,0) = KJ*(F2233-F2332) - twothird*muJ23*F11 +
              (twonine*F11*(-F12*F3123-F13*F2132+F22*F3113+F33*F1221)
               +(F2233-F2332)*SS_1)*muJ53;
  // dTdF12
  dTdF(0,1) = KJ*(-F2133+F3123) - muJ23*F12 + (F3123-F2133)*SS*muJ53;
  // dTdF13
  dTdF(0,2) = KJ*(F2132-F3122) - muJ23*F13 + (-F3122+F2132)*SS*muJ53;
  // dTdF21
  dTdF(1,0) = KJ*(-F1233+F1332) - muJ23*F21 + (-F1233+F1332)*SS*muJ53;
  // dTdF22
  dTdF(1,1) = KJ*(F1133-F3113) - twothird*muJ23*F22 +
              (twonine*F22*(F33*F1221-F13*F2132-F12*F3123+F11*F2332)
              +(-F3113+F1133)*SS_2)*muJ53;
  // dTdF23
  dTdF(1,2) = KJ*(-F1132+F3112) - muJ23*F23 + (F3112-F1132)*SS*muJ53;
  // dTdF31
  dTdF(2,0) = KJ*(F1223-F1322) + 2.0*muJ23*F31 + (F1223-F1322)*SS*muJ53;
  // dTdF32
  dTdF(2,1) = KJ*(-F1123+F2113)+ 2.0*muJ23*F32 + (-F1123+F2113)*SS*muJ53;
  // dTdF33
  dTdF(2,2) = KJ*(F1122-F1221) + 2.0*twothird*muJ23*F33 +
             (fournine*F33*(-F22*F3113+F13*F2132+F12*F3123-F11*F2332)+
              (F1122-F1221)*SS_3)*muJ53;
}
