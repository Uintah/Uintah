
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModelFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ShellMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/Short27.h> // for Fracture
#include <Packages/Uintah/Core/Grid/NodeIterator.h> // just added
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Math/MinMax.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/NotFinished.h>
#include <sgi_stl_warnings_off.h>
#include <fstream>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using std::cerr;
using namespace Uintah;
using namespace SCIRun;

#define FRACTURE
#undef FRACTURE

////////////////////////////////////////////////////////////////////////////////
//
// Constructor
//
ShellMaterial::ShellMaterial(ProblemSpecP& ps,  MPMLabel* Mlb, int n8or27)
{
  // Read Material Constants
  ps->require("bulk_modulus", d_initialData.Bulk);
  ps->require("shear_modulus",d_initialData.Shear);

  // Initialize labels
  lb = Mlb;

  // Set up support size for interpolation
  d_8or27 = n8or27;
  NGN = 1;
  if (d_8or27 == 27) NGN = 2;

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

////////////////////////////////////////////////////////////////////////////////
//
// Make sure all labels are correctly relocated
//
void 
ShellMaterial::addParticleState(std::vector<const VarLabel*>& from,
				std::vector<const VarLabel*>& to)
{
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

  from.push_back(lb->pDeformationMeasureLabel);
  from.push_back(lb->pStressLabel);

  to.push_back(lb->pDeformationMeasureLabel_preReloc);
  to.push_back(lb->pStressLabel_preReloc);
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

  ParticleVariable<Matrix3> pDefGrad, pStress;
  new_dw->allocateAndPut(pDefGrad, lb->pDeformationMeasureLabel, pset);
  new_dw->allocateAndPut(pStress,  lb->pStressLabel,             pset);

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

    pDefGrad[*iter] = One;
    pStress[*iter] = Zero;
  }

  computeStableTimestep(patch, matl, new_dw);
}

void 
ShellMaterial::allocateCMData(DataWarehouse* new_dw,
			      ParticleSubset* subset,
			      map<const VarLabel*, ParticleVariableBase*>* newState)
{
  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  Matrix3 One, Zero(0.0); One.Identity();


  ParticleVariable<Vector>  pRotRate; 
  ParticleVariable<Matrix3> pDefGradTop, pDefGradCen, pDefGradBot, 
                            pStressTop, pStressCen, pStressBot;
  new_dw->allocateTemporary(pRotRate, subset);
  new_dw->allocateTemporary(pDefGradTop, subset);
  new_dw->allocateTemporary(pDefGradCen, subset);
  new_dw->allocateTemporary(pDefGradBot, subset);
  new_dw->allocateTemporary(pStressTop, subset);
  new_dw->allocateTemporary(pStressCen, subset);
  new_dw->allocateTemporary(pStressBot, subset);

  ParticleVariable<Matrix3> pDefGrad, pStress;
  new_dw->allocateTemporary(pDefGrad, subset);
  new_dw->allocateTemporary(pStress,  subset);

  ParticleSubset::iterator iter = subset->begin();
  for(; iter != subset->end(); iter++) {
    particleIndex pidx = *iter;
    pRotRate[pidx]    = Vector(0.0,0.0,0.0);
    pDefGradTop[pidx] = One;
    pDefGradCen[pidx] = One;
    pDefGradBot[pidx] = One;
    pStressTop[pidx]  = Zero;
    pStressCen[pidx]  = Zero;
    pStressBot[pidx]  = Zero;

    pDefGrad[*iter] = One;
    pStress[*iter] = Zero;
  }

   (*newState)[pNormalRotRateLabel]=pRotRate.clone();
   (*newState)[pDefGradTopLabel]=pDefGradTop.clone();
   (*newState)[pDefGradCenLabel]=pDefGradCen.clone();
   (*newState)[pDefGradBotLabel]=pDefGradBot.clone();
   (*newState)[pStressTopLabel]=pStressTop.clone();
   (*newState)[pStressCenLabel]=pStressCen.clone();
   (*newState)[pStressBotLabel]=pStressBot.clone();
   (*newState)[lb->pDeformationMeasureLabel]=pDefGrad.clone();
   (*newState)[lb->pStressLabel]=pStress.clone();
 
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
  new_dw->put(delt_vartype(delT_new), lb->delTLabel);
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
  if (d_8or27==27) 
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
  constParticleVariable<Vector> pRotRate, pSize;
  constNCVariable<double> gMass;

  // Create arrays for the grid data
  NCVariable<Vector> gRotRate;

  // Loop thru patches
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch, gan, NGN, 
						     lb->pXLabel);

    // Get the required data
    old_dw->get(pMass,          lb->pMassLabel,          pset);
    old_dw->get(pX,             lb->pXLabel,             pset);
    if(d_8or27==27)
      old_dw->get(pSize,        lb->pSizeLabel,          pset);
    old_dw->get(pRotRate,       pNormalRotRateLabel,     pset);
    new_dw->get(gMass,          lb->gMassLabel, dwi,     patch, gan, NGN);

    // Allocate arrays for the grid data
    new_dw->allocateAndPut(gRotRate, lb->gNormalRotRateLabel, dwi,patch);
    gRotRate.initialize(Vector(0,0,0));

    // Interpolate particle data to Grid data.  Attempt to conserve 
    // angular momentum (I_grid*omega_grid =  S*I_particle*omega_particle).
    IntVector ni[MAX_BASIS];
    double S[MAX_BASIS];
    Vector pMom(0.0,0.0,0.0);
    for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); 
	 iter++){
      particleIndex idx = *iter;

      // Get the node indices that surround the cell
      if(d_8or27==27) patch->findCellAndWeights27(pX[idx], ni, S, pSize[idx]);
      else            patch->findCellAndWeights(pX[idx], ni, S);

      // Calculate momentum
      pMom = pRotRate[idx]*pMass[idx];

      // Add each particles contribution to the grid rotation rate
      for(int k = 0; k < d_8or27; k++) {
	if(patch->containsNode(ni[k])) gRotRate[ni[k]] += pMom * S[k];
      }
    } // End of particle loop

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
      gRotRate[*iter] /= gMass[*iter];
    }
  }  // End loop over patches
}

////////////////////////////////////////////////////////////////////////////////
//
// Create task graph for each time step after initialization
//
void 
ShellMaterial::addComputesAndRequires(Task* task,
				      const MPMMaterial* matl,
				      const PatchSet*) const
{
  Ghost::GhostType  gnone = Ghost::None;
  Ghost::GhostType  gac   = Ghost::AroundCells;
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW, lb->pXLabel,                  matlset, gnone);
  task->requires(Task::OldDW, lb->pMassLabel,               matlset, gnone);
  if (d_8or27 == 27)
    task->requires(Task::OldDW, lb->pSizeLabel,             matlset, gnone);
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
  task->requires(Task::OldDW, lb->pStressLabel,             matlset, gnone);
  task->requires(Task::OldDW, lb->pDeformationMeasureLabel, matlset, gnone);
  task->requires(Task::NewDW, lb->gVelocityLabel,           matlset, gac, NGN);
  task->requires(Task::NewDW, lb->gNormalRotRateLabel,      matlset, gac, NGN);
  task->requires(Task::OldDW, lb->delTLabel);

  task->computes(lb->pStressLabel_preReloc,             matlset);
  task->computes(lb->pDeformationMeasureLabel_preReloc, matlset);
  task->computes(lb->pVolumeDeformedLabel,              matlset);

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

    // Read the datawarehouse
    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    // Read from datawarehouses 
    constParticleVariable<double>  pMass, pThickTop, pThickBot, pThickTop0,
                                   pThickBot0; 
    constParticleVariable<Point>   pX; 
    constParticleVariable<Vector>  pSize, pVelocity, pRotRate, pNormal; 
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
    if (d_8or27 == 27)
      old_dw->get(pSize,     lb->pSizeLabel,               pset);
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
    old_dw->get(delT,        lb->delTLabel);
    new_dw->get(gVelocity,   lb->gVelocityLabel,      dwi, patch, gac, NGN);
    new_dw->get(gRotRate,    lb->gNormalRotRateLabel, dwi, patch, gac, NGN);

    // Allocate for updated variables in new_dw 
    ParticleVariable<double>  pVolume_new, pThickTop_new, pThickBot_new,
                              pThickTop0_new, pThickBot0_new; 
    ParticleVariable<Matrix3> pDefGradTop_new, pDefGradBot_new, pDefGradCen_new,
                              pStressTop_new, pStressCen_new, pStressBot_new, 
                              pStress_new, pDefGrad_new;
    new_dw->allocateAndPut(pVolume_new,    lb->pVolumeDeformedLabel,      pset);
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
    new_dw->allocateAndPut(pAvMoment,  pAverageMomentLabel,     pset);
    new_dw->allocateAndPut(pNDotAvSig, pNormalDotAvStressLabel, pset);
    new_dw->allocateAndPut(pRotMass,   pRotMassLabel,           pset);

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

      // Find the surrounding nodes, interpolation functions and derivatives
      IntVector ni[MAX_BASIS];
      Vector d_S[MAX_BASIS];
      if (d_8or27 == 27)
        patch->findCellAndShapeDerivatives27(pX[idx], ni, d_S, pSize[idx]);
      else
        patch->findCellAndShapeDerivatives(pX[idx], ni, d_S);

      // Calculate the spatial gradient of the velocity and the 
      // normal rotation rate
      Matrix3 velGrad(0.0), rotGrad(0.0);
      for(int k = 0; k < d_8or27; k++) {
	Vector gvel = gVelocity[ni[k]];
	Vector grot = gRotRate[ni[k]];
	for (int j = 0; j<3; j++){
	  double d_SXoodx = d_S[k][j] * oodx[j];
	  for (int i = 0; i<3; i++) {
            velGrad(i+1,j+1) += gvel[i] * d_SXoodx;
            rotGrad(i+1,j+1) += grot[i] * d_SXoodx;
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
      //computeShellElasticStress(defGradTop_new, sigTop);
      //computeShellElasticStress(defGradCen_new, sigCen);
      //computeShellElasticStress(defGradBot_new, sigBot);

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
      if (!computePlaneStressAndDefGrad(defGradTop_new, sigTop)) {
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
      if (!computePlaneStressAndDefGrad(defGradCen_new, sigCen)) {
        cerr << "Normal = " << pNormal[idx] << endl;
        cerr << "R = " << R << endl;
        cerr << "defGradCen = " << defGradCen_new << endl;
        cerr << "SigCen = " << sigCen << endl;
        exit(1);
      }
      if (!computePlaneStressAndDefGrad(defGradBot_new, sigBot)) {
        cerr << "Normal = " << pNormal[idx] << endl;
        cerr << "R = " << R << endl;
        cerr << "defGradBot = " << defGradBot_new << endl;
        cerr << "SigBot = " << sigBot << endl;
        exit(1);
      }

      // Calculate the change in thickness in the direction of
      // the normal
      double zTopInc = 0.5*(defGradTop_new(3,3)+defGradCen_new(3,3));
      double zBotInc = 0.5*(defGradBot_new(3,3)+defGradCen_new(3,3));
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
    }

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel);
    new_dw->put(sum_vartype(strainEnergy), lb->StrainEnergyLabel);
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
  if(d_8or27==27) 
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

    // Get stuff from datawarehouse
    constParticleVariable<Point>   pX;
    constParticleVariable<Vector>  pSize;
    constParticleVariable<Matrix3> pAvMoment;
    old_dw->get(pX,         lb->pXLabel,                      pset);
    if(d_8or27==27)
      old_dw->get(pSize,    lb->pSizeLabel,                   pset);
    new_dw->get(pAvMoment,  pAverageMomentLabel,              pset);

    // Allocate stuff to be written to datawarehouse
    NCVariable<Vector> gRotMoment;
    new_dw->allocateAndPut(gRotMoment, lb->gNormalRotMomentLabel,  
			   dwi, patch);
    gRotMoment.initialize(Vector(0,0,0));

    // Loop thru particles
    IntVector ni[MAX_BASIS]; 
    double S[MAX_BASIS]; Vector d_S[MAX_BASIS];
    for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); 
	 iter++){
      particleIndex idx = *iter;
  
      // Get the node indices that surround the cell and the derivatives
      // of the interpolation functions
      if(d_8or27==27)
	patch->findCellAndWeightsAndShapeDerivatives27(pX[idx], ni, S, d_S,
						       pSize[idx]);
      else
	patch->findCellAndWeightsAndShapeDerivatives(pX[idx], ni, S, d_S);

      // Loop thru nodes
      for (int k = 0; k < d_8or27; k++){
	if(patch->containsNode(ni[k])){
	  Vector gradS(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],
		       d_S[k].z()*oodx[2]);
	  gRotMoment[ni[k]] -= (gradS*pAvMoment[idx]);
	}
      }
    }
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
  if(d_8or27==27) 
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
  Ghost::GhostType  gnone = Ghost::None;
  int dwi = matl->getDWIndex();

  // Loop thru patches
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    // Get stuff from datawarehouse
    constParticleVariable<Point>   pX;
    constParticleVariable<double>  pRotMass;
    constParticleVariable<Vector>  pSize, pNormal, pNDotAvSig;
    constNCVariable<Vector>        gRotMoment;
    old_dw->get(pX,          lb->pXLabel,                      pset);
    if(d_8or27==27)
      old_dw->get(pSize,     lb->pSizeLabel,                   pset);
    old_dw->get(pNormal,     lb->pNormalLabel,                 pset);
    new_dw->get(pRotMass,    pRotMassLabel,                    pset);
    new_dw->get(pNDotAvSig,  pNormalDotAvStressLabel,          pset);
    new_dw->get(gRotMoment,  lb->gNormalRotMomentLabel, dwi, patch, gnone, 0);

    // Create variables for the results
    ParticleVariable<Vector> pRotAcc;
    new_dw->allocateAndPut(pRotAcc, pNormalRotAccLabel, pset);

    // Loop thru particles
    IntVector ni[MAX_BASIS]; 
    double S[MAX_BASIS]; Vector d_S[MAX_BASIS];
    for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); 
	 iter++){
      particleIndex idx = *iter;
  
      // Get the node indices that surround the cell and the derivatives
      // of the interpolation functions
      if(d_8or27==27)
	patch->findCellAndWeightsAndShapeDerivatives27(pX[idx], ni, S, d_S,
						       pSize[idx]);
      else
	patch->findCellAndWeightsAndShapeDerivatives(pX[idx], ni, S, d_S);

      // Calculate the in-surface identity tensor
      Matrix3 nn(pNormal[idx], pNormal[idx]);
      Matrix3 Is = One - nn;

      // Loop thru nodes
      pRotAcc[idx] = Vector(0.0,0.0,0.0);
      for (int k = 0; k < d_8or27; k++) pRotAcc[idx] += gRotMoment[ni[k]]*S[k];
      pRotAcc[idx] -= pNDotAvSig[idx];
      pRotAcc[idx] /= pRotMass[idx];
      pRotAcc[idx] = pRotAcc[idx]*Is; // project to surface
    }
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
  task->requires(Task::NewDW, lb->pVolumeDeformedLabel,    matlset, gnone);
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
  old_dw->get(delT, lb->delTLabel);

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
    new_dw->get(pVol,      lb->pVolumeDeformedLabel,    pset);
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
				 const MPMMaterial* matl)
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
				 const MPMMaterial* matl)
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
  // Calculate the rotation angle
  double phi = r.length()*delT;
  if (phi == 0.0) return Matrix3(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0);

  // Create vector a = (n x r)/|(n x r)|
  Vector a = Cross(n,r);
  ASSERT(a.length() > 0.0);  
  a /= (a.length());

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
ShellMaterial::computeShellElasticStress(Matrix3& F, Matrix3& sig)
{
  // Initialize bulk, shear
  double bulk = d_initialData.Bulk;
  double shear = d_initialData.Shear;
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
ShellMaterial::computePlaneStressAndDefGrad(Matrix3& F, Matrix3& sig)
{
  // Initialize bulk, shear
  double bulk = d_initialData.Bulk;
  double shear = d_initialData.Shear;
  Matrix3 One; One.Identity();

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

  // Initial guess for F(3,3), delta
  double delta = 1.0;
  double epsilon = 1.e-14;
  F(3,3) = 1.0/(F(1,1)*F(2,2));
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
    Fp(3,3) = 1.01*F(3,3);
    Jp = Fp.Determinant();
    if (!(Jp > 0.0)) {
       cerr << "** ERROR ** Fp = " << Fp << " det Fp = " << Jp << endl;
       return false;
    }
    //ASSERT(Jp > 0.0);
    pp = (0.5*bulk)*(Jp - 1.0/Jp);
    bp = (Fp*Fp.Transpose())/pow(Jp, 2.0/3.0);
    sp = (bp - One*(bp.Trace()/3.0))*(shear/Jp);
    sig33p = pp + sp(3,3);

    // Right value
    Fm(3,3) = 0.99*F(3,3);
    Jm = Fm.Determinant();
    if (!(Jm > 0.0)) {
       cerr << "** ERROR ** Fm = " << Fm << " det Fm = " << Jm << endl;
       return false;
    }
    //ASSERT(Jm > 0.0);
    pm = (0.5*bulk)*(Jm - 1.0/Jm);
    bm = (Fm*Fm.Transpose())/pow(Jm, 2.0/3.0);
    sm = (bm - One*(bm.Trace()/3.0))*(shear/Jm);
    sig33m = pm + sm(3,3);

    // Calculate slope and increment F(3,3)
    slope = (Fp(3,3)-Fm(3,3))/(sig33p-sig33m);
    delta = -sig(3,3)*slope;
    F(3,3) += delta;

  } while (fabs(delta) > epsilon);
  sig(3,3) = 0.0;
  return true;
}
