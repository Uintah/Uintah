/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include "HyperElasticPlastic.h"
#include <CCA/Components/MPM/Crack/FractureDefine.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModelFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/DamageModelFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMEquationOfStateFactory.h>
#include <Core/Grid/Patch.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Math/MinMax.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Math/Matrix3.h>
#include <Core/Math/Short27.h> //for Fracture
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Malloc/Allocator.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityState.h>
#include <iostream>

using std::cerr;
using namespace Uintah;

HyperElasticPlastic::HyperElasticPlastic(ProblemSpecP& ps, 
                                         MPMLabel* Mlb, 
                                         int n8or27)
{
  lb = Mlb;

  ps->require("bulk_modulus",d_initialData.Bulk);
  ps->require("shear_modulus",d_initialData.Shear);
  d_useMPMICEModifiedEOS = false;
  ps->get("useModifiedEOS",d_useMPMICEModifiedEOS); 
  d_erosionAlgorithm = "none";
  
  d_plasticity = PlasticityModelFactory::create(ps);
  if(!d_plasticity){
    ostringstream desc;
    desc << "An error occured in the PlasticityModelFactory that has \n"
         << " slipped through the existing bullet proofing. Please tell \n"
         << " Biswajit.  "<< endl;
    throw ParameterNotFound(desc.str());
  }

  d_damage = DamageModelFactory::create(ps);
  if(!d_damage){
    ostringstream desc;
    desc << "An error occured in the DamageModelFactory that has \n"
         << " slipped through the existing bullet proofing. Please tell \n"
         << " Biswajit.  "<< endl;
    throw ParameterNotFound(desc.str());
  }
  
  d_eos = MPMEquationOfStateFactory::create(ps);
  if(!d_eos){
    ostringstream desc;
    desc << "An error occured in the EquationOfStateFactory that has \n"
         << " slipped through the existing bullet proofing. Please tell \n"
         << " Biswajit.  "<< endl;
    throw ParameterNotFound(desc.str());
  }

  d_8or27 = n8or27;
  switch(d_8or27) {
  case 8:
    NGN = 1; break;
  case 27:
    NGN = 2; break;
  default:
    NGN = 1; break;
  }

  pBbarElasticLabel = VarLabel::create("p.bbarElastic",
        ParticleVariable<Matrix3>::getTypeDescription());
  pPlasticStrainLabel = VarLabel::create("p.plasticStrain",
        ParticleVariable<double>::getTypeDescription());
  pDamageLabel = VarLabel::create("p.damage",
        ParticleVariable<double>::getTypeDescription());

  pBbarElasticLabel_preReloc = VarLabel::create("p.bbarElastic+",
        ParticleVariable<Matrix3>::getTypeDescription());
  pPlasticStrainLabel_preReloc = VarLabel::create("p.plasticStrain+",
        ParticleVariable<double>::getTypeDescription());
  pDamageLabel_preReloc = VarLabel::create("p.damage+",
        ParticleVariable<double>::getTypeDescription());

}

HyperElasticPlastic::HyperElasticPlastic(const HyperElasticPlastic* cm)
{
  lb = cm->lb;
  d_8or27 = cm->d_8or27;

  d_initialData.Bulk = cm->d_initialData.Bulk;
  d_initialData.Shear = cm->d_initialData.Shear;
  d_useMPMICEModifiedEOS = cm->d_useMPMICEModifiedEOS;
  d_erosionAlgorithm = cm->d_erosionAlgorithm ;
  
  d_plasticity = PlasticityModelFactory::createCopy(cm->d_plasticity);
  d_damage = DamageModelFactory::createCopy(cm->d_damage);
  d_eos = MPMEquationOfStateFactory::createCopy(cm->d_eos);

  pBbarElasticLabel = VarLabel::create("p.bbarElastic",
        ParticleVariable<Matrix3>::getTypeDescription());
  pPlasticStrainLabel = VarLabel::create("p.plasticStrain",
        ParticleVariable<double>::getTypeDescription());
  pDamageLabel = VarLabel::create("p.damage",
        ParticleVariable<double>::getTypeDescription());

  pBbarElasticLabel_preReloc = VarLabel::create("p.bbarElastic+",
        ParticleVariable<Matrix3>::getTypeDescription());
  pPlasticStrainLabel_preReloc = VarLabel::create("p.plasticStrain+",
        ParticleVariable<double>::getTypeDescription());
  pDamageLabel_preReloc = VarLabel::create("p.damage+",
        ParticleVariable<double>::getTypeDescription());

}

HyperElasticPlastic::~HyperElasticPlastic()
{
  // Destructor 
  VarLabel::destroy(pBbarElasticLabel);
  VarLabel::destroy(pPlasticStrainLabel);
  VarLabel::destroy(pDamageLabel);

  VarLabel::destroy(pBbarElasticLabel_preReloc);
  VarLabel::destroy(pPlasticStrainLabel_preReloc);
  VarLabel::destroy(pDamageLabel_preReloc);

  delete d_plasticity;
  delete d_damage;
  delete d_eos;
}

void 
HyperElasticPlastic::addParticleState(std::vector<const VarLabel*>& from,
                                      std::vector<const VarLabel*>& to)
{
  from.push_back(lb->pDeformationMeasureLabel);
  from.push_back(lb->pStressLabel);

  to.push_back(lb->pDeformationMeasureLabel_preReloc);
  to.push_back(lb->pStressLabel_preReloc);

  // Local variables
  from.push_back(pBbarElasticLabel);
  from.push_back(pPlasticStrainLabel);
  from.push_back(pDamageLabel);

  to.push_back(pBbarElasticLabel_preReloc);
  to.push_back(pPlasticStrainLabel_preReloc);
  to.push_back(pDamageLabel_preReloc);

  // Erosion stuff
  if (d_erosionAlgorithm != "none") {
    from.push_back(lb->pErosionLabel);
    to.push_back(lb->pErosionLabel_preReloc);
  }

  // Add the particle state for the plasticity model
  d_plasticity->addParticleState(from, to);
}

void 
HyperElasticPlastic::initializeCMData(const Patch* patch,
                                      const MPMMaterial* matl,
                                      DataWarehouse* new_dw)
{
  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  Matrix3 one, zero(0.); one.Identity();

  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  ParticleVariable<Matrix3> pDeformGrad, pStress;
  new_dw->allocateAndPut(pDeformGrad,    lb->pDeformationMeasureLabel, pset);
  new_dw->allocateAndPut(pStress,        lb->pStressLabel,             pset);

  // Local variables
  ParticleVariable<Matrix3> pBbarElastic;
  ParticleVariable<double>  pPlasticStrain, pDamage;
  new_dw->allocateAndPut(pBbarElastic,   pBbarElasticLabel,            pset);
  new_dw->allocateAndPut(pPlasticStrain, pPlasticStrainLabel,          pset);
  new_dw->allocateAndPut(pDamage,        pDamageLabel,                 pset);

  for(ParticleSubset::iterator iter =pset->begin();iter != pset->end(); iter++){

    // To fix : For a material that is initially stressed we need to
    // modify the left Cauchy-Green and stress tensors to comply with the
    // initial stress state
    pDeformGrad[*iter] = one;
    pStress[*iter] = zero;

    pBbarElastic[*iter] = one;
    pPlasticStrain[*iter] = 0.0;
    pDamage[*iter] = 0.0;
  }

  // Initialize the data for the plasticity model
  d_plasticity->initializeInternalVars(pset, new_dw);

  computeStableTimestep(patch, matl, new_dw);
}

void HyperElasticPlastic::allocateCMDataAddRequires(Task* task,
                                                   const MPMMaterial* matl,
                                                   const PatchSet* patch,
                                                   MPMLabel* lb) const
{
  //const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW, lb->pDeformationMeasureLabel, Ghost::None);
  task->requires(Task::OldDW, lb->pStressLabel,             Ghost::None);

  // Local variables
  task->requires(Task::OldDW, pBbarElasticLabel,            Ghost::None);
  task->requires(Task::OldDW, pPlasticStrainLabel,          Ghost::None);
  task->requires(Task::OldDW, pDamageLabel,                 Ghost::None);
}


void 
HyperElasticPlastic::allocateCMDataAdd(DataWarehouse* new_dw,
                                       ParticleSubset* addset,
                                       map<const VarLabel*, ParticleVariableBase*>* newState,
                                       ParticleSubset* delset,
                                       DataWarehouse* old_dw)
{
  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  Matrix3 zero(0.); 
  ParticleSubset::iterator n,o;

  ParticleVariable<Matrix3> pDeformGrad, pStress;
  new_dw->allocateTemporary(pDeformGrad,    addset);
  new_dw->allocateTemporary(pStress,        addset);

  constParticleVariable<Matrix3> o_DeformGrad,o_Stress;
  old_dw->get(o_DeformGrad,    lb->pDeformationMeasureLabel, delset);
  old_dw->get(o_Stress,        lb->pStressLabel,             delset);

  // Local variables
  ParticleVariable<Matrix3> pBbarElastic;
  ParticleVariable<double>  pPlasticStrain, pDamage;
  new_dw->allocateTemporary(pBbarElastic,   addset);
  new_dw->allocateTemporary(pPlasticStrain, addset);
  new_dw->allocateTemporary(pDamage,        addset);

  constParticleVariable<Matrix3> o_BbarElastic;
  constParticleVariable<double>  o_PlasticStrain, o_Damage;
  old_dw->get(o_BbarElastic,   pBbarElasticLabel,            delset);
  old_dw->get(o_PlasticStrain, pPlasticStrainLabel,          delset);
  old_dw->get(o_Damage,        pDamageLabel,                 delset);
  
  n = addset->begin();
  for (o=delset->begin(); o != delset->end(); o++, n++) {
    pDeformGrad[*n] = o_DeformGrad[*o];
    pStress[*n] = zero;

    pBbarElastic[*n] = o_BbarElastic[*o];
    pPlasticStrain[*n] = o_PlasticStrain[*o];
    pDamage[*n] = o_Damage[*o];
  }

  (*newState)[lb->pDeformationMeasureLabel] = pDeformGrad.clone();
  (*newState)[lb->pStressLabel] = pStress.clone();

  (*newState)[pBbarElasticLabel] = pBbarElastic.clone();
  (*newState)[pPlasticStrainLabel] = pPlasticStrain.clone();
  (*newState)[pDamageLabel] = pDamage.clone();

  // Initialize the data for the plasticity model
  d_plasticity->allocateCMDataAdd(new_dw,addset, newState,delset,old_dw);
}

void 
HyperElasticPlastic::computeStableTimestep(const Patch* patch,
                                           const MPMMaterial* matl,
                                           DataWarehouse* new_dw)
{
  // This is only called for the initial timestep - all other timesteps
  // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int dwi = matl->getDWIndex();

  // Retrieve the array of constitutive parameters
  ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);

  constParticleVariable<double> pMass, pVolume;
  constParticleVariable<Vector> pVelocity;

  new_dw->get(pMass,     lb->pMassLabel,     pset);
  new_dw->get(pVolume,   lb->pVolumeLabel,   pset);
  new_dw->get(pVelocity, lb->pVelocityLabel, pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double shear = d_initialData.Shear;
  double bulk = d_initialData.Bulk;

  ParticleSubset::iterator iter = pset->begin(); 
  for(; iter != pset->end(); iter++){
    particleIndex idx = *iter;

    // Compute wave speed at each particle, store the maximum
    Vector pVel = pVelocity[idx];
    if(pMass[idx] > 0){
      c_dil = sqrt((bulk + 4.*shear/3.)*pVolume[idx]/pMass[idx]);
    }
    else{
      c_dil = 0.0;
      pVel = Vector(0.0,0.0,0.0);
    }
    WaveSpeed=Vector(Max(c_dil+fabs(pVel.x()),WaveSpeed.x()),
                     Max(c_dil+fabs(pVel.y()),WaveSpeed.y()),
                     Max(c_dil+fabs(pVel.z()),WaveSpeed.z()));
  }

  WaveSpeed = dx/WaveSpeed;
  double delT_new = WaveSpeed.minComponent();
  new_dw->put(delt_vartype(delT_new), lb->delTLabel);
}

void 
HyperElasticPlastic::computeStressTensor(const PatchSubset* patches,
                                         const MPMMaterial* matl,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw)
{
  // General stuff
  Matrix3 one, zero(0.0); one.Identity(); 

  Matrix3 tensorL; // Velocity gradient
  Matrix3 tensorD; // Rate of deformation tensor
  Matrix3 tensorEta; // Deviatoric part of rate of deformation tensor
  Matrix3 tensorF_new; // Deformation gradient
  Matrix3 trialBbarElastic; // Trial vol. preserving elastic Cauchy Green tensor
  Matrix3 tensorFinc; // Increment of the deformation gradient tensor
  Matrix3 trialS; // Trial deviatoric stress tensor
  Matrix3 tensorS; // Actual deviatoric stress tensor
  Matrix3 normal; // Normal to yield surface
  Matrix3 tensorFbar; // Volume preserving part of relative deformation tensor

  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double trialSNorm = 0.0;
  double flowStress = 0.0;
  double totalStrainEnergy = 0.0;
  double bulk  = d_initialData.Bulk;
  double shear = d_initialData.Shear;
  double Tm = matl->getMeltTemperature();
  double rho_0 = matl->getInitialDensity();
  double oneThird = (1.0/3.0);
  double sqrtTwoThird = sqrt(2.0/3.0);

  // Loop thru patches
  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);

    // Get grid size
    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    // Get the set of particles
    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    // Get the deformation gradient (F)
    constParticleVariable<Matrix3> pDeformGrad;
    old_dw->get(pDeformGrad, lb->pDeformationMeasureLabel, pset);

    // Get the particle location, particle size, particle mass, particle volume
    constParticleVariable<Point> px;
    constParticleVariable<Vector> psize;
    constParticleVariable<double> pMass, pVolume;
    old_dw->get(px, lb->pXLabel, pset);
    if(d_8or27==27) old_dw->get(psize, lb->pSizeLabel, pset);
    old_dw->get(pMass, lb->pMassLabel, pset);
    old_dw->get(pVolume, lb->pVolumeLabel, pset);

    // Get the velocity from the grid and particle velocity
    constParticleVariable<Vector> pVelocity;
    constNCVariable<Vector> gVelocity;
    old_dw->get(pVelocity, lb->pVelocityLabel, pset);
    Ghost::GhostType  gac = Ghost::AroundCells;
    new_dw->get(gVelocity, lb->gVelocityLabel, dwi, patch, gac, NGN);

    // Get the particle stress and temperature
    constParticleVariable<Matrix3> pStress;
    constParticleVariable<double> pTemperature;
    old_dw->get(pStress, lb->pStressLabel, pset);
    old_dw->get(pTemperature, lb->pTemperatureLabel, pset);

    // Get the time increment (delT)
    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel);

#ifdef FRACTURE
    constParticleVariable<Short27> pgCode;
    new_dw->get(pgCode, lb->pgCodeLabel, pset);
    constNCVariable<Vector> GVelocity;
    new_dw->get(GVelocity,lb->GVelocityLabel, dwi, patch, gac, NGN);
#endif

    // Get the left Cauchy Green tensor (bBar) and the particle damage state
    constParticleVariable<Matrix3> pBbarElastic;
    constParticleVariable<double>  pPlasticStrain, pDamage;
    old_dw->get(pBbarElastic,   pBbarElasticLabel,   pset);
    old_dw->get(pPlasticStrain, pPlasticStrainLabel, pset);
    old_dw->get(pDamage,        pDamageLabel,        pset);

    // Create and allocate arrays for storing the updated information
    ParticleVariable<Matrix3> pDeformGrad_new, pStress_new;
    ParticleVariable<double>  pVolume_new;
    new_dw->allocateAndPut(pDeformGrad_new,  
                           lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->allocateAndPut(pStress_new,      
                           lb->pStressLabel_preReloc,             pset);
    new_dw->allocateAndPut(pVolume_new, 
                           lb->pVolumeDeformedLabel,              pset);

    // Local variables
    ParticleVariable<Matrix3> pBbarElastic_new;
    ParticleVariable<double>  pPlasticStrain_new, pDamage_new;
    new_dw->allocateAndPut(pBbarElastic_new, 
                           pBbarElasticLabel_preReloc,            pset);
    new_dw->allocateAndPut(pPlasticStrain_new,      
                           pPlasticStrainLabel_preReloc,          pset);
    new_dw->allocateAndPut(pDamage_new,      
                           pDamageLabel_preReloc,                 pset);

    // Get the plastic strain
    d_plasticity->getInternalVars(pset, old_dw);
    d_plasticity->allocateAndPutInternalVars(pset, new_dw);

    // Loop thru particles
    ParticleSubset::iterator iter = pset->begin(); 
    for( ; iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Check if the damage is greater than the cut-off value
      // Then reset everything and return
      if (d_damage->hasFailed(pDamage[idx])) {
         pDeformGrad_new[idx] = one;
         pStress_new[idx] = zero;
         pVolume_new[idx]=pMass[idx]/rho_0;

         pBbarElastic_new[idx] = one;
         pPlasticStrain_new[idx] = pPlasticStrain[idx];
         pDamage_new[idx] = pDamage[idx];

         d_plasticity->updateElastic(idx);
         Vector pVel = pVelocity[idx];
         double c_dil = sqrt((bulk + 4.0*shear/3.0)*
                              pVolume_new[idx]/pMass[idx]);
         WaveSpeed=Vector(Max(c_dil+fabs(pVel.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pVel.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pVel.z()),WaveSpeed.z()));
         continue;
      }

      // Calculate the velocity gradient (L) from the grid velocity
#ifdef FRACTURE   
      short pgFld[27];
      for(int k=0; k<27; k++) 
         pgFld[k]=pgCode[idx][k];
      if (d_8or27==27) 
        tensorL = computeVelocityGradient(patch, oodx, px[idx], psize[idx], 
                                          pgFld, gVelocity, GVelocity);
      else 
        tensorL = computeVelocityGradient(patch, oodx, px[idx], 
                                          pgFld, gVelocity, GVelocity);
#else
      if (d_8or27==27)
        tensorL = computeVelocityGradient(patch, oodx, px[idx], psize[idx], 
                                          gVelocity);
      else
        tensorL = computeVelocityGradient(patch, oodx, px[idx], gVelocity);
#endif
      // Calculate rate of deformation tensor (D) and spin tensor (W)
      tensorD = (tensorL + tensorL.Transpose())*0.5;
      tensorEta = tensorD - one*(tensorD.Trace()/3.0);

      // Compute the deformation gradient increment using the time_step
      // velocity gradient
      // F_n^np1 = dudx * dt + Identity
      tensorFinc = tensorL*delT + one;
      double Jinc = tensorFinc.Determinant();

      // Update the deformation gradient tensor to its time n+1 value.
      pDeformGrad_new[idx] = tensorFinc*pDeformGrad[idx];
      double J = pDeformGrad_new[idx].Determinant();
      tensorF_new = pDeformGrad_new[idx];
      //cout << "J = " << J << "\n Updated deformation gradient =\n " 
      //     << pDeformGrad_new[idx] << endl;

      // get the volume preserving part of the deformation gradient increment
      tensorFbar = tensorFinc*pow(Jinc,-oneThird);
      //cout << "fbar = \n" << tensorFbar << endl;

      // predict the elastic part of the volume preserving part of the left
      // Cauchy-Green deformation tensor
      trialBbarElastic = tensorFbar*(pBbarElastic[idx]*tensorFbar.Transpose());
      double traceBbarElastic = oneThird*trialBbarElastic.Trace();
      //cout << "Tr(bbar^el) = " << traceBbarElastic 
      //     << "\n bbar^el_trial = " << trialBbarElastic << endl;

      // Compute the trial deviatoric stress
      // trialS is equal to the shear modulus times dev(bElBar)
      // and calculate the norm of the deviatoric stress 
      // (assuming isotropic yield surface)
      trialS = (trialBbarElastic - one*traceBbarElastic)*shear;
      trialSNorm = trialS.Norm();
      //cout << "Norm(s_trial) = " << trialSNorm 
      //     << "\n s_trial = " << trialS << endl;

      // Calculate the plastic strain rate and plastic strain
      double plasticStrainRate = sqrt(tensorEta.NormSquared()/1.5);
      plasticStrainRate = max(plasticStrainRate, d_tol);
      double plasticStrain = pPlasticStrain[idx] + plasticStrainRate*delT;

      // compute pressure, temperature, density
      double pressure = pStress[idx].Trace()/3.0;
      double temperature = pTemperature[idx];
      double rho_cur = rho_0/J;

      // Set up the PlasticityState
      PlasticityState* state = scinew PlasticityState();
      state->plasticStrainRate = plasticStrainRate;
      state->plasticStrain = plasticStrain;
      state->pressure = pressure;
      state->temperature = temperature;
      state->density = rho_cur;
      state->initialDensity = rho_0;
      state->bulkModulus = bulk ;
      state->initialBulkModulus = bulk;
      state->shearModulus = shear ;
      state->initialShearModulus = shear;
      state->meltingTemp = Tm ;
      state->initialMeltTemp = Tm;
    
      // Calculate the shear modulus and the melting temperature at the
      // start of the time step
      double mu_cur = d_plasticity->computeShearModulus(state);
      double Tm_cur = d_plasticity->computeMeltingTemp(state);

      // Update the plasticity state
      state->shearModulus = mu_cur ;
      state->meltingTemp = Tm_cur ;

      // get the hydrostatic part of the stress .. the pressure should ideally
      // be obtained from a strain energy functional of the form U'(J)
      // which is usually satisfied by equations of states that may or may not
      // satisfy small strain elasticity
      double p = d_eos->computePressure(matl, state, tensorF_new, tensorD, 
                                        delT);
      Matrix3 tensorHy = one*p;

      // Calculate the flow stress
      flowStress = d_plasticity->computeFlowStress(state, delT, d_tol, 
                                                   matl, idx);
      flowStress *= sqrtTwoThird;
      //cout << "Flow Stress = " << flowStress << endl;

      // Check for plastic loading
      if(trialSNorm > flowStress){

        // Plastic case
        // Calculate delGamma
        double Ielastic = oneThird*traceBbarElastic;
        double muBar = mu_cur*Ielastic;
        double delGamma = (trialSNorm - flowStress)/(2.0*muBar);
        //cout << "Ie = " << Ielastic << " mubar = " << muBar 
        //     << " delgamma = " << delGamma << endl;

        // Calculate normal
        normal = trialS/trialSNorm;
        //cout << " Normal = \n" << normal << endl;

        // The actual deviatoric stress
        tensorS = trialS - normal*2.0*muBar*delGamma;

        // Update deviatoric part of elastic left Cauchy-Green tensor
        pBbarElastic_new[idx] = tensorS/mu_cur + one*Ielastic;

        // Update the plastic strain
        pPlasticStrain_new[idx] = plasticStrain;

        // Calculate the updated scalar damage parameter
        pDamage_new[idx] = d_damage->computeScalarDamage(plasticStrainRate, 
                                                         tensorS, 
                                                         pTemperature[idx],
                                                         delT, matl, d_tol, 
                                                         pDamage[idx]);

        // Update internal variables
        d_plasticity->updatePlastic(idx, delGamma);

      } else {

        // Elastic case
        tensorS = trialS;

        // Update deviatoric part of elastic left Cauchy-Green tensor
        pBbarElastic_new[idx] = trialBbarElastic;

        // Update the plastic strain
        pPlasticStrain_new[idx] = pPlasticStrain[idx];

        // Update the scalar damage parameter
        pDamage_new[idx] = pDamage[idx];

        // Update the internal variables
        d_plasticity->updateElastic(idx);
      }

      //cout << "tensorS = \n" << tensorS << endl << "tensorHy = \n" 
      //     << tensorHy << endl;

      // Compute the total Cauchy stress = 
      // (Kirchhoff stress/J) (volumetric + deviatoric)
      pStress_new[idx] = tensorHy + tensorS/J;
      //cout << "Updated stress =\n " << pStress_new[idx] << endl;

      // Update the volume
      pVolume_new[idx]=pMass[idx]/rho_cur;

      // Compute the strain energy for all the particles
      double U = 0.5*bulk*(0.5*(J*J - 1.0) - log(J));
      double W = 0.5*mu_cur*(pBbarElastic_new[idx].Trace() - 3.0);
      double e = (U + W)*pVolume_new[idx]/J;
      totalStrainEnergy += e;

      // Compute wave speed at each particle, store the maximum
      Vector pVel = pVelocity[idx];
      double c_dil = sqrt((bulk + 4.*mu_cur/3.)*pVolume_new[idx]/pMass[idx]);
      WaveSpeed=Vector(Max(c_dil+fabs(pVel.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pVel.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pVel.z()),WaveSpeed.z()));
    }
    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel);
    new_dw->put(sum_vartype(totalStrainEnergy), lb->StrainEnergyLabel);
  }
}

void HyperElasticPlastic::carryForward(const PatchSubset* patches,
                                       const MPMMaterial* matl,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    constParticleVariable<Matrix3> pDeformGrad, pStress;
    constParticleVariable<double>  pmass;
    old_dw->get(pDeformGrad,       lb->pDeformationMeasureLabel, pset);
    old_dw->get(pStress,           lb->pStressLabel,             pset);
    old_dw->get(pmass,             lb->pMassLabel,               pset);

    ParticleVariable<Matrix3>      pDeformGrad_new, pStress_new;
    ParticleVariable<double>       pvolume_deformed;
    new_dw->allocateAndPut(pDeformGrad_new,
                           lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->allocateAndPut(pStress_new,
                           lb->pStressLabel_preReloc,             pset);
    new_dw->allocateAndPut(pvolume_deformed, 
                           lb->pVolumeDeformedLabel,              pset);

    // Local variables
    constParticleVariable<Matrix3> pBbarElastic;
    constParticleVariable<double>  pPlasticStrain, pDamage;
    old_dw->get(pBbarElastic,      pBbarElasticLabel,            pset);
    old_dw->get(pPlasticStrain,    pPlasticStrainLabel,          pset);
    old_dw->get(pDamage,           pDamageLabel,                 pset);

    ParticleVariable<Matrix3>      pBbarElastic_new;
    ParticleVariable<double>       pPlasticStrain_new, pDamage_new;
    new_dw->allocateAndPut(pBbarElastic_new, 
                           pBbarElasticLabel_preReloc,           pset);
    new_dw->allocateAndPut(pPlasticStrain_new,
                           pPlasticStrainLabel_preReloc,         pset);
    new_dw->allocateAndPut(pDamage_new,
                           pDamageLabel_preReloc,                pset);


    // Get the plastic strain
    d_plasticity->getInternalVars(pset, old_dw);
    d_plasticity->initializeInternalVars(pset, new_dw);

    double rho_orig = matl->getInitialDensity();
    for(ParticleSubset::iterator iter = pset->begin();
                                 iter != pset->end(); iter++){
      particleIndex idx = *iter;
      pDeformGrad_new[idx] = pDeformGrad[idx];
      pStress_new[idx] = pStress[idx];
      pvolume_deformed[idx]=(pmass[idx]/rho_orig);

      pBbarElastic_new[idx] = pBbarElastic[idx];
      pPlasticStrain_new[idx] = pPlasticStrain[idx];
      pDamage_new[idx] = pDamage[idx];
    }

    new_dw->put(delt_vartype(1.e10), lb->delTLabel);
    new_dw->put(sum_vartype(0.),     lb->StrainEnergyLabel);
  }
}

void 
HyperElasticPlastic::addInitialComputesAndRequires(Task* task,
                                                   const MPMMaterial* matl,
                                                   const PatchSet* patch) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(pBbarElasticLabel,   matlset);
  task->computes(pPlasticStrainLabel, matlset);
  task->computes(pDamageLabel,        matlset);
 
  // Add internal evolution variables computed by plasticity model
  d_plasticity->addInitialComputesAndRequires(task, matl, patch);
}

void 
HyperElasticPlastic::addComputesAndRequires(Task* task,
                                            const MPMMaterial* matl,
                                            const PatchSet* patch) const
{
  Ghost::GhostType  gac   = Ghost::AroundCells;
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW, lb->delTLabel);
  task->requires(Task::OldDW, lb->pXLabel, matlset,Ghost::None);
  if(d_8or27==27)
    task->requires(Task::OldDW, lb->pSizeLabel, matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pMassLabel,  matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pVolumeLabel,  matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pTemperatureLabel, matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pVelocityLabel, matlset,Ghost::None);
  task->requires(Task::NewDW, lb->gVelocityLabel,  matlset,gac, NGN);

  task->requires(Task::OldDW, lb->pStressLabel, matlset,Ghost::None);
  task->requires(Task::OldDW, lb->pDeformationMeasureLabel,matlset,Ghost::None);
#ifdef FRACTURE
  task->requires(Task::NewDW,  lb->pgCodeLabel,    matlset, Ghost::None);
  task->requires(Task::NewDW,  lb->GVelocityLabel, matlset, gac, NGN);
#endif

  task->computes(lb->pStressLabel_preReloc,             matlset);
  task->computes(lb->pDeformationMeasureLabel_preReloc, matlset);
  task->computes(lb->pVolumeDeformedLabel,              matlset);

  // Variables local to this model
  task->requires(Task::OldDW, pBbarElasticLabel,   matlset,Ghost::None);
  task->requires(Task::OldDW, pPlasticStrainLabel, matlset,Ghost::None);
  task->requires(Task::OldDW, pDamageLabel,        matlset,Ghost::None);

  task->computes(pBbarElasticLabel_preReloc,            matlset);
  task->computes(pPlasticStrainLabel_preReloc,          matlset);
  task->computes(pDamageLabel_preReloc,                 matlset);

  // Add internal evolution variables computed by plasticity model
  d_plasticity->addComputesAndRequires(task, matl, patch);
}

void 
HyperElasticPlastic::addComputesAndRequires(Task* ,
                                            const MPMMaterial* ,
                                            const PatchSet* ,
                                            const bool ) const
{
}


// Needed by MPMICE
double 
HyperElasticPlastic::computeRhoMicroCM(double pressure,
                                       const double p_ref,
                                       const MPMMaterial* matl)
{
  double bulk = d_initialData.Bulk;
  double rho_orig = matl->getInitialDensity();
  double p_gauge = pressure - p_ref;

  double rho_cur;
  if(d_useMPMICEModifiedEOS && p_gauge < 0.0) {
    double A = p_ref;           // MODIFIED EOS
    double n = p_ref/bulk;
    rho_cur = rho_orig*pow(pressure/A,n);
  } else {                      // STANDARD EOS
    rho_cur = rho_orig*(p_gauge/bulk + sqrt((p_gauge/bulk)*(p_gauge/bulk) +1));
  }
  return rho_cur;
}

// Needed by MPMICE
void 
HyperElasticPlastic::computePressEOSCM(double rho_cur,double& pressure,
                                       double p_ref,  
                                       double& dp_drho, double& C0_sq,
                                       const MPMMaterial* matl)
{
  double rho_orig = matl->getInitialDensity();
  double bulk = d_initialData.Bulk;

  if(d_useMPMICEModifiedEOS && rho_cur < rho_orig){
    double A = p_ref;           // MODIFIED EOS
    double n = bulk/p_ref;
    pressure = A*pow(rho_cur/rho_orig,n);
    dp_drho  = (bulk/rho_orig)*pow(rho_cur/rho_orig,n-1);
    C0_sq    = dp_drho;         // speed of sound squared
  } else {                      // STANDARD EOS            
    double p_g = .5*bulk*(rho_cur/rho_orig - rho_orig/rho_cur);
    pressure   = p_ref + p_g;
    dp_drho    = .5*bulk*(rho_orig/(rho_cur*rho_cur) + 1./rho_orig);
    C0_sq      = bulk/rho_cur;  // speed of sound squared
  }
}

// Needed by MPMICE
double 
HyperElasticPlastic::getCompressibility()
{
  return 1.0/d_initialData.Bulk;
}



