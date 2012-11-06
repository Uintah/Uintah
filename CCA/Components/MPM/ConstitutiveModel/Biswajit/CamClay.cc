/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#ifdef __APPLE__
// This is a hack.  gcc 3.3 #undefs isnan in the cmath header, which
// make the isnan function not work.  This define makes the cmath header
// not get included since we do not need it anyway.
#  define _CPP_CMATH
#endif

#include <CCA/Components/MPM/ConstitutiveModel/Biswajit/CamClay.h>
#include <CCA/Components/MPM/ConstitutiveModel/Biswajit/Models/YieldConditionFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/Biswajit/Models/InternalVariableModelFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/Biswajit/Models/PressureModelFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/Biswajit/Models/ShearModulusModelFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/Biswajit/Models/ModelState.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/LinearInterpolator.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/ParticleSubset.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/Gaussian.h>
#include <Core/Math/Matrix3.h>
#include <Core/Math/SymmMatrix3.h>
#include <Core/Math/FastMatrix.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/DebugStream.h>
#include <cmath>
#include <iostream>

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Exceptions/ConvergenceFailure.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/InternalError.h>


using namespace std;
using namespace Uintah;

static DebugStream cout_CC("SSEP",false);
static DebugStream cout_CC1("SSEP1",false);
static DebugStream CSTi("SSEPi",false);
static DebugStream CSTir("SSEPir",false);

CamClay::CamClay(ProblemSpecP& ps, MPMFlags* Mflag)
  : ConstitutiveModel(Mflag)
{
  d_eos = UintahBB::PressureModelFactory::create(ps);
  if(!d_eos){
    ostringstream desc;
    desc << "**ERROR** Internal error while creating CamClay->PressureModelFactory." << endl;
    throw InternalError(desc.str(), __FILE__, __LINE__);
  }

  d_shear = UintahBB::ShearModulusModelFactory::create(ps);
  if (!d_shear) {
    ostringstream desc;
    desc << "**ERROR** Internal error while creating CamClay->ShearModulusModelFactory." << endl;
    throw InternalError(desc.str(), __FILE__, __LINE__);
  }
  
  d_yield = UintahBB::YieldConditionFactory::create(ps);
  if(!d_yield){
    ostringstream desc;
    desc << "**ERROR** Internal error while creating CamClay->YieldConditionFactory." << endl;
    throw InternalError(desc.str(), __FILE__, __LINE__);
  }

  d_intvar = UintahBB::InternalVariableModelFactory::create(ps);
  if(!d_intvar){
    ostringstream desc;
    desc << "**ERROR** Internal error while creating CamClay->InternalVariableModelFactory." << endl;
    throw InternalError(desc.str(), __FILE__, __LINE__);
  }

  initializeLocalMPMLabels();

}

CamClay::CamClay(const CamClay* cm) :
  ConstitutiveModel(cm)
{
  d_eos = UintahBB::PressureModelFactory::createCopy(cm->d_eos);
  d_shear = UintahBB::ShearModulusModelFactory::createCopy(cm->d_shear);
  d_yield = UintahBB::YieldConditionFactory::createCopy(cm->d_yield);
  d_intvar = UintahBB::InternalVariableModelFactory::createCopy(cm->d_intvar);
  
  initializeLocalMPMLabels();
}

CamClay::~CamClay()
{
  // Destructor 
  VarLabel::destroy(pStrainLabel);
  VarLabel::destroy(pElasticStrainLabel);
  VarLabel::destroy(pDeltaGammaLabel);

  VarLabel::destroy(pStrainLabel_preReloc);
  VarLabel::destroy(pElasticStrainLabel_preReloc);
  VarLabel::destroy(pDeltaGammaLabel_preReloc);

  delete d_eos;
  delete d_shear;
  delete d_yield;
  delete d_intvar;
}


void CamClay::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","cam_clay");
  }
  
  d_eos->outputProblemSpec(cm_ps);
  d_shear->outputProblemSpec(cm_ps);
  d_yield->outputProblemSpec(cm_ps);
  d_intvar->outputProblemSpec(cm_ps);
}


CamClay* CamClay::clone()
{
  return scinew CamClay(*this);
}


void
CamClay::initializeLocalMPMLabels()
{
  pStrainLabel = VarLabel::create("p.strain",
    ParticleVariable<Matrix3>::getTypeDescription());
  pElasticStrainLabel = VarLabel::create("p.elasticStrain",
    ParticleVariable<Matrix3>::getTypeDescription());
  pDeltaGammaLabel = VarLabel::create("p.deltaGamma",
    ParticleVariable<double>::getTypeDescription());

  pStrainLabel_preReloc = VarLabel::create("p.strain+",
    ParticleVariable<Matrix3>::getTypeDescription());
  pElasticStrainLabel_preReloc = VarLabel::create("p.elasticStrain+",
    ParticleVariable<Matrix3>::getTypeDescription());
  pDeltaGammaLabel_preReloc = VarLabel::create("p.deltaGamma+",
    ParticleVariable<double>::getTypeDescription());
}

void 
CamClay::addParticleState(std::vector<const VarLabel*>& from,
                          std::vector<const VarLabel*>& to)
{
  // Add the local particle state data for this constitutive model.
  from.push_back(pStrainLabel);
  from.push_back(pElasticStrainLabel);
  from.push_back(pDeltaGammaLabel);

  to.push_back(pStrainLabel_preReloc);
  to.push_back(pElasticStrainLabel_preReloc);
  to.push_back(pDeltaGammaLabel_preReloc);

  // Add the particle state for the internal variable models
  d_intvar->addParticleState(from, to);
}

void 
CamClay::addInitialComputesAndRequires(Task* task,
                                       const MPMMaterial* matl,
                                       const PatchSet* patch) const
{
  const MaterialSubset* matlset = matl->thisMaterial();

  task->computes(pStrainLabel, matlset);
  task->computes(pElasticStrainLabel, matlset);
  task->computes(pDeltaGammaLabel, matlset);
 
  // Add internal evolution variables computed by internal variable model
  d_intvar->addInitialComputesAndRequires(task, matl, patch);
}

void 
CamClay::initializeCMData(const Patch* patch,
                          const MPMMaterial* matl,
                          DataWarehouse* new_dw)
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  initSharedDataForExplicit(patch, matl, new_dw);
  computeStableTimestep(patch, matl, new_dw);

  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  //cout << "Initialize CM Data in CamClay" << endl;
  Matrix3 one, zero(0.); one.Identity();

  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  ParticleVariable<Matrix3>  pStrain, pElasticStrain; 
  ParticleVariable<double>   pDeltaGamma;

  new_dw->allocateAndPut(pStrain, pStrainLabel, pset);
  new_dw->allocateAndPut(pElasticStrain, pElasticStrainLabel, pset);
  new_dw->allocateAndPut(pDeltaGamma, pDeltaGammaLabel, pset);

  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){

    pStrain[*iter] = zero;
    pElasticStrain[*iter] = zero;
    pDeltaGamma[*iter] = 0.0;
  }

  // Initialize the data for the internal variable model
  d_intvar->initializeInternalVariable(pset, new_dw);
}

void 
CamClay::computeStableTimestep(const Patch* patch,
                               const MPMMaterial* matl,
                               DataWarehouse* new_dw)
{
  // This is only called for the initial timestep - all other timesteps
  // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int matlindex = matl->getDWIndex();

  // Retrieve the array of constitutive parameters
  ParticleSubset* pset = new_dw->getParticleSubset(matlindex, patch);

  constParticleVariable<double> pMass, pVol_new;
  constParticleVariable<Vector> pVelocity;

  new_dw->get(pMass,     lb->pMassLabel,     pset);
  new_dw->get(pVol_new,  lb->pVolumeLabel,   pset);
  new_dw->get(pVelocity, lb->pVelocityLabel, pset);

  double c_dil = 0.0;
  Vector waveSpeed(1.e-12,1.e-12,1.e-12);

  double shear = d_shear->computeInitialShearModulus();
  double bulk = d_eos->computeInitialBulkModulus();

  ParticleSubset::iterator iter = pset->begin(); 
  for(; iter != pset->end(); iter++){
    particleIndex idx = *iter;

    // Compute wave speed at each particle, store the maximum
    Vector pvelocity_idx = pVelocity[idx];
    if(pMass[idx] > 0){
      // ** WARNING ** assuming incrementally linear elastic
      //               this is the volumetric wave speed
      c_dil = sqrt((bulk + 4.0*shear/3.0)*pVol_new[idx]/pMass[idx]);
    } else {
      c_dil = 0.0;
      pvelocity_idx = Vector(0.0,0.0,0.0);
    }
    waveSpeed=Vector(Max(c_dil+fabs(pvelocity_idx.x()),waveSpeed.x()),
                     Max(c_dil+fabs(pvelocity_idx.y()),waveSpeed.y()),
                     Max(c_dil+fabs(pvelocity_idx.z()),waveSpeed.z()));
  }

  waveSpeed = dx/waveSpeed;
  double delT_new = waveSpeed.minComponent();
  new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
}

void 
CamClay::addComputesAndRequires(Task* task,
                                const MPMMaterial* matl,
                                const PatchSet* patches) const
{
  // Add the computes and requires that are common to all explicit 
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  Ghost::GhostType  gnone = Ghost::None;
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForHypoExplicit(task, matlset, patches);

  // Other constitutive model and input dependent computes and requires
  task->requires(Task::OldDW, pStrainLabel,      matlset, gnone);
  task->requires(Task::OldDW, pElasticStrainLabel,    matlset, gnone);
  task->requires(Task::OldDW, pDeltaGammaLabel,    matlset, gnone);

  task->computes(pStrainLabel_preReloc,    matlset);
  task->computes(pElasticStrainLabel_preReloc,  matlset);
  task->computes(pDeltaGammaLabel_preReloc,  matlset);

  // Add internal evolution variables computed by internal variable model
  d_intvar->addComputesAndRequires(task, matl, patches);
}

void 
CamClay::computeStressTensor(const PatchSubset* patches,
                             const MPMMaterial* matl,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw)
{
  // Constants
  Matrix3 one; one.Identity(); Matrix3 zero(0.0);
  double sqrtThreeTwo = sqrt(1.5);
  double sqrtTwoThird = 1.0/sqrtThreeTwo;
  Ghost::GhostType  gac = Ghost::AroundCells;
  Vector waveSpeed(1.e-12,1.e-12,1.e-12);
  double rho_0 = matl->getInitialDensity();
  double totalStrainEnergy = 0.0;

  // Strain variables  (declared later)
  // Matrix3 strain(0.0);                  // Total strain
  // double strain_v = 0.0;                // Volumeric strain (eps_v)
  // Matrix3 strain_dev(0.0);              // Deviatoric strain (e)
  // double strain_dev_norm = 0.0;         // ||e||
  // double strain_s = 0.0;                // eps_s = sqrt(2/3) ||e|| 

  // Matrix3 strain_elast_tr(0.0);         // Trial elastic strain
  // double strain_elast_v_tr(0.0);        // Trial volumetric elastic strain
  // Matrix3 strain_elast_devtr(0.0);      // Trial deviatoric elastic strain
  // double strain_elast_devtr_norm = 0.0; // ||ee||
  // double strain_elast_s_tr = 0.0;       // epse_s = sqrt(2/3) ||ee||

  // double strain_elast_v_n = 0.0;        // last volumetric elastic strain
  // Matrix3 strain_elast_dev_n(0.0);      // last devaitoric elastic strain
  // double strain_elast_dev_n_norm = 0.0;
  // double strain_elast_s_n = 0.0;

  // Plasticity related variables
  // Matrix3 nn(0.0);                    // Plastic flow direction n = ee/||ee||

  // Newton iteration constants
  double tolr = 1.0e-4; //1e-4
  double tolf = 1.0e-8;
  int iter_break = 100;

  // Loop thru patches
  for(int patchIndex=0; patchIndex<patches->size(); patchIndex++){
    const Patch* patch = patches->get(patchIndex);

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<Vector>    d_S(interpolator->size());
    vector<double>    S(interpolator->size());
    
    // Get grid size
    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    // Get the set of particles
    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    // GET GLOBAL DATA 

    // Get the deformation gradient (F)
    constParticleVariable<Matrix3>  pDefGrad;
    old_dw->get(pDefGrad, lb->pDeformationMeasureLabel, pset);

    // Get the particle location, particle size, particle mass, particle volume
    constParticleVariable<Point>  px;
    constParticleVariable<Matrix3> psize;
    constParticleVariable<double> pMass, pVol_old;
    old_dw->get(px,       lb->pXLabel,      pset);
    old_dw->get(psize,    lb->pSizeLabel,   pset);
    old_dw->get(pMass,    lb->pMassLabel,   pset);
    old_dw->get(pVol_old, lb->pVolumeLabel, pset);

    // Get the velocity from the grid and particle velocity
    constParticleVariable<Vector> pVelocity;
    constNCVariable<Vector>       gVelocity;
    old_dw->get(pVelocity, lb->pVelocityLabel, pset);
    new_dw->get(gVelocity, lb->gVelocityStarLabel, dwi, patch, gac, NGN);

    // Get the particle stress 
    constParticleVariable<Matrix3> pStress_old;
    old_dw->get(pStress_old, lb->pStressLabel,       pset);

    // Get the time increment (delT)
    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    // GET LOCAL DATA 
    constParticleVariable<Matrix3> pStrain_old, pElasticStrain_old; 
    constParticleVariable<double>  pDeltaGamma_old;
    old_dw->get(pStrain_old,       pStrainLabel,       pset);
    old_dw->get(pElasticStrain_old,     pElasticStrainLabel,     pset);
    old_dw->get(pDeltaGamma_old,        pDeltaGammaLabel,        pset);

    // Create and allocate arrays for storing the updated information
    // GLOBAL
    ParticleVariable<Matrix3> pDefGrad_new, pStress_new;
    ParticleVariable<double>  pVol_new;
    ParticleVariable<double>  pdTdt;

    new_dw->allocateAndPut(pDefGrad_new,  
                           lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->allocateAndPut(pStress_new,      
                           lb->pStressLabel_preReloc,             pset);
    new_dw->allocateAndPut(pVol_new, 
                           lb->pVolumeLabel_preReloc,             pset);
    new_dw->allocateAndPut(pdTdt, lb->pdTdtLabel_preReloc,        pset);

    // LOCAL
    ParticleVariable<Matrix3>  pStrain_new, pElasticStrain_new; 
    ParticleVariable<double> pDeltaGamma_new;
    new_dw->allocateAndPut(pStrain_new,      
                           pStrainLabel_preReloc,            pset);
    new_dw->allocateAndPut(pElasticStrain_new,      
                           pElasticStrainLabel_preReloc,          pset);
    new_dw->allocateAndPut(pDeltaGamma_new,      
                           pDeltaGammaLabel_preReloc,             pset);

    // Get the internal variable and allocate space for the updated internal 
    // variables
    constParticleVariable<double> pPc;
    ParticleVariable<double> pPc_new;
    d_intvar->getInternalVariable(pset, old_dw, pPc);
    d_intvar->allocateAndPutInternalVariable(pset, new_dw, pPc_new);

    // Loop thru particles
    ParticleSubset::iterator iter = pset->begin(); 
    for( ; iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Assign zero internal heating by default - modify if necessary.
      pdTdt[idx] = 0.0;

      //-----------------------------------------------------------------------
      // Stage 1:
      //-----------------------------------------------------------------------
      // Calculate the velocity gradient (L) from the grid velocity
      Matrix3 velGrad(0.0);
      if(!flag->d_axisymmetric){
        // Get the node indices that surround the cell
        interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],pDefGrad[idx]);

        computeVelocityGradient(velGrad,ni,d_S, oodx, gVelocity);
      } else {  // axi-symmetric kinematics
        // Get the node indices that surround the cell
        interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,
                                                                   psize[idx],pDefGrad[idx]);
        // x -> r, y -> z, z -> theta
        computeAxiSymVelocityGradient(velGrad,ni,d_S,S,oodx,gVelocity,px[idx]);
      }

      // Calculate rate of deformation tensor (D)
      Matrix3 rateOfDef_new = (velGrad + velGrad.Transpose())*0.5;

      // Compute the deformation gradient increment using the time_step
      // velocity gradient F_n^np1 = dudx * dt + Identity
      // Update the deformation gradient tensor to its time n+1 value.
      // *TO DO* Compute defGradInc more accurately using previous timestep velGrad
      //         and mid point rule
      Matrix3 defGradInc = velGrad*delT + one;
      Matrix3 defGrad_new = defGradInc*pDefGrad[idx];
      pDefGrad_new[idx] = defGrad_new;
      double J_new = defGrad_new.Determinant();

      // Check 1: Check for negative Jacobian (determinant of deformation gradient)
      if (!(J_new > 0.0)) {
        cerr << getpid() 
             << "**ERROR** Negative Jacobian of deformation gradient" 
             << " in particle " << idx << endl;
        cerr << "l = " << velGrad << endl;
        cerr << "F_old = " << pDefGrad[idx] << endl;
        cerr << "F_inc = " << defGradInc << endl;
        cerr << "F_new = " << defGrad_new << endl;
        cerr << "J_old = " << pDefGrad[idx].Determinant() << endl;
        cerr << "J_new = " << J_new << endl;
        throw ParameterNotFound("**ERROR**:InvalidValue: J < 0.0", __FILE__, __LINE__);
      }

      // Calculate the current mass density and deformed volume
      double rho_cur = rho_0/J_new;
      pVol_new[idx]=pMass[idx]/rho_cur;

      // Compute polar decompositions of F_old and F_new (F = RU)
      // pDefGrad[idx].polarDecompositionRMB(rightStretch_old, rotation_old);
      Matrix3 rightStretch_new; rightStretch_new.Identity();
      Matrix3 rotation_new; rotation_new.Identity();
      defGrad_new.polarDecompositionRMB(rightStretch_new, rotation_new);

      // Unrotate the spatial rate of deformation tensor and elastic strain
      rateOfDef_new = (rotation_new.Transpose())*(rateOfDef_new*rotation_new);
      Matrix3 elasticStrain_old = (rotation_new.Transpose())*(pElasticStrain_old[idx]*rotation_new);

      // Calc volumetric and deviatoric elastic strains at beginninging of timestep (t_n)
      double strain_elast_v_n = elasticStrain_old.Trace();
      Matrix3 strain_elast_dev_n = elasticStrain_old - one*(strain_elast_v_n/3.0);
      double strain_elast_dev_n_norm = strain_elast_dev_n.Norm();
      double strain_elast_s_n = sqrtTwoThird*strain_elast_dev_n_norm;
      
      cout << "idx = " << idx 
           << " t_n: eps_v_e = " << strain_elast_v_n << " eps_s_e = " << strain_elast_s_n << endl;

      // Compute strain increment from rotationally corrected rate of deformation
      // (Forward Euler)
      Matrix3 strainInc = rateOfDef_new*delT; 

      // Calculate the total strain  
      //   Volumetric strain &  Deviatoric strain
      Matrix3 strain = pStrain_old[idx] + strainInc;
      double strain_v = strain.Trace();
      Matrix3 strain_dev = strain - one*(strain_v/3.0);

      // Trial elastic strain
      //   Volumetric elastic strain &  Deviatoric elastic strain
      Matrix3 strain_elast_tr = elasticStrain_old + strainInc;
      double strain_elast_v_tr = strain_elast_tr.Trace();
      Matrix3 strain_elast_devtr = strain_elast_tr - one*(strain_elast_v_tr/3.0);
      double strain_elast_devtr_norm = strain_elast_devtr.Norm();
      double strain_elast_s_tr = sqrtTwoThird*strain_elast_devtr_norm;

      // Set up the ModelState (for t_n)
      UintahBB::ModelState* state = scinew UintahBB::ModelState();
      state->density             = rho_cur;
      state->initialDensity      = rho_0;
      state->volume              = pVol_new[idx];
      state->initialVolume       = pMass[idx]/rho_0;
      state->elasticStrain = strain_elast_tr;
      state->epse_v = strain_elast_v_tr;
      state->epse_s = strain_elast_s_tr;
      state->elasticStrainTrial = strain_elast_tr;
      state->epse_v_tr = strain_elast_v_tr;
      state->epse_s_tr = strain_elast_s_tr;
      state->p_c = pPc[idx];

      // Compute mu and q
      double mu = d_shear->computeShearModulus(state);
      state->shearModulus = mu;
      double q = d_shear->computeQ(state);
      state->q = q;

      // Compute p and bulk modulus
      double p = d_eos->computePressure(matl, state, zero, zero, 0.0);
      double bulk = d_eos->computeBulkModulus(rho_0, rho_cur);
      state->p = p;
      
      // compute the local sound wave speed
      double c_dil = sqrt((bulk + 4.0*mu/3.0)/rho_cur);

      // Get internal state variable (p_c)
      double pc_n = d_intvar->computeInternalVariable(state);
      state->p_c = pc_n;
      pPc_new[idx] = pc_n;
        
      //-----------------------------------------------------------------------
      // Stage 2: Elastic-plastic stress update
      //-----------------------------------------------------------------------

      // Compute plastic flow direction (n = ee/||ee||)
      // Magic to deal with small strains
      double small = 1.0e-12;
      double oo_strain_elast_s_tr =  (strain_elast_s_tr > small) ? 1.0/strain_elast_s_tr : 1.0;
      Matrix3 nn = strain_elast_devtr*(sqrtTwoThird*oo_strain_elast_s_tr);
      
      // Calculate yield function
      double ftrial = d_yield->evalYieldCondition(state);

      small = 1.0e-8; // **WARNING** Should not be hard coded (use d_tol)
     
      double strain_elast_v = strain_elast_v_n;
      double strain_elast_s = strain_elast_s_n;
      double pc = pc_n;
      if (ftrial > small) { // Plastic loading

        double fyield = ftrial;
        double strain_elast_v_k = strain_elast_v;
        double strain_elast_s_k = strain_elast_s;
        double delgamma = pDeltaGamma_old[idx];

        // Derivatives
        double dfdp = d_yield->computeVolStressDerivOfYieldFunction(state);
        double dfdq = d_yield->computeDevStressDerivOfYieldFunction(state);

        // Residual
        double rv = strain_elast_v - strain_elast_v_tr + delgamma*dfdp;
        double rs = strain_elast_s - strain_elast_s_tr + delgamma*dfdq;
        double rf = fyield;

        // Set up Newton iterations
        double rtolv = 1.0;
        double rtols = 1.0;
        double rtolf = 1.0;
        int klocal = 0;
 
        // Do Newton iterations
        state->elasticStrain = elasticStrain_old;
        state->elasticStrainTrial = strain_elast_tr;
        while ((rtolv > tolr) || (rtols > tolr) || (rtolf > tolf)) 
        {
          klocal++;
         
          // Compute needed derivatives
          double dpdepsev = d_eos->computeDpDepse_v(state);
          double dpdepses = d_eos->computeDpDepse_s(state);
          double dqdepsev = dpdepses;
          double dqdepses = d_shear->computeDqDepse_s(state);
          double dpcdepsev = d_intvar->computeVolStrainDerivOfInternalVariable(state);

          // Compute derivatives of residuals
          double dr1_dx1 = 1.0 + delgamma*(2.0*dpdepsev - dpcdepsev);
          double dr1_dx2 = 2.0*delgamma*dpdepses;
          double dr1_dx3 = dfdp;

          // dfdq = 2q/M^2 => 2/M^2 = 1/q dfdq
          double dr2_dx1 = (delgamma*dqdepsev*dfdq)/(state->q);
          double dr2_dx2 = 1.0 + (delgamma*dqdepses*dfdq)/(state->q);
          double dr2_dx3 = dfdq;

          double dr3_dx1 = dfdq*dqdepsev + dfdp*dpdepsev - state->p*dpcdepsev;
          double dr3_dx2 = dfdq*dqdepses + dfdp*dpdepses;

          FastMatrix A_MAT(2, 2), inv_A_MAT(2,2);
          A_MAT(0, 0) = dr1_dx1;
          A_MAT(0, 1) = dr1_dx2;
          A_MAT(1, 0) = dr2_dx1;
          A_MAT(1, 1) = dr2_dx2;

          inv_A_MAT.destructiveInvert(A_MAT);

          vector<double> B_MAT(2), C_MAT(2), AinvB(2), rvs_vec(2), Ainvrvs(2);
          B_MAT[0] = dr1_dx3; 
          B_MAT[1] = dr2_dx3;

          C_MAT[0] = dr3_dx1;
          C_MAT[1] = dr3_dx2;

          rvs_vec[0] = rv;
          rvs_vec[1] = rs;

          inv_A_MAT.multiply(B_MAT, AinvB);

          inv_A_MAT.multiply(rvs_vec, Ainvrvs);

          double denom = C_MAT[0]*AinvB[0] + C_MAT[1]*AinvB[1];
          double deldelgamma = 0.0;
          if (fabs(denom) > 1e-20) {
            deldelgamma = (-C_MAT[0]*Ainvrvs[0]-C_MAT[1]*Ainvrvs[1] + rf)/denom;
          } else {
            deldelgamma = 0.0;
          }
          vector<double> delvoldev(2);
          delvoldev[0] = -Ainvrvs[0] - AinvB[0]*deldelgamma;
          delvoldev[1] = -Ainvrvs[1] - AinvB[1]*deldelgamma;

          //std::cout << "deldelgamma = " << deldelgamma 
          //          << " delvoldev = " << delvoldev[0] << " , " << delvoldev[1] << endl;

          // update
          strain_elast_v = strain_elast_v_k + delvoldev[0];
          strain_elast_s = strain_elast_s_k + delvoldev[1];
          strain_elast_v_k = strain_elast_v;
          if (strain_elast_s < 0.0) {
             strain_elast_s = strain_elast_s_k;
          }
          strain_elast_s_k = strain_elast_s;
          double delgamma_old = delgamma;
          delgamma += deldelgamma;
          if (delgamma < 0.0) delgamma = delgamma_old;
        
          state->epse_v = strain_elast_v;
          state->epse_s = strain_elast_s;

          mu = d_shear->computeShearModulus(state);
          q = d_shear->computeQ(state);
          p = d_eos->computePressure(matl, state, zero, zero, 0.0);
          pc = d_intvar->computeInternalVariable(state);

          state->shearModulus = mu;
          state->q = q;
          state->p = p;
          state->p_c = pc;
          pPc_new[idx] = pc;

          dfdp = d_yield->computeVolStressDerivOfYieldFunction(state);
          dfdq = d_yield->computeDevStressDerivOfYieldFunction(state);

          fyield = d_yield->evalYieldCondition(state);

          // update residual
          rv = strain_elast_v - strain_elast_v_tr + delgamma*dfdp;
          rs = strain_elast_s - strain_elast_s_tr + delgamma*dfdq;
          rf = fyield;

          // calculate tolerances
          rtolv=fabs(delvoldev[0]);
          rtols=fabs(delvoldev[1]);
          rtolf=fabs(deldelgamma);


          // Check max iters
          if (klocal == iter_break) {
            ostringstream desc;
            desc << "**ERROR** Newton iterations did not converge" 
                 << " rtolv = " << rtolv << " rtols = " << rtols << " rtolf = " << rtolf 
                 << " klocal = " << klocal << endl;
            throw ConvergenceFailure(desc.str(), iter_break, rtolf, tolf, __FILE__, __LINE__);
          }

        } // End of Newton-Raphson while

        if ((delgamma < 0.0) && (fabs(delgamma) > 1.0e-10)) {
          ostringstream desc;
          desc << "**ERROR** delgamma less than 0.0 in local converged solution." << endl;
          throw ConvergenceFailure(desc.str(), klocal, rtolf, delgamma, __FILE__, __LINE__);
        }

        // update stress
        pStress_new[idx] = one*p + nn*(sqrtTwoThird*q);

        // update elastic strain
        pElasticStrain_new[idx] = nn*(sqrtThreeTwo*strain_elast_s) + one*(strain_elast_v/3.0);

        // update delta gamma (plastic strain)
        pDeltaGamma_new[idx] = pDeltaGamma_old[idx] + delgamma;

        std::cout << "Plastic: t_{n+1}:  eps_v_e = " << strain_elast_v 
                  << " eps_s_e = " << strain_elast_s << " f_n+1 = " << fyield << endl;
        std::cout << "          pqpc = [" << p << " " << q << " " << pc <<"]" << endl;

      } else { // Elastic range

        // update stress from trial elastic strain
        pStress_new[idx] = one*p + nn*(sqrtTwoThird*q);

        // update elastic strain from trial value
        pElasticStrain_new[idx] = strain_elast_tr;
        strain_elast_v = strain_elast_v_tr;
        strain_elast_s = strain_elast_s_tr;

        std::cout << "Elastic: t_{n+1}:  eps_v_e = " << strain_elast_v << " eps_s_e = " << strain_elast_s 
                  << endl;
        std::cout << "  pqpc = [" << p << " " << q << " " << pc <<"]" << endl;

        // update delta gamma (plastic strain increament)
        pDeltaGamma_new[idx] = pDeltaGamma_old[idx];
      }

      //-----------------------------------------------------------------------
      // Stage 4:
      //-----------------------------------------------------------------------
      // Rotate back to spatial configuration
      pStress_new[idx] = rotation_new*(pStress_new[idx]*rotation_new.Transpose());
      pElasticStrain_new[idx] = rotation_new*(pElasticStrain_new[idx]*rotation_new.Transpose());

      // Compute the strain energy 
      double W_vol = d_eos->computeStrainEnergy(state);
      double W_dev = d_shear->computeStrainEnergy(state);
      totalStrainEnergy = (W_vol + W_dev)*pVol_new[idx];

      // Compute wave speed at each particle, store the maximum
      Vector pVel = pVelocity[idx];
      waveSpeed=Vector(Max(c_dil+fabs(pVel.x()),waveSpeed.x()),
                       Max(c_dil+fabs(pVel.y()),waveSpeed.y()),
                       Max(c_dil+fabs(pVel.z()),waveSpeed.z()));

      delete state;
    }  // end loop over particles

    waveSpeed = dx/waveSpeed;
    double delT_new = waveSpeed.minComponent();

    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
    
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(totalStrainEnergy), lb->StrainEnergyLabel);
    }
    delete interpolator;
  }

  if (cout_CC.active()) 
    cout_CC << getpid() << "... End." << endl;

}

void 
CamClay::carryForward(const PatchSubset* patches,
                      const MPMMaterial* matl,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    // Carry forward the data common to all constitutive models 
    // when using RigidMPM.
    // This method is defined in the ConstitutiveModel base class.
    carryForwardSharedData(pset, old_dw, new_dw, matl);

    // Carry forward the data local to this constitutive model 
    constParticleVariable<Matrix3>  pStrain, pElasticStrain;
    constParticleVariable<double>   pDeltaGamma; 

    old_dw->get(pStrain,         pStrainLabel,         pset);
    old_dw->get(pElasticStrain,  pElasticStrainLabel,  pset);
    old_dw->get(pDeltaGamma,     pDeltaGammaLabel,     pset);

    ParticleVariable<Matrix3>       pStrain_new, pElasticStrain_new; 
    ParticleVariable<double>        pDeltaGamma_new;

    new_dw->allocateAndPut(pStrain_new,      
                           pStrainLabel_preReloc,            pset);
    new_dw->allocateAndPut(pElasticStrain_new,      
                           pElasticStrainLabel_preReloc,     pset);
    new_dw->allocateAndPut(pDeltaGamma_new,      
                           pDeltaGammaLabel_preReloc,        pset);

    // Get and copy the internal variables
    constParticleVariable<double> pPc;
    d_intvar->getInternalVariable(pset, old_dw, pPc);
    d_intvar->allocateAndPutRigid(pset, new_dw, pPc);

    for(ParticleSubset::iterator iter = pset->begin();
        iter != pset->end(); iter++){
      particleIndex idx = *iter;
      pStrain_new[idx] = pStrain[idx];
      pElasticStrain_new[idx] = pElasticStrain[idx];
      pDeltaGamma_new[idx] = pDeltaGamma[idx];
    }

    new_dw->put(delt_vartype(1.e10), lb->delTLabel, patch->getLevel());
    
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(0.),     lb->StrainEnergyLabel);
    }
  }
}

void 
CamClay::allocateCMDataAddRequires(Task* task,
                                   const MPMMaterial* matl,
                                   const PatchSet* patch,
                                   MPMLabel* lb) const
{
  const MaterialSubset* matlset = matl->thisMaterial();

  // Allocate the variables shared by all constitutive models
  // for the particle convert operation
  // This method is defined in the ConstitutiveModel base class.
  addSharedRForConvertExplicit(task, matlset, patch);

  // Add requires local to this model
  Ghost::GhostType  gnone = Ghost::None;
  task->requires(Task::NewDW, pStrainLabel_preReloc,         matlset, gnone);
  task->requires(Task::NewDW, pElasticStrainLabel_preReloc,  matlset, gnone);
  task->requires(Task::NewDW, pDeltaGammaLabel_preReloc,     matlset, gnone);
  d_intvar->allocateCMDataAddRequires(task,matl,patch,lb);
}

void 
CamClay::allocateCMDataAdd(DataWarehouse* new_dw,
                           ParticleSubset* addset,
                           map<const VarLabel*, 
                           ParticleVariableBase*>* newState,
                           ParticleSubset* delset,
                           DataWarehouse* old_dw)
{
  // Copy the data common to all constitutive models from the particle to be 
  // deleted to the particle to be added. 
  // This method is defined in the ConstitutiveModel base class.
  copyDelToAddSetForConvertExplicit(new_dw, delset, addset, newState);
  
  // Copy the data local to this constitutive model from the particles to 
  // be deleted to the particles to be added
  ParticleSubset::iterator n,o;

  ParticleVariable<Matrix3>  pStrain, pElasticStrain; 
  ParticleVariable<double>   pDeltaGamma;

  constParticleVariable<Matrix3>  o_Strain, o_ElasticStrain; 
  constParticleVariable<double>   o_DeltaGamma;

  new_dw->allocateTemporary(pStrain,addset);
  new_dw->allocateTemporary(pElasticStrain,addset);
  new_dw->allocateTemporary(pDeltaGamma,addset);

  new_dw->get(o_Strain,pStrainLabel_preReloc,delset);
  new_dw->get(o_ElasticStrain,pElasticStrainLabel_preReloc,delset);
  new_dw->get(o_DeltaGamma,pDeltaGammaLabel_preReloc,delset);

  n = addset->begin();
  for (o=delset->begin(); o != delset->end(); o++, n++) {
    pStrain[*n] = o_Strain[*o];
    pElasticStrain[*n] = o_ElasticStrain[*o];
    pDeltaGamma[*n] = o_DeltaGamma[*o];
  }

  (*newState)[pStrainLabel]=pStrain.clone();
  (*newState)[pElasticStrainLabel]=pElasticStrain.clone();
  (*newState)[pDeltaGammaLabel]=pDeltaGamma.clone();
  
  // Initialize the data for the internal variable model
  d_intvar->allocateCMDataAdd(new_dw,addset, newState, delset, old_dw);
}


double CamClay::computeRhoMicroCM(double pressure,
                                  const double p_ref,
                                  const MPMMaterial* matl,
                                  double temperature,
                                  double rho_guess)
{
  double rho_orig = matl->getInitialDensity();
  pressure -= p_ref;
  double rho_cur = d_eos->computeDensity(rho_orig, pressure);

  if (std::isnan(rho_cur)) {
    ostringstream desc;
    desc << "rho_cur = " << rho_cur << " pressure = " << pressure
         << " p_ref = " << p_ref << " rho_orig = " << rho_orig << endl;
    throw InvalidValue(desc.str(), __FILE__, __LINE__);
  }

  return rho_cur;
}

void CamClay::computePressEOSCM(double rho_cur,double& pressure,
                                double p_ref,  
                                double& dp_drho, double& csquared,
                                const MPMMaterial* matl,
                                double temperature)
{
  double rho_orig = matl->getInitialDensity();
  d_eos->computePressure(rho_orig, rho_cur, pressure, dp_drho, csquared);
  pressure += p_ref;

  if (std::isnan(pressure)) {
    ostringstream desc;
    desc << "rho_cur = " << rho_cur << " pressure = " << pressure
         << " p_ref = " << p_ref << " dp_drho = " << dp_drho << endl;
    throw InvalidValue(desc.str(), __FILE__, __LINE__);
  }
}

double CamClay::getCompressibility()
{
  return 1.0/d_eos->initialBulkModulus();
}

void
CamClay::scheduleCheckNeedAddMPMMaterial(Task* task,
                                         const MPMMaterial* ,
                                         const PatchSet* ) const
{
  task->computes(lb->NeedAddMPMMaterialLabel);
}

void CamClay::checkNeedAddMPMMaterial(const PatchSubset* patches,
                                      const MPMMaterial* matl,
                                      DataWarehouse* ,
                                      DataWarehouse* new_dw)
{
  if (cout_CC.active()) {
    cout_CC << getpid() << "checkNeedAddMPMMaterial: In : Matl = " << matl
            << " id = " << matl->getDWIndex() <<  " patch = "
            << (patches->get(0))->getID();
  }

  double need_add=0.;
  new_dw->put(sum_vartype(need_add),     lb->NeedAddMPMMaterialLabel);
}
