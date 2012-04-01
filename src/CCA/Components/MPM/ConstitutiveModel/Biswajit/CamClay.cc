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
  addSharedCRForExplicit(task, matlset, patches);

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

  // Deformation variables
  Matrix3 velGrad(0.0);
  Matrix3 rateOfDef_new(0.0);
  Matrix3 defGrad_new; defGrad_new.Identity(); 
  Matrix3 defGradInc; defGradInc.Identity();
  Matrix3 strainInc(0.0);

  // Strain variables
  Matrix3 strain(0.0);                  // Total strain
  double strain_v = 0.0;                // Volumeric strain (eps_v)
  Matrix3 strain_dev(0.0);              // Deviatoric strain (e)
  double strain_dev_norm = 0.0;         // ||e||
  double strain_s = 0.0;                // eps_s = sqrt(2/3) ||e|| 

  Matrix3 strain_elast_tr(0.0);         // Trial elastic strain
  double strain_elast_v_tr(0.0);        // Trial volumetric elastic strain
  Matrix3 strain_elast_devtr(0.0);      // Trial deviatoric elastic strain
  double strain_elast_devtr_norm = 0.0; // ||ee||
  double strain_elast_s_tr = 0.0;       // epse_s = sqrt(2/3) ||ee||

  double strain_elast_v_n = 0.0;        // last volumetric elastic strain
  Matrix3 strain_elast_dev_n(0.0);      // last devaitoric elastic strain
  double strain_elast_dev_n_norm = 0.0;
  double strain_elast_s_n = 0.0;

  double strain_elast_v = 0.0;        
  double strain_elast_s = 0.0;        

  // Plasticity related variables
  Matrix3 flow(0.0);                    // Plastic flow direction n = ee/||ee||

  // Newton iteration constants
  double tolr = 1.0e-8; //1e-4
  double tola = 1.0e-8; //1e-8
  double tola_f = 1.0e-1; //1e-8
  int iter_break = 20;

  // Newton iteration variables
  double strain_elast_v_k = 0.0;
  double strain_elast_s_k = 0.0;
  double delgamma = 0.0;
  double pc = 0.0;
  double fyield = 0.0;
  double rv = 0.0, rs = 0.0, rf = 0.0;  // residuals
  double rv0 = 0.0, normrv0 = 0.0, normrv = 0.0;
  double rs0 = 0.0, normrs0 = 0.0, normrs = 0.0;
  double rf0 = 0.0, normrf0 = 0.0, normrf = 0.0;

  // Newton iteration computed variables
  double dfdp = 0.0, dfdq = 0.0;
  double dpdepsev = 0.0, dpdepses = 0.0, dqdepsev = 0.0, dqdepses = 0.0, dpcdepsev = 0.0;
  double d2fdpdepsev = 0.0, d2fdpdepses = 0.0, drvdepsev = 0.0, drvdepses = 0.0, drvdgamma = 0.0;
  double d2fdqdepsev = 0.0, d2fdqdepses = 0.0, drsdepsev = 0.0, drsdepses = 0.0, drsdgamma = 0.0;
  double drfdepsev = 0.0, drfdepses = 0.0;

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
    constParticleVariable<Vector> psize;
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

    new_dw->allocateAndPut(pDefGrad_new,  
                           lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->allocateAndPut(pStress_new,      
                           lb->pStressLabel_preReloc,             pset);
    new_dw->allocateAndPut(pVol_new, 
                           lb->pVolumeLabel_preReloc,             pset);

    // LOCAL
    ParticleVariable<Matrix3>  pStrain_new, pElasticStrain_new; 
    ParticleVariable<double> pDeltaGamma_new;
    new_dw->allocateAndPut(pStrain_new,      
                           pStrainLabel_preReloc,            pset);
    new_dw->allocateAndPut(pElasticStrain_new,      
                           pElasticStrainLabel_preReloc,          pset);
    new_dw->allocateAndPut(pDeltaGamma_new,      
                           pDeltaGammaLabel_preReloc,             pset);

    // Get the nternal variable and allocate space for the updated internal 
    // variables
    d_intvar->getInternalVariable(pset, old_dw);
    d_intvar->allocateAndPutInternalVariable(pset, new_dw);

    // Loop thru particles
    ParticleSubset::iterator iter = pset->begin(); 
    for( ; iter != pset->end(); iter++){
      particleIndex idx = *iter;

      //-----------------------------------------------------------------------
      // Stage 1:
      //-----------------------------------------------------------------------
      // Calculate the velocity gradient (L) from the grid velocity
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

      // Compute the deformation gradient increment using the time_step
      // velocity gradient F_n^np1 = dudx * dt + Identity
      // Update the deformation gradient tensor to its time n+1 value.
      // *TO DO* Compute defGradInc more accurately using previous timestep velGrad
      //         and mid point rule
      defGradInc = velGrad*delT + one;
      defGrad_new = defGradInc*pDefGrad[idx];
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

      // Calculate the current density and deformed volume
      double rho_cur = rho_0/J_new;
      pVol_new[idx]=pMass[idx]/rho_cur;

      // Compute polar decomposition of F (F = RU)
      //pDefGrad[idx].polarDecompositionRMB(rightStretch, rotation);

      // Calculate rate of deformation tensor (D)
      rateOfDef_new = (velGrad + velGrad.Transpose())*0.5;
      strainInc = rateOfDef_new*delT; // **WARNING** not rotationally corrected

      // Calculate the total strain  (**WARNING** not rotationally corrected)
      //   Volumetric strain &  Deviatoric strain
      strain = pStrain_old[idx] + strainInc;
      strain_v = strain.Trace();
      strain_dev = strain - one*(strain_v/3.0);
      strain_dev_norm = strain_dev.Norm();
      strain_s = sqrtTwoThird*strain_dev_norm;

      // Trial elastic strain
      //   Volumetric elastic strain &  Deviatoric elastic strain
      strain_elast_tr = pElasticStrain_old[idx] + strainInc;
      strain_elast_v_tr = strain_elast_tr.Trace();
      strain_elast_devtr = strain_elast_tr - one*(strain_elast_v_tr/3.0);
      strain_elast_devtr_norm = strain_elast_devtr.Norm();
      strain_elast_s_tr = sqrtTwoThird*strain_elast_devtr_norm;

      // Previous volumetric and deviatoric elastic strains
      strain_elast_v_n = pElasticStrain_old[idx].Trace();
      strain_elast_dev_n = pElasticStrain_old[idx] - one*(strain_elast_v_n/3.0);
      strain_elast_dev_n_norm = strain_elast_dev_n.Norm();
      strain_elast_s_n = sqrtTwoThird*strain_elast_dev_n_norm;
      
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
      double pc_n = d_intvar->computeInternalVariable(state, delT, matl, idx);
      state->p_c = pc_n;
        
      //-----------------------------------------------------------------------
      // Stage 2: Elastic-plastic stress update
      //-----------------------------------------------------------------------

      // Compute plastic flow direction (n = ee/||ee||)
      // Magic to deal with small strains
      double small = 1.0e-12;
      double oo_strain_elast_s_tr =  (strain_elast_s_tr > small) ? 1.0/strain_elast_s_tr : 1.0;
      flow = strain_elast_devtr*(sqrtTwoThird*oo_strain_elast_s_tr);
      
      // Calculate yield function
      double ftrial = d_yield->evalYieldCondition(state);

      small = 1.0e-8; // **WARNING** Should not be hard coded (use d_tol)
      if (ftrial > small) { // Plastic loading

        fyield = ftrial;
        strain_elast_v = strain_elast_v_n;
        strain_elast_s = strain_elast_s_n;
        strain_elast_v_k = strain_elast_v;
        strain_elast_s_k = strain_elast_s;
        delgamma = pDeltaGamma_old[idx];
        pc = pc_n;

        // Derivatives
        dfdp = d_yield->computeVolStressDerivOfYieldFunction(state);
        dfdq = d_yield->computeDevStressDerivOfYieldFunction(state);

        // Residual
        rv = strain_elast_v - strain_elast_v_tr + delgamma*dfdp;
        rs = strain_elast_s - strain_elast_s_tr + delgamma*dfdq;
        rf = fyield;

        // Set up Newton iterations
        rv0 = rv;
        normrv0 = (fabs(rv0) < small) ? fabs(strain_elast_v_tr) : fabs(rv0);
        normrv = normrv0;
        rs0 = rs;
        normrs0 = (fabs(rs0) < small) ? fabs(strain_elast_s_tr) : fabs(rs0);
        normrs = normrs0;
        rf0 = rf;
        normrf0 = (fabs(rf0) < small) ? 1.0 : fabs(rf0);
        normrf = normrf0;
        
        double rtolv = 1.0;
        double rtols = 1.0;
        double rtolf = 1.0;
        int klocal = 0;
 
        // Do Newton iterations
        state->elasticStrain = pElasticStrain_old[idx];
        state->elasticStrainTrial = strain_elast_tr;
        while (( (rtolv > tolr) || (rtols > tolr) || (rtolf > tolr) ) 
              && ( (normrv > tola) || (normrs > tola) || (normrf > tola_f) ))
        {
          klocal++;
         
          // calc deldelgamma
          dpdepsev = d_eos->computeDpDepse_v(state);
          dpdepses = d_eos->computeDpDepse_s(state);
          dqdepsev = dpdepses;
          dqdepses = d_shear->computeDqDepse_s(state);
          dpcdepsev = d_intvar->computeVolStrainDerivOfInternalVariable(state);
            
          d2fdpdepsev = d_yield->computeVolStrainDerivOfDfDp(state, d_eos, d_shear, d_intvar);
          d2fdpdepses = d_yield->computeDevStrainDerivOfDfDp(state, d_eos, d_shear, d_intvar);
          drvdepsev = 1.0 + delgamma*d2fdpdepsev;
          drvdepses = delgamma*d2fdpdepses;
          drvdgamma = dfdp;

          d2fdqdepsev = d_yield->computeVolStrainDerivOfDfDq(state, d_eos, d_shear, d_intvar);
          d2fdqdepses = d_yield->computeDevStrainDerivOfDfDq(state, d_eos, d_shear, d_intvar);
          drsdepsev = delgamma*d2fdqdepsev;
          drsdepses = 1.0 + delgamma*d2fdqdepses;
          drsdgamma = dfdq;

          drfdepsev = d_yield->computeVolStrainDerivOfYieldFunction(state, d_eos, d_shear, d_intvar);
          drfdepses = d_yield->computeDevStrainDerivOfYieldFunction(state, d_eos, d_shear, d_intvar);
          // drfdgamma = 0.0;

          FastMatrix A_MAT(2, 2), inv_A_MAT(2,2);
          A_MAT(0, 0) = drvdepsev;
          A_MAT(0, 1) = drvdepses;
          A_MAT(1, 0) = drsdepsev;
          A_MAT(1, 1) = drsdepses;
          inv_A_MAT.destructiveInvert(A_MAT);
          vector<double> B_MAT(2), C_MAT(2), AinvB(2), rvs_vec(2), Ainvrvs(2);
          B_MAT[0] = drvdgamma; 
          B_MAT[1] = drsdgamma;
          C_MAT[0] = drfdepsev;
          C_MAT[1] =  drfdepses;
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

          // update
          strain_elast_v = strain_elast_v_k + delvoldev[0];
          strain_elast_s = strain_elast_s_k + delvoldev[1];
          strain_elast_v_k = strain_elast_v;
          if (strain_elast_s < 0.0) {
            strain_elast_s = strain_elast_s_k;
          }
          strain_elast_s_k = strain_elast_s;
          delgamma = delgamma + deldelgamma;
        
          state->epse_v = strain_elast_v;
          state->epse_s = strain_elast_s;

          mu = d_shear->computeShearModulus(state);
          q = d_shear->computeQ(state);
          p = d_eos->computePressure(matl, state, zero, zero, 0.0);
          pc = d_intvar->computeInternalVariable(state, delT, matl, idx);
          state->shearModulus = mu;
          state->q = q;
          state->p = p;
          state->p_c = pc;

          dfdp = d_yield->computeVolStressDerivOfYieldFunction(state);
          dfdq = d_yield->computeDevStressDerivOfYieldFunction(state);

          fyield = d_yield->evalYieldCondition(state);

          // update residual
          rv = strain_elast_v - strain_elast_v_tr + delgamma*dfdp;
          rs = strain_elast_s - strain_elast_s_tr + delgamma*dfdq;
          rf = fyield;
        
          // calculate tolerances
          normrv=fabs(rv);
          normrs=fabs(rs);
          normrf=fabs(rf);
          rtolv=fabs(rv)/normrv0;
          rtols=fabs(rs)/normrs0;
          rtolf=fabs(rf)/normrf0;

          // Check max iters
          if (klocal == iter_break) {
            ostringstream desc;
            desc << "**ERROR** Newton iterations did not converge" 
                 << " normrv = " << normrv << " normrs = " << normrs 
                 << " normrf = " << normrf << " rtolv = " << rtolv 
                 << " rtols = " << rtols << " rtolf = " << rtolf 
                 << " klocal = " << klocal << endl;
            throw ConvergenceFailure(desc.str(), iter_break, normrf, rtolf, __FILE__, __LINE__);
          }

        } // End of Newton-Raphson while

        double delgammaneg = 0.0;
        if ((delgamma < 0.0) && (fabs(delgamma) > 1.0e-10)) {
          delgammaneg = delgamma;
          ostringstream desc;
          desc << "**ERROR** delgamma less that 0.0 in local converged solution." << endl;
          throw ConvergenceFailure(desc.str(), klocal, delgamma, normrf, __FILE__, __LINE__);
        }

        // update stress
        pStress_new[idx] = one*p + flow*(sqrtTwoThird*q);

        // update elastic strain
        pElasticStrain_new[idx] = flow*(sqrtThreeTwo*strain_elast_s) + one*(strain_elast_v/3.0);

        // update delta gamma (plastic strain)
        pDeltaGamma_new[idx] = pDeltaGamma_old[idx] + delgamma;

      } else { // Elastic range

        // update stress from trial elastic strain
        pStress_new[idx] = one*p + flow*(sqrtTwoThird*q);

        // update elastic strain from trial value
        pElasticStrain_new[idx] = strain_elast_tr;
        strain_elast_v = strain_elast_v_tr;
        strain_elast_s = strain_elast_s_tr;

        // update delta gamma (plastic strain increament)
        pDeltaGamma_new[idx] = pDeltaGamma_old[idx];
      }

      //-----------------------------------------------------------------------
      // Stage 4:
      //-----------------------------------------------------------------------
      // Rotate the stress/backStress back to the laboratory coordinates
      // Update the stress/back stress

      // Use new rotation
      // defGrad_new.polarDecompositionRMB(rightStretch, rotation);

      // sigma_new = (rotation*sigma_new)*(rotation.Transpose());
      // pStress_new[idx] = sigma_new;
        
      // Rotate the deformation rate back to the laboratory coordinates
      // rateOfDef_new = (rotation*rateOfDef_new)*(rotation.Transpose());

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

    // Get the internal variables
    d_intvar->getInternalVariable(pset, old_dw);
    d_intvar->allocateAndPutRigid(pset, new_dw);

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
