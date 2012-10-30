/* MIT LICENSE

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

/* ARENISCA INTRO

This source code is for a simplified constitutive model, named ``Arenisca'',
which has some of the basic features needed for modeling geomaterials.
To better explain the source code, the comments in this file frequently refer
to the equations in the following three references:
1. The Arenisca manual,
2. R.M.	Brannon and S. Leelavanichkul, "A multi-stage return algorithm for
   solving the classical damage component of constitutive models for rocks,
   ceramics, and other rock-like media", International Journal of Fracture,
   163, pp.133-149, 2010, and
3. R.M.	Brannon, "Elements of Phenomenological Plasticity: Geometrical Insight,
   Computational Algorithms, and Topics in Shock Physics", Shock Wave Science
   and Technology Reference Library: Solids I, Springer 2: pp. 189-274, 2007.

As shown in "fig:AreniscaYieldSurface" of the Arenisca manual, Arenisca is
a two-surface plasticity model combining a linear Drucker-Prager
pressure-dependent strength (to model influence of friction at microscale
sliding surfaces) and a cap yield function (to model influence of microscale
porosity).

*/ 

/* NOTES
  1207 code control passed from Dr. Ali to James Colovos

  This is VERSION 0.1 120826.1339
*/

//----------------suggested max line width (72char)-------------------->

//----------JC DEFINE SECTION----------
//#define JC_ARENISCA_VERSION 0.1  //120826.1339
#define JC_ARENISCA_VERSION 0.2  //120826.0827
//#define JC_USE_BB_STATE_UPDATE
#define JC_USE_BB_DEFGRAD_UPDATE 2
//#define JC_DEBUG_SMALL_TIMESTEP
//#define JC_DEBUG_PARTICLE 249108234247
//#define JC_DEBUG_DEBUG_PARTICLE 42501762304
//#define JC_EPV
#define JC_FREEZE_PARTICLE
//#define JC_MAX_NESTED_RETURN
//#define JC_DEBUG_FR_OUTSIDE_CAP
//#define JC_LIMITER_PRINT

// INCLUDE SECTION: tells the preprocessor to include the necessary files
#include <CCA/Components/MPM/ConstitutiveModel/Arenisca.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#ifdef JC_USE_BB_STATE_UPDATE
#include <CCA/Components/MPM/ConstitutiveModel/Biswajit/Models/InternalVariableModelFactory.h>
#include <CCA/Components/MPM/ConstitutiveModel/Biswajit/Models/ModelState.h>
#endif
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/Matrix3.h>
#include <Core/Malloc/Allocator.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <sci_values.h>
#include <iostream>

using std::cerr;

using namespace Uintah;
using namespace std;

// Requires the necessary input parameters CONSTRUCTORS
Arenisca::Arenisca(ProblemSpecP& ps, MPMFlags* Mflag)
  : ConstitutiveModel(Mflag)
{
  cout << "In Arenisca ver"<< JC_ARENISCA_VERSION;
#ifdef JC_DEBUG_PARTICLE
  cout << ",JC_DEBUG_PARTICLE=" << JC_DEBUG_PARTICLE ;
#endif
#ifdef JC_USE_BB_STATE_UPDATE
  cout << ",JC_USE_BB_STATE_UPDATE";
#endif
#ifdef JC_USE_BB_DEFGRAD_UPDATE
  cout << ",JC_USE_BB_DEFGRAD_UPDATE=" << JC_USE_BB_DEFGRAD_UPDATE;
#endif
#ifdef JC_DEBUG_SMALL_TIMESTEP
  cout << ",JC_DEBUG_SMALL_TIMESTEP";
#endif
#ifdef JC_EPV
  cout << ",JC_EPV";
#endif
#ifdef JC_FREEZE_PARTICLE
  cout << ",JC_FREEZE_PARTICLE";
#endif
#ifdef JC_MAX_NESTED_RETURN
  cout << ",JC_MAX_NESTED_RETURN";
#endif
#ifdef JC_DEBUG_FR_OUTSIDE_CAP
  cout << ",JC_DEBUG_FR_OUTSIDE_CAP";
#endif
#ifdef JC_LIMITER_PRINT
  cout << ",JC_LIMITER_PRINT";
#endif
  cout << endl;
  
  ps->require("FSLOPE",d_initialData.FSLOPE);
  ps->require("FSLOPE_p",d_initialData.FSLOPE_p);
  ps->require("hardening_modulus",d_initialData.hardening_modulus);
  ps->require("CR",d_initialData.CR);
  ps->require("p0_crush_curve",d_initialData.p0_crush_curve);
  ps->require("p1_crush_curve",d_initialData.p1_crush_curve);
  ps->require("p3_crush_curve",d_initialData.p3_crush_curve);
  ps->require("p4_fluid_effect",d_initialData.p4_fluid_effect);
  ps->require("fluid_B0",d_initialData.fluid_B0);
  ps->require("fluid_pressure_initial",d_initialData.fluid_pressure_initial);
  ps->require("subcycling_characteristic_number",d_initialData.subcycling_characteristic_number);
  ps->require("kinematic_hardening_constant",d_initialData.kinematic_hardening_constant);
  ps->require("PEAKI1",d_initialData.PEAKI1);
  ps->require("B0",d_initialData.B0);
  ps->require("G0",d_initialData.G0);
  
#ifdef JC_USE_BB_STATE_UPDATE
  d_intvar = UintahBB::InternalVariableModelFactory::create(ps);
  if(!d_intvar){
    ostringstream desc;
    desc << "**ERROR** Internal error while creating Arenisca_BB->InternalVariableModelFactory." << endl;
    throw InternalError(desc.str(), __FILE__, __LINE__);
  }
#endif
  initializeLocalMPMLabels();
}
Arenisca::Arenisca(const Arenisca* cm)
  : ConstitutiveModel(cm)
{
  cout << "in Arenisca2";
  d_initialData.FSLOPE = cm->d_initialData.FSLOPE;
  d_initialData.FSLOPE_p = cm->d_initialData.FSLOPE_p;
  d_initialData.hardening_modulus = cm->d_initialData.hardening_modulus;
  d_initialData.CR = cm->d_initialData.CR;
  d_initialData.p0_crush_curve = cm->d_initialData.p0_crush_curve;
  d_initialData.p1_crush_curve = cm->d_initialData.p1_crush_curve;
  d_initialData.p3_crush_curve = cm->d_initialData.p3_crush_curve;
  d_initialData.p4_fluid_effect = cm->d_initialData.p4_fluid_effect;
  d_initialData.fluid_B0 = cm->d_initialData.fluid_B0;
  d_initialData.fluid_pressure_initial = cm->d_initialData.fluid_pressure_initial;
  d_initialData.subcycling_characteristic_number = cm->d_initialData.subcycling_characteristic_number;
  d_initialData.kinematic_hardening_constant = cm->d_initialData.kinematic_hardening_constant;
  d_initialData.PEAKI1 = cm->d_initialData.PEAKI1;
  d_initialData.B0 = cm->d_initialData.B0;
  d_initialData.G0 = cm->d_initialData.G0;
  
#ifdef JC_USE_BB_STATE_UPDATE
  d_intvar = UintahBB::InternalVariableModelFactory::createCopy(cm->d_intvar);
#endif
  
  initializeLocalMPMLabels();
}
// DESTRUCTOR
Arenisca::~Arenisca()
{
  VarLabel::destroy(pPlasticStrainLabel);
  VarLabel::destroy(pPlasticStrainLabel_preReloc);
  VarLabel::destroy(pPlasticStrainVolLabel);
  VarLabel::destroy(pPlasticStrainVolLabel_preReloc);
  VarLabel::destroy(pElasticStrainVolLabel);
  VarLabel::destroy(pElasticStrainVolLabel_preReloc);
  VarLabel::destroy(pKappaLabel);
  VarLabel::destroy(pBackStressLabel);
  VarLabel::destroy(pBackStressLabel_preReloc);
  VarLabel::destroy(pBackStressIsoLabel);
  VarLabel::destroy(pBackStressIsoLabel_preReloc);
  VarLabel::destroy(pKappaFlagLabel);
  VarLabel::destroy(pKappaFlagLabel_preReloc);
  VarLabel::destroy(pLocalizedLabel);
  VarLabel::destroy(pLocalizedLabel_preReloc);
  
#ifdef JC_USE_BB_STATE_UPDATE
    delete d_intvar;
#endif
}

//adds problem specification values to checkpoint data for restart
void Arenisca::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","Arenisca");
  }
  cm_ps->appendElement("FSLOPE",d_initialData.FSLOPE);
  cm_ps->appendElement("FSLOPE_p",d_initialData.FSLOPE_p);
  cm_ps->appendElement("hardening_modulus",d_initialData.hardening_modulus);
  cm_ps->appendElement("CR",d_initialData.CR);
  cm_ps->appendElement("p0_crush_curve",d_initialData.p0_crush_curve);
  cm_ps->appendElement("p1_crush_curve",d_initialData.p1_crush_curve);
  cm_ps->appendElement("p3_crush_curve",d_initialData.p3_crush_curve);
  cm_ps->appendElement("p4_fluid_effect",d_initialData.p4_fluid_effect);
  cm_ps->appendElement("fluid_B0",d_initialData.fluid_B0);
  cm_ps->appendElement("fluid_pressure_initial",d_initialData.fluid_pressure_initial);
  cm_ps->appendElement("subcycling_characteristic_number",d_initialData.subcycling_characteristic_number);
  cm_ps->appendElement("kinematic_hardening_constant",d_initialData.kinematic_hardening_constant);
  cm_ps->appendElement("PEAKI1",d_initialData.PEAKI1);
  cm_ps->appendElement("B0",d_initialData.B0);
  cm_ps->appendElement("G0",d_initialData.G0);

#ifdef JC_USE_BB_STATE_UPDATE
   d_intvar->outputProblemSpec(cm_ps);
#endif
}

Arenisca* Arenisca::clone()
{
  return scinew Arenisca(*this);
}

void Arenisca::initializeCMData(const Patch* patch,
                                   const MPMMaterial* matl,
                                   DataWarehouse* new_dw)
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  initSharedDataForExplicit(patch, matl, new_dw);
  
  // Allocates memory for internal state variables at beginning of run.
  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(),patch);
  ParticleVariable<double> pPlasticStrain;
  ParticleVariable<double> pPlasticStrainVol;
  ParticleVariable<double> pElasticStrainVol;
  ParticleVariable<double> pKappa;
  ParticleVariable<Matrix3> pBackStress;
  ParticleVariable<Matrix3> pBackStressIso;
  ParticleVariable<double> pKappaFlag;
  ParticleVariable<int> pLocalized;
  new_dw->allocateAndPut(pPlasticStrain,    pPlasticStrainLabel,    pset);
  new_dw->allocateAndPut(pPlasticStrainVol, pPlasticStrainVolLabel, pset);
  new_dw->allocateAndPut(pElasticStrainVol, pElasticStrainVolLabel, pset);
  new_dw->allocateAndPut(pKappa,            pKappaLabel,            pset);
  new_dw->allocateAndPut(pBackStress,       pBackStressLabel,       pset);
  new_dw->allocateAndPut(pBackStressIso,    pBackStressIsoLabel,    pset);
  new_dw->allocateAndPut(pKappaFlag,        pKappaFlagLabel,        pset);
  new_dw->allocateAndPut(pLocalized,        pLocalizedLabel,        pset);
  ParticleSubset::iterator iter = pset->begin();
  Matrix3 Identity;
  Identity.Identity();
  for(;iter != pset->end();iter++){
    pPlasticStrain[*iter] = 0.0;
    pPlasticStrainVol[*iter] = 0.0;
    pElasticStrainVol[*iter] = 0.0;
    pKappa[*iter] = ( d_initialData.p0_crush_curve +
      d_initialData.CR*d_initialData.FSLOPE*d_initialData.PEAKI1 )/
      (d_initialData.CR*d_initialData.FSLOPE+1.0);
    pBackStress[*iter].set(0.0);
    pBackStressIso[*iter] = Identity*d_initialData.fluid_pressure_initial;
    pKappaFlag[*iter] = 0.0;
    pLocalized[*iter] = 0.0;
  }
  computeStableTimestep(patch, matl, new_dw);
#ifdef JC_USE_BB_STATE_UPDATE
  d_intvar->initializeInternalVariable(pset, new_dw);
#endif
}

//T2D: remove function?
void Arenisca::allocateCMDataAdd(DataWarehouse* new_dw,
                                         ParticleSubset* addset,
          map<const VarLabel*, ParticleVariableBase*>* newState,
                                         ParticleSubset* delset,
                                         DataWarehouse* old_dw)
{
  //Needed by ConstitutiveModelFactory.cc:116
  #ifdef JC_USE_BB_STATE_UPDATE
    d_intvar->allocateCMDataAdd(new_dw,addset, newState, delset, old_dw);
  #endif
}

// Compute stable timestep based on both the particle velocities
// and wave speed
void Arenisca::computeStableTimestep(const Patch* patch,
                                             const MPMMaterial* matl,
                                             DataWarehouse* new_dw)
{
  Vector dx = patch->dCell();
  int dwi = matl->getDWIndex();

  // Get the particles in the current patch
  ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);

  // Get particles mass, volume, and velocity
  constParticleVariable<double> pmass, pvolume;
  constParticleVariable<Vector> pvelocity;
  new_dw->get(pmass,     lb->pMassLabel,     pset);
  new_dw->get(pvolume,   lb->pVolumeLabel,   pset);
  new_dw->get(pvelocity, lb->pVelocityLabel, pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
//  Vector idvel(1,1,1);
//T2D: Include fluid effects in initial timestep calculation
  double bulk = d_initialData.B0;
  double shear= d_initialData.G0;

  // loop over the particles
  for(ParticleSubset::iterator iter = pset->begin();
      iter != pset->end(); iter++){
     particleIndex idx = *iter;

     // Compute wave speed + particle velocity at each particle,
     // store the maximum
     c_dil = sqrt((bulk+4.0*shear/3.0)*pvolume[idx]/pmass[idx]);
/*     if(c_dil+fabs(pvelocity[idx].x()) > WaveSpeed.x()) idvel.x(idx);
     if(c_dil+fabs(pvelocity[idx].y()) > WaveSpeed.y()) idvel.y(idx);
     if(c_dil+fabs(pvelocity[idx].z()) > WaveSpeed.z()) idvel.z(idx);*/
     WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                      Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                      Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
    }
    
    // Compute the stable timestep based on maximum value of
    // "wave speed + particle velocity"
    WaveSpeed = dx/WaveSpeed;
/*    cout << "pvel.x: " << pvelocity[idvel.x()].x() 
         << "; wavespeed.x: " << WaveSpeed.x() << endl;
    cout << "pvel.y: " << pvelocity[idvel.y()].y() 
         << "; wavespeed.y: " << WaveSpeed.y() << endl;
    cout << "pvel.z: " << pvelocity[idvel.z()].z() 
         << "; wavespeed.z: " << WaveSpeed.z() << endl;*/
    double delT_new = WaveSpeed.minComponent();
    if(delT_new < 1.e-12)
      new_dw->put(delt_vartype(DBL_MAX), lb->delTLabel, patch->getLevel());
    else
      new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
}

// ------------------------------------- BEGIN COMPUTE STRESS TENSOR FUNCTION
/**
Arenisca::computeStressTensor is the core of the Arenisca model which computes
the updated stress at the end of the current timestep along with all other
required data such plastic strain, elastic strain, cap position, etc.

*/
void Arenisca::computeStressTensor(const PatchSubset* patches,
                                           const MPMMaterial* matl,
                                           DataWarehouse* old_dw,
                                           DataWarehouse* new_dw)
{
  // Define some constants
  double one_third = 1.0/(3.0);
  double two_third = 2.0/(3.0);
  double four_third = 4.0/(3.0);
  double sqrt_three = sqrt(3.0);
  double one_sqrt_three = 1.0/sqrt_three;

  // Global loop over each patch
  for(int p=0;p<patches->size();p++){
	
    // Declare and initial value assignment for some variables
    const Patch* patch = patches->get(p);
    Matrix3 Identity,D,tensorL(0.0);
    Matrix3 velGrad_old(0.0);
    Identity.Identity();
    double J,c_dil=0.0,se=0.0;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};
	
    // Declare the interpolator variables (CPDI, GIMP, linear, etc)
    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());  ///< nodes affected by particle
    vector<Vector> d_S(interpolator->size());    ///< gradient of grid shape fxn
    vector<double> S(interpolator->size());      ///< grid shape fxn (T2D: needed?)

    // Get particle subset for the current patch
    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    // Declare some particle ISVs and field variables
    ParticleVariable<Matrix3>      deformationGradient_new;
    constParticleVariable<Matrix3> deformationGradient;
    constParticleVariable<Matrix3> stress_old;
    ParticleVariable<Matrix3>      stress_new;
    constParticleVariable<Point>   px;
    constParticleVariable<double>  pmass;
    ParticleVariable<double>       pvolume,
                                   p_q;
    constParticleVariable<Vector>  pvelocity;
	  constParticleVariable<Matrix3> psize;
    ParticleVariable<double>       pdTdt;
    constParticleVariable<double>  pPlasticStrain;
    ParticleVariable<double>       pPlasticStrain_new;
    constParticleVariable<double>  pPlasticStrainVol;
    ParticleVariable<double>       pPlasticStrainVol_new;
    constParticleVariable<double>  pElasticStrainVol;
    ParticleVariable<double>       pElasticStrainVol_new;
    constParticleVariable<double>  pKappa;
    ParticleVariable<double>       pKappa_new;
    constParticleVariable<Matrix3> pBackStress;
    ParticleVariable<Matrix3>      pBackStress_new;
    constParticleVariable<Matrix3> pBackStressIso;
    ParticleVariable<Matrix3>      pBackStressIso_new;
    constParticleVariable<double>  pKappaFlag;
    ParticleVariable<double>       pKappaFlag_new;
    constParticleVariable<int>     pLocalized;
    ParticleVariable<int>          pLocalized_new;
    ParticleVariable<Matrix3>      velGrad,
                                   rotation,
                                   trial_stress;
    ParticleVariable<double>       f_trial,
                                   rho_cur;
    delt_vartype                   delT;
#ifdef JC_USE_BB_STATE_UPDATE
    d_intvar->getInternalVariable(pset, old_dw, pKappa);
    d_intvar->allocateAndPutInternalVariable(pset, new_dw, pKappa_new);
#endif
    // Get, allocate, and put the particle variables
    old_dw->get(delT,              lb->delTLabel,          getLevel(patches));
    old_dw->get(pPlasticStrain,    pPlasticStrainLabel,    pset);
    old_dw->get(pPlasticStrainVol, pPlasticStrainVolLabel, pset);
    old_dw->get(pElasticStrainVol, pElasticStrainVolLabel, pset);
    old_dw->get(pKappa,            pKappaLabel,            pset);
    old_dw->get(pBackStress,       pBackStressLabel,       pset);
    old_dw->get(pBackStressIso,    pBackStressIsoLabel,    pset);
    old_dw->get(pKappaFlag,        pKappaFlagLabel,       pset);
    old_dw->get(pLocalized,        pLocalizedLabel,        pset);
	//all of the above must be initialized in initializeCMData()
    new_dw->allocateAndPut(pPlasticStrain_new,    pPlasticStrainLabel_preReloc,    pset);
    new_dw->allocateAndPut(pPlasticStrainVol_new, pPlasticStrainVolLabel_preReloc, pset);
    new_dw->allocateAndPut(pElasticStrainVol_new, pElasticStrainVolLabel_preReloc, pset);
    new_dw->allocateAndPut(pKappa_new,            pKappaLabel_preReloc,            pset);
    new_dw->allocateAndPut(pBackStress_new,       pBackStressLabel_preReloc,       pset);
    new_dw->allocateAndPut(pBackStressIso_new,    pBackStressIsoLabel_preReloc,    pset);
    new_dw->allocateAndPut(pKappaFlag_new,       pKappaFlagLabel_preReloc,       pset);
    new_dw->allocateAndPut(pLocalized_new,        pLocalizedLabel_preReloc,        pset);
    Ghost::GhostType  gac   = Ghost::AroundCells;
	// variables needed for computing kinematics and timestep size
	//T2D: rearrange by use
    old_dw->get(px,                  lb->pXLabel,                  pset);
    old_dw->get(pmass,               lb->pMassLabel,               pset);
    old_dw->get(psize,               lb->pSizeLabel,               pset);
    old_dw->get(pvelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);
    old_dw->get(stress_old,          lb->pStressLabel,             pset);
    new_dw->allocateAndPut(stress_new, lb->pStressLabel_preReloc,             pset);
    new_dw->allocateAndPut(pvolume,    lb->pVolumeLabel_preReloc,             pset);
    new_dw->allocateAndPut(pdTdt,      lb->pdTdtLabel_preReloc,               pset);
    new_dw->allocateAndPut(deformationGradient_new,
                                       lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->allocateAndPut(p_q,        lb->p_qLabel_preReloc,                 pset);
    new_dw->allocateTemporary(velGrad,      pset);
    new_dw->allocateTemporary(rotation,     pset);
    new_dw->allocateTemporary(trial_stress, pset);
    new_dw->allocateTemporary(f_trial,      pset);
    new_dw->allocateTemporary(rho_cur,      pset);
    
    //get particle ID incase simulation errs
       // Get the particle IDs, useful in case a simulation goes belly up
    constParticleVariable<long64> pParticleID; 
    old_dw->get(pParticleID, lb->pParticleIDLabel, pset);

    // Get the Arenisca model parameters
    const double FSLOPE = d_initialData.FSLOPE;      //yield function
    const double FSLOPE_p = d_initialData.FSLOPE_p;  //flow function
    const double hardening_modulus = d_initialData.hardening_modulus;
    const double CR = d_initialData.CR;
    const double p0_crush_curve = d_initialData.p0_crush_curve;
    const double p1_crush_curve = d_initialData.p1_crush_curve;
    const double p3_crush_curve = d_initialData.p3_crush_curve;
    const double p4_fluid_effect = d_initialData.p4_fluid_effect;
    const double subcycling_characteristic_number = d_initialData.subcycling_characteristic_number;
    const double fluid_B0 = d_initialData.fluid_B0;
    const double kinematic_hardening_constant = d_initialData.kinematic_hardening_constant;
    const double PEAKI1 = d_initialData.PEAKI1;
//    double bulk = d_initialData.B0;
    const double shear= d_initialData.G0;

    // Get the initial density
    double rho_orig = matl->getInitialDensity();

    // Get the deformation gradients first.
    constNCVariable<Vector> gvelocity;  //NCV=Node Centered Variable
    new_dw->get(gvelocity, lb->gVelocityStarLabel,dwi,patch,gac,NGN);

    // Loop over the particles of the current patch to compute particle
    // deformation gradient, volume, and density
    for(ParticleSubset::iterator iter=pset->begin();iter!=pset->end();iter++){
      particleIndex idx = *iter;

      //re-zero the velocity gradient:
      pLocalized_new[idx]=pLocalized[idx];
      tensorL.set(0.0);
      if(!flag->d_axisymmetric){
        // Get the node indices that surround the cell
        interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],
                                                   deformationGradient[idx]);
        computeVelocityGradient(tensorL,ni,d_S, oodx, gvelocity);
      } else {  // axi-symmetric kinematics
        // Get the node indices that surround the cell
        interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,
                                   psize[idx],deformationGradient[idx]);
        // x -> r, y -> z, z -> theta
        computeAxiSymVelocityGradient(tensorL,ni,d_S,S,oodx,gvelocity,px[idx]);
      }
      velGrad_old=velGrad[idx];
      velGrad[idx]=tensorL;

      int num_scs = 1;
      Matrix3 one; one.Identity();
#ifdef JC_USE_BB_DEFGRAD_UPDATE
      // Improve upon first order estimate of deformation gradient
      Matrix3 Amat = (velGrad_old + velGrad[idx])*(0.5*delT);
      Matrix3 Finc = Amat.Exponential(JC_USE_BB_DEFGRAD_UPDATE);
      Matrix3 Fnew = Finc*deformationGradient[idx];
      deformationGradient_new[idx] = Fnew;
#else
      // Update the deformation gradient in a new way using subcycling
      Matrix3 F=deformationGradient[idx];
      double Lnorm_dt = tensorL.Norm()*delT;
      num_scs = max(1,2*((int) Lnorm_dt));
      if(num_scs > 1000){
        cout << "NUM_SCS = " << num_scs << endl;
      }
      double dtsc = delT/(double (num_scs));
      Matrix3 OP_tensorL_DT = one + tensorL*dtsc;
      for(int n=0;n<num_scs;n++){
        F=OP_tensorL_DT*F;
      }
      deformationGradient_new[idx]=F;
      // Update the deformation gradient, Old First Order Way
      // deformationGradient_new[idx]=(tensorL*delT+Identity)*deformationGradient[idx];
#endif
      
      // Compute the Jacobian and delete the particle in the case of negative Jacobian
      J = deformationGradient_new[idx].Determinant();
      if (J<=0){
        cout<<"ERROR, negative J! "<<endl;
        cout<<"pParticleID="<<pParticleID[idx]<<endl;
        cout<<"deformationGradient_new="<<deformationGradient_new[idx]<<endl<<endl;
        cout<<"J= "<<J<<endl;
        cout<<"Fnew= "<<deformationGradient_new[idx]<<endl;
        cout<<"Fold= "<<deformationGradient[idx]<<endl;
        cout<<"L= "<<tensorL<<endl;
        cout<<"num_scs= "<<num_scs<<endl;
        pLocalized_new[idx] = -999;
        cout<<"DELETING Arenisca particle " << endl;
        J=1;
        deformationGradient_new[idx] = one;
        //throw InvalidValue("**ERROR**:Negative Jacobian", __FILE__, __LINE__);
      }
#ifdef JC_FREEZE_PARTICLE
      if (J>4000){
        cout << "WARNING, massive $J! " << "J=" << J
        //     <<", DELETING particleID=" <<pParticleID[idx]<< endl; 
        //pLocalized_new[idx] = -999;
             <<",FREEZING particleID="<<pParticleID[idx]<<endl;
        J = deformationGradient[idx].Determinant();
        deformationGradient_new[idx] = deformationGradient[idx];
      }
#endif

      // Update particle volume and density
      pvolume[idx]=(pmass[idx]/rho_orig)*J;
      rho_cur[idx] = rho_orig/J;
      
    }

    // Compute the initial value of R=\kappa-X (see "fig:CapEccentricity" and
    // "eq:initialValueOfR" in the Arenisca manual)
    double cap_r_initial = CR*FSLOPE*(PEAKI1-p0_crush_curve)/(1.0+CR*FSLOPE);

    // Define two limitations for \kappa and X (see "eq:limitationForX" and
    // "eq:limitationForKappa" in the Arenisca manual)
    double min_kappa = 1.0e5 * p0_crush_curve;
    double max_X = 0.00001 * p0_crush_curve;
#ifdef JC_USE_BB_STATE_UPDATE
    UintahBB::ModelState* state = scinew UintahBB::ModelState();
    state->local_var[0] = 0.0;            // kappa value
    state->local_var[1] = cap_r_initial;  // initial cap radius
    state->local_var[2] = max_X;          // max_X
    state->local_var[3] = 0.0;            // eps_v
    state->local_var[4] = 0.0;            // Delta eps_v
    state->local_var[5] = 1.0;            // scale factor for Delta eps_v    
#endif

   
    // Loop over the particles of the current patch to update particle
    // stress at the end of the current timestep along with all other
    // required data such plastic strain, elastic strain, cap position, etc.
#ifdef JC_DEBUG_SMALL_TIMESTEP
	  Vector idvel(1,1,1);   // temp
    Vector vbulk(1,1,1);   // temp
	  Vector vshear(1,1,1);  // temp
#endif
    for(ParticleSubset::iterator iter = pset->begin();iter!=pset->end();iter++){
      particleIndex idx = *iter;  //patch index
      
        // A parameter to consider the thermal effects of the plastic work which
        // is not coded in the current source code. Further development of Arenisca
        // may ativate this feature.
        pdTdt[idx] = 0.0;

        // pKappaFlag is a flag indicating if the particle has met any of the
        // limit values on \kappa and X. (see "eq:limitationForX"
        // and "eq:limitationForKappa" in the Arenisca manual)
        // 1: met the max_X limitation, 2: met the min_kappa limitation, 0: otherwise
        pKappaFlag_new[idx] = pKappaFlag[idx];

        
        // Apply the hardening modulus for the Drucker-Prager part
        //T2D: add reference to manual
        double PEAKI1_hardening = PEAKI1*FSLOPE + hardening_modulus*pPlasticStrain[idx];

        // Compute the current value of cap_radius=R=\kappa-X (see "fig:CapEccentricity" and
        // "eq:initialValueOfR" in the Arenisca manual)
        double pKappa1=pKappa[idx];
        double cap_radius=-CR*(FSLOPE*pKappa1-PEAKI1_hardening);
  #ifdef JC_USE_BB_STATE_UPDATE
        double cap_radius_fac = FSLOPE*pKappa1 - PEAKI1_hardening;
        cap_radius=-CR*cap_radius_fac;
  #endif

        // Apply the limitation for R=\kappa-X (see "eq:limitationForR" in the Arenisca manual).
        // The condition of "pKappa1>PEAKI1_hardening/FSLOPE" in the following IF condition
        // indicates that the limitation should be applied if cap_radius<0.
        // cond_fixed_cap_radius is a flag which indicates if the limit has been met or not?
        int cond_fixed_cap_radius = 0;
        if (cap_radius<0.1*cap_r_initial || pKappa1>PEAKI1_hardening/FSLOPE){
  #ifdef JC_LIMITER_PRINT
          cout << "$R";
  #endif
          cout<<",pKappa1_old="<<pKappa1;
          pKappa1 = pKappa1 - cap_radius + 0.1*cap_r_initial;
          cout<<",pKappa1_new="<<pKappa1;
          cap_radius=0.1*cap_r_initial;
          cond_fixed_cap_radius = 1;
        }

        // Compute the symmetric part of the velocity gradient
        Matrix3 D = (velGrad[idx] + velGrad[idx].Transpose())*.5;

        // Use polar decomposition to compute the rotation and stretch tensors
        Matrix3 tensorR, tensorU;
        deformationGradient[idx].polarDecompositionRMB(tensorU, tensorR);
        rotation[idx]=tensorR;

//#ifdef jc_freeze_particle
//        if(tensorU.MaxAbsElem()>15){
//          pLocalized_new[idx] = -999;  //T2D: May need more here
//          cout << ",maxUhit";
//        }
//#endif
          
        // Compute the unrotated symmetric part of the velocity gradient
        D = (tensorR.Transpose())*(D*tensorR);

        // Compute the effective bulk modulus based on the fluid effects
        //  See Section 4 of the user guide for more information on the fluid effects
        double bulk_temp = exp(p3_crush_curve+p4_fluid_effect+pPlasticStrainVol[idx]);
        double bulk_initial = d_initialData.B0;
        double bulk = bulk_initial + fluid_B0*
             ( exp(p3_crush_curve+p4_fluid_effect)-1.0 ) * bulk_temp
             / ( (bulk_temp-1.0)*(bulk_temp-1.0) );

        // Apply the limitation for the effective bulk modulus
        // (see "eq:limitationForKe" in the Arenisca manual).
        if (bulk>5.0*bulk_initial) {
          //cout << "patch=" << patch->getID() << ",idx=" << idx << ":limit5xbulk_initial" ;
  #ifdef JC_LIMITER_PRINT
          cout << "$B";
  #endif
  #ifdef JC_DEBUG_PARTICLE //LIMITER
          if(pParticleID[idx]==JC_DEBUG_PARTICLE)
            cout << ",EBLim,p="<<patch->getID()<<",idx="<<idx;
  #endif
          bulk = 5.0*bulk_initial;
        }  //T2D: use min statement

        // Compute the lame constant using the bulk and shear moduli
        double lame = bulk - two_third*shear;

        // Compute the unrotated stress at the first of the current timestep
        Matrix3 unrotated_stress = (tensorR.Transpose())*(stress_old[idx]*tensorR);

        // Compute the unrotated trial stress
        Matrix3 stress_diff = (Identity*lame*(D.Trace()*delT) + D*delT*2.0*shear);
        trial_stress[idx] = unrotated_stress + stress_diff;

        // Compute shifted trial stress based on the back stress
        trial_stress[idx] = trial_stress[idx] - pBackStress[idx];

        // Compute the value of the yield function at the trial stress
        f_trial[idx] = YieldFunction(trial_stress[idx],FSLOPE,pKappa1,cap_radius,PEAKI1_hardening);

        // initial assignment for the updated values of plastic strains, volumetric
        // part of the plastic strain, volumetric part of the elastic strain, \kappa,
        // and the backstress. tentative assumption of elasticity
        pPlasticStrain_new[idx]    = pPlasticStrain[idx];
        pPlasticStrainVol_new[idx] = pPlasticStrainVol[idx];
        pElasticStrainVol_new[idx] = pElasticStrainVol[idx] + D.Trace()*delT;
        pKappa_new[idx]            = pKappa1;
        pBackStress_new[idx]       = pBackStress[idx];
        pBackStressIso_new[idx]    = pBackStressIso[idx];

        // Compute stress invariants of the trial stress
        double I1_trial,J2_trial;
        Matrix3 S_trial;
        computeInvariants(trial_stress[idx], S_trial, I1_trial, J2_trial);

        // Declare and assign two variables for evolving back stress tensor
        Matrix3 deltaBackStress;
        Matrix3 deltaBackStressIso;
        deltaBackStress.set(0.0);  //tentative assumption of elasticity
        deltaBackStressIso.set(0.0);

        // Check if the stress is elastic or plastic?
        if (f_trial[idx]<0){  // elastic

          // An elastic step: the updated stres at the end of the current time step
          // is equal to the trial stress. otherwise, the plasticity return algrithm would be used.
          stress_new[idx] = trial_stress[idx];  //T2D: make trial_stress a local variable

        }else{  // plastic

          // An elasto-plasic/fully plastic step: the plasticity return algrithm should be used.
          // The nested return algorithm is used (Brannon & Leelavanichkul 2010) in Arenisca.

          // Determine a characteristic length of the yield surface.
          // If Arenisca is used as the Drucker-Prager model, which is determined by a large negative
          // \kappa value (pKappa1<-1.0e80), the characteristic length is two times the value of
          // sqrt(J2) at I1=0 (T2D: use fraction of sqrt(J2) at a value of B0/1000 for softening),
          // and if it leads to a small value the chracteristic length equals
          // two times PEAKI1. If two-surface Arenisca is used, the minumum of the following two
          // values is considered as the characteristic length: "PEAKI1-X" and "2*(FSLOPE*X-PEAKI1)" 
          double clen;
          double PI1_h_over_FSLOPE = PEAKI1_hardening/FSLOPE;
          int char_length_yeild_surface_flag;
          if (pKappa1<-1.0e80){  //no cap
            if (I1_trial<0.0){
              clen = abs(PI1_h_over_FSLOPE-FSLOPE*I1_trial);
              char_length_yeild_surface_flag=1;
            } else {
              clen = abs(PI1_h_over_FSLOPE);
              char_length_yeild_surface_flag=2;
            }
          } else {  //cap
            if (PI1_h_over_FSLOPE-(pKappa1-cap_radius)
                < -2.0*(FSLOPE*(pKappa_new[idx]-cap_radius)-PI1_h_over_FSLOPE)){
              clen = PI1_h_over_FSLOPE-(pKappa1-cap_radius);
              char_length_yeild_surface_flag=3;
            } else {
              clen = -1*(FSLOPE*(pKappa_new[idx]-cap_radius)//T2D: should this be 2*__?
                                                -PI1_h_over_FSLOPE);
              char_length_yeild_surface_flag=4;
            }
          }

#ifdef JC_DEBUG_PARTICLE // print characteristic length of yeild surface
          if(pParticleID[idx]==JC_DEBUG_PARTICLE){
            cout << " clen=" << clen 
                 << "@" << char_length_yeild_surface_flag;
            cout << ", test@3="<<PI1_h_over_FSLOPE-(pKappa1-cap_radius);
            cout << ", test@4="<<(-1*(FSLOPE*(pKappa_new[idx]-cap_radius)
                                     -PI1_h_over_FSLOPE));
          }
#endif
          // If the characteristic length gets a negative value, it means that there is an issue
          // with the yield surface, which should be reported.
          if (clen<=0.0) {
            cout<<"ERROR! in clen"<<endl;
            cout<<"pParticleID="<<pParticleID[idx]<<endl;
            cout<<"deformationGradient_new="<<deformationGradient_new[idx]<<endl<<endl;
            cout<<"clen="<<clen<<endl;
            cout<<"pKappa_new[idx]="<<pKappa_new[idx]<<endl;
            cout<<"cap_radius="<<cap_radius<<endl;
            cout<<"PEAKI1_hardening/FSLOPE="<<PI1_h_over_FSLOPE<<endl;
            cout<<"idx="<<idx<<endl;
            throw InvalidValue("**ERROR**:in characteristic length of yield surface (clen)",
                               __FILE__, __LINE__);
          }
//---------------------------------------------------------------------------------
//JC_BEGIN SUBCYCLING
          // Compute the un-shifted trial stress
          trial_stress[idx] = trial_stress[idx] + pBackStress[idx];
          
          // Compute total number of cycles in the plasticity subcycling
          double sqrt_mag_trial_stress=sqrt(trial_stress[idx].Norm());
          double num_subcycles = floor (sqrt_mag_trial_stress
             /(sqrt(clen)/subcycling_characteristic_number) + 1);

          if(num_subcycles > 1000){
            cout << "WARNING: $num_subcycles="<<num_subcycles
                 << " exceeds 1000 maximum for pID="<<pParticleID[idx] << endl;
            num_subcycles=1000;
          }

          // Remove the new changes from the trial stress so we can apply the changes in each sub-cycle
          trial_stress[idx] = trial_stress[idx] - stress_diff; //T2D: consider setting trial_s to stress at beginning of ts
          
          // Changes in the trial stress in each sub-cycle assuming the elastic behavior
          stress_diff = stress_diff/num_subcycles;
          
          //initial assignment
          stress_new[idx] = trial_stress[idx];
          Matrix3 trial_stress_loop;
#ifdef JC_DEBUG_PARTICLE // print characteristic length of yeild surface
          if(pParticleID[idx]==JC_DEBUG_PARTICLE)
            cout << ",smts="<<sqrt_mag_trial_stress<<",nSC=" << num_subcycles;
#endif
          // Loop over sub-cycles in the plasticity return algorithm
          for (int subcycle_counter=0 ; subcycle_counter<=num_subcycles-1 ; subcycle_counter++){
//JC_IF_FAIL_RESTART_WITH_MORE_SUBCYCLES
#ifdef JC_DEBUG_PARTICLE  // Print number of subcycles
            if(pParticleID[idx]==JC_DEBUG_PARTICLE)
              cout << endl << "  SCc=" << subcycle_counter << "/"<<num_subcycles-1;
                  //<< ",tsr="<<trial_stress[idx].Norm();
#endif
            // Compute the trial stress for the current sub-cycle
            trial_stress[idx] = stress_new[idx];
            trial_stress[idx] = trial_stress[idx] + stress_diff;
            
            // Compute the shifted trial stress
            trial_stress[idx] = trial_stress[idx] - pBackStress_new[idx];
  #ifdef JC_DEBUG_PARTICLE  // Print number of subcycles
            if(pParticleID[idx]==JC_DEBUG_PARTICLE){
              Matrix3 S_trial_test;
              double I1_trial_test,J2_trial_test;
              computeInvariants(trial_stress[idx], S_trial_test, I1_trial_test, J2_trial_test);
              cout << ",I1_test=" << I1_trial_test << ",J2_test="<<J2_trial_test;
            }
  #endif
            // 'condition_return_to_vertex' flag defines if we should do vertex treatment for the
            // particle at the current timestep or not? 1: the trial stress will return back to the
            // vertex, 0: the trial stress will not return back to the vertex.
            int condition_return_to_vertex=0;

            // Check if the vertex treatment is necessary or not? If I1>PEAKI1, we may need
            // vertex treatment otherwise, we do not need vertex treatment.
            if (I1_trial>PI1_h_over_FSLOPE){  //I1 to right of vertex

              // In the case of hydrostatic tensile loading, the vertex treatment is needed.
              if (J2_trial<1.0e-10*clen){  // the case of hydrostatic tensile loading

                // The updated stress should be the vertex.
                stress_new[idx] = Identity*PI1_h_over_FSLOPE*one_third;

                // 'condition_return_to_vertex' is set to one, which means that the trial stress
                // is returned back to the vertex.
                condition_return_to_vertex = 1;

              }else{  // test the case of non-hydrostatic tensile loading

                // To determine if we should apply vertex treatment or not, here it is checked
                // if the trial stress is between two P tensors, the projection direction tensor,
                // (see Eq. 22 and 24 in Brannon & Leelavanichkul 2010) at the vertex or not.
                // If yes, we will do vertex treatment.

                // Declare the variables needed for vertex treatment
                double P_component_1,P_component_2;
                double relative_stress_to_vertex_1,relative_stress_to_vertex_2;
                Matrix3 relative_stress_to_vertex,relative_stress_to_vertex_deviatoric;
                Matrix3 unit_tensor_vertex_1;
                Matrix3 unit_tensor_vertex_2;
                Matrix3 P,M,P_deviatoric;

                // Compute the relative trial stress in respect with the vertex
                relative_stress_to_vertex = trial_stress[idx] - Identity*PI1_h_over_FSLOPE*one_third;

                // Compute two unit tensors of the stress space
                unit_tensor_vertex_1 = Identity/sqrt_three; //T2D: use other 1/root3
                unit_tensor_vertex_2 = S_trial/sqrt(2.0*J2_trial);  

                // Compute the unit tensor in the direction of the plastic strain
                M = ( Identity*FSLOPE_p + S_trial*(1.0/(2.0*sqrt(J2_trial))) )/sqrt(3.0*FSLOPE_p*FSLOPE_p + 0.5);

                // Compute the projection direction tensor
                P = (Identity*lame*(M.Trace()) + M*2.0*shear);

                // Compute the components of P tensor in respect with two unit_tensor_vertex
                P_component_1 = P.Trace()/sqrt_three;
                P_deviatoric = P - unit_tensor_vertex_1*P_component_1;
                int counter_1_fix=0;
                int counter_2_fix=0;
                for (int counter_1=0 ; counter_1<=2 ; counter_1++){
                  for (int counter_2=0 ; counter_2<=2 ; counter_2++){
                    if (fabs(unit_tensor_vertex_2(counter_1,counter_2))>
                        fabs(unit_tensor_vertex_2(counter_1_fix,counter_2_fix))){
                      counter_1_fix = counter_1;
                      counter_2_fix = counter_2;
                    }
                  }
                }
                P_component_2 = P_deviatoric(counter_1_fix,counter_2_fix)/
                                unit_tensor_vertex_2(counter_1_fix,counter_2_fix);

                // Compute the components of relative_stress_to_vertex in respect with
                // two unit_tensor_vertex
                relative_stress_to_vertex_1 = relative_stress_to_vertex.Trace()*one_sqrt_three;
                relative_stress_to_vertex_deviatoric = relative_stress_to_vertex -
                                              unit_tensor_vertex_1*relative_stress_to_vertex_1;
                relative_stress_to_vertex_2 =
                            relative_stress_to_vertex_deviatoric(counter_1_fix,counter_2_fix)/
                            unit_tensor_vertex_2(counter_1_fix,counter_2_fix);

                // Check if the stress_trial is in the vertex zone or not?
                if ( ((relative_stress_to_vertex_1*P_component_2 + relative_stress_to_vertex_2*P_component_1)/
                   (P_component_1*P_component_1) >=0 ) ){

                  // The updated stress should be the vertex.
                  stress_new[idx] = Identity*one_third*PI1_h_over_FSLOPE;
      #ifdef JC_DEBUG_PARTICLE
                  if(pParticleID[idx]==JC_DEBUG_PARTICLE)
                    cout << ",901:stress_new="<<stress_new[idx];
      #endif

                  // 'condition_return_to_vertex' is set to one, which means that the trial stress
                  // is returned back to the vertex.
                  condition_return_to_vertex = 1;
                }

              }

              // If the trial stress is returned back to the vertex, other required particle variables
              // such as the plastic strain, elastic strain, \kappa, back stress, etc. should also 
              // be updated.
              // T2D: cite manual.
              if (condition_return_to_vertex == 1) {
                // Compute two coefficients that are used in calculation of strain from stress
                double shear_inverse = 0.5/shear;
                double nu_over_E = (-1.0)*lame/(2.0*shear*(2.0*shear+3.0*lame));

                // Compute the difference between the stress tensor at the beginning and end of
                //the current time step
                Matrix3 diff_stress_iteration = trial_stress[idx] - stress_new[idx];
                
  #ifdef JC_DEBUG_PARTICLE  // Print number of subcycles
                if(pParticleID[idx]==JC_DEBUG_PARTICLE)
                  cout << ",dsr="<<diff_stress_iteration.Norm();
  #endif
                
                // Compute the plastic strain during the current timestep
                Matrix3 strain_iteration = (Identity*nu_over_E*(diff_stress_iteration.Trace())
                                           + diff_stress_iteration*shear_inverse);

                // Update the plastic strain magnitude
                pPlasticStrain_new[idx] = pPlasticStrain[idx] +strain_iteration.Norm();

                // Update the volumetric part of the plastic strain
                pPlasticStrainVol_new[idx] = pPlasticStrainVol[idx] + strain_iteration.Trace();
      #ifdef JC_DEBUG_PARTICLE  // Print number of subcycles
                if(pParticleID[idx]==JC_DEBUG_PARTICLE)
                  cout << ",epv="<<pPlasticStrainVol_new[idx];
      #endif      
      #ifdef JC_EPV
                if (pPlasticStrainVol_new[idx]<-p3_crush_curve) {
                  cout<<"ERROR! in pPlasticStrainVol_new"<<endl;
                  cout<<"pParticleID="<<pParticleID[idx]<<endl;
                  cout<<"deformationGradient_new="<<deformationGradient_new[idx]<<endl<<endl;
                  cout<<"epv=" << pPlasticStrainVol_new[idx]<<endl;
                  cout<<"pKappa_new[idx]="<<pKappa_new[idx]<<endl;
                  throw InvalidValue("**ERROR**:in pPlasticStrainVol calculation",
                                     __FILE__, __LINE__);
                }
      #endif
                // Update the volumetric part of the elastic strain
                pElasticStrainVol_new[idx] = pElasticStrainVol_new[idx] - strain_iteration.Trace();

                // Update the back stress (see "eq:backStressFluidEffect" in the Arenisca manual)
                pBackStress_new[idx] = Identity*( -3.0*fluid_B0*
                                       ((exp(pPlasticStrainVol_new[idx])-1.0)
                                        * exp(p3_crush_curve+p4_fluid_effect))
                                       /(exp(p3_crush_curve+p4_fluid_effect +
                                         pPlasticStrainVol_new[idx])-1.0) )*
                                       (pPlasticStrainVol_new[idx]);
                if (isnan(pBackStress_new[idx].Trace())) {  //Check pBackStress for nan
                  cerr << "ParticleID = " << pParticleID[idx] << " pBackStress_new[idx] = " << pBackStress_new[idx] << endl;
                  cout<<"deformationGradient_new="<<deformationGradient_new[idx]<<endl<<endl;
                  cout<<"epv=" << pPlasticStrainVol_new[idx]<<endl;
                  cout<<"pKappa_new[idx]="<<pKappa_new[idx]<<endl;
                  throw InvalidValue("**ERROR**: Nan in stress_iteration value", __FILE__, __LINE__);
                }
                
      #ifdef JC_DEBUG_PARTICLE  // Print pPlasticStrainVol
                if(pParticleID[idx]==JC_DEBUG_PARTICLE)
                  cout << ",alpha=I*" << ( -3.0*fluid_B0*
                                           ((exp(pPlasticStrainVol_new[idx])-1.0)
                                           * exp(p3_crush_curve+p4_fluid_effect))
                                           /(exp(p3_crush_curve+p4_fluid_effect +
                                                 pPlasticStrainVol_new[idx])-1.0) )*
                                          (pPlasticStrainVol_new[idx]);
                
      #endif
                // Update \kappa (= the position of the cap) see "eq:evolutionOfKappaFluidEffect" in the
                // Arenisca manual. (Also, see "fig:AreniscaYieldSurface" in the Arenisca manual)

                // Declare and assign some axiliary variables
                double pKappa_temp = exp(p3_crush_curve+p4_fluid_effect+pPlasticStrainVol[idx]);
                double pKappa_temp1 = exp(p3_crush_curve+pPlasticStrainVol[idx]);
                double var1;

                // Compute the var1 which relates dX/de to d(kappa)/de: d(kappa)/de=dX/de * (1/var1)
                // Consider the limitation for R=\kappa-X (see "eq:limitationForR" in the Arenisca manual).
                // 'cond_fixed_cap_radius' is a variable which indicates if the limit has been met or not?
                if (cond_fixed_cap_radius==0) {  //T2D: limiter print?
                  var1 = 1.0+FSLOPE*CR;
                } else {
                  var1 = 1.0;
                }
      #ifdef JC_USE_BB_STATE_UPDATE
                // Update the model state
                state->local_var[0] = pKappa1;                  // kappa value
                state->local_var[1] = cap_radius;                  // cap radius
                state->local_var[2] = max_X;                       // max_X
                state->local_var[3] = pPlasticStrainVol[idx];      // eps_v
                state->local_var[4] = strain_iteration.Trace();    // Delta eps_v
                state->local_var[5] = var1;                      // scale factor for Delta eps_v
                
                // Compute internal variable
                pKappa_new[idx] = d_intvar->computeInternalVariable(state);
                // Set the kappa state flag
                if (pKappa_new[idx] == (max_X + cap_radius)) pKappaFlag_new[idx] = 1.0;
      #else
                // Update \kappa (pKappa1) in the case of 
                if (pKappa1-cap_radius-p0_crush_curve<0) {  // X < p0
                  // Update \kappa in the case of X < p0
                  // (see "fig:AreniscaYieldSurface" in the Arenisca manual)
                  // (see "eq:evolutionOfKappaFluidEffect" in the Arenisca manual)
                  double exp_p3_p4_m1 = exp(p3_crush_curve+p4_fluid_effect)-1.0;
                  pKappa_new[idx] = pKappa1 + ( exp(-p1_crush_curve*
                                    (pKappa1-cap_radius-p0_crush_curve))
                                    /( p3_crush_curve*p1_crush_curve ) -
                                    3.0*fluid_B0*(exp_p3_p4_m1)*pKappa_temp
                                    /( (pKappa_temp-1.0)*(pKappa_temp-1.0) ) +
                                    3.0*fluid_B0*(exp_p3_p4_m1)*pKappa_temp1
                                    /( (pKappa_temp1-1.0)*(pKappa_temp1-1.0) ) )
                                    *strain_iteration.Trace()/var1;
       #ifdef JC_DEBUG_PARTICLE
                  if(pParticleID[idx]==JC_DEBUG_PARTICLE)
                    cout<<",X<p0,Knew="<<pKappa_new[idx];
       #endif
                   //cout<<",pKappa1="<<pKappa1<<",cap_radius="<<cap_radius<<",pKappa_temp1="<<pKappa_temp1
                   //    <<",pKappa_temp="<<pKappa_temp<<",Tr(strain_itr)="<<strain_iteration.Trace()
                   //    <<",exp(X-p0)="<<exp(-p1_crush_curve*(pKappa1-cap_radius-p0_crush_curve));
                } else if (pKappa1-cap_radius<max_X) {  // p0 <= X < max_X
                  // Update \kappa in the cae of p0 <= X < max_X
                  // (see "fig:AreniscaYieldSurface" in the Arenisca manual)
                  // (see "eq:evolutionOfKappaFluidEffect1" in the Arenisca manual)
                  // (for the limitation of max_X see "eq:limitationForX" in the Arenisca manual)
                  double exp_p3_p4_m1 = exp(p3_crush_curve+p4_fluid_effect)-1.0;
                  pKappa_new[idx] = pKappa1 + ( pow( (pKappa1-cap_radius)/p0_crush_curve,
                                               1-p0_crush_curve*p1_crush_curve*p3_crush_curve )
                                               /( p3_crush_curve*p1_crush_curve ) -
                                               3.0*fluid_B0*exp_p3_p4_m1*pKappa_temp
                                               /( (pKappa_temp-1.0)*(pKappa_temp-1.0) ) +
                                               3.0*fluid_B0*exp_p3_p4_m1*pKappa_temp1
                                               /( (pKappa_temp1-1.0)*(pKappa_temp1-1.0) ) )
                                               *strain_iteration.Trace()/var1;
  #ifdef JC_DEBUG_PARTICLE
                  if(pParticleID[idx]==JC_DEBUG_PARTICLE)
                     cout<<",p0<=X<max_X,Knew="<<pKappa_new[idx];
  #endif
                } else {  // X >= max_X
                  // Update \kappa in the cae of X >= max_X
                  // (see "fig:AreniscaYieldSurface" in the Arenisca manual)
                  // In this case it is assumed that X=max_X
                  // (for the limitation of max_X see "eq:limitationForX" in the Arenisca manual)
                  // pKappaFlag is a particle variable variable which defines if the particle
                  // meet any of the limitation for \kappa and X or not?
                  // pKappaFlag=1: means that the particle met the max_X limitation
                  pKappa_new[idx] = max_X + cap_radius;
                  pKappaFlag_new[idx] = 1.0;
      #ifdef JC_LIMITER_PRINT
                  cout << "$X";
      #endif
  #ifdef JC_DEBUG_PARTICLE
                  if(pParticleID[idx]==JC_DEBUG_PARTICLE)
                    cout<<",X>=max_X,Knew="<<pKappa_new[idx];
  #endif
                }
      #endif
                // \kappa limeter ensures \kappa >= \kappa_min (pKappa_new[idx]>=min_kappa)
                if (pKappa_new[idx]<min_kappa){
                  // Apply the lower limit for \kappa.
                  // (for the limitation of min_kappa see "eq:limitationForKappa"
                  // in the Arenisca manual)
                  // pKappaFlag is a particle variable variable which defines if the particle
                  // meet any of the limitation for \kappa and X or not?
                  // pKappaFlag=2: means that the particle met the min_kappa limitation.
                  pKappa_new[idx] = min_kappa;
                  pKappaFlag_new[idx] = 2.0;
      #ifdef JC_LIMITER_PRINT
                  cout << "$K";
      #endif
  #ifdef JC_DEBUG_PARTICLE
                  if(pParticleID[idx]==JC_DEBUG_PARTICLE)
                    cout << ",Klim";
  #endif
                }

                // Re-calculate the axiliary varable 'PEAKI1_hardening'
                // because 'pPlasticStrain_new[idx]' has been updated 
                PEAKI1_hardening = PEAKI1*FSLOPE + hardening_modulus*pPlasticStrain_new[idx];
      #ifdef JC_USE_BB_STATE_UPDATE
                cap_radius_fac = FSLOPE*pKappa_new[idx] - PEAKI1_hardening;
      #endif
                // R limiter ensures R >= 0.1*Rinit (cap_radius >= 0.1*cap_r_initial)
                if (cond_fixed_cap_radius==0) {
                  // Consider the limitation for R=\kappa-X (see "eq:limitationForR" in the Arenisca manual).
                  // 'cond_fixed_cap_radius' is a variable which indicates if the limit has been met or not?
                  // If the limit has been met, updated \kappa should be modified.
                  cap_radius=CR*abs(FSLOPE*pKappa_new[idx]-PEAKI1_hardening);
                  if (cap_radius<0.1*cap_r_initial || pKappa_new[idx]>PEAKI1_hardening/FSLOPE) {
                    pKappa_new[idx] = pKappa_new[idx] - cap_radius + 0.1*cap_r_initial;
                    cap_radius=0.1*cap_r_initial;
                    cond_fixed_cap_radius = 1;
      #ifdef JC_LIMITER_PRINT
                    cout << "$R";
      #endif
  #ifdef JC_DEBUG_PARTICLE
                    //cout<<",cond_fixed_cap_radius="<<cond_fixed_cap_radius;
                    if(pParticleID[idx]==JC_DEBUG_PARTICLE)
                      cout << ",Rlim";
  #endif
                  }
                }
                
                // X limiter ensures X < maxX (pKappa_new - cap_radius < max_X)
                if (pKappa_new[idx]>max_X+cap_radius) {
                  // Apply the upper limit for X. 
                  // (for the limitation of max_X see "eq:limitationForX" in the Arenisca manual)
                  // pKappaFlag is a particle variable variable which defines if the particle
                  // meet any of the limitation for \kappa and X or not?
                  // pKappaFlag=1: means that the particle met the max_X limitation
                  // If the limit has been met, updated \kappa should be modified.
                  pKappa_new[idx]=max_X+cap_radius;
                  pKappaFlag_new[idx] = 1.0;
        #ifdef JC_LIMITER_PRINT
                  cout << "$X";
        #endif
  #ifdef JC_DEBUG_PARTICLE  // USE COLOVOS IN DEFINE NAME
                  //cout<<",pKappaFlag_new[idx]=" << pKappaFlag_new[idx];
                  if(pParticleID[idx]==JC_DEBUG_PARTICLE)
                    cout << ",Xlim";
  #endif
                }
                //compute the un-shifted stress
                stress_new[idx] = stress_new[idx] + pBackStress_new[idx];
              }
            } //end of if for I1 to right of vertex

            // 'condition_return_to_vertex' variable defines if we should do vertex treatment for the
            // particle at the current timestep or not? (condition_return_to_vertex == 0) means that
            // the trial stress is not returned back to the vertex so the nested return algorithm 
            // (Brannon & Leelavanichkul 2010) should be used for returning back the trial stress to
            // the yield surface.
            if (condition_return_to_vertex == 0){

#ifdef JC_DEBUG_PARTICLE // Print shifted trial stress for current subcycle
              //if(pParticleID[idx]==JC_DEBUG_PARTICLE)
              //  cout << "  cSC=" << subcycle_counter << ",sftrStress="<<trial_stress[idx];
#endif
          //BEGIN NESTED RETURN ALGORITHM
              // Declare and initialize some needed variables in the nested return algorithm
              double gamma = 0.0;;
              double I1_iteration,J2_iteration;
              double beta_cap,FSLOPE_cap;
              double f_new_loop = 1e99;
              Matrix3 pBackStress_loop = pBackStress_new[idx];
              double pKappa_loop = pKappa_new[idx];
              int counter = 0;
              Matrix3 P,M,G;
              Matrix3 stress_iteration=trial_stress[idx];
              trial_stress_loop = trial_stress[idx];
              Matrix3 S_iteration;
              Matrix3 plasStrain_loop;
              plasStrain_loop.set(0.0);

              // Nested return algorithm (Brannon & Leelavanichkul 2010) is an iterative method.
              // The maximum allowed number of iterations is defined here. Large numbers may lead
              // very slow simulations.
              int max_number_of_iterations = 9;

              // Loop for the nested return algorithm (Brannon & Leelavanichkul 2010).
              // The loop will continue until the trial stress is returned back to the yield
              // surface or the number of iterations exeeds the maximum number.

              while(fabs(f_new_loop)>9.9e-2*clen
                    && counter<=max_number_of_iterations ){//T2D: change 9.9e-2 to const, test how small

                // Number of completed iterations
              
                counter=counter+1;
                trial_stress_loop = stress_iteration;

                if (isnan(stress_iteration.Trace())) {  //Check stress_iteration for nan
                  cerr << "ParticleID = " << pParticleID[idx] << " stress_iteration = " << stress_iteration << endl;
                  //cerr<<"I1_iter="<<I1_iteration_err<<endl;
                  //cerr<<"J2_iter="<<J2_iteration_err<<endl;
                  throw InvalidValue("**ERROR**: Nan in stress_iteration value", __FILE__, __LINE__);
                }
                // Compute the invariants of the trial stress for the current subcycle in the loop
                computeInvariants(stress_iteration, S_iteration, I1_iteration, J2_iteration);

                if (I1_iteration>PI1_h_over_FSLOPE){  //1st right of PEAKI1 & above vertex treatment
        #ifdef JC_DEBUG_PARTICLE // Print shifted trial stress for current subcycle
                  if(pParticleID[idx]==JC_DEBUG_PARTICLE)
                    cout << ",FR1";
        #endif
                  // Fast return algorithm in the case of I1>PEAKI1 (see "fig:AreniscaYieldSurface"
                  // in the Arenisca manual). In this case, the fast returned position is the vertex.
                  stress_iteration = Identity*(PI1_h_over_FSLOPE)/3.0;
                  // Fast return algorithm in other cases (see "fig:AreniscaYieldSurface"
                  // in the Arenisca manual). In this case, the radial fast returning is used.
                  double I1_near_PEAKI1=PEAKI1_hardening/FSLOPE-1e-11*clen;
        #ifdef JC_DEBUG_PARTICLE // Print shifted trial stress for current subcycle
                  if(pParticleID[idx]==JC_DEBUG_PARTICLE){
                    cout << ",I1_near_PEAKI1=" <<I1_near_PEAKI1;
                  }
        #endif
                  stress_iteration = stress_iteration + S_iteration*
                                  ((PEAKI1_hardening-FSLOPE*I1_near_PEAKI1)/
                                   sqrt(J2_iteration)-1);

                }
                else if ( (I1_iteration<pKappa_loop-0.9*cap_radius)
                         || (I1_iteration<pKappa_loop && J2_iteration<0.01) ){  //2nd outside cap
        #ifdef JC_DEBUG_PARTICLE // Print shifted trial stress for current subcycle
                  if(pParticleID[idx]==JC_DEBUG_PARTICLE)
                    cout << ",FR2"<< ",I1_iter="<<I1_iteration<<",J2_iter="<<J2_iteration;
        #endif
                  // Fast return algorithm in the case of I1<X+0.1R (see "fig:CapEccentricity"
                  // in the Arenisca manual) OR ( I1<\kappa && J2<0.01)

                  // Declare some needed variables for the fast return algorithm.
                  Matrix3 stress_iteration_temp;
                  double I1_iteration1;
                  double I1_iteration2;
                  double I1_iteration3;
                  double I1_diff;
                  double f_iteration2;
                  int counter_temp=0;

                  // Compute the invariants of the fast returned stress in the loop
                  computeInvariants(stress_iteration, S_iteration, I1_iteration, J2_iteration);

                  // Compute the yield function of the fast returned stress in the loop
                  f_iteration2=YieldFunction(stress_iteration,FSLOPE,pKappa_loop,
                                             cap_radius,PEAKI1_hardening);

                  if(f_iteration2<0.0){

                    // If the fast returned stress in the loop is inside the yield surface,
                    // find two stress positions, one should be inside the yield surface and another
                    // should be outside the yield surface.

                    // \kappa-2R is outside the yield surface
                    I1_iteration1=pKappa_loop-2.0*cap_radius;

                    // I1_iteration is inside the yield surface
                    I1_iteration2=I1_iteration;

                  }else{

                    // If the fast returned stress in the loop is not inside the yield surface,
                    // find two stress positions, one should be inside the yield surface and another
                    // should be outside the yield surface.

                    // I1_iteration is outside the yield surface
                    I1_iteration1=I1_iteration;

                    // We start with a value for I1_iteration2 which is guessed to be inside
                    // the yield surface.
                    if (pKappa_loop>PI1_h_over_FSLOPE) {
                      I1_iteration2=(pKappa_loop-cap_radius + PI1_h_over_FSLOPE)/2.0;
                    } else {
                      I1_iteration2=pKappa_loop;
                    }

                    // Check if the selected value for I1_iteration2 is inside the yield surface or not?
                    stress_iteration_temp = stress_iteration + Identity*I1_iteration*one_third*
                                             (I1_iteration2/I1_iteration-1.0);
                    f_new_loop=YieldFunction(stress_iteration_temp,FSLOPE,pKappa_loop,
                                                 cap_radius,PEAKI1_hardening);

                    if (f_new_loop>=0.0){  //outside yield surface

                      // If the selected value for I1_iteration2 is not inside the yield surface,
                      // find a suitable value for I1_iteration2 which is inside the yield surface.
                      Matrix3 S_iteration_temp;
                      double I1_iteration_temp;
                      double J2_iteration_temp;
                      double var1=1.0;

                      // Compute the invariants of the stress related to I1_iteration2
                      computeInvariants(stress_iteration_temp, S_iteration_temp, I1_iteration_temp, J2_iteration_temp);
                      Matrix3 stress_iteration_temp_old = stress_iteration_temp;

                      // Loop to find a suitable value for I1_iteration2 which is inside the yield surface.
                      int counter_I1_iteration2=0;
                      while (f_new_loop>=0.0){

                        // If after 1000 cycles, a suitable value for I1_iteration2 which is inside the
                        // yield surface is not found, an error should be reported.
                        counter_I1_iteration2++;

                        if (counter_I1_iteration2>1000) {
                          cout<<"ERROR! in fast return algorithm"<<endl;
                          cout<<"pParticleID="<<pParticleID[idx]<<endl;
                          cout<<"deformationGradient_new="<<deformationGradient_new[idx]<<endl<<endl;
                          throw InvalidValue("**ERROR**:in fast return algorithm",
                                                               __FILE__, __LINE__);
                        }

                        // Compute a new value for the stress related to I1_iteration2
                        beta_cap = sqrt( 1.0 - (pKappa_loop-I1_iteration_temp)*(pKappa_loop-I1_iteration_temp)/
                                 ( (cap_radius)*(cap_radius) ) );
                        var1=var1*0.5;
                        stress_iteration_temp = stress_iteration_temp_old + S_iteration_temp*(sqrt(var1)-1);

                        // Compute the yield function at the stress related to I1_iteration2
                        f_new_loop=YieldFunction(stress_iteration_temp,FSLOPE,pKappa_loop,cap_radius,PEAKI1_hardening);

                      }

                      // Update the fast returned stress in the loop
                      beta_cap = sqrt( 1.0 - (pKappa_loop-I1_iteration)*(pKappa_loop-I1_iteration)/
                               ( (cap_radius)*(cap_radius) ) );
                      stress_iteration = stress_iteration + S_iteration*(sqrt(var1)-1);
                      computeInvariants(stress_iteration, S_iteration, I1_iteration, J2_iteration);
                      f_new_loop=YieldFunction(stress_iteration,FSLOPE,pKappa_loop,cap_radius,PEAKI1_hardening);

                      // If the fast returned stress in the loop is inside the yield surface,
                      // re-compute I1_iteration1 and I1_iteration2.
                      if (f_new_loop<=0.0){
                        I1_iteration1=pKappa_loop-2.0*cap_radius;
                        I1_iteration2=I1_iteration;
                      }

                    }
                  }
                  // Compute the mid-value between I1_iteration1 and I1_iteration2
                  I1_diff=I1_iteration2-I1_iteration1;
                  I1_iteration3=I1_iteration1+(I1_diff)*0.5;

                  // Compute the stress related to I1_iteration3
                  stress_iteration_temp = stress_iteration + Identity*I1_iteration*one_third
                                              *(I1_iteration3/I1_iteration-1.0);

                  // Compute the invariants of the stress related to I1_iteration3
                  f_iteration2=YieldFunction(stress_iteration_temp,FSLOPE,pKappa_loop,
                                                            cap_radius,PEAKI1_hardening);

                  // Loop to finally find the fast returned back stress

                  int counter_temp_max=100;
                  while ((I1_diff>1.0e-11*clen && counter_temp<counter_temp_max)
                         || counter_temp==0) {  //bisector/midpoint zero finding alg

                    counter_temp++;

#ifdef JC_DEBUG_PARTICLE // Print shifted trial stress for current subcycle
                    if(pParticleID[idx]==JC_DEBUG_PARTICLE
                        && counter_temp>=counter_temp_max)
                      cout << ",maxiter_fastretbackstr";
            #ifdef JC_DEBUG_FR_OUTSIDE_CAP
                    if(pParticleID[idx]==JC_DEBUG_PARTICLE)
                      cout << endl << "    c_t=" << counter_temp << ",f_iteration2=" << f_iteration2
                           << ",I1_iterL=" <<I1_iteration1 <<",I1_iterM="<<I1_iteration3
                           << ",I1_iterH="<<I1_iteration2<<",H-L="<<I1_iteration2-I1_iteration1;
            #endif
#endif
                    if (f_iteration2<0.0){

                      // I1_iteration1 is outside the yield surface and I1_iteration2 is inside the
                      // yield surface. We want to find a point, between these two points, which is
                      // on the yield surface. If "I1_iteration3=(I1_iteration1+I1_iteration2)*0.5"
                      // is inside the yield surface, we put I1_iteration2=I1_iteration3.
                      I1_iteration2=I1_iteration3;
                    } else {

                      // I1_iteration1 is outside the yield surface and I1_iteration2 is inside the
                      // yield surface. We want to find a point, between these two points, which is
                      // on the yield surface. If "I1_iteration3=(I1_iteration1+I1_iteration2)*0.5"
                      // is outside the yield surface, we put I1_iteration1=I1_iteration3.
                      I1_iteration1=I1_iteration3;
                    }
                    // Compute the mid-value between I1_iteration1 and I1_iteration2
                    I1_diff=I1_iteration2-I1_iteration1;
                    I1_iteration3=I1_iteration1+(I1_diff)*0.5;

                    // Compute the stress related to I1_iteration3
                    stress_iteration_temp = stress_iteration + Identity*I1_iteration*one_third
                                                      *(I1_iteration3/I1_iteration-1.0);

                  // Compute the invariants of the stress related to I1_iteration3
                    f_iteration2=YieldFunction(stress_iteration_temp,FSLOPE,pKappa_loop,
                                                        cap_radius,PEAKI1_hardening);

                  }
                  if(sqrt(abs(f_iteration2))<2.0e-6*clen)  //bisection method worked
                    stress_iteration = stress_iteration_temp;
                  else{  //bisection method ill-conditioned
          #ifdef JC_DEBUG_PARTICLE // Print shifted trial stress for current subcycle
                    //if(pParticleID[idx]==JC_DEBUG_PARTICLE)
                    //  cout << ",ILL_COND_BISECTION";
          #endif
            #ifdef JC_DEBUG_FR_OUTSIDE_CAP
                    cerr<<"ERROR: Fast returned stress not on yield surface"<<endl;
                    cerr<<"ParticleID = " << pParticleID[idx] << endl<<endl;
                    cerr<<"f_iteration2 = " << f_iteration2<<endl; 
                    cerr<< "1.e-11*clen="<<1.e-11*clen<<endl;
                    cerr<<"pKappa_loop="<<pKappa_loop<<endl;
                    cerr<<"R="<<cap_radius<<endl;
                    cerr<<"PEAKI1="<<PEAKI1_hardening/FSLOPE<<endl;
                    cerr<<"FSLOPE="<<FSLOPE<<endl;
                    cerr<<"stress_iteration="<<stress_iteration<<endl;
                    throw InvalidValue("**ERROR**: Fast returned stress not on yield surface", __FILE__, __LINE__);
            #else
                    //cerr<<"WARNING: Fast returned stress not on yield surface for pID="<<pParticleID[idx]<<endl;
            #endif
                    beta_cap = sqrt( 1.0 - (pKappa_loop-I1_iteration2)*(pKappa_loop-I1_iteration2)/
                                     ( (cap_radius)*(cap_radius) ) );
                    stress_iteration = stress_iteration + S_iteration*
                                       ((PEAKI1_hardening-FSLOPE*I1_iteration2)*
                                        beta_cap/sqrt(J2_iteration)-1);
                    stress_iteration = stress_iteration_temp;//T2D check this
                  }
                }
                else if (I1_iteration<pKappa_loop){  //3rd above cap
            #ifdef JC_DEBUG_PARTICLE // Print shifted trial stress for current subcycle
                  if(pParticleID[idx]==JC_DEBUG_PARTICLE)
                    cout << ",FR3";
            #endif
                  // Fast return algorithm in the case of I1<\kappa (see "fig:AreniscaYieldSurface"
                  // in the Arenisca manual). In this case, the radial fast returning is used.
                  beta_cap = sqrt( 1.0 - (pKappa_loop-I1_iteration)*(pKappa_loop-I1_iteration)/
                           ( (cap_radius)*(cap_radius) ) );
                  stress_iteration = stress_iteration + S_iteration*
                                     ((PEAKI1_hardening-FSLOPE*I1_iteration)*
                                      beta_cap/sqrt(J2_iteration)-1);

                }
                else{  //4th above normal drucker-p
            #ifdef JC_DEBUG_PARTICLE // Print shifted trial stress for current subcycle
                  if(pParticleID[idx]==JC_DEBUG_PARTICLE)
                    cout << ",FR4";
            #endif
                  // Fast return algorithm in other cases (see "fig:AreniscaYieldSurface"
                  // in the Arenisca manual). In this case, the radial fast returning is used.
                     stress_iteration = stress_iteration + S_iteration*
                                      ((PEAKI1_hardening-FSLOPE*I1_iteration)/
                                       sqrt(J2_iteration)-1);

                }
                if (isnan(stress_iteration.Trace())) {  //Check stress_iteration for nan
                  cerr << "ParticleID = " << pParticleID[idx] << " stress_iteration = " << stress_iteration << endl;
                  throw InvalidValue("**ERROR**: Nan in stress_iteration value", __FILE__, __LINE__);
                }
                // Compute the invariants of the fast returned stress in the loop
                computeInvariants(stress_iteration, S_iteration, I1_iteration, J2_iteration);
//end of fast return
                if (I1_iteration>=pKappa_loop){

                  // Compute the gradient of the yield surface and the unit tensor in the
                  // direction of the plastic strain at the fast returned stress for the case
                  // of I1>=\kappa (see "fig:AreniscaYieldSurface" in the Arenisca manual).
                  // Also see Eqs. 14, 15, 17, and 18 in 'Brannon & Leelavanichkul 2010'.
                  G = Identity*(-2.0)*FSLOPE*(FSLOPE*I1_iteration-PEAKI1_hardening) + S_iteration;
          //#ifdef JC_DEBUG_PARTICLE // Print shifted trial stress for current subcycle
          //        if(pParticleID[idx]==JC_DEBUG_PARTICLE)
          //          cout << endl<<"BEFORE:FSLOPE_p="<<FSLOPE_p<<",FSLOPE="<<FSLOPE<<",I1_iteration="<<I1_iteration
          //               << ",PEAKI1_hardening="<<PEAKI1_hardening<<",S_iteration="<<S_iteration
          //               <<",F*I1-PI1h="<<FSLOPE*I1_iteration-PEAKI1_hardening
          //               <<",F*I1="<<FSLOPE*I1_iteration<<endl;
          //#endif
                  M = Identity*(-2.0)*FSLOPE_p*(FSLOPE*I1_iteration-PEAKI1_hardening) + S_iteration;
          #ifdef JC_DEBUG_PARTICLE // Print shifted trial stress for current subcycle
                  if(pParticleID[idx]==JC_DEBUG_PARTICLE)
                    cout << endl<<"M="<<M<<",M.Norm()="<<M.Norm()<<",F*I1-PI1h="<<FSLOPE*I1_iteration-PEAKI1_hardening<<endl;
          #endif
                  M = M/M.Norm();
          #ifdef JC_DEBUG_PARTICLE // Print shifted trial stress for current subcycle
                  if(pParticleID[idx]==JC_DEBUG_PARTICLE)
                    cout << ",FSLOPE_p="<<FSLOPE_p<<",FSLOPE="<<FSLOPE<<",I1_iteration="<<I1_iteration
                         << ",PEAKI1_hardening="<<PEAKI1_hardening<<",S_iteration="<<S_iteration<<endl;
          #endif

                }else{

                  // Compute the gradient of the yield surface and the unit tensor in the
                  // direction of the plastic strain at the fast returned stress for the case
                  // of I1<\kappa (see "fig:AreniscaYieldSurface" in the Arenisca manual).
                  // Also see Eqs. 14, 15, 17, and 18 in 'Brannon & Leelavanichkul 2010'.
                  beta_cap = 1.0 - (pKappa_loop-I1_iteration)*(pKappa_loop-I1_iteration)/
                             ( (cap_radius)*(cap_radius) );
                  double FS_I1_i_PI1_h = FSLOPE*I1_iteration-PEAKI1_hardening;
                  FSLOPE_cap = -2.0*(FS_I1_i_PI1_h)*(FS_I1_i_PI1_h)
                                   *(pKappa_loop-I1_iteration)/( cap_radius*cap_radius ) 
                               -2.0*FSLOPE*beta_cap*(FSLOPE*I1_iteration-PEAKI1_hardening);
                  G = Identity*FSLOPE_cap + S_iteration;
                  M = G/G.Norm();
                  if (G.Norm()<1.e-10) {
                    Matrix3 var_Mat3(0.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0,0.0);
                    G = Identity*2.0 + var_Mat3/sqrt_three;
                    M = Identity*2.0 + var_Mat3/sqrt_three;
                  }
                  M = M/M.Norm();

                }
                if (isnan(M.Trace())) {  //Check M for nan
                  cerr << "ERRORpID = " << pParticleID[idx] << ", pKappa_loop= " << pKappa_loop 
                       <<", I1_iteration=" << I1_iteration << endl;
                  throw InvalidValue("**ERROR**: Nan in M value", __FILE__, __LINE__);
                }
                if (isnan(G.Trace())) {  //Check G for nan
                  cerr << "ERRORpID = " << pParticleID[idx] << ", pKappa_loop= " << pKappa_loop 
                       <<", I1_iteration=" << I1_iteration << endl;
                  throw InvalidValue("**ERROR**: Nan in M value", __FILE__, __LINE__);
                }

                // Compute the back stress tensor
                double deltaBackStressIso_temp = exp(p3_crush_curve+pPlasticStrainVol_new[idx]);
                deltaBackStress = stress_iteration*kinematic_hardening_constant;
                deltaBackStressIso = Identity*( 3.0*fluid_B0*0*
                                    (exp(p3_crush_curve+p4_fluid_effect)-1.0) * deltaBackStressIso_temp
                                    /( (deltaBackStressIso_temp-1.0)*(deltaBackStressIso_temp-1.0) ) );

                // Compute the projection direction tensor at the fast returned stress position
                // See Eq. 24 in 'Brannon & Leelavanichkul 2010'.
                double I1_M,J2_M;
                Matrix3 S_M;
                computeInvariants(M, S_M, I1_M, J2_M);
                P = (Identity*lame*(M.Trace()) + M*2.0*shear)
                    -deltaBackStressIso*M.Trace()-deltaBackStress*sqrt(J2_M);

                // Compute the multiplier Gamma
                // See Eq. 35 in 'Brannon & Leelavanichkul 2010'.
                   gamma = ( G.Contract(trial_stress_loop-stress_iteration) )/( G.Contract(P) );

                // Loop to apply hardening in calculation of multiplier Gamma
                int condGamma = 1;
                double pKappa_loop_old = pKappa_loop;
                double gamma_old=gamma;
                int counter_gamma1=0;
                double hardeningEns=-1e99;
                while (condGamma == 1) {

                  // If after 1000 cycles, gamma is not converged, an error should be reported.
                  counter_gamma1++;
                  if (counter_gamma1>1000) {
                    cerr<<"ERROR! in nested retuen algorithm"<<endl;
                    cerr<<"pParticleID="<<pParticleID[idx]<<endl;
                    cerr<<"deformationGradient_new="<<deformationGradient_new[idx]<<endl<<endl;
                    cerr<<"hardeningEns="<<hardeningEns<<endl;
                    cerr<<"f_new_loop="<<f_new_loop<<endl;
                    cerr<<"sqrt(abs(f_new_loop))="<<sqrt(abs(f_new_loop))<<endl;
                    cerr<<"1e-4*clen="<<1e-4*clen<<endl;
                    cerr<<"pKappa_loop="<<pKappa_loop<<endl;
                    cerr<<"R="<<cap_radius<<endl;
                    cerr<<"gamma="<<gamma<<endl;
                    cerr<<"gamma_old="<<gamma_old<<endl;
                    cerr<<"M="<<M<<endl;
                    cerr<<"stress_iteration="<<stress_iteration<<endl;
                    throw InvalidValue("**ERROR**:in nested retuen algorithm",
                                                         __FILE__, __LINE__);
                  }

                  // Compute new trial stress for the current subcycle in the loop
                  // See Eq. 22 in 'Brannon & Leelavanichkul 2010'.
                     stress_iteration = trial_stress_loop - P*gamma;

                  // Compute the un-shifted new trial stress for the current subcycle in the loop
                  stress_iteration = stress_iteration + pBackStress_loop;
                  trial_stress_loop = trial_stress_loop + pBackStress_loop;

                  double I1_plasStrain,J2_plasStrain;
                  Matrix3 S_plasStrain;

                  // Compute the plastic strain increment based on the unit tensor in the
                  // direction of the plastic strain and the multiplier Gamma
                  plasStrain_loop = M*gamma;
                  if (isnan(plasStrain_loop.Trace())) {  //Check plasStrain_loop for nan
                    cerr << "ERRORpID = " << pParticleID[idx] << ", M = " << M <<", gamma=" << gamma << endl;
                    throw InvalidValue("**ERROR**: Nan in plasStrain_loop value", __FILE__, __LINE__);
                  }

                  // Compute the invariants of the plastic strain increment
                  computeInvariants(plasStrain_loop, S_plasStrain, I1_plasStrain, J2_plasStrain);

                  // Compute back stress increments
                  deltaBackStressIso_temp = exp(p3_crush_curve+(pPlasticStrainVol_new[idx]
                                             +plasStrain_loop.Trace()));
                  deltaBackStress = stress_iteration*kinematic_hardening_constant
                                    *sqrt(J2_plasStrain);
                  deltaBackStressIso = Identity*( -3.0*fluid_B0*
                                      ((exp(pPlasticStrainVol_new[idx])-1.0)
                                      * exp(p3_crush_curve+p4_fluid_effect))
                                      /(exp(p3_crush_curve+p4_fluid_effect
                                      +(pPlasticStrainVol_new[idx]+plasStrain_loop.Trace()))-1.0) )
                                      *(pPlasticStrainVol_new[idx]+plasStrain_loop.Trace());
                 
                  pBackStress_loop = deltaBackStressIso;
                  
                  if (isnan(pBackStress_loop.Trace())) {  //Check pBackStress_loop for nan
                    cerr << "ERRORpID = " << pParticleID[idx] << " pBackStress_loop = " << pBackStress_loop << endl;
                    throw InvalidValue("**ERROR**: Nan in pBackStress_loop value", __FILE__, __LINE__);
                  }

                  // Compute the shifted new trial stress for the current subcycle in the loop
                  stress_iteration = stress_iteration - pBackStress_loop;
                  trial_stress_loop = trial_stress_loop - pBackStress_loop;

                  //double hardeningEns;
                  double hardeningEnsCond=-1.0;
                  double FS_I1_i_PI1_h = FSLOPE*I1_iteration-PEAKI1_hardening;

                  if (I1_iteration>=pKappa_loop){

                    // Compute the hardening ensemble for the case of I1>=\kappa 
                    // (see "fig:AreniscaYieldSurface" in the Arenisca manual).
                    // Also, see Eq. 6.53 in 'Brannon 2007'.
                    hardeningEns = -2.0*hardening_modulus*FS_I1_i_PI1_h/G.Norm();

                  }else{

                    // Compute the hardening ensemble for the case of I1<\kappa 
                    // (see "fig:AreniscaYieldSurface" in the Arenisca manual).

                    // Declare and initialize some auxiliaryvariables
                    beta_cap = 1.0 - (pKappa_loop-I1_iteration)*(pKappa_loop-I1_iteration)/
                               ( (cap_radius)*(cap_radius) );
                    double pKappa_tempA = exp(p3_crush_curve+p4_fluid_effect
                                          +pPlasticStrainVol_new[idx]+(M*gamma).Trace());
                    double pKappa_tempA1 = exp(p3_crush_curve+pPlasticStrainVol_new[idx]
                                           +(M*gamma).Trace());
                    double pKappa_tempA2;

                    if (cond_fixed_cap_radius==0) {

                      // Consider the limitation for R=\kappa-X
                      // (see "eq:limitationForR" in the Arenisca manual).
                      // cond_fixed_cap_radius is a variable which indicates
                      // if the limit has been met or not?
                      // Compute auxiliary variable in the case that the limit has not been met.
                      pKappa_tempA2 = 2.0/G.Norm()*CR*(FS_I1_i_PI1_h*FS_I1_i_PI1_h*FS_I1_i_PI1_h)
                                              *(pKappa_loop-I1_iteration)
                                              /( cap_radius*cap_radius*cap_radius*(1.0+FSLOPE*CR) );

                    } else {

                      // Compute auxiliary variable in the case that the limit has been met.
                      pKappa_tempA2 = -2.0*FS_I1_i_PI1_h*FS_I1_i_PI1_h
                                               *(pKappa_loop-I1_iteration)
                                               /( G.Norm()*cap_radius*cap_radius );

                    }

                    // Compute the hardening ensemble (see Eq. 6.53 in 'Brannon 2007').
                    hardeningEnsCond = -2.0*beta_cap*FS_I1_i_PI1_h
                                       *hardening_modulus/G.Norm()
                                   +pKappa_tempA2
                                       *( exp(-p1_crush_curve*(pKappa_loop-cap_radius-p0_crush_curve))
                                          /( p3_crush_curve*p1_crush_curve ) -
                                          3.0*fluid_B0*0*(exp(p3_crush_curve+p4_fluid_effect)-1.0)
                                          *pKappa_tempA
                                          /( (pKappa_tempA-1.0)*(pKappa_tempA-1.0) ) +
                                          3.0*fluid_B0*0*(exp(p3_crush_curve+p4_fluid_effect)-1.0)
                                          *pKappa_tempA1
                                          /( (pKappa_tempA1-1.0)*(pKappa_tempA1-1.0) ) )
                                       *M.Trace();

                    if (pKappa_loop-cap_radius-p0_crush_curve<0 || hardeningEnsCond>0) {

                      // In the case of X<p_0 (see "fig:AreniscaYieldSurface" in the Arenisca manual),
                      // consider the hardening ensemble.
                      hardeningEns = hardeningEnsCond;

                    } else {

                      // In the case of X>p_0 (see "fig:AreniscaYieldSurface" in the Arenisca manual),
                      // do not consider the full hardening ensemble. Consider only the Drucker-Prager
                      // hardening ensemble. This may slow down the convergence of the plasticity return
                      // algorithm but, it increases its robustness.
                      hardeningEns = -2.0*beta_cap*(FSLOPE*I1_iteration-PEAKI1_hardening)
                                         *hardening_modulus/G.Norm();
                    }
                  }

                  if (hardeningEns<0.0) {

                    // In the case of moving cap toward the vertex, do not consider the hardening ensemble.
                    // This may slow down the convergence of the plasticity return
                    // algorithm but, it increases its robustness.
                    hardeningEns = 0.0;
                    condGamma = 0;
                  }

                  // Re-compute the multiplier Gamma
                  // See Eq. 35 in 'Brannon & Leelavanichkul 2010'.
                  Matrix3 G_unit = G/G.Norm();
                  gamma=(G_unit.Contract(P)/( G_unit.Contract(P)+hardeningEns ))*gamma;

                  // Re-compute new trial stress for the current subcycle in the loop
                  // See Eq. 22 in 'Brannon & Leelavanichkul 2010'.
                     stress_iteration = trial_stress_loop - P*gamma;

                  // Update \kappa (= the position of the cap) see "eq:evolutionOfKappaFluidEffect"
                  // in the Arenisca manual

                  // Declare and assign some axiliary variables
                  double pKappa_temp = exp(p3_crush_curve+p4_fluid_effect+pPlasticStrainVol_new[idx]
                                       +(M*gamma).Trace());
                  double pKappa_temp1 = exp(p3_crush_curve+pPlasticStrainVol_new[idx]
                                        +(M*gamma).Trace());
                  double var1;

                  // Compute the var1 which relates dX/de to d(kappa)/de: d(kappa)/de=dX/de * (1/var1)
                  // Consider the limitation for R=\kappa-X (see "eq:limitationForR" in the Arenisca manual).
                  // 'cond_fixed_cap_radius' is a variable which indicates if the limit has been met or not?
                  if (cond_fixed_cap_radius==0) {
                    var1 = 1.0+FSLOPE*CR;
                  } else {
                    var1 = 1.0;
                  }
            #ifdef JC_USE_BB_STATE_UPDATE
                  // Update the model state
                state->local_var[0] = pKappa_loop_old;              // kappa value
                state->local_var[1] = cap_radius;                  // cap radius
                state->local_var[2] = max_X;                       // max_X
                state->local_var[3] = pPlasticStrainVol_new[idx];  // eps_v
                state->local_var[4] = (M*gamma).Trace();           // Delta eps_v
                state->local_var[5] = var1;                        // scale factor for Delta eps_v
                
                pKappa_loop = d_intvar->computeInternalVariable(state);
                
                // Set the kappa state flag
                if (pKappa_loop == (max_X + cap_radius)) pKappaFlag_new[idx] = 1.0;
                
                if (pKappa_loop<min_kappa){
                  pKappa_loop = min_kappa;
                  pKappaFlag_new[idx] = 2.0;
              #ifdef JC_LIMITER_PRINT
                  cout << "$K";
              #endif
                }
            #else
                  // Update \kappa in the case of: 
                  if (pKappa_loop-cap_radius-p0_crush_curve<0) {  //X < p0
                    // (see "fig:AreniscaYieldSurface" in the Arenisca manual)
                    // (see "eq:evolutionOfKappaFluidEffect" in the Arenisca manual)
                    pKappa_loop = pKappa_loop_old + ( exp(-p1_crush_curve
                                   *(pKappa_loop-cap_radius-p0_crush_curve))
                                   /( p3_crush_curve*p1_crush_curve ) -
                                   3.0*fluid_B0*(exp(p3_crush_curve+p4_fluid_effect)-1.0)*pKappa_temp
                                   /( (pKappa_temp-1.0)*(pKappa_temp-1.0) ) +
                                   3.0*fluid_B0*(exp(p3_crush_curve+p4_fluid_effect)-1.0)*pKappa_temp1
                                   /( (pKappa_temp1-1.0)*(pKappa_temp1-1.0) ) )
                                   *(M*gamma).Trace()/var1;
            
                    // Apply the lower limit for \kappa. 
                    // (for the limitation of min_kappa see "eq:limitationForKappa"
                    // in the Arenisca manual)
                    // pKappaFlag is a particle flag variable which defines if the particle
                    // meet any of the limitation for \kappa and X or not?
                    // pKappaFlag=2: means that the particle met the min_kappa limitation.
                    if (pKappa_loop<min_kappa){
                      pKappa_loop = min_kappa;
                      pKappaFlag_new[idx] = 2.0;
                #ifdef JC_LIMITER_PRINT
                      cout << "$K";
                #endif
                    }

                  } else if (pKappa_loop-cap_radius-max_X<0) {  //p0 <= X < max_X
                    // (see "fig:AreniscaYieldSurface" in the Arenisca manual)
                    // (see "eq:evolutionOfKappaFluidEffect1" in the Arenisca manual)
                    // (for the limitation of max_X see "eq:limitationForX" in the Arenisca manual)
                    pKappa_loop = pKappa_loop + ( pow( abs( (pKappa_loop-cap_radius)/p0_crush_curve ),
                                     1-p0_crush_curve*p1_crush_curve*p3_crush_curve )
                                     /( p3_crush_curve*p1_crush_curve ) -
                                     3.0*fluid_B0*(exp(p3_crush_curve+p4_fluid_effect)-1.0)*pKappa_temp
                                     /( (pKappa_temp-1.0)*(pKappa_temp-1.0) ) +
                                     3.0*fluid_B0*(exp(p3_crush_curve+p4_fluid_effect)-1.0)*pKappa_temp1
                                     /( (pKappa_temp1-1.0)*(pKappa_temp1-1.0) ) )
                                     *(M*gamma).Trace()/var1;

                  } else {  //X >= max_X
                    // (see "fig:AreniscaYieldSurface" in the Arenisca manual)
                    // In this case it is assumed that X=max_X
                    // (for the limitation of max_X see "eq:limitationForX" in the Arenisca manual)
                    // pKappaFlag is a particle variable variable which defines if the particle
                    // meet any of the limitation for \kappa and X or not?
                    // pKappaFlag=1: means that the particle met the max_X limitation
                    pKappa_loop = max_X + cap_radius;
                    pKappaFlag_new[idx] = 1.0;
                #ifdef JC_LIMITER_PRINT
                    cout << "$R";
                #endif
                  }
#endif
                  // Re-calculate the axiliary varable 'PEAKI1_hardening'
                  // because '(M*gamma).Norm()' has been updated 
                  PEAKI1_hardening = PEAKI1*FSLOPE + hardening_modulus*(pPlasticStrain_new[idx]
                                     +(M*gamma).Norm());

                  // Consider the limitation for R=\kappa-X (see "eq:limitationForR" in the Arenisca manual).
                  // 'cond_fixed_cap_radius' is a variable which indicates if the limit has been met or not?
                  // If the limit has been met, updated \kappa should be modified.
                  if (cond_fixed_cap_radius==0) {
                    double cap_radius_old=cap_radius;
                    cap_radius=-CR*(FSLOPE*pKappa_loop-PEAKI1_hardening);
                #ifdef JC_USE_BB_STATE_UPDATE
                    cap_radius_fac=FSLOPE*pKappa_loop-PEAKI1_hardening;
                    cap_radius=-CR*cap_radius_fac;
                #endif
                    if (cap_radius<0.1*cap_r_initial || pKappa_loop>PEAKI1_hardening/FSLOPE) {
                #ifdef JC_LIMITER_PRINT
                      cout << "$R";
                #endif
                      pKappa_loop = pKappa_loop - cap_radius_old + 0.1*cap_r_initial;
                      cap_radius=0.1*cap_r_initial;
                      cond_fixed_cap_radius=1;
                    }
                  }

                  // Apply the upper limit for X. 
                  if (pKappa_loop>max_X+cap_radius){
                    // (for the limitation of max_X see "eq:limitationForX" in the Arenisca manual)
                    // pKappaFlag is a particle variable variable which defines if the particle
                    // meet any of the limitation for \kappa and X or not?
                    // pKappaFlag=1: means that the particle met the max_X limitation
                    // If the limit has been met, updated \kappa should be modified.
                    pKappa_loop = max_X+cap_radius;
                    pKappaFlag_new[idx] = 1.0;
                #ifdef JC_LIMITER_PRINT
                    cout << "$X";
                #endif
                  }

                  // Compute the yield function at the returned back stress in the loop
                  f_new_loop=YieldFunction(stress_iteration,FSLOPE,pKappa_loop,
                                           cap_radius,PEAKI1_hardening);

                  // If the returned back stress is inside the yield surface, gamma 
                  // should be decreased.
                  if (hardeningEns>0.0 && gamma>0.0 && f_new_loop<0.0 && 
//                      sqrt(abs(f_new_loop))>1.0e-4*char_length_yield_surface ) {
                      sqrt(abs(f_new_loop))>1.0e-4*clen ) {
                    gamma = gamma/2.0;
                  } else {
                    condGamma = 0;
                  }

                }

                f_new_loop=sqrt(abs(f_new_loop));
 
    #ifdef JC_DEBUG_PARTICLE // Print shifted trial stress for current subcycle
                if(pParticleID[idx]==JC_DEBUG_PARTICLE){
                  cout << endl << "   cNR="<<counter<<",f="<<f_new_loop
                    <<",9.9e-2*clen="<<9.9e-2*clen;
                }
                if(pParticleID[idx]==JC_DEBUG_PARTICLE
                    && fabs(f_new_loop)>9.9e-2*clen && counter>max_number_of_iterations)
                  cout <<",maxNR";
    #endif
                #ifdef JC_MAX_NESTED_RETURN
                if (fabs(f_new_loop)>9.9e-2*clen && counter>max_number_of_iterations) {
                  cerr<<endl << "ERROR: Nested return loop not converged" << endl;
                  cerr<<"ParticleID = " << pParticleID[idx] << endl <<endl;
                  cerr<<"f_new_loop = " << f_new_loop << endl;
                  cerr<<"9.9e-2*clen="<<9.9e-2*clen<<endl;
                  cerr<<"pKappa_loop= "<<pKappa_loop<<endl;
                  cerr<<"gamma="<<gamma<<endl;
                  cerr<<"condGamma="<<condGamma<<endl;
                  cerr<<"PEAKI1="<<PEAKI1_hardening/FSLOPE<<endl;
                  cerr<<"R="<<cap_radius<<endl;
                  cerr<<"FSLOPE="<<FSLOPE<<endl;
                  cerr<<"stress_iteration="<<stress_iteration<<endl;
                  Matrix3 S_iteration_err;
                  double I1_iteration_err,J2_iteration_err;
                  computeInvariants(stress_iteration, S_iteration_err, I1_iteration_err, J2_iteration_err);
                  cerr<<"I1_iter="<<I1_iteration_err<<endl;
                  cerr<<"J2_iter="<<J2_iteration_err<<endl;
                  throw InvalidValue("**ERROR**: Nested return loop not converged", __FILE__, __LINE__);
                }
                #endif

              }//END OF NR LOOP
              if (isnan(pBackStress_loop.Trace())) {  //Check pBackStress_loop for nan
                cerr << "ERRORpID = " << pParticleID[idx] << " pBackStress_loop = " << pBackStress_loop << endl;
                throw InvalidValue("**ERROR**: Nan in pBackStress_loop value", __FILE__, __LINE__);
              }
              if (isnan(pKappa_loop)) {  //Check pKappa_loop for nan
                cerr << "ERRORpID = " << pParticleID[idx] << " pKappa_loop = " << pKappa_loop << endl;
                throw InvalidValue("**ERROR**: Nan in pKappa_loop value", __FILE__, __LINE__);
              }
              if (isnan(stress_iteration.Trace())) {  //Check stress_iteration for nan
                cerr << "ERRORpID = " << pParticleID[idx] << " stress_iteration = " << stress_iteration << endl;
                throw InvalidValue("**ERROR**: Nan in stress_iteration value", __FILE__, __LINE__);
              }
              if (isnan(trial_stress_loop.Trace())) {  //Check trial_stress_loop for nan
                cerr << "ERRORpID = " << pParticleID[idx] << " trial_stress_loop = " << trial_stress_loop << endl;
                throw InvalidValue("**ERROR**: Nan in trial_stress_loop value", __FILE__, __LINE__);
              }

              // Transfer the back stress, \kappa, and final stress in the current subcycle
              // to the associated particle variables
              pBackStress_new[idx] = pBackStress_loop;
              pKappa_new[idx] = pKappa_loop;
              stress_new[idx] = stress_iteration;

              // Compute two coefficients that are used in calculation of strain from stress
              double shear_inverse = 0.5/shear;
              double nu_over_E = (-1.0)*lame/(2.0*shear*(2.0*shear+3.0*lame));

              // Compute the difference between the stress tensor at the beginning and end of
              // the current subcycle
              Matrix3 diff_stress_iteration = trial_stress_loop - stress_new[idx];
              Matrix3 strain_iteration = (Identity*nu_over_E*(diff_stress_iteration.Trace()) +
                                          diff_stress_iteration*shear_inverse);
              if (isnan(trial_stress_loop.Trace())) {  //Check trial_stress_loop for nan
                cerr << "ERRORpID = " << pParticleID[idx] << " trial_stress_loop = " << trial_stress_loop << endl;
                throw InvalidValue("**ERROR**: Nan in trial_stress_loop value", __FILE__, __LINE__);
              }

              // Update the plastic strain magnitude
              pPlasticStrain_new[idx] = pPlasticStrain_new[idx] + strain_iteration.Norm();

              // Update the volumetric part of the plastic strain
              pPlasticStrainVol_new[idx] = pPlasticStrainVol_new[idx] + strain_iteration.Trace();
  #ifdef JC_DEBUG_PARTICLE  // Print number of subcycles
              if(pParticleID[idx]==JC_DEBUG_PARTICLE)
                cout << ",epv="<<pPlasticStrainVol_new[idx];
  #endif                 
  #ifdef JC_EPV
              if (pPlasticStrainVol_new[idx]<-p3_crush_curve) {
                cout<<"ERROR! in pPlasticStrainVol_new"<<endl;
                cout<<"pParticleID="<<pParticleID[idx]<<endl;
                cout<<"deformationGradient_new="<<deformationGradient_new[idx]<<endl<<endl;
                cout<<"epv=" << pPlasticStrainVol_new[idx]<<endl;
                cout<<"pKappa_new[idx]="<<pKappa_new[idx]<<endl;
                throw InvalidValue("**ERROR**:in pPlasticStrainVol calculation",
                __FILE__, __LINE__);
              }
  #endif
              // Update the volumetric part of the elastic strain
              pElasticStrainVol_new[idx] = pElasticStrainVol_new[idx] - strain_iteration.Trace();
                                           
              stress_new[idx] = stress_new[idx] + pBackStress_new[idx];//---------------

#ifdef JC_DEBUG_PARTICLE // Print end of subcyclcling routine for non-vertex treatment
             // if(pParticleID[idx]==JC_DEBUG_PARTICLE)
             //   cout << ",eoSC=" << subcycle_counter <<endl;
#endif
            }  //END Nested Return Routine for non-vertex treatment 
//            // Compute the shifted stress
//            stress_new[idx] = stress_new[idx] - pBackStress_new[idx];
          }  //END OF SUBCYCLING ROUTINE 
          // Compute the shifted stress
          stress_new[idx] = stress_new[idx] - pBackStress_new[idx];

          // Compute the yield function at the returned back stress to check
          // if it correctly returned back to the yield surface or not?
          double f_new=YieldFunction(stress_new[idx],FSLOPE,pKappa_new[idx],
                                      cap_radius,PEAKI1_hardening);
          f_new=sqrt(abs(f_new));

          // Compute the invariants of the returned stress
          double J2_new,I1_new;
          Matrix3 S_new;
          computeInvariants(stress_new[idx], S_new, I1_new, J2_new);

          // If X is larger than PEAKI1 send error message to the host code.
          if (pKappa_new[idx]-cap_radius>PEAKI1_hardening/FSLOPE) {
            cerr<<"ERROR! pKappa-R>PEAKI1 "<<endl;
            cerr<<"pParticleID="<<pParticleID[idx]<<endl;
            cerr<<"deformationGradient_new="<<deformationGradient_new[idx]<<endl<<endl;
            cerr<<"J2_new= "<<J2_new<<endl;
            cerr<<"I1_new= "<<I1_new<<endl;
            cerr<<"pKappa_new[idx]= "<<pKappa_new[idx]<<endl;
            cerr<<"f_new= "<<f_new<<endl;
            cerr<<"clen= "<<clen<<endl;
            cerr<<"PEAKI1="<<PEAKI1_hardening/FSLOPE<<endl;
            cerr<<"R="<<cap_radius<<endl;
            cerr<<"FSLOPE="<<FSLOPE<<endl;
            throw InvalidValue("**ERROR**:pKappa-R>PEAKI1 ", __FILE__, __LINE__);
          }

          // If the new stress is not on the yield surface send error message to the host code.s
          if (sqrt(abs(f_new))>1.0e-1*clen)
          {
            cerr<<"ERROR!  did not return to <1e-1 yield surface (Arenisca.cc)"<<endl;
            cerr<<"pParticleID="<<pParticleID[idx]<<endl;
            cerr<<"deformationGradient_new="<<deformationGradient_new[idx]<<endl<<endl;
            cerr<<"deformationGradient_old="<<deformationGradient[idx]<<endl;
            //cerr<<"condition_return_to_vertex="<<condition_return_to_vertex<<endl;
            cerr<<"J2_new= "<<J2_new<<endl;
            cerr<<"I1_new= "<<I1_new<<endl;
            cerr<<"pKappa_new[idx]= "<<pKappa_new[idx]<<endl;
            cerr<<"pKappaFlag_new[idx]= "<<pKappaFlag_new[idx]<<endl;
            cerr<<"f_new= "<<f_new<<endl;
            cerr<<"clen= "<<clen<<endl;
            cerr<<"PEAKI1="<<PEAKI1_hardening/FSLOPE<<endl;
            cerr<<"R="<<cap_radius<<endl;
            cerr<<"FSLOPE="<<FSLOPE<<endl;
            cerr<<"pvelocity[idx]="<<pvelocity[idx]<<endl;
            cerr<<"rho_cur[idx]="<<rho_cur[idx]<<endl;
            cerr<<"pBackStress_new[idx]="<<pBackStress_new[idx]<<endl;
            cerr<<"stress_old[idx]="<<stress_old[idx]<<endl;
            cerr<<"stress_new[idx]="<<stress_new[idx]<<endl;
          throw InvalidValue("**ERROR**:did not return to yield surface ",
                             __FILE__, __LINE__);
          }
        }

        // Compute the unshifted stress from the shifted stress
        stress_new[idx] = stress_new[idx] + pBackStress_new[idx];

        // Compute the total strain energy and the stable timestep based on both
        // the particle velocities and wave speed.

        // Use polar decomposition to compute the rotation and stretch tensors
        deformationGradient_new[idx].polarDecompositionRMB(tensorU, tensorR);
        rotation[idx]=tensorR;

        // Compute the rotated stress at the end of the current timestep
        stress_new[idx] = (rotation[idx]*stress_new[idx])*(rotation[idx].Transpose());

        // Compute wave speed + particle velocity at each particle,
        // store the maximum
        c_dil = sqrt((bulk+four_third*shear)/(rho_cur[idx]));
#ifdef JC_DEBUG_SMALL_TIMESTEP
        if(c_dil+fabs(pvelocity[idx].x()) > WaveSpeed.x())
          {
          idvel.x(idx);
          vbulk.x(bulk);
          vshear.x(shear);
          }
        if(c_dil+fabs(pvelocity[idx].y()) > WaveSpeed.y())
        {
          idvel.y(idx);
          vbulk.y(bulk);
          vshear.y(shear);
        }
        if(c_dil+fabs(pvelocity[idx].z()) > WaveSpeed.z())
        {
          idvel.z(idx);
          vbulk.z(bulk);
          vshear.z(shear);
        }
#endif
        WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                         Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                         Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));

        // Compute artificial viscosity term
        if (flag->d_artificial_viscosity) {
          double dx_ave = (dx.x() + dx.y() + dx.z())*one_third;
          double c_bulk = sqrt(bulk/rho_cur[idx]);
          p_q[idx] = artificialBulkViscosity(D.Trace(), c_bulk, rho_cur[idx], dx_ave);
        } else {
          p_q[idx] = 0.;
        }

        // Compute the averaged stress
        Matrix3 AvgStress = (stress_new[idx] + stress_old[idx])*0.5;
                            
  #ifdef JC_DEBUG_PARTICLE // Print plastic work //T2D UNFINISHED!!
        // Verify plastic work is positive
        //Given: AvgStress, delT, stress_new, stress_old, bulk, shear,
        if(pParticleID[idx]==JC_DEBUG_PARTICLE){
          Matrix3 AvgStressRate = (stress_new[idx] + stress_old[idx])/delT;
          Matrix3 AvgStressRateIso = one_third*AvgStressRate.Trace()*Identity;
          Matrix3 EdE = D-0.5/shear*(AvgStressRate-AvgStressRateIso)-one_third/bulk*AvgStressRateIso;
          Matrix3 PSR = (pPlasticStrain_new[idx] + pPlasticStrain[idx])/delT;  //AvgPlasticStrainRate
          double pPlasticWork = (PSR(0,0)*AvgStress(0,0) +
                                 PSR(1,1)*AvgStress(1,1) +
                                 PSR(2,2)*AvgStress(2,2) +
                             2.*(PSR(0,1)*AvgStress(0,1) +
                                 PSR(0,2)*AvgStress(0,2) +
                                 PSR(1,2)*AvgStress(1,2)));
          //cout << ",pPlasticWork=" << pPlasticWork;
        }
  #endif

        // Compute the strain energy associated with the particle
        double e = (D(0,0)*AvgStress(0,0) +
                    D(1,1)*AvgStress(1,1) +
                    D(2,2)*AvgStress(2,2) +
                2.*(D(0,1)*AvgStress(0,1) +
                    D(0,2)*AvgStress(0,2) +
                    D(1,2)*AvgStress(1,2))) * pvolume[idx]*delT;

        // Accumulate the total strain energy
        se += e;
#ifdef JC_DEBUG_PARTICLE
      if(pParticleID[idx]==JC_DEBUG_PARTICLE)
        cout << endl;

#endif

    }
#ifdef JC_DEBUG_PARTICLE // Undefine
#undef JC_DEBUG_PARTICLE
#endif
                                         
    // Compute the stable timestep based on maximum value of
    // "wave speed + particle velocity"
    WaveSpeed = dx/WaveSpeed; //Variable now holds critical timestep (not speed)

    double delT_new = WaveSpeed.minComponent();
#ifdef JC_DEBUG_SMALL_TIMESTEP
        //cout <<"delT_new="<<delT_new;
        //cout <<"dx="<<dx<<endl;
    if(delT_new==WaveSpeed.x()){
      cout << "pvel.x=" << pvelocity[idvel.x()].x() 
           << ",wavespeed.x=" << WaveSpeed.x()
           << ",bulk=" << vbulk.x() 
           << ",rho=" << rho_cur[idvel.x()]  << endl;
    }
    else if(delT_new==WaveSpeed.y()){
    cout << "pvel.y: " << pvelocity[idvel.y()].y() 
         << ",wavespeed.y=" << WaveSpeed.y()
         << ",bulk=" << vbulk.y() 
         << ",rho=" << rho_cur[idvel.y()] << endl;
    }
    else if(delT_new==WaveSpeed.z()){
      cout << "pvel.z: " << pvelocity[idvel.z()].z() 
           << ",wavespeed.z=" << WaveSpeed.z()
           << ",bulk=" << vbulk.z() 
           << ",rho=" << rho_cur[idvel.z()]  << endl;
    }
    else
      cout << "ERROR in JC_DEBUG_SMALL_TIMESTEP" <<endl;
#endif
  
    // Put the stable timestep and total strain enrgy
    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(se),        lb->StrainEnergyLabel);
    }

    delete interpolator;

  }
} // -----------------------------------END OF COMPUTE STRESS TENSOR FUNCTION

void Arenisca::computeInvariants(const Matrix3& stress, Matrix3& S,  double& I1, double& J2)
{
  // Compute the invariants of a second-order tensor

  Matrix3 Identity;
  Identity.Identity();

  // Compute the first invariants
  I1 = stress.Trace();  //Pa

  // Compute the deviatoric part of the tensor
  S = stress - Identity*(I1/3.0);  //Pa

  // Compute the first invariants
  J2 = 0.5*S.Contract(S);  //Pa^2
}

double Arenisca::YieldFunction(const Matrix3& stress, const double& FSLOPE, const double& kappa,
                               const double& cap_radius, const double&PEAKI1)
{
  // Compute the yield function.
  // See "fig:AreniscaYieldSurface" in the Arenisca manual.

  Matrix3 S;
  double I1,J2,b,var1,var2;

  // Compute the invariants of the stress tensor
  computeInvariants(stress,S,I1,J2);

  // Compute an auxiliary variable
  double FSI1_PI1 =  FSLOPE*I1 - PEAKI1;

  if (I1>kappa){

    // If I1>kappa, a linear Drucker-Prager yield function is calculated.
    // See "eq:DruckerPragerPart" in the Arenisca manual.
    double sqrtJ2=sqrt(J2);
    var1 = sqrtJ2 - FSI1_PI1;
    var2 = sqrtJ2 + FSI1_PI1;
    if (var1<0.0 && var2>0.0 ) {
      return -var1*var2;
    } else {
      return var1*var2;
    }

  }else{

    // If I1<kappa, the yield function is obtained by multiplying the linear Drucker-Prager
    // yield function and a cap function.
    // See "eq:CapPartOfTheYieldSurface" in the Arenisca manual.
    b = 1.0 - ((kappa-I1)*(kappa-I1))/((cap_radius)*(cap_radius));
    return J2 - b * FSI1_PI1*FSI1_PI1;

  }
}

void Arenisca::addRequiresDamageParameter(Task* task,
                                     const MPMMaterial* matl,
                                     const PatchSet* ) const
{
  // Require the damage parameter
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::NewDW, pLocalizedLabel_preReloc,matlset,Ghost::None);
}


void Arenisca::getDamageParameter(const Patch* patch,
                             ParticleVariable<int>& damage,
                             int dwi,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw)
{
  // Get the damage parameter
  ParticleSubset* pset = old_dw->getParticleSubset(dwi,patch);
  constParticleVariable<int> pLocalized;
  new_dw->get(pLocalized, pLocalizedLabel_preReloc, pset);

  ParticleSubset::iterator iter;
  // Loop over the particle in the current patch.
  for (iter = pset->begin(); iter != pset->end(); iter++) {
    damage[*iter] = pLocalized[*iter];
  }
}

void Arenisca::carryForward(const PatchSubset* patches,
                                    const MPMMaterial* matl,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw)
{
  // Carry forward the data.
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    // Carry forward the data common to all constitutive models
    // when using RigidMPM.
    // This method is defined in the ConstitutiveModel base class.
    carryForwardSharedData(pset, old_dw, new_dw, matl);
    
#ifdef JC_USE_BB_STATE_UPDATE
    // Get and copy the internal variables
    constParticleVariable<double> pKappa;
    d_intvar->getInternalVariable(pset, old_dw, pKappa);
    d_intvar->allocateAndPutRigid(pset, new_dw, pKappa);
#endif

    // Carry forward the data local to this constitutive model
    new_dw->put(delt_vartype(1.e10), lb->delTLabel, patch->getLevel());

    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(0.),     lb->StrainEnergyLabel);
    }
  }

}

//When a particle is pushed from patch to patch, carry information needed for the particle
void Arenisca::addParticleState(std::vector<const VarLabel*>& from,
                                        std::vector<const VarLabel*>& to)
{
  // Push back all the particle variables associated with Arenisca.
  // Important to keep from and to lists in same order!
  from.push_back(pPlasticStrainLabel);
  from.push_back(pPlasticStrainVolLabel);
  from.push_back(pElasticStrainVolLabel);
  from.push_back(pKappaLabel);
  from.push_back(pBackStressLabel);
  from.push_back(pBackStressIsoLabel);
  from.push_back(pKappaFlagLabel);
  from.push_back(pLocalizedLabel);
  to.push_back(  pPlasticStrainLabel_preReloc);
  to.push_back(  pPlasticStrainVolLabel_preReloc);
  to.push_back(  pElasticStrainVolLabel_preReloc);
  to.push_back(  pKappaLabel_preReloc);
  to.push_back(  pBackStressLabel_preReloc);
  to.push_back(  pBackStressIsoLabel_preReloc);
  to.push_back(  pKappaFlagLabel_preReloc);
  to.push_back(  pLocalizedLabel_preReloc);
  
#ifdef JC_USE_BB_STATE_UPDATE
  // Add the particle state for the internal variable models
  d_intvar->addParticleState(from, to);
#endif
}

//T2D: move up
void Arenisca::addInitialComputesAndRequires(Task* task,
                                            const MPMMaterial* matl,
                                            const PatchSet* patch) const
{

  // Add the computes and requires that are common to all explicit
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();

  // Other constitutive model and input dependent computes and requires
  task->computes(pPlasticStrainLabel,    matlset);
  task->computes(pPlasticStrainVolLabel, matlset);
  task->computes(pElasticStrainVolLabel, matlset);
  task->computes(pKappaLabel,            matlset);
  task->computes(pBackStressLabel,       matlset);
  task->computes(pBackStressIsoLabel,    matlset);
  task->computes(pKappaFlagLabel,        matlset);
  task->computes(pLocalizedLabel,        matlset);

#ifdef JC_USE_BB_STATE_UPDATE
  // Add internal evolution variables computed by internal variable model
  d_intvar->addInitialComputesAndRequires(task, matl, patch);
#endif
}

void Arenisca::addComputesAndRequires(Task* task,
                                              const MPMMaterial* matl,
                                              const PatchSet* patches ) const
{
  // Add the computes and requires that are common to all explicit
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForHypoExplicit(task, matlset, patches);
  task->requires(Task::OldDW, pPlasticStrainLabel,    matlset, Ghost::None);
  task->requires(Task::OldDW, pPlasticStrainVolLabel, matlset, Ghost::None);
  task->requires(Task::OldDW, pElasticStrainVolLabel, matlset, Ghost::None);
  task->requires(Task::OldDW, pKappaLabel,            matlset, Ghost::None);
  task->requires(Task::OldDW, pBackStressLabel,       matlset, Ghost::None);
  task->requires(Task::OldDW, pBackStressIsoLabel,    matlset, Ghost::None);
  task->requires(Task::OldDW, pKappaFlagLabel,        matlset, Ghost::None);
  task->requires(Task::OldDW, pLocalizedLabel,        matlset, Ghost::None);
  task->requires(Task::OldDW, lb->pParticleIDLabel,   matlset, Ghost::None);
  task->computes(pPlasticStrainLabel_preReloc,    matlset);
  task->computes(pPlasticStrainVolLabel_preReloc, matlset);
  task->computes(pElasticStrainVolLabel_preReloc, matlset);
  task->computes(pKappaLabel_preReloc,            matlset);
  task->computes(pBackStressLabel_preReloc,       matlset);
  task->computes(pBackStressIsoLabel_preReloc,    matlset);
  task->computes(pKappaFlagLabel_preReloc,        matlset);
  task->computes(pLocalizedLabel_preReloc,        matlset);
  
#ifdef JC_USE_BB_STATE_UPDATE
  // Add internal evolution variables computed by internal variable model
  d_intvar->addComputesAndRequires(task, matl, patches);
#endif
}

//T2D: Throw exception that this is not supported
void Arenisca::addComputesAndRequires(Task* ,
                                   const MPMMaterial* ,
                                   const PatchSet* ,
                                   const bool ) const
{
	
}

//T2D: Throw exception that this is not supported
double Arenisca::computeRhoMicroCM(double pressure,
                                      const double p_ref,
                                           const MPMMaterial* matl,
                                           double temperature,
                                           double rho_guess)
{
  double rho_orig = matl->getInitialDensity();
  double p_gauge = pressure - p_ref;
  double rho_cur;
  double bulk = d_initialData.B0;

  rho_cur = rho_orig/(1-p_gauge/bulk);

  return rho_cur;

//#if 1
  cout << "NO VERSION OF computeRhoMicroCM EXISTS YET FOR Arenisca"<<endl;
//#endif

}

//T2D: Throw exception that this is not supported
void Arenisca::computePressEOSCM(double rho_cur,double& pressure,
                                         double p_ref,
                                         double& dp_drho, double& tmp,
                                         const MPMMaterial* matl,
                                         double temperature)
{

  double bulk = d_initialData.B0;
  double shear = d_initialData.G0;
  double rho_orig = matl->getInitialDensity();

  double p_g = .5*bulk*(rho_cur/rho_orig - rho_orig/rho_cur);
  pressure = p_ref + p_g;
  dp_drho  = .5*bulk*(rho_orig/(rho_cur*rho_cur) + 1./rho_orig);
  tmp = (bulk + 4.*shear/3.)/rho_cur;  // speed of sound squared

  cout << "NO VERSION OF computePressEOSCM EXISTS YET FOR Arenisca"
       << endl;
}

//T2D: Throw exception that this is not supported
double Arenisca::getCompressibility()
{
  cout << "NO VERSION OF computePressEOSCM EXISTS YET FOR Arenisca"
       << endl;
  return 1.0;
}

// Initialize all labels of the particle variables associated with Arenisca.
void Arenisca::initializeLocalMPMLabels()
{
  pPlasticStrainLabel = VarLabel::create("p.plasticStrain",
    ParticleVariable<double>::getTypeDescription());
  pPlasticStrainLabel_preReloc = VarLabel::create("p.plasticStrain+",
    ParticleVariable<double>::getTypeDescription());
  pPlasticStrainVolLabel = VarLabel::create("p.plasticStrainVol",
    ParticleVariable<double>::getTypeDescription());
  pPlasticStrainVolLabel_preReloc = VarLabel::create("p.plasticStrainVol+",
    ParticleVariable<double>::getTypeDescription());
  pElasticStrainVolLabel = VarLabel::create("p.elasticStrainVol",
    ParticleVariable<double>::getTypeDescription());
  pElasticStrainVolLabel_preReloc = VarLabel::create("p.elasticStrainVol+",
    ParticleVariable<double>::getTypeDescription());
  pKappaLabel = VarLabel::create("p.kappa",
    ParticleVariable<double>::getTypeDescription());
  pKappaLabel_preReloc = VarLabel::create("p.kappa+",
    ParticleVariable<double>::getTypeDescription());
  pBackStressLabel = VarLabel::create("p.BackStress",
    ParticleVariable<Matrix3>::getTypeDescription());
  pBackStressLabel_preReloc = VarLabel::create("p.BackStress+",
    ParticleVariable<Matrix3>::getTypeDescription());
  pBackStressIsoLabel = VarLabel::create("p.BackStressIso",
    ParticleVariable<Matrix3>::getTypeDescription());
  pBackStressIsoLabel_preReloc = VarLabel::create("p.BackStressIso+",
    ParticleVariable<Matrix3>::getTypeDescription());
  pKappaFlagLabel = VarLabel::create("p.kappaState",
    ParticleVariable<double>::getTypeDescription());
  pKappaFlagLabel_preReloc = VarLabel::create("p.kappaState+",
    ParticleVariable<double>::getTypeDescription());
  pLocalizedLabel = VarLabel::create("p.localized",
    ParticleVariable<int>::getTypeDescription());
  pLocalizedLabel_preReloc = VarLabel::create("p.localized+",
    ParticleVariable<int>::getTypeDescription());

}
