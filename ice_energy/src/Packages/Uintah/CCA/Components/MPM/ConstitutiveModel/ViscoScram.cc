#include "ViscoScram.h"
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/Short27.h> //for Fracture
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h> // just added
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Math/Rand48.h>
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>

#include <Core/Malloc/Allocator.h>
#include <Core/Util/Endian.h>
#include <Core/Util/DebugStream.h>
#include <Core/Math/MinMax.h>

#include <sgi_stl_warnings_off.h>
#include <fstream>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using std::cerr;
using namespace Uintah;
using namespace SCIRun;

static DebugStream dbg("VS", false);
static DebugStream dbgSig("VSSig", false);

ViscoScram::ViscoScram(ProblemSpecP& ps, MPMLabel* Mlb,  MPMFlags* Mflag)
  : ConstitutiveModel(Mlb,Mflag)
{
  ps->require("PR",d_initialData.PR);
  //d_initialData.CoefThermExp = 12.5e-5;  // strains per K
  d_initialData.CoefThermExp = 0.0;  // keep from breaking RT
  ps->get("CoeffThermalExpansion", d_initialData.CoefThermExp);
  ps->require("CrackParameterA",d_initialData.CrackParameterA);
  ps->require("CrackPowerValue",d_initialData.CrackPowerValue);
  ps->require("CrackMaxGrowthRate",d_initialData.CrackMaxGrowthRate);
  ps->require("StressIntensityF",d_initialData.StressIntensityF);
  ps->require("CrackFriction",d_initialData.CrackFriction);
  ps->require("InitialCrackRadius",d_initialData.InitialCrackRadius);
  ps->require("CrackGrowthRate",d_initialData.CrackGrowthRate);
  ps->require("G1",d_initialData.G[0]);
  ps->require("G2",d_initialData.G[1]);
  ps->require("G3",d_initialData.G[2]);
  ps->require("G4",d_initialData.G[3]);
  ps->require("G5",d_initialData.G[4]);
  ps->require("RTau1",d_initialData.RTau[0]);
  ps->require("RTau2",d_initialData.RTau[1]);
  ps->require("RTau3",d_initialData.RTau[2]);
  ps->require("RTau4",d_initialData.RTau[3]);
  ps->require("RTau5",d_initialData.RTau[4]);
  ps->require("Beta",d_initialData.Beta);
  ps->require("Gamma",d_initialData.Gamma);
  ps->require("DCp_DTemperature",d_initialData.DCp_DTemperature);
  d_random = false;
  ps->get("randomize_parameters", d_random);
  d_doTimeTemperature = false;
  ps->get("use_time_temperature_equation", d_doTimeTemperature);
  d_useModifiedEOS = false;
  ps->get("useModifiedEOS",d_useModifiedEOS);
  d_useObjectiveRate = false;
  ps->get("useObjectiveRate",d_useObjectiveRate);

  // Time-temperature data for relaxtion time calculation
  d_tt.T0_WLF = 298.0;
  ps->get("T0", d_tt.T0_WLF);
  d_tt.C1_WLF = 6.5;
  ps->get("C1", d_tt.C1_WLF);
  d_tt.C2_WLF = 120.0;
  ps->get("C2", d_tt.C2_WLF);

  pVolChangeHeatRateLabel = VarLabel::create("p.volHeatRate",
      ParticleVariable<double>::getTypeDescription());
  pViscousHeatRateLabel   = VarLabel::create("p.veHeatRate",
      ParticleVariable<double>::getTypeDescription());
  pCrackHeatRateLabel     = VarLabel::create("p.crHeatRate",
      ParticleVariable<double>::getTypeDescription());
  pCrackRadiusLabel       = VarLabel::create("p.crackRad",
      ParticleVariable<double>::getTypeDescription());
  pStatedataLabel         = VarLabel::create("p.pStatedata_vs",
      ParticleVariable<StateData>::getTypeDescription());
  pRandLabel              = VarLabel::create("p.rand",
      ParticleVariable<double>::getTypeDescription() );
  pStrainRateLabel        = VarLabel::create("p.deformRate",
      ParticleVariable<Matrix3>::getTypeDescription() );

  pVolChangeHeatRateLabel_preReloc = VarLabel::create("p.volHeatRate+",
      ParticleVariable<double>::getTypeDescription());
  pViscousHeatRateLabel_preReloc   = VarLabel::create("p.veHeatRate+",
      ParticleVariable<double>::getTypeDescription());
  pCrackHeatRateLabel_preReloc     = VarLabel::create("p.crHeatRate+",
      ParticleVariable<double>::getTypeDescription());
  pCrackRadiusLabel_preReloc       = VarLabel::create("p.crackRad+",
      ParticleVariable<double>::getTypeDescription());
  pStatedataLabel_preReloc         = VarLabel::create("p.pStatedata_vs+",
      ParticleVariable<StateData>::getTypeDescription());
  pRandLabel_preReloc              = VarLabel::create("p.rand+",
      ParticleVariable<double>::getTypeDescription());
  pStrainRateLabel_preReloc        = VarLabel::create("p.deformRate+",
      ParticleVariable<Matrix3>::getTypeDescription());

  
  // The following are precomputed once for use with ICE.
  double G = d_initialData.G[0] + d_initialData.G[1] +
             d_initialData.G[2] + d_initialData.G[3] + d_initialData.G[4];
  d_bulk = (2.*G*(1. + d_initialData.PR))/(3.*(1.-2.*d_initialData.PR));

}

ViscoScram::ViscoScram(const ViscoScram* cm)
{
  
  lb = cm->lb;
  flag = cm->flag;
  NGN = cm->NGN ;
  d_bulk = cm->d_bulk;
  d_random = cm->d_random;
  d_useModifiedEOS = cm->d_useModifiedEOS ;
  d_doTimeTemperature = cm->d_doTimeTemperature;
  d_useObjectiveRate = cm->d_useObjectiveRate;

  d_initialData.PR = cm->d_initialData.PR;
  d_initialData.CoefThermExp = cm->d_initialData.CoefThermExp; 
  d_initialData.CrackParameterA = cm->d_initialData.CrackParameterA;
  d_initialData.CrackPowerValue = cm->d_initialData.CrackPowerValue;
  d_initialData.CrackMaxGrowthRate = cm->d_initialData.CrackMaxGrowthRate;
  d_initialData.StressIntensityF = cm->d_initialData.StressIntensityF;
  d_initialData.CrackFriction = cm->d_initialData.CrackFriction;
  d_initialData.InitialCrackRadius = cm->d_initialData.InitialCrackRadius;
  d_initialData.CrackGrowthRate = cm->d_initialData.CrackGrowthRate;
  d_initialData.G[0] = cm->d_initialData.G[0];
  d_initialData.G[1] = cm->d_initialData.G[1];
  d_initialData.G[2] = cm->d_initialData.G[2];
  d_initialData.G[3] = cm->d_initialData.G[3];
  d_initialData.G[4] = cm->d_initialData.G[4];
  d_initialData.RTau[0] = cm->d_initialData.RTau[0];
  d_initialData.RTau[1] = cm->d_initialData.RTau[1];
  d_initialData.RTau[2] = cm->d_initialData.RTau[2];
  d_initialData.RTau[3] = cm->d_initialData.RTau[3];
  d_initialData.RTau[4] = cm->d_initialData.RTau[4];
  d_initialData.Beta = cm->d_initialData.Beta;
  d_initialData.Gamma = cm->d_initialData.Gamma;
  d_initialData.DCp_DTemperature = cm->d_initialData.DCp_DTemperature;

  // Time-temperature data for relaxtion time calculation
  d_tt.T0_WLF = cm->d_tt.T0_WLF;
  d_tt.C1_WLF = cm->d_tt.C1_WLF;
  d_tt.C2_WLF = cm->d_tt.C2_WLF;

  pVolChangeHeatRateLabel = VarLabel::create("p.volHeatRate",
      ParticleVariable<double>::getTypeDescription());
  pViscousHeatRateLabel   = VarLabel::create("p.veHeatRate",
      ParticleVariable<double>::getTypeDescription());
  pCrackHeatRateLabel     = VarLabel::create("p.crHeatRate",
      ParticleVariable<double>::getTypeDescription());
  pCrackRadiusLabel       = VarLabel::create("p.crackRad",
      ParticleVariable<double>::getTypeDescription());
  pStatedataLabel         = VarLabel::create("p.pStatedata_vs",
      ParticleVariable<StateData>::getTypeDescription());
  pRandLabel              = VarLabel::create("p.rand",
      ParticleVariable<double>::getTypeDescription() );
  pStrainRateLabel        = VarLabel::create("p.deformRate",
      ParticleVariable<Matrix3>::getTypeDescription() );

  pVolChangeHeatRateLabel_preReloc = VarLabel::create("p.volHeatRate+",
      ParticleVariable<double>::getTypeDescription());
  pViscousHeatRateLabel_preReloc   = VarLabel::create("p.veHeatRate+",
      ParticleVariable<double>::getTypeDescription());
  pCrackHeatRateLabel_preReloc     = VarLabel::create("p.crHeatRate+",
      ParticleVariable<double>::getTypeDescription());
  pCrackRadiusLabel_preReloc       = VarLabel::create("p.crackRad+",
      ParticleVariable<double>::getTypeDescription());
  pStatedataLabel_preReloc         = VarLabel::create("p.pStatedata_vs+",
      ParticleVariable<StateData>::getTypeDescription());
  pRandLabel_preReloc              = VarLabel::create("p.rand+",
      ParticleVariable<double>::getTypeDescription() );
  pStrainRateLabel_preReloc        = VarLabel::create("p.deformRate+",
      ParticleVariable<Matrix3>::getTypeDescription());
}

ViscoScram::~ViscoScram()
{
  // Destructor
  VarLabel::destroy(pVolChangeHeatRateLabel);
  VarLabel::destroy(pViscousHeatRateLabel);
  VarLabel::destroy(pCrackHeatRateLabel);
  VarLabel::destroy(pCrackRadiusLabel);
  VarLabel::destroy(pStatedataLabel);
  VarLabel::destroy(pRandLabel);
  VarLabel::destroy(pStrainRateLabel);

  VarLabel::destroy(pVolChangeHeatRateLabel_preReloc);
  VarLabel::destroy(pViscousHeatRateLabel_preReloc);
  VarLabel::destroy(pCrackHeatRateLabel_preReloc);
  VarLabel::destroy(pCrackRadiusLabel_preReloc);
  VarLabel::destroy(pStatedataLabel_preReloc);
  VarLabel::destroy(pRandLabel_preReloc);
  VarLabel::destroy(pStrainRateLabel_preReloc);
}


ViscoScram* ViscoScram::clone()
{
  return scinew ViscoScram(*this);
}

void 
ViscoScram::addInitialComputesAndRequires(Task* task,
                                          const MPMMaterial* matl,
                                          const PatchSet*) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(pVolChangeHeatRateLabel, matlset);
  task->computes(pViscousHeatRateLabel,   matlset);
  task->computes(pCrackHeatRateLabel,     matlset);
  task->computes(pCrackRadiusLabel,       matlset);
  task->computes(pStatedataLabel,         matlset);
  task->computes(pRandLabel,              matlset);
  task->computes(pStrainRateLabel,        matlset);
}

void 
ViscoScram::initializeCMData(const Patch* patch,
                             const MPMMaterial* matl,
                             DataWarehouse* new_dw)
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  initSharedDataForExplicit(patch, matl, new_dw);

  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  Matrix3 zero(0.);

  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  ParticleVariable<double>    pVolChangeHeatRate;
  ParticleVariable<double>    pViscousHeatRate;
  ParticleVariable<double>    pCrackHeatRate;
  ParticleVariable<double>    pCrackRadius;
  ParticleVariable<StateData> pStatedata;
  ParticleVariable<double>    pRand;
  ParticleVariable<Matrix3>   pStrainRate;

  new_dw->allocateAndPut(pVolChangeHeatRate, pVolChangeHeatRateLabel, pset);
  new_dw->allocateAndPut(pViscousHeatRate,   pViscousHeatRateLabel,   pset);
  new_dw->allocateAndPut(pCrackHeatRate,     pCrackHeatRateLabel,     pset);
  new_dw->allocateAndPut(pCrackRadius,       pCrackRadiusLabel,       pset);
  new_dw->allocateAndPut(pStatedata,         pStatedataLabel,         pset);
  new_dw->allocateAndPut(pRand,              pRandLabel,              pset);
  new_dw->allocateAndPut(pStrainRate,        pStrainRateLabel,        pset);

  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
    particleIndex idx = *iter;
    pVolChangeHeatRate[idx] = 0.0;
    pViscousHeatRate[idx] = 0.0;
    pCrackHeatRate[idx] = 0.0;
    pCrackRadius[idx] = d_initialData.InitialCrackRadius;
    for(int imaxwell=0; imaxwell<5; imaxwell++){
      pStatedata[idx].DevStress[imaxwell] = zero;
    }
    if (d_random) pRand[idx] = drand48();
    else pRand[idx] = .5;
    pStrainRate[idx] = zero;
  }

  computeStableTimestep(patch, matl, new_dw);
}


void 
ViscoScram::computeStableTimestep(const Patch* patch,
                                  const MPMMaterial* matl,
                                  DataWarehouse* new_dw)
{
  // This is only called for the initial timestep - all other timesteps
  // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int dwi = matl->getDWIndex();

  // Retrieve the array of constitutive parameters
  ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
  constParticleVariable<double> pMass, pVol;
  constParticleVariable<Vector> pVelocity;

  new_dw->get(pMass,     lb->pMassLabel,     pset);
  new_dw->get(pVol,      lb->pVolumeLabel,   pset);
  new_dw->get(pVelocity, lb->pVelocityLabel, pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double G = d_initialData.G[0] + d_initialData.G[1] +
    d_initialData.G[2] + d_initialData.G[3] + d_initialData.G[4];
  double bulk = (2.*G*(1. + d_initialData.PR))/(3.*(1.-2.*d_initialData.PR));

  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
    particleIndex idx = *iter;

    // Compute wave speed at each particle, store the maximum
    c_dil = sqrt((bulk + 4.*G/3.)*pVol[idx]/pMass[idx]);
    WaveSpeed=Vector(Max(c_dil+fabs(pVelocity[idx].x()),WaveSpeed.x()),
                     Max(c_dil+fabs(pVelocity[idx].y()),WaveSpeed.y()),
                     Max(c_dil+fabs(pVelocity[idx].z()),WaveSpeed.z()));
  }
  WaveSpeed = dx/WaveSpeed;

  double delT_new = WaveSpeed.minComponent();
  //Timesteps larger than 1 microsecond cause VS to be unstable
  delT_new = min(1.e-6, delT_new);
  new_dw->put(delt_vartype(patch->getLevel()->adjustDelt(delT_new)), 
              lb->delTLabel);
}

void 
ViscoScram::addComputesAndRequires(Task* task,
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

  task->requires(Task::OldDW, lb->pTempPreviousLabel, matlset, gnone); 
  task->requires(Task::NewDW, lb->pTempCurrentLabel,  matlset, gnone); 

  task->requires(Task::OldDW, pCrackRadiusLabel, matlset, gnone);
  task->requires(Task::OldDW, pStatedataLabel,   matlset, gnone);
  task->requires(Task::OldDW, pRandLabel,        matlset, gnone);

  task->computes(pVolChangeHeatRateLabel_preReloc, matlset);
  task->computes(pViscousHeatRateLabel_preReloc,   matlset);
  task->computes(pCrackHeatRateLabel_preReloc,     matlset);
  task->computes(pCrackRadiusLabel_preReloc,       matlset);
  task->computes(pStatedataLabel_preReloc,         matlset);
  task->computes(pRandLabel_preReloc,              matlset);
  task->computes(pStrainRateLabel_preReloc,        matlset);
}

void 
ViscoScram::computeStressTensor(const PatchSubset* patches,
                                const MPMMaterial* matl,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  // Constants
  Matrix3 Identity; Identity.Identity();
  Matrix3 zero(0.),One(1.);
  double onethird = (1.0/3.0);
  double onesixth = (1.0/6.0);
  double sqrtopf=sqrt(1.5);
  double sqrtPI = sqrt(M_PI);
  Ghost::GhostType gac = Ghost::AroundCells;
  int dwi = matl->getDWIndex();

  // Material constants
  double rho_0 = matl->getInitialDensity();
  double Cp0   = matl->getSpecificHeat();
  double cf    = d_initialData.CrackFriction;
  double nu    = d_initialData.PR;
  double alpha = d_initialData.CoefThermExp;
  double cdot0 = d_initialData.CrackGrowthRate;
  double K_I   = d_initialData.StressIntensityF;
  double mm    = d_initialData.CrackPowerValue;
  double arad  = d_initialData.CrackParameterA;
  double arad3 = arad*arad*arad;

  // Do thermal expansion?
  if(!flag->d_doThermalExpansion){
    alpha = 0;
  }

  double Gmw[5];
  Gmw[0]=d_initialData.G[0];
  Gmw[1]=d_initialData.G[1];
  Gmw[2]=d_initialData.G[2];
  Gmw[3]=d_initialData.G[3];
  Gmw[4]=d_initialData.G[4];
  double RTau[5];
  double T0 = d_tt.T0_WLF;
  double C1 = d_tt.C1_WLF;
  double C2 = d_tt.C2_WLF;

  // Define particle and grid variables
  delt_vartype delT;
  old_dw->get(delT, lb->delTLabel, getLevel(patches));

  constParticleVariable<Short27>   pgCode;
  constParticleVariable<double>    pMass, pVol, pTemperature;
  constParticleVariable<double>    pCrackRadius;
  constParticleVariable<Point>     pX;
  constParticleVariable<Vector>    pVelocity, pSize;
  constParticleVariable<Matrix3>   pDefGrad, pStress;
  constNCVariable<Vector>          gVelocity, Gvelocity;
  constParticleVariable<double>    pTempPrev, pTempCur;

  ParticleVariable<double>    pVol_new, pIntHeatRate_new;
  ParticleVariable<Matrix3>   pDefGrad_new, pStress_new, pStrainRate_new;
  ParticleVariable<double>    pVolHeatRate_new, pVeHeatRate_new;
  ParticleVariable<double>    pCrHeatRate_new, pCrackRadius_new;
  ParticleVariable<double>    pRand;
  ParticleVariable<StateData> pStatedata;


  // Loop thru patches
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni;
    ni.reserve(interpolator->size());
    vector<Vector> d_S;
    d_S.reserve(interpolator->size());


    // initialize strain energy and wavespeed to zero
    double se = 0;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

    // Get patch information

    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    // Get material information
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    // Get the particle and grid data for the current patch
    old_dw->get(pX,                  lb->pXLabel,                  pset);
    old_dw->get(pMass,               lb->pMassLabel,               pset);
    old_dw->get(pVol,                lb->pVolumeLabel,             pset);
    old_dw->get(pTemperature,        lb->pTemperatureLabel,        pset);
    old_dw->get(pSize,               lb->pSizeLabel,               pset);

    old_dw->get(pVelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(pDefGrad,            lb->pDeformationMeasureLabel, pset);
    old_dw->get(pStress,             lb->pStressLabel,             pset);
    old_dw->get(pCrackRadius,        pCrackRadiusLabel,            pset);
    if (flag->d_fracture) {
      new_dw->get(pgCode,            lb->pgCodeLabel,              pset);
      new_dw->get(Gvelocity,         lb->GVelocityLabel, dwi, patch, gac, NGN);
    }
    new_dw->get(gVelocity,           lb->gVelocityLabel, dwi, patch, gac, NGN);

    old_dw->get(pTempPrev,           lb->pTempPreviousLabel,       pset); 
    new_dw->get(pTempCur,            lb->pTempCurrentLabel,        pset); 

    // Allocate arrays for the updated particle data for the current patch
    new_dw->allocateAndPut(pVol_new,         
                           lb->pVolumeDeformedLabel,              pset);
    new_dw->allocateAndPut(pIntHeatRate_new, 
                           lb->pInternalHeatRateLabel_preReloc,   pset);
    new_dw->allocateAndPut(pDefGrad_new,     
                           lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->allocateAndPut(pStress_new,      
                           lb->pStressLabel_preReloc,             pset);
    new_dw->allocateAndPut(pVolHeatRate_new, 
                           pVolChangeHeatRateLabel_preReloc,      pset);
    new_dw->allocateAndPut(pVeHeatRate_new,  
                           pViscousHeatRateLabel_preReloc,        pset);
    new_dw->allocateAndPut(pCrHeatRate_new,  
                           pCrackHeatRateLabel_preReloc,          pset);
    new_dw->allocateAndPut(pCrackRadius_new, 
                           pCrackRadiusLabel_preReloc,            pset);
    new_dw->allocateAndPut(pStrainRate_new, 
                           pStrainRateLabel_preReloc,             pset);
    new_dw->allocateAndPut(pRand,        
                           pRandLabel_preReloc,                   pset);
    new_dw->allocateAndPut(pStatedata,   
                           pStatedataLabel_preReloc,              pset);
    old_dw->copyOut(pRand,           pRandLabel,                   pset);
    old_dw->copyOut(pStatedata,      pStatedataLabel,              pset);
    ASSERTEQ(pset, pStatedata.getParticleSubset());

    // Loop thru particles in the patch
    ParticleSubset::iterator iter = pset->begin();
    for(; iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Assign zero internal heating by default - modify if necessary.
      pIntHeatRate_new[idx] = 0.0;

      // Randomize the shear moduli of the elements of each particle
      Gmw[0]=d_initialData.G[0]*(1.+.4*(pRand[idx]-.5));
      Gmw[1]=d_initialData.G[1]*(1.+.4*(pRand[idx]-.5));
      Gmw[2]=d_initialData.G[2]*(1.+.4*(pRand[idx]-.5));
      Gmw[3]=d_initialData.G[3]*(1.+.4*(pRand[idx]-.5));
      Gmw[4]=d_initialData.G[4]*(1.+.4*(pRand[idx]-.5));
      double G = Gmw[0] + Gmw[1] + Gmw[2] + Gmw[3] + Gmw[4];
      double bulk = (2.*G*(1.+ nu))/(3.*(1.-2.*nu));
      //double beta = 3.*alpha*bulk*(1.+.4*(pRand[idx]-.5));

      RTau[0]=d_initialData.RTau[0];
      RTau[1]=d_initialData.RTau[1];
      RTau[2]=d_initialData.RTau[2];
      RTau[3]=d_initialData.RTau[3];
      RTau[4]=d_initialData.RTau[4];
      if (d_doTimeTemperature) {
        // Calculate the temperature dependent the relaxation times tau(ii)
        // First calculate a_T
        double TDiff = pTemperature[idx] - T0;
        double a_T = exp(-C1*TDiff/(C2 + TDiff)); 

        //dbg << "idx = " << idx << " pT = " << pTemperature[idx]
        //    << " T0 = " << T0 << " TDiff = " << TDiff << endl;
        //dbg << " a_T = " << a_T << endl;

        // Then calculate relaxation times and store in an array
        // (Note that shear moduli are already in the array Gi)
        for (int ii = 0; ii < 5; ++ii) {
          //dbg << "Old RTau["<< ii <<"] ="<< RTau[ii] <<" ";
          RTau[ii] /= a_T;
          //dbg << "New RTau["<< ii <<"]=" << RTau[ii] << endl;
        }
      }

      // Get the node indices that surround the cell
      interpolator->findCellAndShapeDerivatives(pX[idx], ni, d_S,pSize[idx]);
      

      Matrix3 pVelGrad(0.0);
      for(int k = 0; k < flag->d_8or27; k++) {
        Vector gvel;
        if (flag->d_fracture) {
          if(pgCode[idx][k]==1) gvel = gVelocity[ni[k]];
          if(pgCode[idx][k]==2) gvel = Gvelocity[ni[k]];
        } else gvel = gVelocity[ni[k]];
        for (int j = 0; j<3; j++){
          for (int i = 0; i<3; i++) {
            pVelGrad(i,j)+=gvel[i] * d_S[k][j] * oodx[j];
          }
        }
      }

      // Compute the deformation gradient increment using the time_step
      // velocity gradient (F_n^np1 = dudx * dt + Identity)
      Matrix3 pDefGradInc = pVelGrad * delT + Identity;
      double Jinc = pDefGradInc.Determinant();

      // Update the deformation gradient tensor to its time n+1 value.
      pDefGrad_new[idx] = pDefGradInc*pDefGrad[idx];
      double J = pDefGrad_new[idx].Determinant();
      if (!(J > 0.0)) {
        cerr << getpid() 
             << "**ERROR** Negative Jacobian of deformation gradient" << endl;
        throw ParameterNotFound("**ERROR**:ViscoScram", __FILE__, __LINE__);
      }
      double rho_cur = rho_0/J;

      // Update the volume
      pVol_new[idx] = Jinc*pVol[idx];

      // Calculate rate of deformation D 
      Matrix3 D = (pVelGrad + pVelGrad.Transpose())*0.5;

      // Get stress at time t_n
      Matrix3 sig_old = pStress[idx];

      // For objective rates (rotation neutralized)
      Matrix3 RR(0.0), UU(0.0), RT(0.0);
      if (d_useObjectiveRate) {

        // Compute polar decomposition of F
        pDefGrad_new[idx].polarDecomposition(UU, RR, 1.0e-12, true);
        RT = RR.Transpose();

        // If we want objective rates, rotate stress and rate of
        // deformation to material coordinates using R where F = RU
        sig_old = RT*(sig_old*RR);
        for (int ii = 0; ii < 5; ++ii) {
          pStatedata[idx].DevStress[ii] = 
            RT*(pStatedata[idx].DevStress[ii]*RR); 
        }
        D = RT*(D*RR);
      }
     
      // Subtract the thermal expansion to get D_e + D_p
      double dT_dt = (pTempCur[idx] - pTempPrev[idx])/delT;
      D -= Identity*(alpha*dT_dt);

      // Compute deviatoric rate DPrime
      Matrix3 DPrime = D - Identity*onethird*D.Trace();

      // Get effective strain rate and Effective deviatoric strain rate
      pStrainRate_new[idx] = D;

      //if (dbg.active()) {
      //  dbg.setf(ios::scientific,ios::floatfield);
      //  dbg.precision(8);
      //  dbg << "Total Strain Rate = [" 
      //      << D(0,0) << " " << D(1,1) << " " << D(2,2) << " "
      //      << D(1,2) << " " << D(2,0) << " " << D(0,1) << "]" << endl;
      //}

      double EDeff = sqrtopf*DPrime.Norm();

      // Sum of old deviatoric stresses
      Matrix3 DevStress = pStatedata[idx].DevStress[0] +
                          pStatedata[idx].DevStress[1] +
                          pStatedata[idx].DevStress[2] +
                          pStatedata[idx].DevStress[3] +
                          pStatedata[idx].DevStress[4];

      // old total stress norm
      double EffStress = sqrtopf*sig_old.Norm();

      //if (dbg.active()) {
      //  dbg << "D.Norm() = " << D.Norm()
      //      << " Ddev.Norm() = " << DPrime.Norm()
      //      << " Sig.Norm() = " << pStress[idx].Norm()
      //      << " SigDev.Norm() = " << DevStress.Norm() << endl;
      //}

      //old deviatoric stress norm
      double DevStressNormSq = DevStress.NormSquared();
      double DevStressNorm = sqrt(DevStressNormSq);

      // old effective deviatoric stress
      double EffDevStress = sqrtopf*DevStressNorm;

      // Baseline
      double vres_a = 0.90564746;
      double vres_b =-2.90178468;
      // Aged
      //      double vres_a = 0.90863805;
      //      double vres_b =-2.5061966;

      double vres = 0.0;
      if(EDeff > 1.e-8){
        vres = exp(vres_a*log(EDeff) + vres_b);
      }

      double sig_kk = sig_old.Trace();
      double sig_m = -onethird*sig_kk;

      int compflag = 0;
      if (sig_m < 0.0) compflag = -1;

      EffStress    = (1+compflag)*EffDevStress - compflag*EffStress;
      vres        *= ((1 + compflag) - cdot0*compflag);
      double sigmae = sqrt(DevStressNormSq - compflag*(3*sig_m*sig_m));

      // Stress intensity factor
      double crad   = pCrackRadius[idx];
      ASSERT(crad != 0.0);
      double sqrtc  = sqrt(crad);
      double sif    = sqrtopf*sqrtPI*sqrtc*sigmae;

      // Modification to include friction on crack faces
      double xmup   = (1 + compflag)*sqrt(45./(2.*(3. - 2.*cf*cf)))*cf;
      double a      = xmup*sig_m*sqrtc;
      double b      = 1. + a/K_I;
      double termm  = 1. + M_PI*a*b/K_I;
      double rko    = K_I*sqrt(termm);
      double skp    = rko*sqrt(1. + (2./mm));
      double sk1    = skp*pow((1. + (mm/2.)), 1./mm);

      if(vres > d_initialData.CrackMaxGrowthRate){
        vres = d_initialData.CrackMaxGrowthRate;
      }
      //if (dbg.active()) {
      //  dbg  << "vres = " << vres << " sif = " << sif 
      //       << " sigmae = " << sigmae << endl;
      //  dbg  << "crad = " << crad << " xmup = " << xmup << " a = " << a
      //       << " b = " << b << " termm = " << termm << " rko = " << rko
      //       << " skp = " << skp << " sk1 = " << sk1 << endl;
      //}

      double cdot,cc,rk1c,rk2c,rk3c,rk4c;

      // cdot is crack speed
      // Use fourth order Runge Kutta integration to find new crack radius
      if(sif < skp ){
        double fac = EffStress/sk1;
        cdot = vres*pow((sif/sk1), mm);
        cc   = vres*delT;
        rk1c = cc*pow(sqrtPI*sqrtc*fac,              mm);
        rk2c = cc*pow(sqrtPI*sqrt(crad+.5*rk1c)*fac, mm);
        rk3c = cc*pow(sqrtPI*sqrt(crad+.5*rk2c)*fac, mm);
        rk4c = cc*pow(sqrtPI*sqrt(crad+rk3c)*fac,    mm);
      }
      else{
        double fac = rko*rko/(M_PI*EffStress*EffStress);
        cdot = vres*(1. - rko*rko/(sif*sif));
        cc   = vres*delT;
        rk1c = cc*(1. - fac/crad);
        rk2c = cc*(1. - fac/(crad+.5*rk1c));
        rk3c = cc*(1. - fac/(crad+.5*rk2c));
        rk4c = cc*(1. - fac/(crad+rk3c));
      }
      //if (dbg.active()) {
      //  dbg << "c = " << crad << " cdot = " << cdot << " cc = " << cc
      //      << " rk1c = " << rk1c << " rk2c = " << rk2c
      //      << " rk3c = " << rk3c << " rk3c = " << rk3c << endl;
      //}

      // Deviatoric stress integration
      for(int imw=0;imw<5;imw++){
        // First Runga-Kutta Term
        double crad_rk = crad;
        double con1 = (3.0*crad_rk*crad_rk*cdot)/arad3;
        double con3 = (crad_rk*crad_rk*crad_rk)/arad3;
        double con2 = 1. + con3;
        Matrix3 DevStressOld = pStatedata[idx].DevStress[imw];
        Matrix3 DevStressS = zero; 
        Matrix3 DevStressT = zero;
        for(int jmaxwell=0;jmaxwell<5;jmaxwell++){
          DevStressS += pStatedata[idx].DevStress[jmaxwell];
          DevStressT += pStatedata[idx].DevStress[jmaxwell]*RTau[jmaxwell];
        }
        Matrix3 rk1 = (DPrime*2.*Gmw[imw] -
                       DevStressOld*RTau[imw] -
                       (DevStressS*con1 +
                        (DPrime*2.*G - DevStressT - DevStressS*con1)*con3/con2)
                       *(Gmw[imw]/G))*delT;

        // Second Runga-Kutta Term
        crad_rk = crad+0.5*rk1c;
        con1 = (3.0*crad_rk*crad_rk*cdot)/arad3;
        con3 = (crad_rk*crad_rk*crad_rk)/arad3;
        con2 = 1. + con3;
        DevStressOld = pStatedata[idx].DevStress[imw] + rk1*.5;
        DevStressS = zero; 
        DevStressT = zero;
        for(int jmaxwell=0;jmaxwell<5;jmaxwell++){
          DevStressS += (pStatedata[idx].DevStress[jmaxwell] + rk1*.5);
          DevStressT += (pStatedata[idx].DevStress[jmaxwell] + rk1*.5)*
            RTau[jmaxwell];
        }
        Matrix3 rk2 = (DPrime*2.*Gmw[imw] - 
                       DevStressOld*RTau[imw] -
                       (DevStressS*con1 +
                        (DPrime*2.*G - DevStressT - DevStressS*con1)*con3/con2)
                       *(Gmw[imw]/G))*delT;

        // Third Runga-Kutta Term
        crad_rk = crad+0.5*rk2c;
        con1 = (3.0*crad_rk*crad_rk*cdot)/arad3;
        con3 = (crad_rk*crad_rk*crad_rk)/arad3;
        con2 = 1. + con3;
        DevStressOld = pStatedata[idx].DevStress[imw] + rk2*.5;
        DevStressS = zero; 
        DevStressT = zero;
        for(int jmaxwell=0;jmaxwell<5;jmaxwell++){
          DevStressS += (pStatedata[idx].DevStress[jmaxwell] + rk2*.5);
          DevStressT += (pStatedata[idx].DevStress[jmaxwell] + rk2*.5)*
            RTau[jmaxwell];
        }
        Matrix3 rk3 = (DPrime*2.*Gmw[imw] -
                       DevStressOld*RTau[imw] -
                       (DevStressS*con1 +
                        (DPrime*2.*G - DevStressT - DevStressS*con1)*con3/con2)
                       *(Gmw[imw]/G))*delT;

        // Fourth Runga-Kutta Term
        crad_rk = crad+rk3c;
        con1 = (3.0*crad_rk*crad_rk*cdot)/arad3;
        con3 = (crad_rk*crad_rk*crad_rk)/arad3;
        con2 = 1. + con3;
        DevStressOld = pStatedata[idx].DevStress[imw] + rk3;
        DevStressS = zero; 
        DevStressT = zero;
        for(int jmaxwell=0;jmaxwell<5;jmaxwell++){
          DevStressS += (pStatedata[idx].DevStress[jmaxwell] + rk3);
          DevStressT += (pStatedata[idx].DevStress[jmaxwell] + rk3)*
            RTau[jmaxwell];
        }
        Matrix3 rk4 = (DPrime*2.*Gmw[imw] -
                       DevStressOld*RTau[imw] -
                       (DevStressS*con1 +
                        (DPrime*2.*G - DevStressT - DevStressS*con1)*con3/con2)
                       *(Gmw[imw]/G))*delT;
        //dbg << "imw = " << imw << endl;
        //dbg << "   rk1 = [" << rk1(0,0) << " " << rk1(1,1) << " " << rk1(2,2)
        //    << rk1(1,2) << " " << rk1(2,0) << " " << rk1(0,1) << endl;
        //dbg << "   rk2 = [" << rk2(0,0) << " " << rk2(1,1) << " " << rk2(2,2)
        //    << rk2(1,2) << " " << rk2(2,0) << " " << rk2(0,1) << endl;
        //dbg << "   rk3 = [" << rk3(0,0) << " " << rk3(1,1) << " " << rk3(2,2)
        //    << rk3(1,2) << " " << rk3(2,0) << " " << rk3(0,1) << endl;
        //dbg << "   rk4 = [" << rk4(0,0) << " " << rk4(1,1) << " " << rk4(2,2)
        //    << rk4(1,2) << " " << rk4(2,0) << " " << rk4(0,1) << endl;

        // Update Maxwell element Deviatoric Stresses
        pStatedata[idx].DevStress[imw] +=
          (rk1 + rk4)*onesixth + (rk2 + rk3)*onethird;
      }

      // Update the Cauchy stress
      //if (dbgSig.active()) {
      //  dbgSig.setf(ios::scientific,ios::floatfield);
      //  dbgSig.precision(8);
      //  dbgSig << " Particle = " << idx << endl;
      //  dbgSig << "  D = [" 
      //         << D(0,0) << " " << D(1,1) << " " << D(2,2) << " "
      //         << D(1,2) << " " << D(2,0) << " " << D(0,1) << "]" << endl;
      //}

      double ekkdot = D.Trace();

      //if (dbgSig.active()) {
      //  dbgSig << "  K = " << bulk << " ekkdot = " << ekkdot 
      //         << " delT = " << delT << endl;
      //  dbgSig << "  pold = " << sig_m ;
      //}

      sig_m = -sig_m; // revert back to standard form
      sig_m += ekkdot*bulk*delT;
      DevStress = pStatedata[idx].DevStress[0]+pStatedata[idx].DevStress[1]+ 
                  pStatedata[idx].DevStress[2]+pStatedata[idx].DevStress[3]+ 
                  pStatedata[idx].DevStress[4];
      pStress_new[idx] = DevStress + Identity*sig_m;

      Matrix3 sig = pStress_new[idx];

      //if (dbgSig.active()) {
      //  dbgSig << " pnew = " << sig_m << endl;
      //  dbgSig << "  S_dev = [" 
      //         << DevStress(0,0) << " " << DevStress(1,1) << " " 
      //         << DevStress(2,2) << " " << DevStress(1,2) << " " 
      //         << DevStress(2,0) << " " << DevStress(0,1) << "]" << endl;
      //  dbgSig << "  sig = [" 
      //         << sig(0,0) << " " << sig(1,1) << " " << sig(2,2) << " " 
      //         << sig(1,2) <<" "<< sig(2,0) << " " << sig(0,1) << "]" << endl;
      //}

      // Update crack radius
      crad += onesixth*(rk1c + rk4c) + onethird*(rk2c + rk3c);
      pCrackRadius_new[idx] = crad;
      //if (dbgSig.active())
      //  dbgSig << " Crack Radius = " << crad << endl;

      ASSERT(crad > 0.0);

      // Update the internal heating rate 
      //double cpnew = Cp0 + d_initialData.DCp_DTemperature*pTemperature[idx];
      //double Cv = cpnew/(1+d_initialData.Beta*pTemperature[idx]);
      double Cv = Cp0;
      double rhoCv = rho_cur*Cv;

      // Update the Viscoelastic work rate
      double svedot = 0.;
      for(int imw=0;imw<5;imw++){
        svedot += pStatedata[idx].DevStress[imw].NormSquared()/(2.*Gmw[imw])
                  *RTau[imw] ;
      }
      pVeHeatRate_new[idx] = svedot/rhoCv;

      // Update the cracking work rate
      Matrix3 sovertau(0.0);
      for (int imw = 0; imw < 5; ++imw) {
        sovertau += pStatedata[idx].DevStress[imw]*RTau[imw];
      }
      double coa3   = (crad*crad*crad)/arad3;
      double topc   = 3.*(coa3/crad)*cdot;
      double oocoa3 = 1.0/(1.0 + coa3);
      Matrix3 SRate = (D*(2.0*G) - DevStress*topc - sovertau)*oocoa3;
      double scrdot = (DevStress.NormSquared()*topc + 
                       DevStress.Contract(SRate)*coa3)/(2.0*G);

      //if (dbg.active()) {
      //  dbg << "SRate = [" 
      //      << SRate(0,0) << " " << SRate(1,1) << " " << SRate(2,2) << " "
      //      << SRate(1,2) << " " << SRate(2,0) << " " << SRate(0,1) << "]"
      //      << endl;
      //  dbg << "Wdot_cr = " << scrdot << endl;
      //  dbg << "rhoCv = " << rhoCv << endl;
      //}

      pCrHeatRate_new[idx] = scrdot/rhoCv;

      //if (dbg.active())
      //  dbg << "pCrHeatRate = " << pCrHeatRate_new[idx] << endl;


      // Update the volume change heat rate
      pVolHeatRate_new[idx] = d_initialData.Gamma*pTemperature[idx]*ekkdot;

      // Update the total internal heat rate
      pIntHeatRate_new[idx] = -pVolHeatRate_new[idx] + pVeHeatRate_new[idx] +
                               pCrHeatRate_new[idx];
      //dbg << "idx = " << " qdot_v = " << pVolHeatRate_new[idx]
      //                 << " qdot_ve = " << pVeHeatRate_new[idx]
      //                 << " qdot_cr = " << pCrHeatRate_new[idx]
      //                 << " qdot = " << pIntHeatRate_new[idx] << endl;

      // Compute the strain energy for all the particles
      sig_old = (pStress_new[idx] + sig_old)*.5;
      se += (D.Contract(sig_old))*pVol_new[idx]*delT;

      if (d_useObjectiveRate) {

        // Rotate everything back to lab coordinates
        pStress_new[idx] = RR*(pStress_new[idx]*RT);
        for (int ii = 0; ii < 5; ++ii) {
          pStatedata[idx].DevStress[ii] = 
            RR*(pStatedata[idx].DevStress[ii]*RT); 
        }
      }

      // Compute wave speed at each particle, store the maximum
      Vector pVel = pVelocity[idx];
      double c_dil = sqrt((bulk + 4.*G/3.)/rho_cur);
      WaveSpeed=Vector(Max(c_dil+fabs(pVel.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pVel.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pVel.z()),WaveSpeed.z()));
    }

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    //Timesteps larger than 1 microsecond cause VS to be unstable
    delT_new = min(1.e-6, delT_new);

    new_dw->put(delt_vartype(patch->getLevel()->adjustDelt(delT_new)), 
                lb->delTLabel);
    new_dw->put(sum_vartype(se),     lb->StrainEnergyLabel);
    delete interpolator;
  }
}

void 
ViscoScram::carryForward(const PatchSubset* patches,
                         const MPMMaterial* matl,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw)
{
  Matrix3 zero(0.0);
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    // Carry forward the data common to all constitutive models 
    // when using RigidMPM.
    // This method is defined in the ConstitutiveModel base class.
    carryForwardSharedData(pset, old_dw, new_dw, matl);

    // Carry forward the data local to this constitutive model 
    constParticleVariable<double>  pCrackRadius;
    old_dw->get(pCrackRadius,    pCrackRadiusLabel,    pset);

    ParticleVariable<double>    pVolHeatRate_new, pVeHeatRate_new;
    ParticleVariable<double>    pCrHeatRate_new, pCrackRadius_new;
    ParticleVariable<Matrix3>   pStrainRate_new;
    ParticleVariable<StateData> pStatedata;
    ParticleVariable<double>    pRand;

    new_dw->allocateAndPut(pVolHeatRate_new, 
                           pVolChangeHeatRateLabel_preReloc,      pset);
    new_dw->allocateAndPut(pVeHeatRate_new,  
                           pViscousHeatRateLabel_preReloc,        pset);
    new_dw->allocateAndPut(pCrHeatRate_new,  
                           pCrackHeatRateLabel_preReloc,          pset);
    new_dw->allocateAndPut(pCrackRadius_new, 
                           pCrackRadiusLabel_preReloc,            pset);
    new_dw->allocateAndPut(pStrainRate_new,         
                           pStrainRateLabel_preReloc,             pset);
    new_dw->allocateAndPut(pStatedata,  
                           pStatedataLabel_preReloc,              pset);
    new_dw->allocateAndPut(pRand,         
                           pRandLabel_preReloc,                   pset);
    old_dw->copyOut(pRand,      pRandLabel,      pset);
    old_dw->copyOut(pStatedata, pStatedataLabel, pset);

    ParticleSubset::iterator iter = pset->begin();
    for(; iter != pset->end(); iter++){
      particleIndex idx = *iter;
      pVolHeatRate_new[idx] = 0.0;
      pVeHeatRate_new[idx]  = 0.0;
      pCrHeatRate_new[idx]  = 0.0;
      pCrackRadius_new[idx] = pCrackRadius[idx];
      pStrainRate_new[idx] = zero;
    }
    new_dw->put(delt_vartype(patch->getLevel()->adjustDelt(1.e10)), 
                lb->delTLabel);
    new_dw->put(sum_vartype(0.),     lb->StrainEnergyLabel);
  }
}
         
void 
ViscoScram::allocateCMDataAddRequires(Task* task,
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
  task->requires(Task::NewDW, pVolChangeHeatRateLabel_preReloc, matlset, gnone);
  task->requires(Task::NewDW, pViscousHeatRateLabel_preReloc,   matlset, gnone);
  task->requires(Task::NewDW, pCrackHeatRateLabel_preReloc,     matlset, gnone);
  task->requires(Task::NewDW, pCrackRadiusLabel_preReloc,       matlset, gnone);
  task->requires(Task::NewDW, pStrainRateLabel_preReloc,        matlset, gnone);
  task->requires(Task::NewDW, pStatedataLabel_preReloc,         matlset, gnone);
  task->requires(Task::NewDW, pRandLabel_preReloc,              matlset, gnone);
}


void 
ViscoScram::allocateCMDataAdd(DataWarehouse* new_dw,
                              ParticleSubset* addset,
                              map<const VarLabel*,
                                  ParticleVariableBase*>* newState,
                              ParticleSubset* delset,
                              DataWarehouse* )
{
  // Copy the data common to all constitutive models from the particle to be 
  // deleted to the particle to be added. 
  // This method is defined in the ConstitutiveModel base class.
  copyDelToAddSetForConvertExplicit(new_dw, delset, addset, newState);
  
  // Copy the data local to this constitutive model from the particles to 
  // be deleted to the particles to be added
  ParticleVariable<double>    pVolChangeHeatRate_add;
  ParticleVariable<double>    pViscousHeatRate_add;
  ParticleVariable<double>    pCrackHeatRate_add;
  ParticleVariable<double>    pCrackRadius_add;
  ParticleVariable<Matrix3>   pStrainRate_add;
  ParticleVariable<StateData> pStatedata_add;
  ParticleVariable<double>    pRand_add;

  constParticleVariable<double>    pVolChangeHeatRate_del;
  constParticleVariable<double>    pViscousHeatRate_del;
  constParticleVariable<double>    pCrackHeatRate_del;
  constParticleVariable<double>    pCrackRadius_del;
  constParticleVariable<Matrix3>   pStrainRate_del;
  constParticleVariable<StateData> pStatedata_del;
  constParticleVariable<double>    pRand_del;

  new_dw->allocateTemporary(pVolChangeHeatRate_add, addset);
  new_dw->allocateTemporary(pViscousHeatRate_add,   addset);
  new_dw->allocateTemporary(pCrackHeatRate_add,     addset);
  new_dw->allocateTemporary(pCrackRadius_add,       addset);
  new_dw->allocateTemporary(pStrainRate_add,        addset);
  new_dw->allocateTemporary(pStatedata_add,         addset);
  new_dw->allocateTemporary(pRand_add,              addset);

  new_dw->get(pVolChangeHeatRate_del, pVolChangeHeatRateLabel_preReloc, delset);
  new_dw->get(pViscousHeatRate_del,   pViscousHeatRateLabel_preReloc,   delset);
  new_dw->get(pCrackHeatRate_del,     pCrackHeatRateLabel_preReloc,     delset);
  new_dw->get(pCrackRadius_del,       pCrackRadiusLabel_preReloc,       delset);
  new_dw->get(pStrainRate_del,        pStrainRateLabel_preReloc,        delset);
  new_dw->get(pStatedata_del,         pStatedataLabel_preReloc,         delset);
  new_dw->get(pRand_del,              pRandLabel_preReloc,              delset);

  ParticleSubset::iterator del = delset->begin();
  ParticleSubset::iterator add = addset->begin();
  for (; del != delset->end(); del++, add++) {
    particleIndex delidx = *del;
    particleIndex addidx = *add;
 
    pVolChangeHeatRate_add[addidx] = pVolChangeHeatRate_del[delidx];
    pVolChangeHeatRate_add[addidx] = pViscousHeatRate_del[delidx];
    pCrackHeatRate_add[addidx] = pCrackHeatRate_del[delidx];
    pCrackRadius_add[addidx] = pCrackRadius_del[delidx];
    pStrainRate_add[addidx] = pStrainRate_del[delidx];
    for(int imaxwell=0; imaxwell<5; imaxwell++){
      pStatedata_add[addidx].DevStress[imaxwell] = 
        pStatedata_del[delidx].DevStress[imaxwell];
    }
    pRand_add[addidx] = pRand_del[delidx];
  }

  (*newState)[pVolChangeHeatRateLabel] = pVolChangeHeatRate_add.clone();
  (*newState)[pViscousHeatRateLabel] = pViscousHeatRate_add.clone();
  (*newState)[pCrackHeatRateLabel] = pCrackHeatRate_add.clone();
  (*newState)[pCrackRadiusLabel] = pCrackRadius_add.clone();
  (*newState)[pStrainRateLabel] = pStrainRate_add.clone();
  (*newState)[pStatedataLabel] = pStatedata_add.clone();
  (*newState)[pRandLabel] = pRand_add.clone();
}

void 
ViscoScram::addParticleState(std::vector<const VarLabel*>& from,
                             std::vector<const VarLabel*>& to)
{
  // Add the local particle state data for this constitutive model.
  from.push_back(pVolChangeHeatRateLabel);
  from.push_back(pViscousHeatRateLabel);
  from.push_back(pCrackHeatRateLabel);
  from.push_back(pCrackRadiusLabel);
  from.push_back(pStrainRateLabel);
  from.push_back(pStatedataLabel);
  from.push_back(pRandLabel);

  to.push_back(pVolChangeHeatRateLabel_preReloc);
  to.push_back(pViscousHeatRateLabel_preReloc);
  to.push_back(pCrackHeatRateLabel_preReloc);
  to.push_back(pCrackRadiusLabel_preReloc);
  to.push_back(pStrainRateLabel_preReloc);
  to.push_back(pStatedataLabel_preReloc);
  to.push_back(pRandLabel_preReloc);
}

double ViscoScram::computeRhoMicroCM(double pressure,
                                     const double p_ref,
                                     const MPMMaterial* matl)
{
  double rho_orig = matl->getInitialDensity();
  double p_gauge = pressure - p_ref;
  double rho_cur;

  if(d_useModifiedEOS && p_gauge < 0.0) {
    double A = p_ref;       // Modified EOS
    double n = p_ref/d_bulk;
    rho_cur  = rho_orig*pow(pressure/A,n);
  }
  else {                      // STANDARD EOS
    double p_g_over_bulk = p_gauge/d_bulk;
    rho_cur=rho_orig*(p_g_over_bulk + sqrt(p_g_over_bulk*p_g_over_bulk +1.));
  }
  return rho_cur;

}

void ViscoScram::computePressEOSCM(double rho_cur,double& pressure,
                                   double p_ref,
                                   double& dp_drho, double& tmp,
                                   const MPMMaterial* matl)
{
  double rho_orig = matl->getInitialDensity();
  double inv_rho_orig = 1./rho_orig;

  if(d_useModifiedEOS && rho_cur < rho_orig){
    double A = p_ref;         // MODIFIED EOS
    double n = d_bulk/p_ref;
    double rho_rat_to_the_n = pow(rho_cur*inv_rho_orig,n);
    pressure = A*rho_rat_to_the_n;
    dp_drho  = (d_bulk/rho_cur)*rho_rat_to_the_n;
    tmp      = dp_drho;       // speed of sound squared
  }
  else {                      // STANDARD EOS            
    double p_g = .5*d_bulk*(rho_cur*inv_rho_orig - rho_orig/rho_cur);
    pressure   = p_ref + p_g;
    dp_drho    = .5*d_bulk*(rho_orig/(rho_cur*rho_cur) + inv_rho_orig);
    tmp        = d_bulk/rho_cur;  // speed of sound squared
  }
}

double ViscoScram::getCompressibility()
{
  return 1.0/d_bulk;
}

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1209
#endif

namespace Uintah {

static
MPI_Datatype
makeMPI_CMData()
{
   ASSERTEQ(sizeof(ViscoScramStateData), sizeof(double)*45);
   MPI_Datatype mpitype;
   MPI_Type_vector(1, 45, 45, MPI_DOUBLE, &mpitype);
   MPI_Type_commit(&mpitype);
   return mpitype;
}

const Uintah::TypeDescription*
fun_getTypeDescription(ViscoScramStateData*)
{
   static Uintah::TypeDescription* td = 0;
   if(!td){
      td = scinew Uintah::TypeDescription(TypeDescription::Other,
                               "ViscoScramStateData", true, &makeMPI_CMData);
   }
   return td;
}

} // End namespace Uintah

namespace SCIRun {
void swapbytes( Uintah::ViscoScramStateData& d)
{
  for (int i = 0; i < 5; i++) swapbytes(d.DevStress[i]);
}
  
} // namespace SCIRun
