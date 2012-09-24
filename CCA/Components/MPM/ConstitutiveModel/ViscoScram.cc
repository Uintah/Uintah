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


#include "ViscoScram.h"
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Math/Matrix3.h>
#include <Core/Math/Short27.h> //for Fracture
#include <Core/Grid/Variables/NodeIterator.h> // just added
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Math/Rand48.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Exceptions/ParameterNotFound.h>

#include <Core/Malloc/Allocator.h>
#include <Core/Util/Endian.h>
#include <Core/Util/DebugStream.h>
#include <Core/Math/MinMax.h>

#include <fstream>
#include <iostream>
#include <iomanip>


using namespace std;
using namespace Uintah;

static DebugStream dbg("VS", false);
static DebugStream dbgSig("VSSig", false);

ViscoScram::ViscoScram(ProblemSpecP& ps,MPMFlags* Mflag)
  : ConstitutiveModel(Mflag)
{
  ps->require("PR",d_initialData.PR);
  if( d_initialData.PR > 0.5 || d_initialData.PR < -1.0 )
  {
    ostringstream msg;
    msg << "ERROR: Poisson Ratio (ViscoSCRAM input parameter 'PR') must be greater than -1 and less than 0.5\n";
    throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
  }
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

  // Murnaghan EOS inputs
  ps->getWithDefault("useMurnahanEOS", d_useMurnahanEOS, false);
  ps->getWithDefault("useBirchMurnaghanEOS", d_useBirchMurnaghanEOS, false);
  if(d_useMurnahanEOS || d_useBirchMurnaghanEOS) {
    ps->require("gamma", d_murnahanEOSData.gamma);
    ps->require("P0",    d_murnahanEOSData.P0);
    ps->require("bulkPrime", d_murnahanEOSData.bulkPrime);
  }

  // JWL EOS inputs
  ps->getWithDefault("useJWLEOS", d_useJWLEOS, false);
  ps->getWithDefault("useJWLCEOS", d_useJWLCEOS, false);
  if(d_useJWLEOS || d_useJWLCEOS) {
    ps->require("A",d_JWLEOSData.A);
    ps->require("B",d_JWLEOSData.B);
    ps->require("R1",d_JWLEOSData.R1);
    ps->require("R2",d_JWLEOSData.R2);
    ps->require("om",d_JWLEOSData.om);
    // takes precedence over Murnaghan and Modified
    d_useMurnahanEOS = false;
    d_useBirchMurnaghanEOS = false;
  }
  if(d_useJWLEOS) {
    d_useJWLCEOS = false;
    ps->require("Cv",d_JWLEOSData.Cv);
 }
  if(d_useJWLCEOS) {
    ps->require("C",d_JWLEOSData.C);
  }
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

ViscoScram::ViscoScram(const ViscoScram* cm) : ConstitutiveModel(cm)
{
  std::cout << "Copying..." << std::endl;
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

  // Murnaghan EOS inputs
  d_useMurnahanEOS = cm->d_useMurnahanEOS;
  d_useBirchMurnaghanEOS = cm->d_useBirchMurnaghanEOS;
  if(d_useMurnahanEOS || d_useBirchMurnaghanEOS) {
    d_murnahanEOSData.gamma     = cm->d_murnahanEOSData.gamma;
    d_murnahanEOSData.P0        = cm->d_murnahanEOSData.P0;
    d_murnahanEOSData.bulkPrime = cm->d_murnahanEOSData.bulkPrime;
  }

  // JWL EOS inputs
  d_useJWLEOS = cm->d_useJWLEOS;
  if(d_useJWLEOS || d_useJWLCEOS){
    d_JWLEOSData.A =   cm->d_JWLEOSData.A;
    d_JWLEOSData.B =   cm->d_JWLEOSData.B;
    d_JWLEOSData.C =   cm->d_JWLEOSData.C;
    d_JWLEOSData.Cv =   cm->d_JWLEOSData.Cv;
    d_JWLEOSData.R1 =   cm->d_JWLEOSData.R1;
    d_JWLEOSData.R2 =   cm->d_JWLEOSData.R2;
    d_JWLEOSData.om =   cm->d_JWLEOSData.om;
 }

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


void ViscoScram::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","visco_scram");
  }

  cm_ps->appendElement("PR",d_initialData.PR);
  cm_ps->appendElement("CoeffThermalExpansion", d_initialData.CoefThermExp);
  cm_ps->appendElement("CrackParameterA",d_initialData.CrackParameterA);
  cm_ps->appendElement("CrackPowerValue",d_initialData.CrackPowerValue);
  cm_ps->appendElement("CrackMaxGrowthRate",d_initialData.CrackMaxGrowthRate);
  cm_ps->appendElement("StressIntensityF",d_initialData.StressIntensityF);
  cm_ps->appendElement("CrackFriction",d_initialData.CrackFriction);
  cm_ps->appendElement("InitialCrackRadius",d_initialData.InitialCrackRadius);
  cm_ps->appendElement("CrackGrowthRate",d_initialData.CrackGrowthRate);
  cm_ps->appendElement("G1",d_initialData.G[0]);
  cm_ps->appendElement("G2",d_initialData.G[1]);
  cm_ps->appendElement("G3",d_initialData.G[2]);
  cm_ps->appendElement("G4",d_initialData.G[3]);
  cm_ps->appendElement("G5",d_initialData.G[4]);
  cm_ps->appendElement("RTau1",d_initialData.RTau[0]);
  cm_ps->appendElement("RTau2",d_initialData.RTau[1]);
  cm_ps->appendElement("RTau3",d_initialData.RTau[2]);
  cm_ps->appendElement("RTau4",d_initialData.RTau[3]);
  cm_ps->appendElement("RTau5",d_initialData.RTau[4]);
  cm_ps->appendElement("Beta",d_initialData.Beta);
  cm_ps->appendElement("Gamma",d_initialData.Gamma);
  cm_ps->appendElement("DCp_DTemperature",d_initialData.DCp_DTemperature);
  cm_ps->appendElement("randomize_parameters", d_random);
  cm_ps->appendElement("use_time_temperature_equation", d_doTimeTemperature);
  cm_ps->appendElement("useModifiedEOS",d_useModifiedEOS);
  cm_ps->appendElement("useObjectiveRate",d_useObjectiveRate);

  // Murnaghan EOS inputs
  cm_ps->appendElement("useMurnahanEOS", d_useMurnahanEOS);
  cm_ps->appendElement("useBirchMurnaghanEOS", d_useBirchMurnaghanEOS);
  if(d_useMurnahanEOS || d_useBirchMurnaghanEOS) {
    cm_ps->appendElement("gamma", d_murnahanEOSData.gamma);
    cm_ps->appendElement("P0",    d_murnahanEOSData.P0);
    cm_ps->appendElement("bulkPrime", d_murnahanEOSData.bulkPrime);
  }

  // JWL EOS inputs
  cm_ps->appendElement("useJWLEOS", d_useJWLEOS);
  cm_ps->appendElement("useJWLCEOS", d_useJWLCEOS);
  if(d_useJWLEOS || d_useJWLCEOS){
    cm_ps->appendElement("A",     d_JWLEOSData.A);
    cm_ps->appendElement("B",     d_JWLEOSData.B);
    cm_ps->appendElement("C",     d_JWLEOSData.C);
    cm_ps->appendElement("Cv",     d_JWLEOSData.Cv);
    cm_ps->appendElement("R1",     d_JWLEOSData.R1);
    cm_ps->appendElement("R2",     d_JWLEOSData.R2);
    cm_ps->appendElement("om",     d_JWLEOSData.om);
  }

  // Time-temperature data for relaxtion time calculation
  cm_ps->appendElement("T0", d_tt.T0_WLF);
  cm_ps->appendElement("C1", d_tt.C1_WLF);
  cm_ps->appendElement("C2", d_tt.C2_WLF);
}


ViscoScram*
ViscoScram::clone()
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
  new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
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
  Matrix3 zero(0.);
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
  double K_0   = d_initialData.StressIntensityF;
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
  constParticleVariable<Point>     px;
  constParticleVariable<Vector>    pVelocity;
  constParticleVariable<Matrix3>   psize;
  constParticleVariable<Matrix3>   pDefGrad, pStress;
  constNCVariable<Vector>          gvelocity, Gvelocity;
  constParticleVariable<double>    pTempPrev;

  ParticleVariable<double>    pVol_new, pdTdt, p_q;
  ParticleVariable<Matrix3>   pDefGrad_new, pStress_new, pStrainRate_new;
  ParticleVariable<double>    pVolHeatRate_new, pVeHeatRate_new;
  ParticleVariable<double>    pCrHeatRate_new, pCrackRadius_new;
  ParticleVariable<double>    pRand;
  ParticleVariable<StateData> pStatedata;

  // Loop thru patches
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<Vector> d_S(interpolator->size());
    vector<double> S(interpolator->size());

    // initialize strain energy and wavespeed to zero
    double se = 0;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

    // Get patch information
    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    // Get material information
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    // Get the particle and grid data for the current patch
    old_dw->get(px,                  lb->pXLabel,                  pset);
    old_dw->get(pMass,               lb->pMassLabel,               pset);
    old_dw->get(pVol,                lb->pVolumeLabel,             pset);
    old_dw->get(pTemperature,        lb->pTemperatureLabel,        pset);
    old_dw->get(psize,               lb->pSizeLabel,               pset);
    old_dw->get(pVelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(pDefGrad,            lb->pDeformationMeasureLabel, pset);
    old_dw->get(pStress,             lb->pStressLabel,             pset);
    old_dw->get(pCrackRadius,        pCrackRadiusLabel,            pset);
    old_dw->get(pTempPrev,           lb->pTempPreviousLabel,       pset); 
    if (flag->d_fracture) {
      new_dw->get(pgCode,            lb->pgCodeLabel,              pset);
      new_dw->get(Gvelocity,         lb->GVelocityStarLabel,dwi,patch,gac,NGN);
    }
    new_dw->get(gvelocity,           lb->gVelocityStarLabel,dwi,patch,gac,NGN);

    // Allocate arrays for the updated particle data for the current patch
    new_dw->allocateAndPut(pVol_new,         
                           lb->pVolumeLabel_preReloc,             pset);
    new_dw->allocateAndPut(pdTdt, 
                           lb->pdTdtLabel_preReloc,               pset);
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
    new_dw->allocateAndPut(p_q,   lb->p_qLabel_preReloc,          pset);

    old_dw->copyOut(pRand,           pRandLabel,                  pset);
    old_dw->copyOut(pStatedata,      pStatedataLabel,             pset);
    ASSERTEQ(pset, pStatedata.getParticleSubset());

    // Loop thru particles in the patch
    ParticleSubset::iterator iter = pset->begin();
    for(; iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Assign zero internal heating by default - modify if necessary.
      pdTdt[idx] = 0.0;

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

      Matrix3 velGrad(0.0);
      short pgFld[27];
      if (flag->d_fracture) {
        for(int k=0; k<27; k++){
          pgFld[k]=pgCode[idx][k];
        }
        // Get the node indices that surround the cell
        interpolator->findCellAndShapeDerivatives(px[idx], ni, d_S,psize[idx],pDefGrad[idx]);
        computeVelocityGradient(velGrad,ni,d_S,oodx,pgFld,gvelocity,Gvelocity);
      } else {
        if(!flag->d_axisymmetric){
         // Get the node indices that surround the cell
         interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],pDefGrad[idx]);

         computeVelocityGradient(velGrad,ni,d_S, oodx, gvelocity);
        } else {  // axi-symmetric kinematics
         // Get the node indices that surround the cell
         interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,
                                                                    psize[idx],pDefGrad[idx]);
         // x -> r, y -> z, z -> theta
         computeAxiSymVelocityGradient(velGrad,ni,d_S,S,oodx,gvelocity,px[idx]);
        }

      }

      // Compute the deformation gradient increment using the time_step
      // velocity gradient (F_n^np1 = dudx * dt + Identity)
      Matrix3 pDefGradInc = velGrad * delT + Identity;
      double Jinc = pDefGradInc.Determinant();

      // Update the deformation gradient tensor to its time n+1 value.
      pDefGrad_new[idx] = pDefGradInc*pDefGrad[idx];
      double J = pDefGrad_new[idx].Determinant();
      if (!(J > 0.0)) {
        pDefGrad_new[idx] = pDefGrad[idx];
        J = pDefGrad_new[idx].Determinant();
        cout << getpid() 
             << "**WARNING** Negative Jacobian of deformation gradient" << endl;
        cout << "particle mass = " << pMass[idx]  << endl;
        cout << "Prev. step def. grad. will be used" << endl;
      }
      double rho_cur = rho_0/J;

      // Update the volume
      pVol_new[idx] = Jinc*pVol[idx];

      // Calculate rate of deformation D 
      Matrix3 D = (velGrad + velGrad.Transpose())*0.5;

      // Get stress at time t_n
      double sigm_old = onethird*(pStress[idx].Trace()); //Eq 5
      Matrix3 sigdev_old = pStress[idx] - Identity*sigm_old;

      // For objective rates (rotation neutralized)
      Matrix3 RR(0.0), UU(0.0), RT(0.0);
      if (d_useObjectiveRate) {

        // Compute polar decomposition of F
        pDefGrad_new[idx].polarDecompositionRMB(UU, RR);
        RT = RR.Transpose();

        // If we want objective rates, rotate stress and rate of
        // deformation to material coordinates using R where F = RU
        sigdev_old = zero;
        for (int ii = 0; ii < 5; ++ii) {
          pStatedata[idx].DevStress[ii] = 
            RT*(pStatedata[idx].DevStress[ii]*RR); 
          sigdev_old += pStatedata[idx].DevStress[ii];
        }
        D = RT*(D*RR);
      }
     
      // Subtract the thermal expansion to get D_e + D_p
      double dT_dt = (pTemperature[idx] - pTempPrev[idx])/delT;
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

      //if (dbg.active()) {
      //  dbg << "D.Norm() = " << D.Norm()
      //      << " Ddev.Norm() = " << DPrime.Norm()
      //      << " Sig.Norm() = " << pStress[idx].Norm()
      //      << " SigDev.Norm() = " << sigdev_old.Norm() << endl;
      //}

      //old deviatoric stress norm
      double DevStressNormSq = sigdev_old.NormSquared();
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

      int compflag = 0;
      //!!NOTE!! This should be >= 0?
      if (sigm_old > 0.0) compflag = -1; 
      vres        *= ((1 + compflag) - cdot0*compflag);
      double sigmae = sqrt(DevStressNormSq - compflag*(3*sigm_old*sigm_old));  //Sij*Sij or Oij*Oij, is this right?

      // Stress intensity factor
      double crad   = pCrackRadius[idx];
      ASSERT(crad >= 0.0);
      double sqrtc  = sqrt(crad);
      double K_I    = sqrtopf*sqrtPI*sqrtc*sigmae; //Eq 26 or 27

      // Modification to include friction on crack faces
      double xmup   = (1 + compflag)*sqrt(45./(2.*(3. - 2.*cf*cf)))*cf; //Equation 31 
      //Next 4 equations are equation 30
      double a      = -xmup*sigm_old*sqrtc;         //eq 30 -numerator 
      double b      = 1. + a/K_0;                   //eq 30 2nd term 
      double termm  = 1. + M_PI*a*b/K_0;            //eq 30 inner sqrt term
      double K_0m   = K_0*sqrt(termm);              //eq 30 K_0m
      double Kprime = K_0m*sqrt(1. + (2./mm));      //eq 28
      double K_1    = Kprime*pow((1. + (mm/2.)), 1./mm); //eq 29

      if(vres > d_initialData.CrackMaxGrowthRate){
        vres = d_initialData.CrackMaxGrowthRate;
      }
      //if (dbg.active()) {
      //  dbg  << "vres = " << vres << " K_I = " << K_I 
      //       << " sigmae = " << sigmae << endl;
      //  dbg  << "crad = " << crad << " xmup = " << xmup << " a = " << a
      //       << " b = " << b << " termm = " << termm << " K_0m = " << K_0m
      //       << " Kprime = " << Kprime << " K_1 = " << K_1 << endl;
      //}

      double cdot,cc,rk1c,rk2c,rk3c,rk4c;

      // cdot is crack speed
      // Use fourth order Runge Kutta integration to find new crack radius
      
      if(K_I < Kprime ){
        double fac = pow(K_I/(K_1 * sqrtc), mm);
        cdot = vres*pow((K_I/K_1), mm); //Equation 24, denomenator is K_1 not K'
        cc   = vres*delT;
        rk1c = cc * fac * pow(sqrtc, mm);
        rk2c = cc * fac * pow(sqrt(crad+.5*rk1c), mm);
        rk3c = cc * fac * pow(sqrt(crad+.5*rk2c), mm);
        rk4c = cc * fac * pow(sqrt(crad+rk3c),    mm);
      }
      else{
        double fac = K_0m*K_0m * crad/(K_I * K_I);
        cdot = vres*(1. - K_0m*K_0m/(K_I*K_I));
        cc   = vres*delT;
        rk1c = cc*(1. - fac/(crad) );
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
      double delTinv = 1.0/delT;
      for(int imw=0;imw<5;imw++){

        // If the relaxation time is smaller than delT, assume that
        // the deviatoric stress in the Maxwell element is zero
        if (d_doTimeTemperature) {
          if (RTau[imw] > 0.1*delTinv) {
            pStatedata[idx].DevStress[imw] = zero;
            continue;
          }
        }

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


      //if (dbgSig.active()) {
      //  dbgSig << "  K = " << bulk << " ekkdot = " << ekkdot 
      //         << " delT = " << delT << endl;
      //  dbgSig << "  pold = " << sigm_old ;
      //}

      if (d_useObjectiveRate) {

        // Rotate everything back 
        for (int ii = 0; ii < 5; ++ii) {
          pStatedata[idx].DevStress[ii] = 
            RR*(pStatedata[idx].DevStress[ii]*RT); 
        }
        D = RR*(D*RT);
      }

      double ekkdot = D.Trace();
      double sigm_new = sigm_old + ekkdot*bulk*delT;
      Matrix3 sigdev_new = zero;
      for (int ii = 0; ii < 5; ++ii) {
         sigdev_new += pStatedata[idx].DevStress[ii];
      }
      pStress_new[idx] = sigdev_new + Identity*sigm_new;

      Matrix3 sig = pStress_new[idx];


      //if (dbgSig.active()) {
      //  dbgSig << " pnew = " << sigm_new << endl;
      //  dbgSig << "  S_dev = [" 
      //         << sigdev_new(0,0) << " " << sigdev_new(1,1) << " " 
      //         << sigdev_new(2,2) << " " << sigdev_new(1,2) << " " 
      //         << sigdev_new(2,0) << " " << sigdev_new(0,1) << "]" << endl;
      //  dbgSig << "  sig = [" 
      //         << sig(0,0) << " " << sig(1,1) << " " << sig(2,2) << " " 
      //         << sig(1,2) <<" "<< sig(2,0) << " " << sig(0,1) << "]" << endl;
      //}

      // Update crack radius
      double deltacrad = onesixth*(rk1c + rk4c) + onethird*(rk2c + rk3c);
      if(deltacrad<0)
        throw InternalError("Error crack is healing\n",__FILE__,__LINE__);

      crad += deltacrad;

      pCrackRadius_new[idx] = crad;
      //if (dbgSig.active())
      //  dbgSig << " Crack Radius = " << crad << endl;

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
        /*
        if (pTemperature[idx] > 450.0) {
          cout << "\tidx = " << idx << " j = " << imw 
               << "\n\t\t S_j:S_j = " << pStatedata[idx].DevStress[imw].NormSquared()
               << " mu_j = " << Gmw[imw] << " tau_j = " << RTau[imw] 
               << " wdot_j = " << svedot << endl;
        }
        */
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
      Matrix3 SRate = (D*(2.0*G) - sigdev_new*topc - sovertau)*oocoa3;
      double scrdot = (sigdev_new.NormSquared()*topc + 
                       sigdev_new.Contract(SRate)*coa3)/(2.0*G);

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

      // Update the total internal heat rate  (this is from Hackett and Bennett,
      // IJNME, 2000, 49:1191-1209)
      pdTdt[idx] = -pVolHeatRate_new[idx] + pVeHeatRate_new[idx] +
                               pCrHeatRate_new[idx];
      /*
      if (pTemperature[idx] > 450.0) {
      cout << "\t idx = " << idx << "\n\t\t qdot_v = " << pVolHeatRate_new[idx]
                       << " T = " << pTemperature[idx] << " Tr(edot) = " << ekkdot
                       << "\n\t\t qdot_ve = " << pVeHeatRate_new[idx]
                       << "\n\t\t qdot_cr = " << pCrHeatRate_new[idx]
                       << "\n\t\t qdot = " << pdTdt[idx] << endl;
      }
      */

      // Compute the strain energy for all the particles
      Matrix3 sigma = (pStress_new[idx] + pStress[idx])*.5;
      se += (D.Contract(sigma))*pVol_new[idx]*delT;

      // Compute wave speed at each particle, store the maximum
      Vector pVel = pVelocity[idx];
      double c_dil = sqrt((bulk + 4.*G/3.)/rho_cur);
      WaveSpeed=Vector(Max(c_dil+fabs(pVel.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pVel.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pVel.z()),WaveSpeed.z()));

      // Compute artificial viscosity term
      if (flag->d_artificial_viscosity) {
        double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;
        double c_bulk = sqrt(bulk/rho_cur);
        Matrix3 D=(velGrad + velGrad.Transpose())*0.5;
        p_q[idx] = artificialBulkViscosity(D.Trace(), c_bulk, rho_cur, dx_ave);
      } else {
        p_q[idx] = 0.;
      }
    }  // end loop over particles

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    //Timesteps larger than 1 microsecond cause VS to be unstable
    delT_new = min(1.e-6, delT_new);

    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
    
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(se),     lb->StrainEnergyLabel);
    }
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
    new_dw->put(delt_vartype(1.e10), lb->delTLabel, patch->getLevel());
    
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(0.),   lb->StrainEnergyLabel);
    }
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


//______________________________________________________________________
//  Computes the reference density using a combined bisection and Newton
//  Method.
//  See:  /src/CCA/Components/MPM/ConstitutiveModels/Docs/ModifiedJWL.pdf
//  for details. ?????

void  ViscoScram::computeRhoRef(const double rho_orig,
                                const double p_ref,
                                const double temperature,
                                const double pressure,
                                double& rho_refrr,
                                double& K0)
{
  double delta_old;
  double delta_new;
  double f       = 0;
  double df_drho = 0;
  double Cv      = d_JWLEOSData.Cv;
  double epsilon = 1.0e-10;
  double rho_min = 0.0;                      // Such that f(min) < 0
  double rho_max = 100000.0; //pressure*1.001*rho_orig/(om*Cv*temperature)*1.3431907e7; // Such that f(max) > 0

  IterationVariables iterVar;
  iterVar.Pressure     = p_ref;
  iterVar.Temperature  = temperature;
  iterVar.SpecificHeat = Cv;
  iterVar.IL           = rho_min;
  iterVar.IR           = rho_max;

  rho_refrr = rho_orig;
  
  
  //double rhoM_start = rhoM;

  int iter = 0;
  double A  = d_JWLEOSData.A;
  double B  = d_JWLEOSData.B;
  double R1 = d_JWLEOSData.R1;
  double R2 = d_JWLEOSData.R2;
  double om = d_JWLEOSData.om;
  while(1){

    double V  = rho_orig/(rho_refrr+1e-100);                       
    double P1 = A*exp(-R1*V);                                      
    double P2 = B*exp(-R2*V);                                      
    double P3 = om*iterVar.SpecificHeat*iterVar.Temperature/V;     
    f = (P1 + P2 + P3) - iterVar.Pressure;

    setInterval(f, rho_refrr, &iterVar);
    if(fabs((iterVar.IL-iterVar.IR)/rho_refrr) < epsilon){
      rho_refrr = (iterVar.IL+iterVar.IR)/2.0;
      break;
    }

    delta_new   = 1e100;
    bool breakOuterLoop = false;
    while(1){
      double V  = rho_orig/(rho_refrr +1e-100);
      double P1 = A*exp(-R1*V);
      double P2 = B*exp(-R2*V);
      double P3 = om*iterVar.SpecificHeat*iterVar.Temperature/V;

      df_drho = (P1*R1*V + P2*R2*V+P3)/rho_refrr;
      delta_old = delta_new;
      delta_new = -f/df_drho;
      rho_refrr += delta_new;

      if(fabs(delta_new/rho_refrr) < epsilon){
        breakOuterLoop = true;
        break;
      }

      if(iter>=100){
        ostringstream warn;
        warn << setprecision(15);
        warn << "ERROR:MPM:ViscoSCRAM:ComputingRho_ref. \n";
        warn << "press= " << pressure << " temp=" << temperature << "\n";
        warn << "delta= " << delta_new << " rhoM= " << rho_refrr << " f = " << f
             <<" df_drho =" << df_drho << "\n";
        throw InternalError(warn.str(), __FILE__, __LINE__);
      }
      
      if(rho_refrr<iterVar.IL ||
         rho_refrr>iterVar.IR ||
         fabs(delta_new) > fabs(delta_old*0.7)){
        break;
      }else{
        double V  = rho_orig/(rho_refrr + 1e-100);
        double P1 = A*exp(-R1*V);
        double P2 = B*exp(-R2*V);
        double P3 = om * iterVar.SpecificHeat * iterVar.Temperature/V;
        f = (P1 + P2 + P3) - iterVar.Pressure;
      }
      setInterval(f, rho_refrr, &iterVar);
      iter++;
    }

    if(breakOuterLoop == true)
      break;
    rho_refrr = (iterVar.IL + iterVar.IR)/2.0;
    iter++;
  }   

 double v = rho_orig/rho_refrr;
 K0 = v*(A*R1*exp(-R1*v)
        +B*R2*exp(-R2*v))
       +temperature*Cv*om/v;

}
//______________________________________________________________________
//

double ViscoScram::computeRhoMicroCM(double pressure,
                                     const double p_ref,
                                     const MPMMaterial* matl,
                                     double temperature,
                                     double rho_guess)
{

  double rho_orig = matl->getInitialDensity();
  double rho_cur;


  double rho_refrr = rho_orig;
  double K0 = d_bulk;
 
  //determining rho_ref so pressure does not go negative
  //modified EOS is used when pressure is lower than p_ref
  if(d_useJWLEOS && d_useModifiedEOS) {
    double K0        = -987654321;
    rho_refrr = -987654321;
    computeRhoRef(rho_orig, p_ref,temperature, pressure, rho_refrr, K0);   
  }
  double p_gauge = pressure - p_ref;

  // For expansion beyond relative volume = 1
  //  Used to prevent negative pressures
  if(d_useModifiedEOS && p_gauge < 0.0) {        // MODIFIED EOS

    double A = p_ref;   
    double n = A/K0;
    rho_cur  = rho_refrr*pow(pressure/A,n);
    
  } else if(d_useJWLEOS) {                        // JWL EOS

    double delta_old;
    double delta_new;
    double f       = 0;
    double df_drho = 0;
    double Cv      = d_JWLEOSData.Cv;
    double epsilon = 1.0e-15;
    double rho_min = 0.0;                      // Such that f(min) < 0
    double rho_max = 100000.0; //pressure*1.001*rho_orig/(om*Cv*temperature)*1.3431907e7; // Such that f(max) > 0
    
    IterationVariables iterVar;
    iterVar.Pressure     = pressure;
    iterVar.Temperature  = temperature;
    iterVar.SpecificHeat = Cv;
    iterVar.IL           = rho_min;
    iterVar.IR           = rho_max;

    double rho_cur = rho_guess <= rho_max ? rho_guess : rho_max/2.0;
    //double rhoM_start = rhoM;

    int iter = 0;
    while(1){
      f = computePJWL(rho_cur,matl->getInitialDensity(), &iterVar);
      setInterval(f, rho_cur, &iterVar);
      if(fabs((iterVar.IL-iterVar.IR)/rho_cur) < epsilon){
        return (iterVar.IL+iterVar.IR)/2.0;
      }

      delta_new   = 1e100;
      while(1){
        df_drho   = computedPdrhoJWL(rho_cur,matl->getInitialDensity(), &iterVar);
        delta_old = delta_new;
        delta_new = -f/df_drho;
        rho_cur  += delta_new;

        if(fabs(delta_new/rho_cur) < epsilon){
          return rho_cur;
        }

        if(iter>=100){
          ostringstream warn;
          warn << setprecision(15);
          warn << "ERROR:MPM:ViscoSCRAM:JWL::computeRhoMicro not converging. \n";
          warn << "press= " << pressure << " temp=" << temperature << "\n";
          warn << "delta= " << delta_new << " rhoM= " << rho_cur << " f = " << f
               <<" df_drho =" << df_drho << "\n";
          throw InternalError(warn.str(), __FILE__, __LINE__);
        }

        if(rho_cur<iterVar.IL || 
           rho_cur>iterVar.IR || 
           fabs(delta_new) > fabs(delta_old*0.7)){
          break;
        }

        f = computePJWL(rho_cur,matl->getInitialDensity(),&iterVar);
        setInterval(f, rho_cur, &iterVar);
        iter++;
      }

      rho_cur = (iterVar.IL+iterVar.IR)/2.0;
      iter++;
    }

  } else if(d_useJWLCEOS) {                // JWLC EOS

    double A    = d_JWLEOSData.A;
    double B    = d_JWLEOSData.B;
    double C    = d_JWLEOSData.C;
    double R1   = d_JWLEOSData.R1;
    double R2   = d_JWLEOSData.R2;
    double rhoM = rho_orig;

    double f;
    double df_drho;
    double relfac  = 0.9;
    double epsilon = 1.e-15;
    double delta   = 1.0;
    int count      = 0;

    double one_plus_omega = 1.0+d_JWLEOSData.om;

    while(fabs(delta/rhoM) > epsilon){
      double inv_rho_rat = rho_orig/rhoM;
      double rho_rat     = rhoM/rho_orig;
      double A_e_to_the_R1_rho0_over_rhoM   = A*exp(-R1*inv_rho_rat);        // A-Term
      double B_e_to_the_R2_rho0_over_rhoM   = B*exp(-R2*inv_rho_rat);        // B-Term
      double C_rho_rat_tothe_one_plus_omega = C*pow(rho_rat,one_plus_omega); // C-Term

      f = (A_e_to_the_R1_rho0_over_rhoM +
           B_e_to_the_R2_rho0_over_rhoM + 
           C_rho_rat_tothe_one_plus_omega) - pressure;

      double rho0_rhoMsqrd = rho_orig/(rhoM*rhoM);
      df_drho = R1*rho0_rhoMsqrd*A_e_to_the_R1_rho0_over_rhoM
              + R2*rho0_rhoMsqrd*B_e_to_the_R2_rho0_over_rhoM
              + (one_plus_omega/rhoM)*C_rho_rat_tothe_one_plus_omega;

      delta = -relfac*(f/df_drho);
      rhoM += delta;
      rhoM  = fabs(rhoM);

      if(count >= 100){

        // The following is here solely to help figure out what was going on
        // at the time the above code failed to converge.  Start over with this
        // copy and print more out.
        delta = 1.;
        rhoM = 2.*rho_orig;
        while(fabs(delta/rhoM) > epsilon){
         double inv_rho_rat = rho_orig/rhoM;
         double rho_rat     = rhoM/rho_orig;
         double A_e_to_the_R1_rho0_over_rhoM   = A * exp(-R1*inv_rho_rat);
         double B_e_to_the_R2_rho0_over_rhoM   = B * exp(-R2*inv_rho_rat);
         double C_rho_rat_tothe_one_plus_omega = C * pow(rho_rat,one_plus_omega);

         f = (A_e_to_the_R1_rho0_over_rhoM +
              B_e_to_the_R2_rho0_over_rhoM +
              C_rho_rat_tothe_one_plus_omega) - pressure;

         double rho0_rhoMsqrd = rho_orig/(rhoM*rhoM);
         df_drho =  R1 * rho0_rhoMsqrd * A_e_to_the_R1_rho0_over_rhoM
                  + R2 * rho0_rhoMsqrd * B_e_to_the_R2_rho0_over_rhoM
                  + (one_plus_omega/rhoM) * C_rho_rat_tothe_one_plus_omega;

         delta = -relfac*(f/df_drho);
         rhoM += delta;
         rhoM  = fabs(rhoM);
         if(count >= 150){
           ostringstream warn;
           warn << "ERROR:MPM:ViscoScram:JWLC::computeRhoMicro not converging. \n";
           warn << "press= " << pressure << "\n";
           warn << "delta= " << delta << " rhoM= " << rhoM << " f = " << f
                <<" df_drho =" << df_drho << "\n";
           throw InternalError(warn.str(), __FILE__, __LINE__);

         }
        count++;
        }
      }
      count++;
    }
    // copy local rhoM to function rho_cur
    rho_cur = rhoM;


  } else if(d_useMurnahanEOS) {    // Murnaghan EOS


    double bulkPrime = d_murnahanEOSData.bulkPrime;
    double P0        = d_murnahanEOSData.P0;
    double gamma     = d_murnahanEOSData.gamma;

    if( pressure >= P0 ) {
      rho_cur = rho_orig * pow((bulkPrime*gamma*(pressure-P0)+1.0),1.0/gamma);
    } else {
      rho_cur = rho_orig * pow((pressure/P0), bulkPrime*P0);
    }


  } else if(d_useBirchMurnaghanEOS) {    // Birch Murnaghan EOS


      // Use normal Birch-Murnaghan EOS
      //  Solved using Newton Method code adapted from JWLC.cc
      double f;                // difference between current and previous function value
      double df_drho;          // rate of change of function value
      double epsilon = 1.e-15; // convergence limit
      double delta   = 1.0;    // change in rhoM each step
      double relfac  = 0.9;
      int count      = 0;      // counter of total iterations
      double rhoM    = rho_orig;
      double rho0    = rho_orig;

      while(fabs(delta/rhoM) > epsilon){  // Main Iterative loop
        // Compute the difference between the previous pressure and the new pressure
        f       = computePBirchMurnaghan(rho0/rhoM) - pressure;

        // Compute the new pressure derivative
        df_drho = computedPdrhoBirchMurnaghan(rhoM, rho0);

        // factor by which to adjust rhoM
        delta = -relfac*(f/df_drho);
        rhoM +=  delta;
        rhoM  =  fabs(rhoM);

        if(count >= 100){
          // The following is here solely to help figure out what was going on
          // at the time the above code failed to converge.  Start over with this
          // copy and print more out.
          delta = 1.0;
          rhoM  = 1.5*rho0;

          while(fabs(delta/rhoM) > epsilon){
            f       = computePBirchMurnaghan(rho0/rhoM) - pressure;
            df_drho = computedPdrhoBirchMurnaghan(rhoM, rho0);

            // determine by how much to change
            delta = -relfac*(f/df_drho);
            rhoM +=  delta;
            rhoM  =  fabs(rhoM);

            // After 50 more iterations finally quit out
            if(count >= 150){
              ostringstream warn;
              warn << std::setprecision(15);
              warn << "ERROR:ICE:BirchMurnaghan::computeRhoMicro(...) not converging. \n";
              warn << "press= " << pressure << "\n";
              warn << "delta= " << delta << " rhoM= " << rhoM << " f = " << f
                   <<" df_drho =" << df_drho << " rho_guess =" << rho_guess << "\n";
              throw InternalError(warn.str(), __FILE__, __LINE__);
           }
           count++;
          }
        }
        count++;
      }
      return rhoM;

   
  } else {                      // STANDARD EOS


    double p_g_over_bulk = p_gauge/d_bulk;
    rho_cur              = rho_orig*(p_g_over_bulk + sqrt(p_g_over_bulk*p_g_over_bulk +1.));


  }

  return rho_cur;

}
void ViscoScram::computePressEOSCM(double rho_cur,
                                   double& pressure,
                                   double p_ref,
                                   double& dp_drho, 
                                   double& tmp,
                                   const MPMMaterial* matl, 
                                   double temperature)
{
  double rho_orig = matl->getInitialDensity();
  double inv_rho_orig = 1.0/rho_orig;

  double rho_refrr = rho_orig;
  double K0 = d_bulk;
  
  //determining rho_ref so pressure does not go negative
  if(d_useJWLEOS && d_useModifiedEOS) {
    double K0        = -987654321;
    rho_refrr = -987654321;
    computeRhoRef(rho_orig, p_ref,temperature, pressure, rho_refrr, K0);  
  }


  // If we are expanding beyond relative volume = 1, then we need to prevent negative pressures
  if(d_useModifiedEOS && rho_cur < rho_refrr) {


    double A = p_ref;         // MODIFIED EOS
    double n = K0/A;
    double invRhoRef = 1.0/rho_refrr;
    double rho_rat_to_the_n = pow(rho_cur*invRhoRef,n);
    pressure = A * rho_rat_to_the_n;
    dp_drho  = (K0/rho_cur)*rho_rat_to_the_n;
    tmp      = dp_drho;       // speed of sound squared


  } else if(d_useJWLEOS) {    // TEMPERATURE DEPENDENT JWL EQUATION OF STATE


    double A  = d_JWLEOSData.A;
    double B  = d_JWLEOSData.B;
    double Cv = d_JWLEOSData.Cv;
    double R1 = d_JWLEOSData.R1;
    double R2 = d_JWLEOSData.R2;
    double om = d_JWLEOSData.om;

    double V  = rho_orig/rho_cur;
    double P1 = A*exp(-R1*V);          // A-Term
    double P2 = B*exp(-R2*V);          // B-Term
    double P3 = om*Cv*temperature/V;   // Ideal solid term

    pressure      = P1 + P2 + P3;
    dp_drho       = (R1*rho_orig*P1 + R2*rho_orig*P2)/(rho_cur*rho_cur)
                  + om*Cv*temperature/rho_orig;
    tmp           = dp_drho;     // speed of sound squared


  } else if(d_useJWLCEOS) {      // TEMPERATURE INDEPENDENT JWL EQUATION OF STATE


    double A = d_JWLEOSData.A;
    double B = d_JWLEOSData.B;
    double C = d_JWLEOSData.C;
    double R1 = d_JWLEOSData.R1;
    double R2 = d_JWLEOSData.R2;

    double one_plus_omega = 1.0+d_JWLEOSData.om;  // Adiabatic index
    double inv_rho_rat    = rho_orig/rho_cur;
    double rho_rat        = rho_cur/rho_orig;
    double A_e_to_the_R1_rho0_over_rhoM   = A*exp(-R1*inv_rho_rat);          // A-Term
    double B_e_to_the_R2_rho0_over_rhoM   = B*exp(-R2*inv_rho_rat);          // B-Term
    double C_rho_rat_tothe_one_plus_omega = C*pow(rho_rat,one_plus_omega);   // C-Term

    pressure = A_e_to_the_R1_rho0_over_rhoM +
               B_e_to_the_R2_rho0_over_rhoM + C_rho_rat_tothe_one_plus_omega;

    double rho0_rhoMsqrd = rho_orig/(rho_cur*rho_cur);
    dp_drho  = R1*rho0_rhoMsqrd*A_e_to_the_R1_rho0_over_rhoM
             + R2*rho0_rhoMsqrd*B_e_to_the_R2_rho0_over_rhoM
             + (one_plus_omega/rho_cur)*C_rho_rat_tothe_one_plus_omega;
    tmp      = dp_drho;       // speed of sound squared


  } else if(d_useMurnahanEOS) {  // 1ST ORDER MURNAGHAN EQUATION OF STATE


    double bulkPrime = d_murnahanEOSData.bulkPrime;
    double P0        = d_murnahanEOSData.P0;
    double gamma     = d_murnahanEOSData.gamma;

    if(rho_cur >= rho_orig) {    // Compression
      pressure = P0 + (1.0/(bulkPrime*gamma))*(pow(rho_cur/rho_orig,gamma)-1.0);
      dp_drho  = (1.0/(bulkPrime*rho_orig))*pow((rho_cur/rho_orig),gamma-1.0);
      tmp      = dp_drho;
    } else {                     // Expansion
      pressure = P0*pow(rho_cur/rho_cur, (1.0/(bulkPrime*P0)));
      dp_drho  = (1.0/(bulkPrime*rho_orig))*pow(rho_cur/rho_orig,(1.0/(bulkPrime*P0)-1.0));
      tmp      = d_bulk/rho_cur;
    } 


  } else if(d_useBirchMurnaghanEOS) { // 3RD ORDER BIRCH-MURNAGHAN EQUATION OF STATE


    if(rho_cur >= rho_orig) {         // Compression
      double v = rho_orig/rho_cur;    // reduced volume
      pressure = d_murnahanEOSData.P0 + computePBirchMurnaghan(v);
      dp_drho  = computedPdrhoBirchMurnaghan(rho_cur, rho_orig);
    } else {                          // Expansion
      pressure = d_murnahanEOSData.P0*pow(rho_cur/rho_cur, (1.0/(d_murnahanEOSData.bulkPrime*d_murnahanEOSData.P0)));
      dp_drho  = (1.0/(d_murnahanEOSData.bulkPrime*rho_orig))*pow(rho_cur/rho_orig,(1.0/(d_murnahanEOSData.bulkPrime*d_murnahanEOSData.P0)-1.0));
      tmp      = d_murnahanEOSData.bulkPrime/rho_cur;
    }


  } else {                      // STANDARD EOS            


    double p_g = 0.5*d_bulk*(rho_cur*inv_rho_orig - rho_orig/rho_cur);
    pressure   = p_ref + p_g;
    dp_drho    = 0.5*d_bulk*(rho_orig/(rho_cur*rho_cur) + inv_rho_orig);
    tmp        = d_bulk/rho_cur;  // speed of sound squared


  }
}

double ViscoScram::getCompressibility()
{
  return 1.0/d_bulk;
}

//_____________________________________________________
// Functions used in solution of the BirchMurnaghan EOS
double ViscoScram::computePBirchMurnaghan(double v)
{
  double K = d_murnahanEOSData.bulkPrime;
  double n = d_murnahanEOSData.gamma;
  double P = 3.0/(2.0*K) * (pow(v,-7.0/3.0) - pow(v,-5.0/3.0))
                             * (1.0 + 0.75*(n-4.0)*(pow(v,-2.0/3.0)-1.0))
           + d_murnahanEOSData.P0;

  return P;

}

double ViscoScram::computedPdrhoBirchMurnaghan(double rho, double rho0)
{
  double v = rho0/rho;

  double K = d_murnahanEOSData.bulkPrime;
  double n = d_murnahanEOSData.gamma;
  double dPdr = 1.5/K * (7.0*rho0/(3.0*pow(v,10.0/3.0)*rho*rho) - 5.0*rho0/(3.0*pow(v,8.0/3.0)*rho*rho))
              * (1.0 + 0.75*(n-4.0)*(1.0/(pow(v,2.0/3.0)) - 1.0)) 
              + (0.75/K * ((1/pow(v,7.0/3.0) - 1.0/pow(v,5.0/3.0))*(n-4.0)*rho0)/(pow(v,5.0/3.0)*rho*rho));

  return dPdr;

}

//____________________________________________________________________________
// Functions used in Newton-Bisection Solver for JWL Temperature Dependent EOS
double ViscoScram::computePJWL(double rhoM, double rho0, IterationVariables *iterVar){
  double A  = d_JWLEOSData.A;
  double B  = d_JWLEOSData.B;
  double R1 = d_JWLEOSData.R1;
  double R2 = d_JWLEOSData.R2;
  double om = d_JWLEOSData.om;

  if(rhoM == 0){
    return -(iterVar->Pressure);
  }
  double V  = rho0/rhoM;
  double P1 = A*exp(-R1*V);
  double P2 = B*exp(-R2*V);
  double P3 = om*iterVar->SpecificHeat*iterVar->Temperature/V;
  return (P1 + P2 + P3) - iterVar->Pressure;
}

double ViscoScram::computedPdrhoJWL(double rhoM, double rho0, IterationVariables *iterVar){
  double A  = d_JWLEOSData.A;
  double B  = d_JWLEOSData.B;
  double R1 = d_JWLEOSData.R1;
  double R2 = d_JWLEOSData.R2;
  double om = d_JWLEOSData.om;

  double V  = rho0/rhoM;
  double P1 = A*exp(-R1*V);
  double P2 = B*exp(-R2*V);
  double P3 = om*iterVar->SpecificHeat*iterVar->Temperature/V;
  return (P1*R1*V + P2*R2*V+P3)/rhoM;
}

// setInterval used in Newton-Bisection Solver for JWL Temperature Dependent EOS
void ViscoScram::setInterval(double f, double rhoM, IterationVariables *iterVar){
  if(f < 0)
    iterVar->IL = rhoM;
  else if(f > 0)
    iterVar->IR = rhoM;
  else if(f ==0){
    iterVar->IL = rhoM;
    iterVar->IR = rhoM;
  }
}
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
