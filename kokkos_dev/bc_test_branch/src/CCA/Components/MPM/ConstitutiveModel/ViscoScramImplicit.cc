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

#include <CCA/Components/MPM/ConstitutiveModel/ViscoScramImplicit.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/Matrix3.h>
#include <Core/Math/Short27.h> //for Fracture
#include <Core/Math/Rand48.h>
#include <Core/Grid/Patch.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Labels/MPMLabel.h>

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/FastMatrix.h>
#include <Core/Malloc/Allocator.h>
#include <fstream>
#include <iostream>

using std::cerr;
using namespace Uintah;

ViscoScramImplicit::ViscoScramImplicit(ProblemSpecP& ps, MPMFlags* Mflag)
  : ConstitutiveModel(Mflag), ImplicitCM()
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

  d_G    = d_initialData.G[0] + d_initialData.G[1] +
             d_initialData.G[2] + d_initialData.G[3] + d_initialData.G[4];
  d_bulk = (2.*d_G*(1. + d_initialData.PR))/(3.*(1.-2.*d_initialData.PR));

}

ViscoScramImplicit::ViscoScramImplicit(const ViscoScramImplicit* cm) 
  : ConstitutiveModel(cm), ImplicitCM(cm)
{
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
  pStatedataLabel         = VarLabel::create("p.pStatedata_vs_implicit",
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
  pStatedataLabel_preReloc         = VarLabel::create("p.pStatedata_vs_implicit+",
      ParticleVariable<StateData>::getTypeDescription());
  pRandLabel_preReloc              = VarLabel::create("p.rand+",
      ParticleVariable<double>::getTypeDescription() );
  pStrainRateLabel_preReloc        = VarLabel::create("p.deformRate+",
      ParticleVariable<Matrix3>::getTypeDescription());
}

ViscoScramImplicit::~ViscoScramImplicit()
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


void ViscoScramImplicit::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
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
  cm_ps->appendElement("T0", d_tt.T0_WLF);
  cm_ps->appendElement("C1", d_tt.C1_WLF);
  cm_ps->appendElement("C2", d_tt.C2_WLF);
}


ViscoScramImplicit* ViscoScramImplicit::clone()
{
  return scinew ViscoScramImplicit(*this);
}

void ViscoScramImplicit::addInitialComputesAndRequires(Task* task,
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

void ViscoScramImplicit::initializeCMData(const Patch* patch,
                                        const MPMMaterial* matl,
                                        DataWarehouse* new_dw)
{
  // Initialize the variables shared by all implicit constitutive models
  // This method is defined in the ImplicitCM base class.
  initSharedDataForImplicit(patch, matl, new_dw);

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
                                                                                
  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){    particleIndex idx = *iter;
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
}

void
ViscoScramImplicit::addParticleState( std::vector<const VarLabel*>& from,
                                       std::vector<const VarLabel*>& to )
{
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

void ViscoScramImplicit::computeStableTimestep(const Patch*,
                                           const MPMMaterial*,
                                           DataWarehouse*)
{
  // Not used in the implicit models
}

void 
ViscoScramImplicit::computeStressTensorImplicit(const PatchSubset* patches,
                                                const MPMMaterial* matl,
                                                DataWarehouse* old_dw,
                                                DataWarehouse* new_dw,
                                                Solver* solver,
                                                const bool )

{
  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<Vector> d_S(interpolator->size());

    IntVector lowIndex = patch->getNodeLowIndex();
    IntVector highIndex = patch->getNodeHighIndex()+IntVector(1,1,1);
    Array3<int> l2g(lowIndex,highIndex);
    solver->copyL2G(l2g,patch);

    Matrix3 Shear,deformationGradientInc,dispGrad,fbar;
    double onethird = (1.0/3.0);
    
    Matrix3 Identity;
    
    Identity.Identity();
    
    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};
    
    int dwi = matl->getDWIndex();

    ParticleSubset* pset;
    constParticleVariable<Point> px;
    constParticleVariable<Matrix3> psize;
    ParticleVariable<Matrix3> deformationGradient_new;
    constParticleVariable<Matrix3> deformationGradient;
    ParticleVariable<Matrix3> pstress_new;
    constParticleVariable<Matrix3> pstress;
    constParticleVariable<double> pvolumeold,pmass;
    ParticleVariable<double> pvolume_deformed,pdTdt;
    constNCVariable<Vector> dispNew;
    
    DataWarehouse* parent_old_dw =
      new_dw->getOtherDataWarehouse(Task::ParentOldDW);
    pset = parent_old_dw->getParticleSubset(dwi, patch);
    parent_old_dw->get(px,                  lb->pXLabel,                  pset);
    parent_old_dw->get(psize,               lb->pSizeLabel,               pset);
    parent_old_dw->get(pstress,             lb->pStressLabel,             pset);
    parent_old_dw->get(pmass,               lb->pMassLabel,               pset);
    parent_old_dw->get(pvolumeold,          lb->pVolumeLabel,             pset);
    parent_old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);
    old_dw->get(dispNew,lb->dispNewLabel,dwi,patch, Ghost::AroundCells,1);
  
    new_dw->allocateAndPut(pstress_new,      lb->pStressLabel_preReloc, pset);
    new_dw->allocateAndPut(pdTdt,            lb->pdTdtLabel_preReloc,   pset);
    new_dw->allocateAndPut(pvolume_deformed, lb->pVolumeDeformedLabel,  pset);
    new_dw->allocateTemporary(deformationGradient_new,pset);

    double G = d_G;
    double K  = d_bulk;

    double rho_orig = matl->getInitialDensity();

    double B[6][24];
    double Bnl[3][24];
    double v[576];

    if(matl->getIsRigid()){
      for(ParticleSubset::iterator iter = pset->begin();
                                   iter != pset->end(); iter++){
        particleIndex idx = *iter;
        pdTdt[idx]=0;
        pstress_new[idx] = Matrix3(0.0);
        pvolume_deformed[idx] = pvolumeold[idx];
      }
    }
    else{
      for(ParticleSubset::iterator iter = pset->begin();
                                   iter != pset->end(); iter++){
        particleIndex idx = *iter;

        dispGrad.set(0.0);
        pdTdt[idx] = 0.;

        // Get the node indices that surround the cell
        interpolator->findCellAndShapeDerivatives(px[idx], ni, d_S,psize[idx],deformationGradient[idx]);
        int dof[24];
        loadBMats(l2g,dof,B,Bnl,d_S,ni,oodx);

        // Calculate the strain (here called D), and deviatoric rate DPrime
        Matrix3 e = (dispGrad + dispGrad.Transpose())*.5;
        Matrix3 ePrime = e - Identity*onethird*e.Trace();

        // This is the (updated) Cauchy stress

        pstress_new[idx] = pstress[idx] + (ePrime*2.*G+Identity*K*e.Trace());

        // Compute the deformation gradient increment using the dispGrad
      
        deformationGradientInc = dispGrad + Identity;

        // Update the deformation gradient tensor to its time n+1 value.
        deformationGradient_new[idx] = deformationGradientInc *
                                       deformationGradient[idx];

        // get the volumetric part of the deformation
        double J = deformationGradient_new[idx].Determinant();

        double E = 9.*K*G/(3.*K+G);
        double PR = (3.*K-E)/(6.*K);
        double C11 = E*(1.-PR)/((1.+PR)*(1.-2.*PR));
        double C12 = E*PR/((1.+PR)*(1.-2.*PR));
        double C44 = G;

        double D[6][6];
      
        D[0][0] = C11;
        D[0][1] = C12;
        D[0][2] = C12;
        D[0][3] = 0.;
        D[0][4] = 0.;
        D[0][5] = 0.;
        D[1][1] = C11;
        D[1][2] = C12;
        D[1][3] = 0.;
        D[1][4] = 0.;
        D[1][5] = 0.;
        D[2][2] = C11;
        D[2][3] = 0.;
        D[2][4] = 0.;
        D[2][5] = 0.;
        D[3][3] = C44;
        D[3][4] = 0.;
        D[3][5] = 0.;
        D[4][4] = C44;
        D[4][5] = 0.;
        D[5][5] = C44;
      
        D[1][0]=D[0][1];
        D[2][0]=D[0][2];
        D[2][1]=D[1][2];
        D[3][0]=D[0][3];
        D[3][1]=D[1][3];
        D[3][2]=D[2][3];
        D[4][0]=D[0][4];
        D[4][1]=D[1][4];
        D[4][2]=D[2][4];
        D[4][3]=D[3][4];
        D[5][0]=D[0][5];
        D[5][1]=D[1][5];
        D[5][2]=D[2][5];
        D[5][3]=D[3][5];
        D[5][4]=D[4][5];
      
        // kmat = B.transpose()*D*B*volold
        double kmat[24][24];
        BtDB(B,D,kmat);
        // kgeo = Bnl.transpose*sig*Bnl*volnew;
        double sig[3][3];
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            sig[i][j]=pstress[idx](i,j);
          }
        }
        double kgeo[24][24];
        BnltDBnl(Bnl,sig,kgeo);

        double volold = (pmass[idx]/rho_orig);
        double volnew = volold*J;

        pvolume_deformed[idx] = volnew;

        for(int ii = 0;ii<24;ii++){
          for(int jj = 0;jj<24;jj++){
            kmat[ii][jj]*=volold;
            kgeo[ii][jj]*=volnew;
          }
        }

        for (int I = 0; I < 24;I++){
          for (int J = 0; J < 24; J++){
            v[24*I+J] = kmat[I][J] + kgeo[I][J];
          }
        }
        solver->fillMatrix(24,dof,24,dof,v);
     }
    }
    delete interpolator;
  }
}


void 
ViscoScramImplicit::computeStressTensorImplicit(const PatchSubset* patches,
                                                const MPMMaterial* matl,
                                                DataWarehouse* old_dw,
                                                DataWarehouse* new_dw)


{
   for(int pp=0;pp<patches->size();pp++){
    double se = 0.0;
    const Patch* patch = patches->get(pp);
    Matrix3 dispGrad,deformationGradientInc,Identity,zero(0.),One(1.);
    double Jinc;
    double onethird = (1.0/3.0);

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<Vector> d_S(interpolator->size());

    Identity.Identity();

    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    // Placeholder stuff for viscoscram goes here
    constParticleVariable<double>    pCrackRadius;
    constParticleVariable<Vector>    pVelocity;
    ParticleVariable<Matrix3>   pStrainRate_new;
    ParticleVariable<double>    pVolHeatRate_new, pVeHeatRate_new;
    ParticleVariable<double>    pCrHeatRate_new, pCrackRadius_new;
    ParticleVariable<double>    pRand;
    ParticleVariable<StateData> pStatedata;

    old_dw->get(pCrackRadius,        pCrackRadiusLabel,            pset);

    // Allocate arrays for the updated particle data for the current patch
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
    old_dw->copyOut(pRand,           pRandLabel,                  pset);
    old_dw->copyOut(pStatedata,      pStatedataLabel,             pset);
    ASSERTEQ(pset, pStatedata.getParticleSubset());

    constParticleVariable<Point> px;
    constParticleVariable<Matrix3> psize;
    constParticleVariable<Matrix3> deformationGradient, pstress;
    ParticleVariable<Matrix3> pstress_new;
    ParticleVariable<Matrix3> deformationGradient_new;
    constParticleVariable<double> pvolume;
    ParticleVariable<double> pvolume_deformed, pdTdt;
    constParticleVariable<Vector> pvelocity;
    constNCVariable<Vector> dispNew;
    delt_vartype delT;

    old_dw->get(px,                  lb->pXLabel,                  pset);
    old_dw->get(psize,               lb->pSizeLabel,               pset);
    old_dw->get(pstress,             lb->pStressLabel,             pset);
    old_dw->get(pvolume,             lb->pVolumeLabel,             pset);
    old_dw->get(pvelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);

    new_dw->get(dispNew,lb->dispNewLabel,dwi,patch,Ghost::AroundCells,1);

    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    new_dw->allocateAndPut(pstress_new,      lb->pStressLabel_preReloc,  pset);
    new_dw->allocateAndPut(pdTdt,            lb->pdTdtLabel_preReloc,    pset);
    new_dw->allocateAndPut(pvolume_deformed, lb->pVolumeDeformedLabel,   pset);
    new_dw->allocateAndPut(deformationGradient_new,
                           lb->pDeformationMeasureLabel_preReloc,        pset);
    double G    = d_G;
    double bulk = d_bulk;

    // Carry Forward the ViscoScram variables
    for(ParticleSubset::iterator iter = pset->begin();iter!=pset->end();iter++){
      particleIndex idx = *iter;
      pCrackRadius_new[idx]=pCrackRadius[idx];
      pVolHeatRate_new[idx] = 0.0;
      pVeHeatRate_new[idx]  = 0.0;
      pCrHeatRate_new[idx]  = 0.0;
      pdTdt[idx]            = 0.0;
    }


    if(matl->getIsRigid()){
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;
        pstress_new[idx] = Matrix3(0.0);
        deformationGradient_new[idx] = Identity;
        pvolume_deformed[idx] = pvolume[idx];
      }
    }
    else{
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;

        dispGrad.set(0.0);

        // Get the node indices that surround the cell
        interpolator->findCellAndShapeDerivatives(px[idx], ni, d_S,psize[idx],deformationGradient[idx]);
        for(int k = 0; k < 8; k++) {
          const Vector& disp = dispNew[ni[k]];
          
          for (int j = 0; j<3; j++){
            for (int i = 0; i<3; i++) {
              dispGrad(i,j) += disp[i] * d_S[k][j]* oodx[j];
            }
          }
        }

      // Calculate the strain (here called D), and deviatoric rate DPrime
      Matrix3 D = (dispGrad + dispGrad.Transpose())*.5;
      Matrix3 DPrime = D - Identity*onethird*D.Trace();
      pStrainRate_new[idx] = D/delT;

      // This is the (updated) Cauchy stress

      pstress_new[idx] = pstress[idx] + (DPrime*2.*G + Identity*bulk*D.Trace());

      // Compute the deformation gradient increment using the time_step
      // velocity gradient
      // F_n^np1 = dudx * dt + Identity
      deformationGradientInc = dispGrad + Identity;

      Jinc = deformationGradientInc.Determinant();

      // Update the deformation gradient tensor to its time n+1 value.
      deformationGradient_new[idx] = deformationGradientInc *
                                     deformationGradient[idx];

      pvolume_deformed[idx]=Jinc*pvolume[idx];

      // Compute the strain energy for all the particles
      Matrix3 AvgStress = (pstress_new[idx] + pstress[idx])*.5;

      double e = (D(0,0)*AvgStress(0,0) +
                  D(1,1)*AvgStress(1,1) +
                  D(2,2)*AvgStress(2,2) +
              2.*(D(0,1)*AvgStress(0,1) +
                  D(0,2)*AvgStress(0,2) +
                  D(1,2)*AvgStress(1,2))) * pvolume_deformed[idx]*delT;
      
      se += e;
      }
      
      if (flag->d_reductionVars->accStrainEnergy ||
          flag->d_reductionVars->strainEnergy) {
        new_dw->put(sum_vartype(se),     lb->StrainEnergyLabel);
      }
    }
    delete interpolator;
   }
}

void ViscoScramImplicit::addComputesAndRequires(Task* task,
                                                 const MPMMaterial* matl,
                                                 const PatchSet* patches,
                                                 const bool ) const
{
  const MaterialSubset* matlset = matl->thisMaterial();

  task->requires(Task::ParentOldDW, lb->pXLabel,         matlset,Ghost::None);
  task->requires(Task::ParentOldDW, lb->pSizeLabel,      matlset,Ghost::None);
  task->requires(Task::ParentOldDW, lb->pMassLabel,      matlset,Ghost::None);
  task->requires(Task::ParentOldDW, lb->pVolumeLabel,    matlset,Ghost::None);
  task->requires(Task::ParentOldDW, lb->pDeformationMeasureLabel,
                                                         matlset,Ghost::None);
  task->requires(Task::OldDW,lb->dispNewLabel,matlset,Ghost::AroundCells,1);

  task->computes(lb->pStressLabel_preReloc,matlset);  
  task->computes(lb->pdTdtLabel_preReloc,  matlset);  
  task->computes(lb->pVolumeDeformedLabel, matlset);

}

void ViscoScramImplicit::addComputesAndRequires(Task* task,
                                                 const MPMMaterial* matl,
                                                 const PatchSet*) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  Ghost::GhostType  gnone = Ghost::None;

  task->requires(Task::OldDW, lb->delTLabel);
  task->requires(Task::OldDW, lb->pXLabel,                 matlset,gnone);
  task->requires(Task::OldDW, lb->pSizeLabel,              matlset,gnone);
  task->requires(Task::OldDW, lb->pMassLabel,              matlset,gnone);
  task->requires(Task::OldDW, lb->pVolumeLabel,            matlset,gnone);
  task->requires(Task::OldDW, lb->pStressLabel,            matlset,gnone);
  task->requires(Task::OldDW, lb->pVelocityLabel,          matlset,gnone);
  task->requires(Task::OldDW, lb->pDeformationMeasureLabel,matlset,gnone);
  task->requires(Task::OldDW, pCrackRadiusLabel,           matlset,gnone);
  task->requires(Task::OldDW, pStatedataLabel,             matlset,gnone);
  task->requires(Task::OldDW, pRandLabel,                  matlset,gnone);
  task->requires(Task::NewDW, lb->dispNewLabel,   matlset,Ghost::AroundCells,1);
                                                                                
  task->computes(lb->pStressLabel_preReloc,                matlset);
  task->computes(lb->pdTdtLabel_preReloc,                  matlset);
  task->computes(lb->pDeformationMeasureLabel_preReloc,    matlset);
  task->computes(lb->pVolumeDeformedLabel,                 matlset);
  task->computes(pVolChangeHeatRateLabel_preReloc,         matlset);
  task->computes(pViscousHeatRateLabel_preReloc,           matlset);
  task->computes(pCrackHeatRateLabel_preReloc,             matlset);
  task->computes(pCrackRadiusLabel_preReloc,               matlset);
  task->computes(pStatedataLabel_preReloc,                 matlset);
  task->computes(pRandLabel_preReloc,                      matlset);
  task->computes(pStrainRateLabel_preReloc,                matlset);
}

// The "CM" versions use the pressure-volume relationship of the CNH model
double ViscoScramImplicit::computeRhoMicroCM(double pressure, 
                                              const double p_ref,
                                              const MPMMaterial* matl,
                                              double temperature,
                                              double rho_guess)
{
  double rho_orig = matl->getInitialDensity();
  double p_gauge = pressure - p_ref;
  double rho_cur;

  rho_cur = rho_orig/(1-p_gauge/d_bulk);

  return rho_cur;
}

void ViscoScramImplicit::computePressEOSCM(const double rho_cur,
                                            double& pressure, 
                                            const double p_ref,
                                            double& dp_drho, double& tmp,
                                            const MPMMaterial* matl,
                                            double temperature)
{
  double rho_orig = matl->getInitialDensity();

  double p_g = d_bulk*(1.0 - rho_orig/rho_cur);
  pressure = p_ref + p_g;
  dp_drho  = d_bulk*rho_orig/(rho_cur*rho_cur);
  tmp = d_bulk/rho_cur;
}

double ViscoScramImplicit::getCompressibility()
{
  return 1.0/d_bulk;
}
