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

#include "ViscoSCRAMHotSpot.h"
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Math/Matrix3.h>
#include <Core/Math/SymmMatrix3.h>
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
#include <Core/Labels/MPMLabel.h>

#include <Core/Malloc/Allocator.h>
#include <Core/Util/Endian.h>
#include <Core/Util/DebugStream.h>
#include <Core/Math/MinMax.h>

#include <fstream>
#include <iostream>

using namespace std;
using namespace Uintah;


static DebugStream dbg("VS_HS", false);

ViscoSCRAMHotSpot::ViscoSCRAMHotSpot(ProblemSpecP& ps, MPMFlags* Mflag)
  :ViscoScram(ps,Mflag)
{
  // Read in the extra material constants
  ps->require("Chi",d_matConst.Chi);
  ps->require("delH",d_matConst.delH);
  ps->require("Z",d_matConst.Z);
  ps->require("EoverR",d_matConst.EoverR);
  ps->require("dynamic_coeff_friction",d_matConst.mu_d);
  ps->require("volfracHE",d_matConst.vfHE);

  // Create labels for the hotspot data
  pHotSpotT1Label    = 
    VarLabel::create("p.hotspotT1",
                     ParticleVariable<double>::getTypeDescription());
  pHotSpotT2Label    = 
    VarLabel::create("p.hotspotT2",
                     ParticleVariable<double>::getTypeDescription());
  pHotSpotPhi1Label  = 
    VarLabel::create("p.hotspotPhi1",
                     ParticleVariable<double>::getTypeDescription());
  pHotSpotPhi2Label  = 
    VarLabel::create("p.hotspotPhi2",
                     ParticleVariable<double>::getTypeDescription());
  pChemHeatRateLabel = 
    VarLabel::create("p.chemHeatRate",
                     ParticleVariable<double>::getTypeDescription());

  pHotSpotT1Label_preReloc    = 
    VarLabel::create("p.hotspotT1+",
                     ParticleVariable<double>::getTypeDescription());
  pHotSpotT2Label_preReloc    = 
    VarLabel::create("p.hotspotT2+",
                     ParticleVariable<double>::getTypeDescription());
  pHotSpotPhi1Label_preReloc  = 
    VarLabel::create("p.hotspotPhi1+",
                     ParticleVariable<double>::getTypeDescription());
  pHotSpotPhi2Label_preReloc  = 
    VarLabel::create("p.hotspotPhi2+",
                     ParticleVariable<double>::getTypeDescription());
  pChemHeatRateLabel_preReloc = 
    VarLabel::create("p.chemHeatRate+",
                     ParticleVariable<double>::getTypeDescription());
}

ViscoSCRAMHotSpot::ViscoSCRAMHotSpot(const ViscoSCRAMHotSpot* cm)
  : ViscoScram(cm)
{
  // Material constants
  d_matConst.Chi = cm->d_matConst.Chi;
  d_matConst.delH = cm->d_matConst.delH;
  d_matConst.Z = cm->d_matConst.Z;
  d_matConst.EoverR = cm->d_matConst.EoverR;
  d_matConst.mu_d = cm->d_matConst.mu_d;
  d_matConst.vfHE = cm->d_matConst.vfHE;

  // Create labels for the hotspot data
  pHotSpotT1Label    = 
    VarLabel::create("p.hotspotT1",
                     ParticleVariable<double>::getTypeDescription());
  pHotSpotT2Label    = 
    VarLabel::create("p.hotspotT2",
                     ParticleVariable<double>::getTypeDescription());
  pHotSpotPhi1Label  = 
    VarLabel::create("p.hotspotPhi1",
                     ParticleVariable<double>::getTypeDescription());
  pHotSpotPhi2Label  = 
    VarLabel::create("p.hotspotPhi2",
                     ParticleVariable<double>::getTypeDescription());
  pChemHeatRateLabel = 
    VarLabel::create("p.chemHeatRate",
                     ParticleVariable<double>::getTypeDescription());

  pHotSpotT1Label_preReloc    = 
    VarLabel::create("p.hotspotT1+",
                     ParticleVariable<double>::getTypeDescription());
  pHotSpotT2Label_preReloc    = 
    VarLabel::create("p.hotspotT2+",
                     ParticleVariable<double>::getTypeDescription());
  pHotSpotPhi1Label_preReloc  = 
    VarLabel::create("p.hotspotPhi1+",
                     ParticleVariable<double>::getTypeDescription());
  pHotSpotPhi2Label_preReloc  = 
    VarLabel::create("p.hotspotPhi2+",
                     ParticleVariable<double>::getTypeDescription());
  pChemHeatRateLabel_preReloc = 
    VarLabel::create("p.chemHeatRate+",
                     ParticleVariable<double>::getTypeDescription());
}

ViscoSCRAMHotSpot::~ViscoSCRAMHotSpot()
{
  // Delete hotspot data
  VarLabel::destroy(pHotSpotT1Label);
  VarLabel::destroy(pHotSpotT2Label);
  VarLabel::destroy(pHotSpotPhi1Label);
  VarLabel::destroy(pHotSpotPhi2Label);
  VarLabel::destroy(pChemHeatRateLabel);

  VarLabel::destroy(pHotSpotT1Label_preReloc);
  VarLabel::destroy(pHotSpotT2Label_preReloc);
  VarLabel::destroy(pHotSpotPhi1Label_preReloc);
  VarLabel::destroy(pHotSpotPhi2Label_preReloc);
  VarLabel::destroy(pChemHeatRateLabel_preReloc);
}

ViscoSCRAMHotSpot* ViscoSCRAMHotSpot::clone()
{
  return scinew ViscoSCRAMHotSpot(*this);
}

void 
ViscoSCRAMHotSpot::addInitialComputesAndRequires(Task* task,
                                                 const MPMMaterial* matl,
                                                 const PatchSet* patches) const
{
  // First setup the standard ViscoScram stuff
  ViscoScram::addInitialComputesAndRequires(task, matl, patches);

  // Set up extra stuff needed for hotspot model
  const MaterialSubset* matlset = matl->thisMaterial();

  //task->requires(Task::NewDW, lb->pTemperatureLabel, matlset, Ghost::None);

  task->computes(pHotSpotT1Label,    matlset);
  task->computes(pHotSpotT2Label,    matlset);
  task->computes(pHotSpotPhi1Label,  matlset);
  task->computes(pHotSpotPhi2Label,  matlset);
  task->computes(pChemHeatRateLabel, matlset);
}

void 
ViscoSCRAMHotSpot::initializeCMData(const Patch* patch,
                                    const MPMMaterial* matl,
                                    DataWarehouse* new_dw)
{
  // First initialize the standard ViscoScram stuff
  ViscoScram::initializeCMData(patch, matl, new_dw);

  // Initialize extra stuff needed for hotspot model
  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  // Need the initial temperature of the particles to initialize the
  // hotspot model
  //constParticleVariable<double> pTemperature;
  //new_dw->get(pTemperature, lb->pTemperatureLabel, pset);

  // Allocate the history variables to be initialized
  ParticleVariable<double> pHotSpotT1, pHotSpotT2, pHotSpotPhi1, pHotSpotPhi2;
  ParticleVariable<double> pChemHeatRate;
  new_dw->allocateAndPut(pHotSpotT1,    pHotSpotT1Label,    pset);
  new_dw->allocateAndPut(pHotSpotT2,    pHotSpotT2Label,    pset);
  new_dw->allocateAndPut(pHotSpotPhi1,  pHotSpotPhi1Label,  pset);
  new_dw->allocateAndPut(pHotSpotPhi2,  pHotSpotPhi2Label,  pset);
  new_dw->allocateAndPut(pChemHeatRate, pChemHeatRateLabel, pset);

  // Loop thru particles and do the initialization
  ParticleSubset::iterator iter = pset->begin();
  for(;iter != pset->end();iter++){

    particleIndex idx = *iter;

    // Set the initial temperature
    //double T = pTemperature[idx];
    double T = 294.0;
    pHotSpotT1[idx] = T;
    pHotSpotT2[idx] = T;
    pHotSpotPhi1[idx] = T;
    pHotSpotPhi1[idx] = T;
    pChemHeatRate[idx] = 0.0;
  }
}

void 
ViscoSCRAMHotSpot::addComputesAndRequires(Task* task,
                                          const MPMMaterial* matl,
                                          const PatchSet* patches) const
{
  // Add the standard ViscoScram computes and requires
  ViscoScram::addComputesAndRequires(task, matl, patches);

  // Other computes and requires needed for the hotspot model
  const MaterialSubset* matlset = matl->thisMaterial();
  Ghost::GhostType  gnone = Ghost::None;

  task->requires(Task::OldDW, pHotSpotT1Label,    matlset, gnone);
  task->requires(Task::OldDW, pHotSpotT2Label,    matlset, gnone);
  task->requires(Task::OldDW, pHotSpotPhi1Label,  matlset, gnone);
  task->requires(Task::OldDW, pHotSpotPhi2Label,  matlset, gnone);
  task->requires(Task::OldDW, pChemHeatRateLabel, matlset, gnone);

  task->computes(pHotSpotT1Label_preReloc,    matlset);
  task->computes(pHotSpotT2Label_preReloc,    matlset);
  task->computes(pHotSpotPhi1Label_preReloc,  matlset);
  task->computes(pHotSpotPhi2Label_preReloc,  matlset);
  task->computes(pChemHeatRateLabel_preReloc, matlset);
}

void 
ViscoSCRAMHotSpot::computeStressTensor(const PatchSubset* patches,
                                       const MPMMaterial* matl,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw)
{
  // Initialize constants
  double onethird = (1.0/3.0);
  double sqrtopf  = sqrt(1.5);
  Matrix3 zero(0.0), Id; Id.Identity();

  // Initial crack growth parameters
  // Baseline PBX 9501
  double vres_a = 0.90564746; double vres_b =-2.90178468;
  // Aged PBX 9501
  // double vres_a = 0.90863805; double vres_b =-2.5061966;

  // Material constants 
  int numMw = 5;
  double Gmw[5];
  double G1 = d_initialData.G[0];
  double G2 = d_initialData.G[1];
  double G3 = d_initialData.G[2];
  double G4 = d_initialData.G[3];
  double G5 = d_initialData.G[4];
  double nu = d_initialData.PR;
  //double alpha = d_initialData.CoefThermExp;
  double rho_0 = matl->getInitialDensity();
  double Cp_0 = matl->getInitialCp();
  double kappa = matl->getThermalConductivity();

  // Define particle and grid variables
  delt_vartype delT;
  constParticleVariable<double>  pMass, pVol, pTemp;
  constParticleVariable<double>  pCrackRadius;
  constParticleVariable<Vector>  pVel;
  constParticleVariable<Matrix3> pDefGrad, pSig;
  constParticleVariable<double>  pHotSpotT1, pHotSpotT2;
  constParticleVariable<double>  pHotSpotPhi1, pHotSpotPhi2;
  constParticleVariable<double>  pVol_new;
  constParticleVariable<Matrix3> pDefGrad_new,velGrad;

  ParticleVariable<double>    pIntHeatRate_new;
  ParticleVariable<Matrix3>   pSig_new;
  ParticleVariable<double>    pVolHeatRate_new, pVeHeatRate_new;
  ParticleVariable<double>    pCrHeatRate_new, pChHeatRate_new;
  ParticleVariable<double>    pCrackRadius_new, pStrainRate_new;
  ParticleVariable<double>    pHotSpotT1_new, pHotSpotT2_new;
  ParticleVariable<double>    pHotSpotPhi1_new, pHotSpotPhi2_new;

  ParticleVariable<StateData> pState;
  ParticleVariable<double>    pRand;

  // Other local variables
  Matrix3 pDefRate(0.0);                       // rate of deformation
  Matrix3 pDefRateDev(0.0);                    // deviatoric rate of deform
  Matrix3 pDefGradInc; pDefGradInc.Identity(); // increment of deform gradient
  Matrix3 sig_old(0.0);                        // old stress
  Matrix3 sig_new(0.0), sigDev_new(0.0);       // new stress+deviatoric stress


  int dwi = matl->getDWIndex();                // Data warehouse index
  old_dw->get(delT, lb->delTLabel, getLevel(patches));

  // Loop through patches
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    // Initialize patch variables
    double se = 0;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

    // Get patch size

    Vector dx = patch->dCell();

    // Get the particle and grid data for the current patch
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    old_dw->get(pMass,         lb->pMassLabel,               pset);
    old_dw->get(pVol,          lb->pVolumeLabel,             pset);
    old_dw->get(pTemp,         lb->pTemperatureLabel,        pset);
    old_dw->get(pVel,          lb->pVelocityLabel,           pset);
    old_dw->get(pDefGrad,      lb->pDeformationMeasureLabel, pset);
    old_dw->get(pSig,          lb->pStressLabel,             pset);
    old_dw->get(pCrackRadius,  pCrackRadiusLabel,            pset);
    old_dw->get(pHotSpotT1,    pHotSpotT1Label,              pset);
    old_dw->get(pHotSpotT2,    pHotSpotT2Label,              pset);
    old_dw->get(pHotSpotPhi1,  pHotSpotPhi1Label,            pset);
    old_dw->get(pHotSpotPhi2,  pHotSpotPhi2Label,            pset);

    // Allocate arrays for the updated particle data for the current patch
    new_dw->get(pVol_new,         lb->pVolumeLabel_preReloc,             pset);
    new_dw->get(velGrad,          lb->pVelGradLabel_preReloc,            pset);
    new_dw->get(pDefGrad_new,     lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->allocateAndPut(pIntHeatRate_new, 
                           lb->pdTdtLabel_preReloc,               pset);
    new_dw->allocateAndPut(pSig_new,      
                           lb->pStressLabel_preReloc,             pset);
    new_dw->allocateAndPut(pVolHeatRate_new, 
                           pVolChangeHeatRateLabel_preReloc,      pset);
    new_dw->allocateAndPut(pVeHeatRate_new,  
                           pViscousHeatRateLabel_preReloc,        pset);
    new_dw->allocateAndPut(pCrHeatRate_new,  
                           pCrackHeatRateLabel_preReloc,          pset);
    new_dw->allocateAndPut(pChHeatRate_new,  
                           pChemHeatRateLabel_preReloc,           pset);
    new_dw->allocateAndPut(pCrackRadius_new, 
                           pCrackRadiusLabel_preReloc,            pset);
    new_dw->allocateAndPut(pStrainRate_new, 
                           pStrainRateLabel_preReloc,             pset);
    new_dw->allocateAndPut(pHotSpotT1_new, 
                           pHotSpotT1Label_preReloc,              pset);
    new_dw->allocateAndPut(pHotSpotT2_new, 
                           pHotSpotT2Label_preReloc,              pset);
    new_dw->allocateAndPut(pHotSpotPhi1_new, 
                           pHotSpotPhi1Label_preReloc,            pset);
    new_dw->allocateAndPut(pHotSpotPhi2_new, 
                           pHotSpotPhi2Label_preReloc,            pset);
    new_dw->allocateAndPut(pRand,        
                           pRandLabel_preReloc,                   pset);
    new_dw->allocateAndPut(pState,   
                           pStatedataLabel_preReloc,              pset);

    old_dw->copyOut(pRand,  pRandLabel,      pset);
    old_dw->copyOut(pState, pStatedataLabel, pset);
    ASSERTEQ(pset, pState.getParticleSubset());

    // Loop thru particles
    ParticleSubset::iterator iter = pset->begin();
    for(; iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Assign zero internal heating by default - modify if necessary.
      pIntHeatRate_new[idx] = 0.0;

      // Vary G from particle to particle and compute G, K, and 3*alpha*K
      double variation = 1.0 + 0.4*(pRand[idx] - 0.5);
      Gmw[0]=G1*variation;
      Gmw[1]=G2*variation;
      Gmw[2]=G3*variation;
      Gmw[3]=G4*variation;
      Gmw[4]=G5*variation;
      double G = Gmw[0] + Gmw[1] + Gmw[2] + Gmw[3] + Gmw[4];
      double K = (2.0*G*(1.0+nu))/(3.0*(1.0-2.0*nu));
      //double alphaK = 3.0*K*(alpha*variation);

      // Calculate rate of deformation, deviatoric rate of deformation
      // and the effective deviatoric strain rate
      pDefRate = (velGrad[idx] + velGrad[idx].Transpose())*.5;
      pStrainRate_new[idx] = pDefRate.Norm();

      if (dbg.active())
        dbg << "Total strain rate = " << pStrainRate_new[idx] << endl;

      pDefRateDev = pDefRate - Id*(onethird*pDefRate.Trace());
      double edotnorm = sqrtopf*pDefRateDev.Norm();
      double vres = 
        (edotnorm > 1.0e-8) ? exp(vres_a*log(edotnorm) + vres_b) : 0.0;

      // Get the old total stress, old total deviatoric stress, and crack radius
      Matrix3 sig_old = pSig[idx];
      double sig_m = onethird*sig_old.Trace();
      double c_old = pCrackRadius[idx];

      // Integrate the evolution equations for the cracks and the element
      // deviatoric stress (using a fourth-order Runge-Kutta scheme)
      FVector Y_old(c_old, pState[idx].DevStress);

      //Matrix3 sigDev_old(0.0);
      // dbg << "Crack Radius (old) = " << c_old << endl;
      //for (int ii = 0 ; ii < 5; ++ii) {
      //  dbg << " Dev Stress (old) [" << ii << "] = " << endl
      //       << pState[idx].DevStress[ii] << endl;
      //  sigDev_old += pState[idx].DevStress[ii];
      //}
      //dbg << "Total Dev Stress (old) = " << endl
      //    << sigDev_old << endl;
      //dbg << " Mean Stress (old) = " << sig_m << endl;
      //dbg << "Total Stress (old) = " << endl << sig_old << endl;

      double cdot_new = 0.0;
      FVector Y_new = integrateRateEquations(Y_old, pDefRateDev, sig_m, 
                                             Gmw, vres, delT, cdot_new);

      // Calculate the updated crack size and element deviatoric stresses
      // and Update total deviatoric stress
      double c_new = Y_new.a;
      pCrackRadius_new[idx] = c_new;
      sigDev_new = zero;
      for (int ii = 0; ii < numMw; ++ii) {
        pState[idx].DevStress[ii] = Y_new.b_n[ii];
        sigDev_new += pState[idx].DevStress[ii];
      }

      // Compute the volumetric part of the stress
      double ekk = pDefRate.Trace();
      double sig_m_new = sig_m + (onethird*K*ekk*delT);

      // Update the Cauchy stress
      pSig_new[idx] = sigDev_new + Id*sig_m_new;
      //dbg << "Crack Radius (new) = " << c_new << endl;
      //for (int ii = 0 ; ii < 5; ++ii) {
      //  dbg << " Dev Stress (new) [" << ii << "] = " << endl
      //       << pState[idx].DevStress[ii] << endl;
      //}
      //dbg << "Total Dev Stress (new) = " << endl
      //    << sigDev_new << endl;
      //dbg << " Mean Stress (new) = " << sig_m_new << endl;
      //dbg << "Total Stress (new) = " << endl << pSig_new[idx] << endl;

      double J = pDefGrad_new[idx].Determinant();

      // Compute the current mass density and volume
      double rho_cur = rho_0/J;

      // Determine the bulk temperature change at a material point
      // assuming adiabatic conditions
      double T_old = pTemp[idx];
      //double Cp = Cp_0 + d_initialData.DCp_DTemperature*T_old;
      //double Cv = Cp/(1+d_initialData.Beta*T_old);
      double Cv = Cp_0;
      double rhoCv = rho_cur*Cv;
      double fac = d_matConst.Chi/rhoCv;

      // Compute viscous work rate
      double wdot_ve = computeViscousWorkRate(numMw, pState[idx].DevStress, 
                                              Gmw);

      // Compute cracking damage work rate
      double wdot_cr = computeCrackingWorkRate(numMw, c_new, 
                                               pState[idx].DevStress,
                                               pDefRateDev, sigDev_new, 
                                               Gmw, cdot_new);
      if (dbg.active())
        dbg << "rhoCv = " << rhoCv << endl;

      // Compute bulk chemical heating rate
      double qdot_ch = computeChemicalHeatRate(rho_cur, T_old);

      // Compute the contributions to the temperature due to each of the
      // components
      double volHeatRate = d_initialData.Gamma*T_old*ekk;
      double veHeatRate = wdot_ve*fac;
      double crHeatRate = wdot_cr*fac;

      if (dbg.active())
        dbg << "pCrHeatRate = " << crHeatRate << endl;

      double chHeatRate = d_matConst.vfHE*qdot_ch;
      pVolHeatRate_new[idx] = volHeatRate;
      pVeHeatRate_new[idx] = veHeatRate;
      pCrHeatRate_new[idx] = crHeatRate;
      pChHeatRate_new[idx] = chHeatRate;

      // Update the internal heating rate
      double totalHeatRate = -volHeatRate+veHeatRate+crHeatRate+chHeatRate;
      pIntHeatRate_new[idx] = totalHeatRate;

      // Evaluate the hot spot model (calculate the updated Temperatures)
      double hotSpotT[4];
      hotSpotT[0] = pHotSpotT1[idx];
      hotSpotT[1] = pHotSpotT2[idx];
      hotSpotT[2] = pHotSpotPhi1[idx];
      hotSpotT[3] = pHotSpotPhi2[idx];
      evaluateHotSpotModel(sig_m_new, pSig_new[idx], pDefRate, hotSpotT,
                           kappa, rho_cur, Cv, delT);
      pHotSpotT1_new[idx] = hotSpotT[0];
      pHotSpotT2_new[idx] = hotSpotT[1]; 
      pHotSpotPhi1_new[idx] = hotSpotT[2];
      pHotSpotPhi2_new[idx] = hotSpotT[3];

      // Compute the strain energy for all the particles
      sig_old = (pSig_new[idx] + sig_old)*.5;
      se += pDefRate.Contract(sig_old)*pVol_new[idx]*delT;

      // Compute wave speed at each particle, store the maximum
      Vector pVel_idx = pVel[idx];
      double c_dil = sqrt((K + 4.*G/3.)*pVol_new[idx]/pMass[idx]);
      WaveSpeed=Vector(Max(c_dil+fabs(pVel_idx.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pVel_idx.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pVel_idx.z()),WaveSpeed.z()));
    }

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    //Timesteps larger than 1 microsecond cause VS to be unstable
    delT_new = min(1.e-6, delT_new);

    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
    
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(se),     lb->StrainEnergyLabel);
    }
  }
}

void 
ViscoSCRAMHotSpot::carryForward(const PatchSubset* patches,
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
    ParticleVariable<double>    pVolHeatRate_new, pVeHeatRate_new;
    ParticleVariable<double>    pCrHeatRate_new, pChHeatRate_new;
    ParticleVariable<double>    pCrackRadius_new, pStrainRate_new;
    ParticleVariable<double>    pHotSpotT1_new, pHotSpotT2_new;
    ParticleVariable<double>    pHotSpotPhi1_new, pHotSpotPhi2_new;
    ParticleVariable<StateData> pState;
    ParticleVariable<double>    pRand;

    new_dw->allocateAndPut(pVolHeatRate_new, 
                           pVolChangeHeatRateLabel_preReloc,      pset);
    new_dw->allocateAndPut(pVeHeatRate_new,  
                           pViscousHeatRateLabel_preReloc,        pset);
    new_dw->allocateAndPut(pCrHeatRate_new,  
                           pCrackHeatRateLabel_preReloc,          pset);
    new_dw->allocateAndPut(pChHeatRate_new,  
                           pChemHeatRateLabel_preReloc,           pset);
    new_dw->allocateAndPut(pCrackRadius_new, 
                           pCrackRadiusLabel_preReloc,            pset);
    new_dw->allocateAndPut(pStrainRate_new, 
                           pStrainRateLabel_preReloc,             pset);
    new_dw->allocateAndPut(pHotSpotT1_new, 
                           pHotSpotT1Label_preReloc,              pset);
    new_dw->allocateAndPut(pHotSpotT2_new, 
                           pHotSpotT2Label_preReloc,              pset);
    new_dw->allocateAndPut(pHotSpotPhi1_new, 
                           pHotSpotPhi1Label_preReloc,            pset);
    new_dw->allocateAndPut(pHotSpotPhi2_new, 
                           pHotSpotPhi2Label_preReloc,            pset);
    new_dw->allocateAndPut(pState,  
                           pStatedataLabel_preReloc,              pset);
    new_dw->allocateAndPut(pRand,         
                           pRandLabel_preReloc,                   pset);
    old_dw->copyOut(pRand,      pRandLabel,      pset);
    old_dw->copyOut(pState,     pStatedataLabel, pset);

    ParticleSubset::iterator iter = pset->begin();
    for(; iter != pset->end(); iter++){
      particleIndex idx = *iter;
      pVolHeatRate_new[idx] = 0.0;
      pVeHeatRate_new[idx]  = 0.0;
      pCrHeatRate_new[idx]  = 0.0;
      pChHeatRate_new[idx]  = 0.0;
      pCrackRadius_new[idx] = 0.0;
      pStrainRate_new[idx] = 0.0;
      pHotSpotT1_new[idx]  = 0.0;
      pHotSpotT2_new[idx]  = 0.0;
      pHotSpotPhi1_new[idx]  = 0.0;
      pHotSpotPhi2_new[idx]  = 0.0;
    }
    new_dw->put(delt_vartype(1.e10), lb->delTLabel, patch->getLevel());
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(0.),   lb->StrainEnergyLabel);
    }
  }
}
         
void 
ViscoSCRAMHotSpot::addParticleState(std::vector<const VarLabel*>& from,
                                    std::vector<const VarLabel*>& to)
{
  // Call the ViscoScram method first
  ViscoScram::addParticleState(from, to);

  // Add the local particle state 
  from.push_back(pChemHeatRateLabel);
  from.push_back(pHotSpotT1Label);
  from.push_back(pHotSpotT2Label);
  from.push_back(pHotSpotPhi1Label);
  from.push_back(pHotSpotPhi2Label);

  to.push_back(pChemHeatRateLabel_preReloc);
  to.push_back(pHotSpotT1Label_preReloc);
  to.push_back(pHotSpotT2Label_preReloc);
  to.push_back(pHotSpotPhi1Label_preReloc);
  to.push_back(pHotSpotPhi2Label_preReloc);
}

///////////////////////////////////////////////////////////////////////////
//
/*! Compute K_I */
//
///////////////////////////////////////////////////////////////////////////
double
ViscoSCRAMHotSpot::computeK_I(double c, double sigEff)
{
  return (sqrt(M_PI*c)*sigEff);
}

///////////////////////////////////////////////////////////////////////////
//
/*! Compute K_{0mu} */
//
///////////////////////////////////////////////////////////////////////////
double 
ViscoSCRAMHotSpot::computeK_0mu(double c, double sig_m)
{
  double K_0 = d_initialData.StressIntensityF;   // K0
  double mu_s = d_initialData.CrackFriction;     // static friction coeff
  double mu_prime = (45.0/(2.0*(3.0-2.0*mu_s*mu_s)))*mu_s;
  double fac = mu_prime*sig_m*sqrt(c)/K_0;
  return (K_0*sqrt(1.0 - M_PI*fac*(1.0-fac)));
}

///////////////////////////////////////////////////////////////////////////
//
/*! Compute K^' */
//
///////////////////////////////////////////////////////////////////////////
double
ViscoSCRAMHotSpot::computeK_prime(double c, double sig_m)
{
  double m = d_initialData.CrackPowerValue;      // Parameter m
  double K_0mu = computeK_0mu(c, sig_m);
  return (K_0mu*sqrt(1.0+2.0/m));
}

///////////////////////////////////////////////////////////////////////////
//
/*! Compute K_1 */
//
///////////////////////////////////////////////////////////////////////////
double 
ViscoSCRAMHotSpot::computeK_1(double c, double sig_m)
{
  double m = d_initialData.CrackPowerValue;      // Parameter m
  double K_prime = computeK_prime(c, sig_m);
  return K_prime*pow((1.0+0.5*m), (1.0/m));
}

///////////////////////////////////////////////////////////////////////////
//
/*! Compute cdot */
//
///////////////////////////////////////////////////////////////////////////
double 
ViscoSCRAMHotSpot::computeCdot(const Matrix3& s, double sig_m, double c, 
                               double vres)
{
  // Constants
  double sqrtopf  = sqrt(1.5);

  // Compute the effective old total stress and old deviatoric 
  // stress norms
  double compflag = (sig_m < 0.0) ? 0.0 : 1.0 ;
  double sdots = s.NormSquared();
  double skk = s.Trace();
  double sigdotsig = sdots + sig_m*(3.0*sig_m + 2.0*skk);
  double sigEff = sqrtopf*sqrt(sigdotsig);
  double sEff = sqrtopf*sqrt(sdots);
  sigEff = (1.0-compflag)*sEff + compflag*sigEff;

  // Compute maximum crack speed
  double vinit = d_initialData.CrackGrowthRate;   // Initial crack growth rate
  double vmax = d_initialData.CrackMaxGrowthRate; // Max crack growth rate
  double m = d_initialData.CrackPowerValue;      // Parameter m
  vres *= ((1 - compflag) + vinit*compflag);
  vres = (vres > vmax) ? vmax : vres;

  // Compute stress intensity factors
  double K_I = computeK_I(c, sigEff);
  double K_prime = computeK_prime(c, sig_m);

  // Solve for new cracking rate
  double cdot = 0.0;
  if (K_I < K_prime) {
    cdot = vres*pow((K_I/K_prime), m);
  } else {
    double K_0mu = computeK_0mu(c, sig_m);
    double K_1 = computeK_1(c, sig_m);
    cdot = vres*(1.0-pow((K_0mu/K_1), 2));
  }

  return cdot;
}

///////////////////////////////////////////////////////////////////////////
//
/*! Compute sdot_n */
//
///////////////////////////////////////////////////////////////////////////
Matrix3 
ViscoSCRAMHotSpot::computeSdot_mw(const Matrix3& edot, const Matrix3& s, 
                                  Matrix3* s_n, double* G_n, double c, 
                                  double cdot, int mwelem, int numMaxwellElem)
{
  // Crack size/rate ratios
  double a = d_initialData.CrackParameterA;
  double ca = (c/a);
  double ca3 = ca*ca*ca;
  double onepca3 = 1.0 + ca3;
  double threeca3cdotoverc = 3.0*ca3*cdot/c;

  // Sums of maxwell elements
  double G = 0.0;
  Matrix3 sovertau(0.0);
  for (int imw = 0; imw < numMaxwellElem; ++imw) {
    G += G_n[imw];
    sovertau += (s_n[imw]*d_initialData.RTau[imw]);
  }

  // Compute total stress rate
  double theta = threeca3cdotoverc/onepca3;
  double psi = 2.0*G/onepca3;
  Matrix3 lambda_theta =  sovertau/onepca3;
  Matrix3 sdot = edot*psi - s*theta - lambda_theta;

  // Compute maxwell element stress rate
  Matrix3 term1 = edot*(2.0*G_n[mwelem]) - 
    s_n[mwelem]*d_initialData.RTau[mwelem];
  Matrix3 term2 = (s*threeca3cdotoverc + sdot*ca3)*(G/G_n[mwelem]);
  Matrix3 sdot_n = term1 - term2;
  return sdot_n;
}

///////////////////////////////////////////////////////////////////////////
//
/*! Evaluate the quantities cdot and sdot_n  */
//
///////////////////////////////////////////////////////////////////////////
ViscoSCRAMHotSpot::FVector
ViscoSCRAMHotSpot::evaluateRateEquations(const ViscoSCRAMHotSpot::FVector& Y, 
                                         const Matrix3& edot,
                                         double sig_m, double* G_n, double vres)
{
  // Get the data from Y ( c and s_n ) and compute total deviatoric stress
  int numMaxwellElem = Y.nn;
  double c = Y.a;
  Matrix3* s_n = scinew Matrix3[numMaxwellElem];
  Matrix3 s(0.0);
  for (int imw = 0; imw < numMaxwellElem; ++imw) {
    s_n[imw] = Y.b_n[imw];
    s += s_n[imw];
  }

  // evaluate the rate equations
  double cdot = computeCdot(s, sig_m, c, vres);
  Matrix3* sdot_n = scinew Matrix3[numMaxwellElem];
  for (int imw = 0; imw < numMaxwellElem; ++imw) {
    sdot_n[imw] = computeSdot_mw(edot, s, s_n, G_n, c, cdot,
                                 imw, numMaxwellElem);
  }

  // Push the rates into the FVector
  FVector Z(cdot, sdot_n);

  // free up memory
  delete [] s_n;
  delete [] sdot_n;

  return Z;
}

///////////////////////////////////////////////////////////////////////////
//
/*! Solve an ordinary differential equation of the form
  dy/dt = f(y,t)
  using a fourth-order Runge-Kutta method
  between t=T and t=T+delT (h = delT) */
//
///////////////////////////////////////////////////////////////////////////
ViscoSCRAMHotSpot::FVector
ViscoSCRAMHotSpot::integrateRateEquations(const ViscoSCRAMHotSpot::FVector& Y0, 
                                          const Matrix3& edot, 
                                          double sig_m, double* G_n,
                                          double vres, double delT,
                                          double& cdot)
{
  ViscoSCRAMHotSpot::FVector Y(Y0);
  ViscoSCRAMHotSpot::FVector f1 = 
    evaluateRateEquations(Y, edot, sig_m, G_n, vres);
  Y = Y + f1*(0.5*delT);
  ViscoSCRAMHotSpot::FVector f2 = 
    evaluateRateEquations(Y, edot, sig_m, G_n, vres);
  Y = Y + f2*(0.5*delT);
  ViscoSCRAMHotSpot::FVector f3 = 
    evaluateRateEquations(Y, edot, sig_m, G_n, vres);
  Y = Y + f3*delT;
  ViscoSCRAMHotSpot::FVector f4 = 
    evaluateRateEquations(Y, edot, sig_m, G_n, vres);
  cdot = f4.a;

  ViscoSCRAMHotSpot::FVector Y_new = Y0 + (f1+(f2+f3)*2.0+f4)*(delT/6.0);
  return Y_new;
}


///////////////////////////////////////////////////////////////////////////
//
/*! Compute viscous work rate */
//
///////////////////////////////////////////////////////////////////////////
double
ViscoSCRAMHotSpot::computeViscousWorkRate(int numElem, Matrix3* s_n, 
                                          double* G_n)
{
  double workrate = 0.0;
  for (int ii = 0; ii < numElem; ++ii) {
    workrate += 
      (s_n[ii].NormSquared()*d_initialData.RTau[ii]/(2.0*G_n[ii])); 
  }
  return workrate;
}

///////////////////////////////////////////////////////////////////////////
//
/*! Compute cracking damage work rate */
//
///////////////////////////////////////////////////////////////////////////
double
ViscoSCRAMHotSpot::computeCrackingWorkRate(int numElem, double c_new, 
                                           Matrix3* s_n_new, 
                                           const Matrix3& edot, 
                                           const Matrix3& s_new, 
                                           double* G_n, double cdot)
{
  int numMaxwellElem = numElem;

  // Crack size/rate ratios
  double a = d_initialData.CrackParameterA;
  double c = c_new;
  double ca = (c/a);
  double ca3 = ca*ca*ca;
  double onepca3 = 1.0 + ca3;
  double threeca3cdotoverc = 3.0*ca3*cdot/c;

  // Sums of maxwell elements
  double G = 0.0;
  Matrix3 sovertau(0.0);
  for (int imw = 0; imw < numMaxwellElem; ++imw) {
    G += G_n[imw];
    sovertau += (s_n_new[imw]*d_initialData.RTau[imw]);
  }

  // Compute total stress rate
  double invonepca3 = 1.0/onepca3;
  double theta = threeca3cdotoverc*invonepca3;
  double psi = 2.0*G/onepca3;
  Matrix3 lambdaTheta =  sovertau*invonepca3;
  Matrix3 sdot = edot*psi - s_new*theta - lambdaTheta;

  if (dbg.active())
    dbg << "SRate = " << endl << sdot << endl;

  // Compute maxwell element stress rate
  double workrate = (threeca3cdotoverc*(s_new.NormSquared()) + 
                     ca3*s_new.Contract(sdot))/(2.0*G);

  if (dbg.active())
    dbg << "Wdot_cr = " << workrate << endl;

  return workrate;
}

///////////////////////////////////////////////////////////////////////////
//
/*! Compute bulk chemicak heating rate */
//
///////////////////////////////////////////////////////////////////////////
double 
ViscoSCRAMHotSpot::computeChemicalHeatRate(double rho, double T_old)
{
  double delH = d_matConst.delH;
  double Z = d_matConst.Z;
  double EoverR = d_matConst.EoverR;

  double qdot = rho*delH*Z*exp(-EoverR/T_old);
  return qdot;
}

///////////////////////////////////////////////////////////////////////////
//
/*! Compute the conduction K matrix for the hotspot model */
//
///////////////////////////////////////////////////////////////////////////
void
ViscoSCRAMHotSpot::computeHotSpotKmatrix(double y1, double y2, double kappa,
                                         FastMatrix& K)
{
  double t3 = 2.0/(-y1+y2)*kappa;
  double t4 = 647.0/270.0*t3;
  double t5 = 352.0/135.0*t3;
  double t6 = 32.0/135.0*t3;
  double t7 = 7.0/270.0*t3;
  double t8 = 544.0/135.0*t3;
  double t9 = 224.0/135.0*t3;
  K(0,0) = t4;
  K(0,1) = -t5;
  K(0,2) = t6;
  K(0,3) = -t7;
  K(1,0) = -t5;
  K(1,1) = t8;
  K(1,2) = -t9;
  K(1,3) = t6;
  K(2,0) = t6;
  K(2,1) = -t9;
  K(2,2) = t8;
  K(2,3) = -t5;
  K(3,0) = -t7;
  K(3,1) = t6;
  K(3,2) = -t5;
  K(3,3) = t4;
}

///////////////////////////////////////////////////////////////////////////
//
/*! Compute the heat capacity matrix C for the hotspot model */
//
///////////////////////////////////////////////////////////////////////////
void
ViscoSCRAMHotSpot::computeHotSpotCmatrix(double y1, double y2, double rho, 
                                         double Cv, FastMatrix& CC)
{
  double t3 = rho*Cv*(-y1+y2)/2.0;
  double t4 = 134.0/945.0*t3;
  double t5 = 4.0/315.0*t3;
  double t6 = 68.0/945.0*t3;
  double t7 = t3/35.0;
  double t8 = 704.0/945.0*t3;
  double t9 = 64.0/315.0*t3;
  CC(0,0) = t4;
  CC(0,1) = t5;
  CC(0,2) = -t6;
  CC(0,3) = t7;
  CC(1,0) = t5;
  CC(1,1) = t8;
  CC(1,2) = t9;
  CC(1,3) = -t6;
  CC(2,0) = -t6;
  CC(2,1) = t9;
  CC(2,2) = t8;
  CC(2,3) = t5;
  CC(3,0) = t7;
  CC(3,1) = -t6;
  CC(3,2) = t5;
  CC(3,3) = t4;
}

///////////////////////////////////////////////////////////////////////////
//
/*! Compute the chemical heat rate matrix Qdot for the hotspot model */
//
///////////////////////////////////////////////////////////////////////////
void
ViscoSCRAMHotSpot::computeHotSpotQdotmatrix(double y1, double y2, 
                                            double rho, double mu_d,
                                            double sig_m, double edotmax, 
                                            double T1, double T2,
                                            double* Qdot)
{
  double t1 = -y1+y2;
  double t2 = rho*d_matConst.delH;
  double t4 = d_matConst.EoverR;
  double t7 = exp(t4/T1);
  double t10 = t2*d_matConst.Z/t7;
  double t14 = exp(t4/T2);
  double t17 = t2*d_matConst.Z/t14;
  double t20 = mu_d*sig_m*edotmax;
  double t21 = t20/9.0;
  double t26 = 8.0/9.0*t20;
  Qdot[0] = t1*(2.0/15.0*t10-t17/45.0-t21)/2.0;
  Qdot[1] = t1*(28.0/45.0*t10+4.0/15.0*t17-t26)/2.0;
  Qdot[2] = t1*(4.0/15.0*t10+28.0/45.0*t17-t26)/2.0;
  Qdot[3] = t1*(-t10/45.0+2.0/15.0*t17-t21)/2.0;
}

///////////////////////////////////////////////////////////////////////////
//
/*! Compute the rate of temperature change at the hot spot */
//
///////////////////////////////////////////////////////////////////////////
void
ViscoSCRAMHotSpot::evaluateTdot(double* T, FastMatrix& k, FastMatrix& C, 
                                double y1, double y2, double rho, double mu_d,
                                double sig_m, double edotmax,
                                double* Tdot) 
{
  double T1 = T[0];
  double T2 = T[1];

  // Compute Qdot
  double Qdot[4];
  computeHotSpotQdotmatrix(y1, y2, rho, mu_d, sig_m, edotmax, T1, T2, Qdot);

  // Compute kT
  double kT[4];
  k.multiply(T, kT);
  
  // Compute kT+Qdot
  double kTplusQdot[4];
  for (int ii = 0; ii < 4; ++ii) kTplusQdot[ii] = kT[ii] + Qdot[ii];

  // Solve for Tdot (C Tdot = kT + Qdot) - Tdot is returned in kTplusQdot
  C.destructiveSolve(kTplusQdot);
  for (int ii = 0; ii < 4; ++ii) Tdot[ii] = kTplusQdot[ii];
}

///////////////////////////////////////////////////////////////////////////
//
/*! Compute the increment of temperature at the hot spot using a 
  Fourth-order Runge-Kutta scheme */
//
///////////////////////////////////////////////////////////////////////////
void
ViscoSCRAMHotSpot::updateHotSpotTemperature(double* T, double y1, double y2, 
                                            double kappa, double rho, 
                                            double Cv, double mu_d,
                                            double sig_m, double edotmax, 
                                            double delT)
{
  FastMatrix kmatrix(4,4);
  computeHotSpotKmatrix(y1, y2, kappa, kmatrix);
  FastMatrix Cmatrix(4,4);
  computeHotSpotCmatrix(y1, y2, rho, Cv, Cmatrix);

  double T_RK[4];
  double Tdot1[4], Tdot2[4], Tdot3[4], Tdot4[4];

  // First RK term
  for (int ii = 0; ii < 4; ++ii) T_RK[ii] = T[ii];
  evaluateTdot(T_RK, kmatrix, Cmatrix, y1, y2, rho, mu_d, 
               sig_m, edotmax, Tdot1);
  
  // Second RK term
  for (int ii = 0; ii < 4; ++ii) T_RK[ii] = T[ii] + Tdot1[ii]*(delT*0.5);
  evaluateTdot(T_RK, kmatrix, Cmatrix, y1, y2, rho, mu_d, 
               sig_m, edotmax, Tdot2);
  
  // Third RK term
  for (int ii = 0; ii < 4; ++ii) T_RK[ii] = T[ii] + Tdot2[ii]*(delT*0.5);
  evaluateTdot(T_RK, kmatrix, Cmatrix, y1, y2, rho, mu_d, 
               sig_m, edotmax, Tdot3);
  
  // Fourth RK term
  for (int ii = 0; ii < 4; ++ii) T_RK[ii] = T[ii] + Tdot3[ii]*delT;
  evaluateTdot(T_RK, kmatrix, Cmatrix, y1, y2, rho, mu_d, 
               sig_m, edotmax, Tdot4);

  // Compute the updated temperature
  for (int ii = 0; ii < 4; ++ii) 
    T[ii] += (Tdot1[ii] + (Tdot2[ii]+Tdot3[ii])*2.0 + Tdot4[ii])*(delT/6.0);
}

///////////////////////////////////////////////////////////////////////////
//
/*! Evaluate the hot spot model */
//
///////////////////////////////////////////////////////////////////////////
void
ViscoSCRAMHotSpot::evaluateHotSpotModel(double sig_m, const Matrix3& sig, 
                                        const Matrix3& edot, double* T,
                                        double kappa, double rho, double Cv, 
                                        double delT)
{
  // For tensile states of stress, no heating is generated
  if (sig_m > 0) return;

  // Compute the eigenvalues and eigenvectors of the deviatoric rate
  // of deformation tensor
  SymmMatrix3 defRateDev(edot);
  Vector eigval(0.0, 0.0, 0.0);
  Matrix3 eigvec(0.0);
  defRateDev.eigen(eigval, eigvec);
  
  // Compute maximum principal rate of deformation
  double edotmax = eigval[0];

  // Get the eigenVector in this direction
  Vector edotvec(eigvec(0,0), eigvec(1,0), eigvec(2,0));

  if (dbg.active())
    dbg << " Max dev edot = " << edotmax << " Direction = " << edotvec << endl;

  // Compute the component of the shear stress in the plane of the "crack"
  // (the component sigCrack(2,3))
  Matrix3 sigCrack = stressInRotatedBasis(sig, edotvec);
  double shearCrackFace = sigCrack(1,2);

  // Check if there is frictional heating
  double mu_s = d_initialData.CrackFriction;
  double mu_d = d_matConst.mu_d;
  if (shearCrackFace <= mu_s*sig_m) mu_d = 0;

  // Compute the updated of temperature at the hotspot using a fourth-order
  // Runge-Kutta method
  double y1 = 0.0; double y2 = 1.0e-3;
  updateHotSpotTemperature(T, y1, y2, kappa, rho, Cv, mu_d,
                           sig_m, edotmax, delT);
}

////////////////////////////////////////////////////////////////////////////////
//
/*! Express a stress tensor in terms of a new set of bases ( with the vector 
  e1Prime being the new direction of e1 ) */
// 
////////////////////////////////////////////////////////////////////////////////
Matrix3 
ViscoSCRAMHotSpot::stressInRotatedBasis(const Matrix3& sig, 
                                        Vector& e1Prime)
{
  Matrix3 sigPrime = sig;

  // Vector e1
  Vector e1(1.0, 0.0, 0.0);

  // Normalize n (make into unit vector)
  e1Prime = e1Prime/e1Prime.length();

  // Calculate the rotation angle (assume n0 and n are unit vectors)
  double phi = acos(Dot(e1Prime,e1)/(e1Prime.length()*e1.length()));
  if (phi == 0.0) return sigPrime;

  // Find the rotation axis
  Vector a = Cross(e1Prime,e1);
  if (a.length() <= 0.0) return sigPrime;
  a /= (a.length());

  // Return the rotation matrix
  Matrix3 R(phi, a);

  // Rotate the stress
  sigPrime = R*sig*R.Transpose();
  
  return sigPrime;
}
