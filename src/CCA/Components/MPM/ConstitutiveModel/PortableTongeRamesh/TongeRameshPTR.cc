/*
 * The MIT License
 *
 * Copyright (c) 2013-2017 The Johns Hopkins University
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

// Adapted from UCNH.cc by Andy Tonge Dec 2011 altonge@gmail.com

#include "TongeRameshPTRCalcs/UMATTR.h"
#include "TongeRameshPTRCalcs/PTR_defs.h"
#include <CCA/Components/MPM/ConstitutiveModel/TongeRamesh_gitInfo.h>
#include "CCA/Components/MPM/ConstitutiveModel/PortableTongeRamesh/TongeRameshPTR.h"

#include <Core/Grid/Patch.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Geometry/Vector.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Math/Matrix3.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
#include <vector>
#include <stdexcept>

using namespace Uintah;

//#define Comer
#undef Comer

// Constructors //
//////////////////
TongeRameshPTR::TongeRameshPTR(ProblemSpecP& ps, MPMFlags* Mflag)
  : ConstitutiveModel(Mflag), ImplicitCM()
{
  proc0cout << "TongeRameshPTR Material model:\n\t Last commit date:"
            << build_date << "\n"
            << "\t Commit sha and message: " << build_git_commit
            << std::endl;

  d_nProps = PTR_NUM_MAT_PARAMS;
  d_matParamArray.reserve(d_nProps);
  for (int i=0; i<d_nProps; ++i){
    std::string matParamName="PTR_" + std::string(PTR_MatParamNames[i]);
    double tmpVal;
    ps->require(matParamName, tmpVal);
    d_matParamArray.push_back(tmpVal);
  }
  initializeLocalMPMLabels(); // Create CM specific labels
}

void
TongeRameshPTR::initializeLocalMPMLabels()
{
  d_numHistVar = PTR_umat_getNumStateVars(d_nProps, d_matParamArray.data());
  // Based on Kayenta interface:
  for( int i=0; i<d_numHistVar; ++i){
    char tmpVarName[80];
    PTR_umat_getStateVarName(i,tmpVarName);
    histVarVect.push_back(VarLabel::create("p.PTR_"+std::string(tmpVarName),
                                         ParticleVariable<double>::getTypeDescription()));
    histVarVect_preReloc.push_back(VarLabel::create("p.PTR_"+std::string(tmpVarName)+"+",
                                                  ParticleVariable<double>::getTypeDescription()));
  }
  pSSELabel = VarLabel::create("p.SpecificStrainEnergy", ParticleVariable<double>::getTypeDescription());
  pSSELabel_preReloc = VarLabel::create("p.SpecificStrainEnergy+",
                                           ParticleVariable<double>::getTypeDescription());
}

void TongeRameshPTR::outputProblemSpec(ProblemSpecP& ps, bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","TongeRameshPTR");
  }

  for(int i=0; i<d_nProps; ++i){
    cm_ps->appendElement(PTR_MatParamNames[i], d_matParamArray[i]);
  }
}

TongeRameshPTR* TongeRameshPTR::clone()
{
	return scinew TongeRameshPTR(*this);
}

TongeRameshPTR::~TongeRameshPTR()
{
  for(int i=0; i<d_numHistVar; ++i){
    VarLabel::destroy(histVarVect_preReloc[i]);
    VarLabel::destroy(histVarVect[i]);
  }
  VarLabel::destroy(pSSELabel_preReloc);
  VarLabel::destroy(pSSELabel);
}

// // Initialization Functions //
// //////////////////////////////

void TongeRameshPTR::carryForward(const PatchSubset* patches,
                               const MPMMaterial* matl,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw)
{
  for(int p=0; p<patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    // Carry forward the data common to all constitutive models
    // when using RigidMPM.
    // This method is defined in the ConstitutiveModel base class.
    carryForwardSharedData(pset, old_dw, new_dw, matl);

    // Carry forward the data local to this constitutive model
    for(int i=0; i<d_numHistVar; ++i){
      ParticleVariable<double> pHistVar_new;
      constParticleVariable<double> pHistVar_old;
      old_dw->get(pHistVar_old, histVarVect[i], pset);
      new_dw->allocateAndPut(pHistVar_new, histVarVect_preReloc[i], pset);
      pHistVar_new.copyData(pHistVar_old);
    }
    ParticleVariable<double> pSSE_new;
    ParticleVariable<int> pLoc_new;
    constParticleVariable<double> pSSE_old;
    constParticleVariable<int> pLoc_old;
    old_dw->get(pSSE_old, pSSELabel, pset);
    new_dw->allocateAndPut(pSSE_new, pSSELabel_preReloc, pset);
    pSSE_new.copyData(pSSE_old);
    pLoc_new.copyData(pLoc_old);
  } // End Patch Loop
}

void TongeRameshPTR::initializeCMData(const Patch* patch,
                                   const MPMMaterial* matl,
                                   DataWarehouse* new_dw)
{
	// Initialize the variables shared by all constitutive models
	// This method is defined in the ConstitutiveModel base class.
	if (flag->d_integrator == MPMFlags::Implicit)
		initSharedDataForImplicit(patch, matl, new_dw);
	else {
		initSharedDataForExplicit(patch, matl, new_dw);
	}
	// Put stuff in here to initialize each particle's
	// constitutive model parameters and deformationMeasure
	Matrix3 Identity;
	Identity.Identity();
	Matrix3 zero(0.0);

	ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

    // Again motivated by Kayenta.cc:
    std::vector<ParticleVariable<double> > HistVars(d_numHistVar);
    for (int i=0; i<d_numHistVar; ++i){
      new_dw->allocateAndPut(HistVars[i], histVarVect[i], pset);
    }
    ParticleVariable<double> pSSE_new;
    ParticleVariable<int> pLoc_new;
    new_dw->allocateAndPut(pSSE_new, pSSELabel, pset);

    constParticleVariable<double> pVolume;
    new_dw->get(pVolume, lb->pVolumeLabel,            pset);


    unsigned int patchID = patch->getID();
    unsigned int matID   = matl->getDWIndex();
    ParticleSubset::iterator iter = pset->begin();
    for(; iter != pset->end(); ++iter){
      unsigned long seedArray[3];
      const int numSeedVals = 3;
      seedArray[0] = matID;
      seedArray[1] = patchID;
      seedArray[2] = *iter;
      std::vector<double> histVarVal(d_numHistVar, 0.0);
      const double dx_ave = cbrt(pVolume[*iter]);
      PTR_umat_getInitialValues(d_numHistVar, histVarVal.data(), d_nProps,
                                d_matParamArray.data(), dx_ave, numSeedVals, seedArray);
      for (int i=0; i<d_numHistVar; ++i){
        (HistVars[i])[*iter] = histVarVal[i];
      }
      pSSE_new[*iter] = 0.0;
      pLoc_new[*iter] = 0;
    }
    
	// If not implicit, compute timestep
	if(!(flag->d_integrator == MPMFlags::Implicit)) {
		// End by computing the stable timestep
		computeStableTimestep(patch, matl, new_dw);
	}
}

// Scheduling Functions //
//////////////////////////
void TongeRameshPTR::addComputesAndRequires(Task* task,
                                            const MPMMaterial* matl,
                                            const PatchSet* patches) const
{
  // Add the computes and requires that are common to all explicit
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();
  if (flag->d_integrator == MPMFlags::Implicit) {
    bool reset = flag->d_doGridReset;
    addSharedCRForImplicit(task, matlset, reset);
  } else {
    addSharedCRForHypoExplicit(task, matlset, patches);
  }

  // Other constitutive model and input dependent computes and requires
  Ghost::GhostType  gnone = Ghost::None;
  for (int i=0; i<d_numHistVar; ++i){
    task->requires(Task::OldDW, histVarVect[i], matlset, gnone);
    task->computes(histVarVect_preReloc[i],     matlset);
  }
  task->requires(Task::OldDW, pSSELabel, matlset, gnone);
  task->computes(pSSELabel_preReloc, matlset);
  task->computes(lb->pLocalizedMPMLabel_preReloc, matlset);
}

void TongeRameshPTR::addComputesAndRequires(Task* task,
                                            const MPMMaterial* matl,
                                            const PatchSet* patches,
                                            const bool recurse,
                                            const bool SchedParent) const
{
  throw ProblemSetupException("This addComputesAndRequires() does not add damage requires"
                              , __FILE__, __LINE__);
}

void TongeRameshPTR::addInitialComputesAndRequires(Task* task,
                                                   const MPMMaterial* matl,
                                                   const PatchSet*) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  // Other constitutive model and input dependent computes and requires
  // task->requires(Task::NewDW, lb->pVolumeLabel, matlset, gnone);
  for (int i=0; i<d_numHistVar; ++i){
    task->computes(histVarVect[i],     matlset);
  }
  task->computes(pSSELabel, matlset);
}

// Compute Functions //
///////////////////////
// RAS throws
void TongeRameshPTR::computePressEOSCM(const double rho_cur,double& pressure,
                                    const double p_ref,
                                    double& dp_drho, double& cSquared,
                                    const MPMMaterial* matl,
                                    double temperature)
{
	throw std::runtime_error("TongeRameshPTR::computePressEOSCM() has not been updated");
}

// The "CM" versions use the pressure-volume relationship of the CNH model
// RAS throws
double TongeRameshPTR::computeRhoMicroCM(double pressure,
                                      const double p_ref,
                                      const MPMMaterial* matl,
                                      double temperature,
                                      double rho_guess)
{
	throw std::runtime_error("TongeRameshPTR::computeRhoMicroCM() has not been updated");
}

void TongeRameshPTR::computeStableTimestep(const Patch* patch,
                                           const MPMMaterial* matl,
                                           DataWarehouse* new_dw)
{
  // This is only called for the initial timestep - all other timesteps
  // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int dwi = matl->getDWIndex();
  // Retrieve the array of constitutive parameters
  ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
  constParticleVariable<Vector> pVelocity;
  new_dw->get(pVelocity, lb->pVelocityLabel, pset);
  double dx_min = std::min(dx.x(), std::min(dx.y(), dx.z()));
  double delT_new = 1.e12;
  double c_dil = sqrt( (d_matParamArray[PTR_BULKMOD_IDX]+4.0/3.0*d_matParamArray[PTR_SHEARMOD_IDX])/
                       d_matParamArray[PTR_DESITY_IDX] );
  for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++) {
    particleIndex idx = *iter;
    double pspeed = pVelocity[idx].length();
    double ans    = dx_min / (pspeed+c_dil);
    delT_new = std::min(delT_new, ans);
  }
  new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
}

void TongeRameshPTR::computeStressTensor(const PatchSubset* patches,
                                      const MPMMaterial* matl,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw)
{
  // Constants
  // double onethird = (1.0/3.0), sqtwthds = sqrt(2.0/3.0);
  Matrix3 Identity;
  Identity.Identity();

  delt_vartype delT;
  old_dw->get(delT, lb->delTLabel, getLevel(patches));

  // Normal patch loop
  for(int pp=0; pp<patches->size(); pp++) {
    const Patch* patch = patches->get(pp);

    double se=0.0;     // Strain energy placeholder
    unsigned long totalLocalizedParticle = 0;

    // Get particle info and patch info
    int dwi              = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    Vector dx            = patch->dCell();
    Vector WaveSpeed(1e-12);
        
    std::vector<constParticleVariable<double> > pHistVars_old(d_numHistVar);
    std::vector<ParticleVariable<double> > pHistVars_new(d_numHistVar);
    for (int i=0; i<d_numHistVar; ++i){
      old_dw->get(pHistVars_old[i],            histVarVect[i], pset);
      new_dw->allocateAndPut(pHistVars_new[i], histVarVect_preReloc[i], pset);
    }
    constParticleVariable<double>  pSSE_old;
    ParticleVariable<double>  pSSE_new;
    ParticleVariable<int>  pLocalized;
    old_dw->get(pSSE_old, pSSELabel, pset);
    new_dw->allocateAndPut(pSSE_new, pSSELabel_preReloc, pset);
    new_dw->allocateAndPut(pLocalized, lb->pLocalizedMPMLabel_preReloc, pset);

    // Particle and grid data universal to model type
    // Old data containers
    constParticleVariable<double>  pMass, pTemperature;
    constParticleVariable<Matrix3> pStress_old;
    constParticleVariable<Vector>  pVelocity;

    // New data containers
    ParticleVariable<Matrix3>      pStress;
    ParticleVariable<double>       p_q,pdTdt; // Rate of change of temperature
    new_dw->allocateAndPut(pStress, lb->pStressLabel_preReloc, pset);
    new_dw->allocateAndPut(pdTdt, lb->pdTdtLabel,   pset);
    new_dw->allocateAndPut(p_q, lb->p_qLabel_preReloc,   pset);

    // Kinematic variables provided by the framework:
    constParticleVariable<Matrix3> pVelGrad, pDefGrad_new;
    constParticleVariable<double>  pVolume_new;

    // Universal Gets
    old_dw->get(pMass,               lb->pMassLabel,               pset);
    old_dw->get(pVelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(pStress_old,         lb->pStressLabel,             pset);
    old_dw->get(pTemperature,        lb->pTemperatureLabel,        pset);

    // Universal Gets for the current timestep:
    new_dw->get(pVelGrad,            lb->pVelGradLabel_preReloc,   pset);
    new_dw->get(pVolume_new,         lb->pVolumeLabel_preReloc,    pset);
    new_dw->get(pDefGrad_new,        lb->pDeformationMeasureLabel_preReloc,pset);

    ParticleSubset::iterator iter = pset->begin();
    for(; iter != pset->end(); iter++) {
      particleIndex idx = *iter;
      Matrix3 stress  = pStress_old[idx];
      Matrix3 L       = pVelGrad[idx];
      Matrix3 defGrad = pDefGrad_new[idx];
      std::vector<double> histVar(d_numHistVar,0.0);
      for(int i = 0; i<d_numHistVar; ++i){
        histVar[i] = pHistVars_old[i][idx];
      }

      // Abaqus uses the Hughes and Winget (1980) approach to finite rotation
      Matrix3 spin   = 0.5*(L-L.Transpose())*delT;
      Matrix3 Q      = Identity + ((Identity - 0.5*spin).Inverse())*spin;
      // Matrix3 Q = Identity;
      stress         = Q*(stress*Q.Transpose());
      Matrix3 D      = 0.5*(L.Transpose() + L);

      double lnJ = log(defGrad.Determinant());
      double DRPLDT;
      // WARNING: This ONLY valid for the PTR_umat (UMATTR.h) becuase
      // that model uses STRAN to compute the volume change ratio
      // only.
      double STRAN[6]  = {lnJ/3.0, lnJ/3.0, lnJ/3.0, 0.0, 0.0, 0.0};
      double DSTRAN[6] = {D(0,0)*delT,
                          D(1,1)*delT,
                          D(2,2)*delT,
                          (D(0,1)+D(1,0))*delT,
                          (D(0,2)+D(2,0))*delT,
                          (D(2,1)+D(1,2))*delT};

      // Prepare other inputs to UMAT call:
      double STRESS[6]  = {stress(0,0), stress(1,1), stress(2,2),
                           0.5*(stress(0,1)+stress(1,0)),
                           0.5*(stress(2,0)+stress(0,2)),
                           0.5*(stress(2,1)+stress(1,2))};
      double DDSDDE[6][6];
      double SSE     = pSSE_old[idx];  // specific strain energy
      double SPD     = 0;  // specific plastic dissipation 
      double SCD     = 0;  // Unused
      double RPL     = 0;  // Output
      double DDSDDT[6];
      double DRPLDE[6];
      double TIME[2] = {200*delT, 201*delT}; // Prevent resetting the flaw dist at time t=0
      double DTIME   = delT;
      double TEMP    = pTemperature[idx];
      double DTEMP;
      double PREDEF(0.0); // unused
      double DPRED(0.0);  // unused
      double CMNAME(10.0); // Unused
      int NDI(3);
      int NSHR(3);
      int NTENS(6);
      int NSTATV(d_numHistVar);
      int NPROPS(d_nProps);
      double COORDS[3] = {0.0,0.0,0.0};
      double DROT[3][3];
      // TO DO: unroll these loops:
      for (int i=0; i<3; ++i){
        for (int j=0; j<3; ++j){
          DROT[j][i] = Q(i,j);
        }
      }
      double PNEWDT(1.0e12); // unused
      double CELENT = cbrt(pVolume_new[idx]);
      double DFGRD0[3][3], DFGRD1[3][3]; // unused
      int NOEL(0);
      int NPT(0);
      int LAYER(0);
      int KSPT(0);
      int KSTEP(0);
      int KINC(0);

      PTR_umat_stressUpdate(STRESS, histVar.data(), DDSDDE,
                            &SSE, &SPD, &SCD, &RPL,
                            DDSDDT, DRPLDE, &DRPLDT,
                            STRAN, DSTRAN, TIME,
                            &DTIME, &TEMP, &DTEMP,
                            &PREDEF, &DPRED, &CMNAME,
                            &NDI, &NSHR, &NTENS, &NSTATV,
                            d_matParamArray.data(), &NPROPS, COORDS,
                            DROT, &PNEWDT, &CELENT,
                            DFGRD0, DFGRD1,
                            &NOEL, &NPT, &LAYER,
                            &KSPT, &KSTEP, &KINC);

      // Copy state variables back to the framework:
      for(int i = 0; i<d_numHistVar; ++i){
        pHistVars_new[i][idx] = histVar[i];
      }
      // Stress:
      pStress[idx](0,0) = STRESS[0];
      pStress[idx](1,1) = STRESS[1];
      pStress[idx](2,2) = STRESS[2];
      pStress[idx](0,1) = STRESS[3];
      pStress[idx](1,0) = STRESS[3];
      pStress[idx](0,2) = STRESS[4];
      pStress[idx](2,0) = STRESS[4];
      pStress[idx](1,2) = STRESS[5];
      pStress[idx](2,1) = STRESS[5];
      pSSE_new[idx]     = SSE;
      pdTdt[idx]        = DTEMP/delT;
      p_q[idx]          = histVar[7];
      pLocalized[idx]   = static_cast<int>(std::floor(histVar[PTR_LOCALIZED_IDX]+0.5));
      if( pLocalized[idx] == 0 ){
        se += SSE*pMass[idx];
      } else {
        ++totalLocalizedParticle;
      }
      double shrMod = (DDSDDE[3][3]+DDSDDE[4][4] + DDSDDE[5][5])/3.0;
      double blkMod = (DDSDDE[0][0]+DDSDDE[0][1] + DDSDDE[0][2] +
                       DDSDDE[1][0]+DDSDDE[1][1] + DDSDDE[1][2] +
                       DDSDDE[2][0]+DDSDDE[2][1] + DDSDDE[2][2]) / 9.0;
      double c_dil = sqrt( (blkMod + 4.0/3.0*shrMod)*pVolume_new[idx]/pMass[idx]);
      WaveSpeed[0] = std::max(WaveSpeed[0], std::abs(pVelocity[idx].x())+c_dil);
      WaveSpeed[1] = std::max(WaveSpeed[1], std::abs(pVelocity[idx].y())+c_dil);
      WaveSpeed[2] = std::max(WaveSpeed[2], std::abs(pVelocity[idx].z())+c_dil);
    } // end loop over particles
    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();

    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(se),        lb->StrainEnergyLabel);
    }
    // new_dw->put(sumlong_vartype(totalLocalizedParticle), lb->TotalLocalizedParticleLabel);
  }
}

// RAS throws
void TongeRameshPTR::computeStressTensor(const PatchSubset* patches,
                                      const MPMMaterial* matl,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw,
                                      Solver* solver,
                                      const bool )

{
	throw std::runtime_error("The TongeRamash material model is not designed to be used with implicit analysis");
}

// Helper Functions //
//////////////////////
double TongeRameshPTR::getCompressibility()
{
  return 1.0/d_matParamArray[PTR_BULKMOD_IDX];
}


void TongeRameshPTR::addParticleState(std::vector<const VarLabel*>& from,
                                   std::vector<const VarLabel*>& to)
{
  // Add the local particle state data for this constitutive model.
  for(int i=0; i<d_numHistVar; ++i){
    from.push_back(histVarVect[i]);
    to.push_back(histVarVect_preReloc[i]);
  }
  from.push_back(pSSELabel);
  to.push_back(pSSELabel_preReloc);
}


// RAS throws
void TongeRameshPTR::computeStressTensorImplicit(const PatchSubset* patches,
        const MPMMaterial* matl,
        DataWarehouse* old_dw,
        DataWarehouse* new_dw)
{

	std::ostringstream msg;
	msg << "\n ERROR: In TongeRameshPTR::computeStressTensorImplicit \n"
	    << "\t This function has not been updated and should not be used. \n";
	throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
}

/*! Compute tangent stiffness matrix */
// ALT Throws
void TongeRameshPTR::computeTangentStiffnessMatrix(const Matrix3& sigdev,
        const double&  mubar,
        const double&  J,
        const double&  bulk,
        double D[6][6])
{
  	std::ostringstream msg;
	msg << "\n ERROR: In TongeRameshPTR::computeTangentStiffnessMatrix \n"
	    << "\t This function has not been updated and should not be used. \n";
	throw ProblemSetupException(msg.str(),__FILE__, __LINE__);

	double twth = 2.0/3.0;
	double frth = 2.0*twth;
	double coef1 = bulk;
	double coef2 = 2.*bulk*log(J);

	for (int ii = 0; ii < 6; ++ii) {
		for (int jj = 0; jj < 6; ++jj) {
			D[ii][jj] = 0.0;
		}
	}
	D[0][0] = coef1 - coef2 + mubar*frth - frth*sigdev(0,0);
	D[0][1] = coef1 - mubar*twth - twth*(sigdev(0,0) + sigdev(1,1));
	D[0][2] = coef1 - mubar*twth - twth*(sigdev(0,0) + sigdev(2,2));
	D[0][3] =  - twth*(sigdev(0,1));
	D[0][4] =  - twth*(sigdev(0,2));
	D[0][5] =  - twth*(sigdev(1,2));
	D[1][1] = coef1 - coef2 + mubar*frth - frth*sigdev(1,1);
	D[1][2] = coef1 - mubar*twth - twth*(sigdev(1,1) + sigdev(2,2));
	D[1][3] =  D[0][3];
	D[1][4] =  D[0][4];
	D[1][5] =  D[0][5];
	D[2][2] = coef1 - coef2 + mubar*frth - frth*sigdev(2,2);
	D[2][3] =  D[0][3];
	D[2][4] =  D[0][4];
	D[2][5] =  D[0][5];
	D[3][3] =  -.5*coef2 + mubar;
	D[4][4] =  D[3][3];
	D[5][5] =  D[3][3];
}

/*! Compute K matrix */
// ALT Throws
void TongeRameshPTR::computeStiffnessMatrix(const double B[6][24],
        const double Bnl[3][24],
        const double D[6][6],
        const Matrix3& sig,
        const double& vol_old,
        const double& vol_new,
        double Kmatrix[24][24])
{
  	std::ostringstream msg;
	msg << "\n ERROR: In TongeRameshPTR::computeStiffnessMatrix \n"
	    << "\t This function has not been updated and should not be used. \n";
	throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
	// Kmat = B.transpose()*D*B*volold
	double Kmat[24][24];
	BtDB(B, D, Kmat);

	// Kgeo = Bnl.transpose*sig*Bnl*volnew;
	double Kgeo[24][24];
	BnlTSigBnl(sig, Bnl, Kgeo);

	/*
	  cout.setf(ios::scientific,ios::floatfield);
	  cout.precision(10);
	  cout << "Kmat = " << std::endl;
	  for(int kk = 0; kk < 24; kk++) {
	  for (int ll = 0; ll < 24; ++ll) {
	  cout << Kmat[ll][kk] << " " ;
	  }
	  cout << std::endl;
	  }
	  cout << "Kgeo = " << std::endl;
	  for(int kk = 0; kk < 24; kk++) {
	  for (int ll = 0; ll < 24; ++ll) {
	  cout << Kgeo[ll][kk] << " " ;
	  }
	  cout << std::endl;
	  }
	*/

	for(int ii = 0; ii<24; ii++) {
		for(int jj = 0; jj<24; jj++) {
			Kmatrix[ii][jj] =  Kmat[ii][jj]*vol_old + Kgeo[ii][jj]*vol_new;
		}
	}
}

// ALT Throws
void TongeRameshPTR::BnlTSigBnl(const Matrix3& sig, const double Bnl[3][24],
                             double Kgeo[24][24]) const
{
  	std::ostringstream msg;
	msg << "\n ERROR: In TongeRameshPTR::BnlTSigBnl \n"
	    << "\t This function has not been updated and should not be used. \n";
	throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
  
	double t1, t10, t11, t12, t13, t14, t15, t16, t17;
	double t18, t19, t2, t20, t21, t22, t23, t24, t25;
	double t26, t27, t28, t29, t3, t30, t31, t32, t33;
	double t34, t35, t36, t37, t38, t39, t4, t40, t41;
	double t42, t43, t44, t45, t46, t47, t48, t49, t5;
	double t50, t51, t52, t53, t54, t55, t56, t57, t58;
	double t59, t6, t60, t61, t62, t63, t64, t65, t66;
	double t67, t68, t69, t7, t70, t71, t72, t73, t74;
	double t75, t77, t78, t8, t81, t85, t88, t9, t90;
	double t79, t82, t83, t86, t87, t89;

	t1  = Bnl[0][0]*sig(0,0);
	t4  = Bnl[0][0]*sig(0,0);
	t2  = Bnl[0][0]*sig(0,1);
	t3  = Bnl[0][0]*sig(0,2);
	t5  = Bnl[1][1]*sig(1,1);
	t8  = Bnl[1][1]*sig(1,1);
	t6  = Bnl[1][1]*sig(1,2);
	t7  = Bnl[1][1]*sig(0,1);
	t9  = Bnl[2][2]*sig(2,2);
	t12 = Bnl[2][2]*sig(2,2);
	t10 = Bnl[2][2]*sig(0,2);
	t11 = Bnl[2][2]*sig(1,2);
	t13 = Bnl[0][3]*sig(0,0);
	t16 = Bnl[0][3]*sig(0,0);
	t14 = Bnl[0][3]*sig(0,1);
	t15 = Bnl[0][3]*sig(0,2);
	t17 = Bnl[1][4]*sig(1,1);
	t20 = Bnl[1][4]*sig(1,1);
	t18 = Bnl[1][4]*sig(1,2);
	t19 = Bnl[1][4]*sig(0,1);
	t21 = Bnl[2][5]*sig(2,2);
	t22 = Bnl[2][5]*sig(0,2);
	t23 = Bnl[2][5]*sig(1,2);
	t24 = Bnl[2][5]*sig(2,2);
	t25 = Bnl[0][6]*sig(0,0);
	t26 = Bnl[0][6]*sig(0,1);
	t27 = Bnl[0][6]*sig(0,2);
	t28 = Bnl[0][6]*sig(0,0);
	t29 = Bnl[1][7]*sig(1,1);
	t30 = Bnl[1][7]*sig(1,2);
	t31 = Bnl[1][7]*sig(0,1);
	t32 = Bnl[1][7]*sig(1,1);
	t33 = Bnl[2][8]*sig(2,2);
	t34 = Bnl[2][8]*sig(0,2);
	t35 = Bnl[2][8]*sig(1,2);
	t36 = Bnl[2][8]*sig(2,2);
	t37 = Bnl[0][9]*sig(0,0);
	t38 = Bnl[0][9]*sig(0,1);
	t39 = Bnl[0][9]*sig(0,2);
	t40 = Bnl[0][9]*sig(0,0);
	t41 = Bnl[1][10]*sig(1,1);
	t42 = Bnl[1][10]*sig(1,2);
	t43 = Bnl[1][10]*sig(0,1);
	t44 = Bnl[1][10]*sig(1,1);
	t45 = Bnl[2][11]*sig(2,2);
	t46 = Bnl[2][11]*sig(0,2);
	t47 = Bnl[2][11]*sig(1,2);
	t48 = Bnl[2][11]*sig(2,2);
	t49 = Bnl[0][12]*sig(0,0);
	t50 = Bnl[0][12]*sig(0,1);
	t51 = Bnl[0][12]*sig(0,2);
	t52 = Bnl[0][12]*sig(0,0);
	t53 = Bnl[1][13]*sig(1,1);
	t54 = Bnl[1][13]*sig(1,2);
	t55 = Bnl[1][13]*sig(0,1);
	t56 = Bnl[1][13]*sig(1,1);
	t57 = Bnl[2][14]*sig(2,2);
	t58 = Bnl[2][14]*sig(0,2);
	t59 = Bnl[2][14]*sig(1,2);
	t60 = Bnl[2][14]*sig(2,2);
	t61 = Bnl[0][15]*sig(0,0);
	t62 = Bnl[0][15]*sig(0,1);
	t63 = Bnl[0][15]*sig(0,2);
	t64 = Bnl[0][15]*sig(0,0);
	t65 = Bnl[1][16]*sig(1,1);
	t66 = Bnl[1][16]*sig(1,2);
	t67 = Bnl[1][16]*sig(0,1);
	t68 = Bnl[1][16]*sig(1,1);
	t69 = Bnl[2][17]*sig(2,2);
	t70 = Bnl[2][17]*sig(0,2);
	t71 = Bnl[2][17]*sig(1,2);
	t72 = Bnl[2][17]*sig(2,2);
	t73 = Bnl[0][18]*sig(0,0);
	t74 = Bnl[0][18]*sig(0,1);
	t75 = Bnl[0][18]*sig(0,2);
	t77 = Bnl[1][19]*sig(1,1);
	t78 = Bnl[1][19]*sig(1,2);
	t79 = Bnl[1][19]*sig(0,1);
	t81 = Bnl[2][20]*sig(2,2);
	t82 = Bnl[2][20]*sig(0,2);
	t83 = Bnl[2][20]*sig(1,2);
	t85 = Bnl[0][21]*sig(0,0);
	t86 = Bnl[0][21]*sig(0,1);
	t87 = Bnl[0][21]*sig(0,2);
	t88 = Bnl[1][22]*sig(1,1);
	t89 = Bnl[1][22]*sig(1,2);
	t90 = Bnl[2][23]*sig(2,2);

	Kgeo[0][0]   = t1*Bnl[0][0];
	Kgeo[0][1]   = t2*Bnl[1][1];
	Kgeo[0][2]   = t3*Bnl[2][2];
	Kgeo[0][3]   = t4*Bnl[0][3];
	Kgeo[0][4]   = t2*Bnl[1][4];
	Kgeo[0][5]   = t3*Bnl[2][5];
	Kgeo[0][6]   = t4*Bnl[0][6];
	Kgeo[0][7]   = t2*Bnl[1][7];
	Kgeo[0][8]   = t3*Bnl[2][8];
	Kgeo[0][9]   = t4*Bnl[0][9];
	Kgeo[0][10]  = t2*Bnl[1][10];
	Kgeo[0][11]  = t3*Bnl[2][11];
	Kgeo[0][12]  = t4*Bnl[0][12];
	Kgeo[0][13]  = t2*Bnl[1][13];
	Kgeo[0][14]  = t3*Bnl[2][14];
	Kgeo[0][15]  = t4*Bnl[0][15];
	Kgeo[0][16]  = t2*Bnl[1][16];
	Kgeo[0][17]  = t3*Bnl[2][17];
	Kgeo[0][18]  = t4*Bnl[0][18];
	Kgeo[0][19]  = t2*Bnl[1][19];
	Kgeo[0][20]  = t3*Bnl[2][20];
	Kgeo[0][21]  = t4*Bnl[0][21];
	Kgeo[0][22]  = t2*Bnl[1][22];
	Kgeo[0][23]  = t3*Bnl[2][23];
	Kgeo[1][0]   = Kgeo[0][1];
	Kgeo[1][1]   = t5*Bnl[1][1];
	Kgeo[1][2]   = t6*Bnl[2][2];
	Kgeo[1][3]   = t7*Bnl[0][3];
	Kgeo[1][4]   = Bnl[1][4]*t8;
	Kgeo[1][5]   = t6*Bnl[2][5];
	Kgeo[1][6]   = t7*Bnl[0][6];
	Kgeo[1][7]   = Bnl[1][7]*t8;
	Kgeo[1][8]   = t6*Bnl[2][8];
	Kgeo[1][9]   = t7*Bnl[0][9];
	Kgeo[1][10]  = Bnl[1][10]*t8;
	Kgeo[1][11]  = t6*Bnl[2][11];
	Kgeo[1][12]  = t7*Bnl[0][12];
	Kgeo[1][13]  = Bnl[1][13]*t8;
	Kgeo[1][14]  = t6*Bnl[2][14];
	Kgeo[1][15]  = t7*Bnl[0][15];
	Kgeo[1][16]  = Bnl[1][16]*t8;
	Kgeo[1][17]  = t6*Bnl[2][17];
	Kgeo[1][18]  = t7*Bnl[0][18];
	Kgeo[1][19]  = Bnl[1][19]*t8;
	Kgeo[1][20]  = t6*Bnl[2][20];
	Kgeo[1][21]  = t7*Bnl[0][21];
	Kgeo[1][22]  = Bnl[1][22]*t8;
	Kgeo[1][23]  = t6*Bnl[2][23];
	Kgeo[2][0]   = Kgeo[0][2];
	Kgeo[2][1]   = Kgeo[1][2];
	Kgeo[2][2]   = t9*Bnl[2][2];
	Kgeo[2][3]   = t10*Bnl[0][3];
	Kgeo[2][4]   = Bnl[1][4]*t11;
	Kgeo[2][5]   = t12*Bnl[2][5];
	Kgeo[2][6]   = t10*Bnl[0][6];
	Kgeo[2][7]   = Bnl[1][7]*t11;
	Kgeo[2][8]   = t12*Bnl[2][8];
	Kgeo[2][9]   = t10*Bnl[0][9];
	Kgeo[2][10]  = Bnl[1][10]*t11;
	Kgeo[2][11]  = t12*Bnl[2][11];
	Kgeo[2][12]  = t10*Bnl[0][12];
	Kgeo[2][13]  = Bnl[1][13]*t11;
	Kgeo[2][14]  = t12*Bnl[2][14];
	Kgeo[2][15]  = t10*Bnl[0][15];
	Kgeo[2][16]  = Bnl[1][16]*t11;
	Kgeo[2][17]  = t12*Bnl[2][17];
	Kgeo[2][18]  = t10*Bnl[0][18];
	Kgeo[2][19]  = t11*Bnl[1][19];
	Kgeo[2][20]  = t12*Bnl[2][20];
	Kgeo[2][21]  = t10*Bnl[0][21];
	Kgeo[2][22]  = t11*Bnl[1][22];
	Kgeo[2][23]  = t12*Bnl[2][23];
	Kgeo[3][0]   = Kgeo[0][3];
	Kgeo[3][1]   = Kgeo[1][3];
	Kgeo[3][2]   = Kgeo[2][3];
	Kgeo[3][3]   = t13*Bnl[0][3];
	Kgeo[3][4]   = t14*Bnl[1][4];
	Kgeo[3][5]   = Bnl[2][5]*t15;
	Kgeo[3][6]   = t16*Bnl[0][6];
	Kgeo[3][7]   = t14*Bnl[1][7];
	Kgeo[3][8]   = Bnl[2][8]*t15;
	Kgeo[3][9]   = t16*Bnl[0][9];
	Kgeo[3][10]  = t14*Bnl[1][10];
	Kgeo[3][11]  = Bnl[2][11]*t15;
	Kgeo[3][12]  = t16*Bnl[0][12];
	Kgeo[3][13]  = t14*Bnl[1][13];
	Kgeo[3][14]  = Bnl[2][14]*t15;
	Kgeo[3][15]  = t16*Bnl[0][15];
	Kgeo[3][16]  = t14*Bnl[1][16];
	Kgeo[3][17]  = Bnl[2][17]*t15;
	Kgeo[3][18]  = t16*Bnl[0][18];
	Kgeo[3][19]  = t14*Bnl[1][19];
	Kgeo[3][20]  = Bnl[2][20]*t15;
	Kgeo[3][21]  = t16*Bnl[0][21];
	Kgeo[3][22]  = t14*Bnl[1][22];
	Kgeo[3][23]  = Bnl[2][23]*t15;
	Kgeo[4][0]   = Kgeo[0][4];
	Kgeo[4][1]   = Kgeo[1][4];
	Kgeo[4][2]   = Kgeo[2][4];
	Kgeo[4][3]   = Kgeo[3][4];
	Kgeo[4][4]   = t17*Bnl[1][4];
	Kgeo[4][5]   = t18*Bnl[2][5];
	Kgeo[4][6]   = t19*Bnl[0][6];
	Kgeo[4][7]   = t20*Bnl[1][7];
	Kgeo[4][8]   = t18*Bnl[2][8];
	Kgeo[4][9]   = t19*Bnl[0][9];
	Kgeo[4][10]  = t20*Bnl[1][10];
	Kgeo[4][11]  = t18*Bnl[2][11];
	Kgeo[4][12]  = t19*Bnl[0][12];
	Kgeo[4][13]  = t20*Bnl[1][13];
	Kgeo[4][14]  = t18*Bnl[2][14];
	Kgeo[4][15]  = t19*Bnl[0][15];
	Kgeo[4][16]  = t20*Bnl[1][16];
	Kgeo[4][17]  = t18*Bnl[2][17];
	Kgeo[4][18]  = t19*Bnl[0][18];
	Kgeo[4][19]  = t20*Bnl[1][19];
	Kgeo[4][20]  = t18*Bnl[2][20];
	Kgeo[4][21]  = t19*Bnl[0][21];
	Kgeo[4][22]  = t20*Bnl[1][22];
	Kgeo[4][23]  = t18*Bnl[2][23];
	Kgeo[5][0]   = Kgeo[0][5];
	Kgeo[5][1]   = Kgeo[1][5];
	Kgeo[5][2]   = Kgeo[2][5];
	Kgeo[5][3]   = Kgeo[3][5];
	Kgeo[5][4]   = Kgeo[4][5];
	Kgeo[5][5]   = t21*Bnl[2][5];
	Kgeo[5][6]   = t22*Bnl[0][6];
	Kgeo[5][7]   = t23*Bnl[1][7];
	Kgeo[5][8]   = t24*Bnl[2][8];
	Kgeo[5][9]   = t22*Bnl[0][9];
	Kgeo[5][10]  = t23*Bnl[1][10];
	Kgeo[5][11]  = t24*Bnl[2][11];
	Kgeo[5][12]  = t22*Bnl[0][12];
	Kgeo[5][13]  = t23*Bnl[1][13];
	Kgeo[5][14]  = t24*Bnl[2][14];
	Kgeo[5][15]  = t22*Bnl[0][15];
	Kgeo[5][16]  = t23*Bnl[1][16];
	Kgeo[5][17]  = t24*Bnl[2][17];
	Kgeo[5][18]  = t22*Bnl[0][18];
	Kgeo[5][19]  = t23*Bnl[1][19];
	Kgeo[5][20]  = t24*Bnl[2][20];
	Kgeo[5][21]  = t22*Bnl[0][21];
	Kgeo[5][22]  = t23*Bnl[1][22];
	Kgeo[5][23]  = t24*Bnl[2][23];
	Kgeo[6][0]   = Kgeo[0][6];
	Kgeo[6][1]   = Kgeo[1][6];
	Kgeo[6][2]   = Kgeo[2][6];
	Kgeo[6][3]   = Kgeo[3][6];
	Kgeo[6][4]   = Kgeo[4][6];
	Kgeo[6][5]   = Kgeo[5][6];
	Kgeo[6][6]   = t25*Bnl[0][6];
	Kgeo[6][7]   = t26*Bnl[1][7];
	Kgeo[6][8]   = t27*Bnl[2][8];
	Kgeo[6][9]   = t28*Bnl[0][9];
	Kgeo[6][10]  = t26*Bnl[1][10];
	Kgeo[6][11]  = t27*Bnl[2][11];
	Kgeo[6][12]  = t28*Bnl[0][12];
	Kgeo[6][13]  = t26*Bnl[1][13];
	Kgeo[6][14]  = t27*Bnl[2][14];
	Kgeo[6][15]  = t28*Bnl[0][15];
	Kgeo[6][16]  = t26*Bnl[1][16];
	Kgeo[6][17]  = t27*Bnl[2][17];
	Kgeo[6][18]  = t28*Bnl[0][18];
	Kgeo[6][19]  = t26*Bnl[1][19];
	Kgeo[6][20]  = t27*Bnl[2][20];
	Kgeo[6][21]  = t28*Bnl[0][21];
	Kgeo[6][22]  = t26*Bnl[1][22];
	Kgeo[6][23]  = t27*Bnl[2][23];
	Kgeo[7][0]   = Kgeo[0][7];
	Kgeo[7][1]   = Kgeo[1][7];
	Kgeo[7][2]   = Kgeo[2][7];
	Kgeo[7][3]   = Kgeo[3][7];
	Kgeo[7][4]   = Kgeo[4][7];
	Kgeo[7][5]   = Kgeo[5][7];
	Kgeo[7][6]   = Kgeo[6][7];
	Kgeo[7][7]   = t29*Bnl[1][7];
	Kgeo[7][8]   = t30*Bnl[2][8];
	Kgeo[7][9]   = t31*Bnl[0][9];
	Kgeo[7][10]  = t32*Bnl[1][10];
	Kgeo[7][11]  = t30*Bnl[2][11];
	Kgeo[7][12]  = t31*Bnl[0][12];
	Kgeo[7][13]  = t32*Bnl[1][13];
	Kgeo[7][14]  = t30*Bnl[2][14];
	Kgeo[7][15]  = t31*Bnl[0][15];
	Kgeo[7][16]  = t32*Bnl[1][16];
	Kgeo[7][17]  = t30*Bnl[2][17];
	Kgeo[7][18]  = t31*Bnl[0][18];
	Kgeo[7][19]  = t32*Bnl[1][19];
	Kgeo[7][20]  = t30*Bnl[2][20];
	Kgeo[7][21]  = t31*Bnl[0][21];
	Kgeo[7][22]  = t32*Bnl[1][22];
	Kgeo[7][23]  = t30*Bnl[2][23];
	Kgeo[8][0]   = Kgeo[0][8];
	Kgeo[8][1]   = Kgeo[1][8];
	Kgeo[8][2]   = Kgeo[2][8];
	Kgeo[8][3]   = Kgeo[3][8];
	Kgeo[8][4]   = Kgeo[4][8];
	Kgeo[8][5]   = Kgeo[5][8];
	Kgeo[8][6]   = Kgeo[6][8];
	Kgeo[8][7]   = Kgeo[7][8];
	Kgeo[8][8]   = t33*Bnl[2][8];
	Kgeo[8][9]   = t34*Bnl[0][9];
	Kgeo[8][10]  = t35*Bnl[1][10];
	Kgeo[8][11]  = t36*Bnl[2][11];
	Kgeo[8][12]  = t34*Bnl[0][12];
	Kgeo[8][13]  = t35*Bnl[1][13];
	Kgeo[8][14]  = t36*Bnl[2][14];
	Kgeo[8][15]  = t34*Bnl[0][15];
	Kgeo[8][16]  = t35*Bnl[1][16];
	Kgeo[8][17]  = t36*Bnl[2][17];
	Kgeo[8][18]  = t34*Bnl[0][18];
	Kgeo[8][19]  = t35*Bnl[1][19];
	Kgeo[8][20]  = t36*Bnl[2][20];
	Kgeo[8][21]  = t34*Bnl[0][21];
	Kgeo[8][22]  = t35*Bnl[1][22];
	Kgeo[8][23]  = t36*Bnl[2][23];
	Kgeo[9][0]   = Kgeo[0][9];
	Kgeo[9][1]   = Kgeo[1][9];
	Kgeo[9][2]   = Kgeo[2][9];
	Kgeo[9][3]   = Kgeo[3][9];
	Kgeo[9][4]   = Kgeo[4][9];
	Kgeo[9][5]   = Kgeo[5][9];
	Kgeo[9][6]   = Kgeo[6][9];
	Kgeo[9][7]   = Kgeo[7][9];
	Kgeo[9][8]   = Kgeo[8][9];
	Kgeo[9][9]   = t37*Bnl[0][9];
	Kgeo[9][10]  = t38*Bnl[1][10];
	Kgeo[9][11]  = t39*Bnl[2][11];
	Kgeo[9][12]  = t40*Bnl[0][12];
	Kgeo[9][13]  = t38*Bnl[1][13];
	Kgeo[9][14]  = t39*Bnl[2][14];
	Kgeo[9][15]  = t40*Bnl[0][15];
	Kgeo[9][16]  = t38*Bnl[1][16];
	Kgeo[9][17]  = t39*Bnl[2][17];
	Kgeo[9][18]  = t40*Bnl[0][18];
	Kgeo[9][19]  = t38*Bnl[1][19];
	Kgeo[9][20]  = t39*Bnl[2][20];
	Kgeo[9][21]  = t40*Bnl[0][21];
	Kgeo[9][22]  = t38*Bnl[1][22];
	Kgeo[9][23]  = t39*Bnl[2][23];
	Kgeo[10][0]  = Kgeo[0][10];
	Kgeo[10][1]  = Kgeo[1][10];
	Kgeo[10][2]  = Kgeo[2][10];
	Kgeo[10][3]  = Kgeo[3][10];
	Kgeo[10][4]  = Kgeo[4][10];
	Kgeo[10][5]  = Kgeo[5][10];
	Kgeo[10][6]  = Kgeo[6][10];
	Kgeo[10][7]  = Kgeo[7][10];
	Kgeo[10][8]  = Kgeo[8][10];
	Kgeo[10][9]  = Kgeo[9][10];
	Kgeo[10][10] = t41*Bnl[1][10];
	Kgeo[10][11] = t42*Bnl[2][11];
	Kgeo[10][12] = t43*Bnl[0][12];
	Kgeo[10][13] = t44*Bnl[1][13];
	Kgeo[10][14] = t42*Bnl[2][14];
	Kgeo[10][15] = t43*Bnl[0][15];
	Kgeo[10][16] = t44*Bnl[1][16];
	Kgeo[10][17] = t42*Bnl[2][17];
	Kgeo[10][18] = t43*Bnl[0][18];
	Kgeo[10][19] = t44*Bnl[1][19];
	Kgeo[10][20] = t42*Bnl[2][20];
	Kgeo[10][21] = t43*Bnl[0][21];
	Kgeo[10][22] = t44*Bnl[1][22];
	Kgeo[10][23] = t42*Bnl[2][23];
	Kgeo[11][0]  = Kgeo[0][11];
	Kgeo[11][1]  = Kgeo[1][11];
	Kgeo[11][2]  = Kgeo[2][11];
	Kgeo[11][3]  = Kgeo[3][11];
	Kgeo[11][4]  = Kgeo[4][11];
	Kgeo[11][5]  = Kgeo[5][11];
	Kgeo[11][6]  = Kgeo[6][11];
	Kgeo[11][7]  = Kgeo[7][11];
	Kgeo[11][8]  = Kgeo[8][11];
	Kgeo[11][9]  = Kgeo[9][11];
	Kgeo[11][10] = Kgeo[10][11];
	Kgeo[11][11] = t45*Bnl[2][11];
	Kgeo[11][12] = t46*Bnl[0][12];
	Kgeo[11][13] = t47*Bnl[1][13];
	Kgeo[11][14] = t48*Bnl[2][14];
	Kgeo[11][15] = t46*Bnl[0][15];
	Kgeo[11][16] = t47*Bnl[1][16];
	Kgeo[11][17] = t48*Bnl[2][17];
	Kgeo[11][18] = t46*Bnl[0][18];
	Kgeo[11][19] = t47*Bnl[1][19];
	Kgeo[11][20] = t48*Bnl[2][20];
	Kgeo[11][21] = t46*Bnl[0][21];
	Kgeo[11][22] = t47*Bnl[1][22];
	Kgeo[11][23] = t48*Bnl[2][23];
	Kgeo[12][0]  = Kgeo[0][12];
	Kgeo[12][1]  = Kgeo[1][12];
	Kgeo[12][2]  = Kgeo[2][12];
	Kgeo[12][3]  = Kgeo[3][12];
	Kgeo[12][4]  = Kgeo[4][12];
	Kgeo[12][5]  = Kgeo[5][12];
	Kgeo[12][6]  = Kgeo[6][12];
	Kgeo[12][7]  = Kgeo[7][12];
	Kgeo[12][8]  = Kgeo[8][12];
	Kgeo[12][9]  = Kgeo[9][12];
	Kgeo[12][10] = Kgeo[10][12];
	Kgeo[12][11] = Kgeo[11][12];
	Kgeo[12][12] = t49*Bnl[0][12];
	Kgeo[12][13] = t50*Bnl[1][13];
	Kgeo[12][14] = t51*Bnl[2][14];
	Kgeo[12][15] = t52*Bnl[0][15];
	Kgeo[12][16] = t50*Bnl[1][16];
	Kgeo[12][17] = t51*Bnl[2][17];
	Kgeo[12][18] = t52*Bnl[0][18];
	Kgeo[12][19] = t50*Bnl[1][19];
	Kgeo[12][20] = t51*Bnl[2][20];
	Kgeo[12][21] = t52*Bnl[0][21];
	Kgeo[12][22] = t50*Bnl[1][22];
	Kgeo[12][23] = t51*Bnl[2][23];
	Kgeo[13][0]  = Kgeo[0][13];
	Kgeo[13][1]  = Kgeo[1][13];
	Kgeo[13][2]  = Kgeo[2][13];
	Kgeo[13][3]  = Kgeo[3][13];
	Kgeo[13][4]  = Kgeo[4][13];
	Kgeo[13][5]  = Kgeo[5][13];
	Kgeo[13][6]  = Kgeo[6][13];
	Kgeo[13][7]  = Kgeo[7][13];
	Kgeo[13][8]  = Kgeo[8][13];
	Kgeo[13][9]  = Kgeo[9][13];
	Kgeo[13][10] = Kgeo[10][13];
	Kgeo[13][11] = Kgeo[11][13];
	Kgeo[13][12] = Kgeo[12][13];
	Kgeo[13][13] = t53*Bnl[1][13];
	Kgeo[13][14] = t54*Bnl[2][14];
	Kgeo[13][15] = t55*Bnl[0][15];
	Kgeo[13][16] = t56*Bnl[1][16];
	Kgeo[13][17] = t54*Bnl[2][17];
	Kgeo[13][18] = t55*Bnl[0][18];
	Kgeo[13][19] = t56*Bnl[1][19];
	Kgeo[13][20] = t54*Bnl[2][20];
	Kgeo[13][21] = t55*Bnl[0][21];
	Kgeo[13][22] = t56*Bnl[1][22];
	Kgeo[13][23] = t54*Bnl[2][23];
	Kgeo[14][0]  = Kgeo[0][14];
	Kgeo[14][1]  = Kgeo[1][14];
	Kgeo[14][2]  = Kgeo[2][14];
	Kgeo[14][3]  = Kgeo[3][14];
	Kgeo[14][4]  = Kgeo[4][14];
	Kgeo[14][5]  = Kgeo[5][14];
	Kgeo[14][6]  = Kgeo[6][14];
	Kgeo[14][7]  = Kgeo[7][14];
	Kgeo[14][8]  = Kgeo[8][14];
	Kgeo[14][9]  = Kgeo[9][14];
	Kgeo[14][10] = Kgeo[10][14];
	Kgeo[14][11] = Kgeo[11][14];
	Kgeo[14][12] = Kgeo[12][14];
	Kgeo[14][13] = Kgeo[13][14];
	Kgeo[14][14] = t57*Bnl[2][14];
	Kgeo[14][15] = t58*Bnl[0][15];
	Kgeo[14][16] = t59*Bnl[1][16];
	Kgeo[14][17] = t60*Bnl[2][17];
	Kgeo[14][18] = t58*Bnl[0][18];
	Kgeo[14][19] = t59*Bnl[1][19];
	Kgeo[14][20] = t60*Bnl[2][20];
	Kgeo[14][21] = t58*Bnl[0][21];
	Kgeo[14][22] = t59*Bnl[1][22];
	Kgeo[14][23] = t60*Bnl[2][23];
	Kgeo[15][0]  = Kgeo[0][15];
	Kgeo[15][1]  = Kgeo[1][15];
	Kgeo[15][2]  = Kgeo[2][15];
	Kgeo[15][3]  = Kgeo[3][15];
	Kgeo[15][4]  = Kgeo[4][15];
	Kgeo[15][5]  = Kgeo[5][15];
	Kgeo[15][6]  = Kgeo[6][15];
	Kgeo[15][7]  = Kgeo[7][15];
	Kgeo[15][8]  = Kgeo[8][15];
	Kgeo[15][9]  = Kgeo[9][15];
	Kgeo[15][10] = Kgeo[10][15];
	Kgeo[15][11] = Kgeo[11][15];
	Kgeo[15][12] = Kgeo[12][15];
	Kgeo[15][13] = Kgeo[13][15];
	Kgeo[15][14] = Kgeo[14][15];
	Kgeo[15][15] = t61*Bnl[0][15];
	Kgeo[15][16] = t62*Bnl[1][16];
	Kgeo[15][17] = t63*Bnl[2][17];
	Kgeo[15][18] = t64*Bnl[0][18];
	Kgeo[15][19] = t62*Bnl[1][19];
	Kgeo[15][20] = t63*Bnl[2][20];
	Kgeo[15][21] = t64*Bnl[0][21];
	Kgeo[15][22] = t62*Bnl[1][22];
	Kgeo[15][23] = t63*Bnl[2][23];
	Kgeo[16][0]  = Kgeo[0][16];
	Kgeo[16][1]  = Kgeo[1][16];
	Kgeo[16][2]  = Kgeo[2][16];
	Kgeo[16][3]  = Kgeo[3][16];
	Kgeo[16][4]  = Kgeo[4][16];
	Kgeo[16][5]  = Kgeo[5][16];
	Kgeo[16][6]  = Kgeo[6][16];
	Kgeo[16][7]  = Kgeo[7][16];
	Kgeo[16][8]  = Kgeo[8][16];
	Kgeo[16][9]  = Kgeo[9][16];
	Kgeo[16][10] = Kgeo[10][16];
	Kgeo[16][11] = Kgeo[11][16];
	Kgeo[16][12] = Kgeo[12][16];
	Kgeo[16][13] = Kgeo[13][16];
	Kgeo[16][14] = Kgeo[14][16];
	Kgeo[16][15] = Kgeo[15][16];
	Kgeo[16][16] = t65*Bnl[1][16];
	Kgeo[16][17] = t66*Bnl[2][17];
	Kgeo[16][18] = t67*Bnl[0][18];
	Kgeo[16][19] = t68*Bnl[1][19];
	Kgeo[16][20] = t66*Bnl[2][20];
	Kgeo[16][21] = t67*Bnl[0][21];
	Kgeo[16][22] = t68*Bnl[1][22];
	Kgeo[16][23] = t66*Bnl[2][23];
	Kgeo[17][0]  = Kgeo[0][17];
	Kgeo[17][1]  = Kgeo[1][17];
	Kgeo[17][2]  = Kgeo[2][17];
	Kgeo[17][3]  = Kgeo[3][17];
	Kgeo[17][4]  = Kgeo[4][17];
	Kgeo[17][5]  = Kgeo[5][17];
	Kgeo[17][6]  = Kgeo[6][17];
	Kgeo[17][7]  = Kgeo[7][17];
	Kgeo[17][8]  = Kgeo[8][17];
	Kgeo[17][9]  = Kgeo[9][17];
	Kgeo[17][10] = Kgeo[10][17];
	Kgeo[17][11] = Kgeo[11][17];
	Kgeo[17][12] = Kgeo[12][17];
	Kgeo[17][13] = Kgeo[13][17];
	Kgeo[17][14] = Kgeo[14][17];
	Kgeo[17][15] = Kgeo[15][17];
	Kgeo[17][16] = Kgeo[16][17];
	Kgeo[17][17] = t69*Bnl[2][17];
	Kgeo[17][18] = t70*Bnl[0][18];
	Kgeo[17][19] = t71*Bnl[1][19];
	Kgeo[17][20] = t72*Bnl[2][20];
	Kgeo[17][21] = t70*Bnl[0][21];
	Kgeo[17][22] = t71*Bnl[1][22];
	Kgeo[17][23] = t72*Bnl[2][23];
	Kgeo[18][0]  = Kgeo[0][18];
	Kgeo[18][1]  = Kgeo[1][18];
	Kgeo[18][2]  = Kgeo[2][18];
	Kgeo[18][3]  = Kgeo[3][18];
	Kgeo[18][4]  = Kgeo[4][18];
	Kgeo[18][5]  = Kgeo[5][18];
	Kgeo[18][6]  = Kgeo[6][18];
	Kgeo[18][7]  = Kgeo[7][18];
	Kgeo[18][8]  = Kgeo[8][18];
	Kgeo[18][9]  = Kgeo[9][18];
	Kgeo[18][10] = Kgeo[10][18];
	Kgeo[18][11] = Kgeo[11][18];
	Kgeo[18][12] = Kgeo[12][18];
	Kgeo[18][13] = Kgeo[13][18];
	Kgeo[18][14] = Kgeo[14][18];
	Kgeo[18][15] = Kgeo[15][18];
	Kgeo[18][16] = Kgeo[16][18];
	Kgeo[18][17] = Kgeo[17][18];
	Kgeo[18][18] = t73*Bnl[0][18];
	Kgeo[18][19] = t74*Bnl[1][19];
	Kgeo[18][20] = t75*Bnl[2][20];
	Kgeo[18][21] = t73*Bnl[0][21];
	Kgeo[18][22] = t74*Bnl[1][22];
	Kgeo[18][23] = t75*Bnl[2][23];
	Kgeo[19][0]  = Kgeo[0][19];
	Kgeo[19][1]  = Kgeo[1][19];
	Kgeo[19][2]  = Kgeo[2][19];
	Kgeo[19][3]  = Kgeo[3][19];
	Kgeo[19][4]  = Kgeo[4][19];
	Kgeo[19][5]  = Kgeo[5][19];
	Kgeo[19][6]  = Kgeo[6][19];
	Kgeo[19][7]  = Kgeo[7][19];
	Kgeo[19][8]  = Kgeo[8][19];
	Kgeo[19][9]  = Kgeo[9][19];
	Kgeo[19][10] = Kgeo[10][19];
	Kgeo[19][11] = Kgeo[11][19];
	Kgeo[19][12] = Kgeo[12][19];
	Kgeo[19][13] = Kgeo[13][19];
	Kgeo[19][14] = Kgeo[14][19];
	Kgeo[19][15] = Kgeo[15][19];
	Kgeo[19][16] = Kgeo[16][19];
	Kgeo[19][17] = Kgeo[17][19];
	Kgeo[19][18] = Kgeo[18][19];
	Kgeo[19][19] = t77*Bnl[1][19];
	Kgeo[19][20] = t78*Bnl[2][20];
	Kgeo[19][21] = t79*Bnl[0][21];
	Kgeo[19][22] = t77*Bnl[1][22];
	Kgeo[19][23] = t78*Bnl[2][23];
	Kgeo[20][0]  = Kgeo[0][20];
	Kgeo[20][1]  = Kgeo[1][20];
	Kgeo[20][2]  = Kgeo[2][20];
	Kgeo[20][3]  = Kgeo[3][20];
	Kgeo[20][4]  = Kgeo[4][20];
	Kgeo[20][5]  = Kgeo[5][20];
	Kgeo[20][6]  = Kgeo[6][20];
	Kgeo[20][7]  = Kgeo[7][20];
	Kgeo[20][8]  = Kgeo[8][20];
	Kgeo[20][9]  = Kgeo[9][20];
	Kgeo[20][10] = Kgeo[10][20];
	Kgeo[20][11] = Kgeo[11][20];
	Kgeo[20][12] = Kgeo[12][20];
	Kgeo[20][13] = Kgeo[13][20];
	Kgeo[20][14] = Kgeo[14][20];
	Kgeo[20][15] = Kgeo[15][20];
	Kgeo[20][16] = Kgeo[16][20];
	Kgeo[20][17] = Kgeo[17][20];
	Kgeo[20][18] = Kgeo[18][20];
	Kgeo[20][19] = Kgeo[19][20];
	Kgeo[20][20] = t81*Bnl[2][20];
	Kgeo[20][21] = t82*Bnl[0][21];
	Kgeo[20][22] = t83*Bnl[1][22];
	Kgeo[20][23] = t81*Bnl[2][23];
	Kgeo[21][0]  = Kgeo[0][21];
	Kgeo[21][1]  = Kgeo[1][21];
	Kgeo[21][2]  = Kgeo[2][21];
	Kgeo[21][3]  = Kgeo[3][21];
	Kgeo[21][4]  = Kgeo[4][21];
	Kgeo[21][5]  = Kgeo[5][21];
	Kgeo[21][6]  = Kgeo[6][21];
	Kgeo[21][7]  = Kgeo[7][21];
	Kgeo[21][8]  = Kgeo[8][21];
	Kgeo[21][9]  = Kgeo[9][21];
	Kgeo[21][10] = Kgeo[10][21];
	Kgeo[21][11] = Kgeo[11][21];
	Kgeo[21][12] = Kgeo[12][21];
	Kgeo[21][13] = Kgeo[13][21];
	Kgeo[21][14] = Kgeo[14][21];
	Kgeo[21][15] = Kgeo[15][21];
	Kgeo[21][16] = Kgeo[16][21];
	Kgeo[21][17] = Kgeo[17][21];
	Kgeo[21][18] = Kgeo[18][21];
	Kgeo[21][19] = Kgeo[19][21];
	Kgeo[21][20] = Kgeo[20][21];
	Kgeo[21][21] = t85*Bnl[0][21];
	Kgeo[21][22] = t86*Bnl[1][22];
	Kgeo[21][23] = t87*Bnl[2][23];
	Kgeo[22][0]  = Kgeo[0][22];
	Kgeo[22][1]  = Kgeo[1][22];
	Kgeo[22][2]  = Kgeo[2][22];
	Kgeo[22][3]  = Kgeo[3][22];
	Kgeo[22][4]  = Kgeo[4][22];
	Kgeo[22][5]  = Kgeo[5][22];
	Kgeo[22][6]  = Kgeo[6][22];
	Kgeo[22][7]  = Kgeo[7][22];
	Kgeo[22][8]  = Kgeo[8][22];
	Kgeo[22][9]  = Kgeo[9][22];
	Kgeo[22][10] = Kgeo[10][22];
	Kgeo[22][11] = Kgeo[11][22];
	Kgeo[22][12] = Kgeo[12][22];
	Kgeo[22][13] = Kgeo[13][22];
	Kgeo[22][14] = Kgeo[14][22];
	Kgeo[22][15] = Kgeo[15][22];
	Kgeo[22][16] = Kgeo[16][22];
	Kgeo[22][17] = Kgeo[17][22];
	Kgeo[22][18] = Kgeo[18][22];
	Kgeo[22][19] = Kgeo[19][22];
	Kgeo[22][20] = Kgeo[20][22];
	Kgeo[22][21] = Kgeo[21][22];
	Kgeo[22][22] = t88*Bnl[1][22];
	Kgeo[22][23] = t89*Bnl[2][23];
	Kgeo[23][0]  = Kgeo[0][23];
	Kgeo[23][1]  = Kgeo[1][23];
	Kgeo[23][2]  = Kgeo[2][23];
	Kgeo[23][3]  = Kgeo[3][23];
	Kgeo[23][4]  = Kgeo[4][23];
	Kgeo[23][5]  = Kgeo[5][23];
	Kgeo[23][6]  = Kgeo[6][23];
	Kgeo[23][7]  = Kgeo[7][23];
	Kgeo[23][8]  = Kgeo[8][23];
	Kgeo[23][9]  = Kgeo[9][23];
	Kgeo[23][10] = Kgeo[10][23];
	Kgeo[23][11] = Kgeo[11][23];
	Kgeo[23][12] = Kgeo[12][23];
	Kgeo[23][13] = Kgeo[13][23];
	Kgeo[23][14] = Kgeo[14][23];
	Kgeo[23][15] = Kgeo[15][23];
	Kgeo[23][16] = Kgeo[16][23];
	Kgeo[23][17] = Kgeo[17][23];
	Kgeo[23][18] = Kgeo[18][23];
	Kgeo[23][19] = Kgeo[19][23];
	Kgeo[23][20] = Kgeo[20][23];
	Kgeo[23][21] = Kgeo[21][23];
	Kgeo[23][22] = Kgeo[22][23];
	Kgeo[23][23] = t90*Bnl[2][23];
}



namespace Uintah
{
/*
  static MPI_Datatype makeMPI_CMData()
  {
  ASSERTEQ(sizeof(TongeRamesh::double), sizeof(double)*0);
  MPI_Datatype mpitype;
  Uintah::MPI::Type_vector(1, 1, 1, MPI_DOUBLE, &mpitype);
  Uintah::MPI::Type_commit(&mpitype);
  return mpitype;
  }

  const TypeDescription* fun_getTypeDescription(TongeRamesh::double*)
  {
  static TypeDescription* td = 0;
  if(!td){
  td = scinew TypeDescription(TypeDescription::Other,
  "TongeRamesh::double",
  true, &makeMPI_CMData);
  }
  return td;
  }
*/
} // End namespace Uintah
