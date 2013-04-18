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

#include <CCA/Components/MPM/ConstitutiveModel/Diamm.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Ports/DataWarehouse.h>

#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Math/Matrix3.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <Core/Containers/StaticArray.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MinMax.h>

#include <sci_defs/uintah_defs.h>

#include <fstream>
#include <iostream>
#include <string>

////////////////////////////////////////////////////////////////////////////////
// The following functions are found in fortran/*.F
//SUBROUTINE YENTA_CALC( NBLK, NINSV, DT, PROP,
  //   $                                   SIGARG, D, SVARG, USM   )

extern "C"{

#if defined( FORTRAN_UNDERSCORE_END )
#  define DMMCHK dmmchk_
#  define DIAMM_CALC diamm_calc_
#  define DMMRXV dmmrxv_
#elif defined( FORTRAN_UNDERSCORE_LINUX )
#  define DMMCHK dmmchk_
#  define DMMRXV dmmrxv_
#  define DIAMM_CALC diamm_calc__
#else // NONE
#  define DMMCHK dmmchk
#  define DIAMM_CALC diamm_calc
#  define DMMRXV dmmrxv
#endif

//#define DMM_ANISOTROPIC
//#undef DMM_ANISOTROPIC

   void DMMCHK( double UI[], double UJ[], double UK[] );
   void DIAMM_CALC( int &nblk, int &ninsv, double &dt,
                                    double UI[], double stress[], double D[],
                                    double svarg[], double &USM );
   void DMMRXV( double UI[], double UJ[], double UK[], int &nx, char* namea[],
                char* keya[], double rinit[], double rdim[], int iadvct[],
                int itype[] );
}

// End fortran functions.
////////////////////////////////////////////////////////////////////////////////
using namespace std; using namespace Uintah;

Diamm::Diamm(ProblemSpecP& ps,MPMFlags* Mflag)
  : ConstitutiveModel(Mflag)
{
  d_NBASICINPUTS=34;
  d_NMGDC=13;

// Total number of properties
  d_NDMMPROP=d_NBASICINPUTS+d_NMGDC;

  // pre-initialize all of the user inputs to zero.
  for(int i = 0; i<d_NDMMPROP; i++){
     UI[i] = 0.;
  }
  // Read model parameters from the input file
  getInputParameters(ps);

  // Check that model parameters are valid and allow model to change if needed

  DMMCHK(UI,UI,&UI[d_NBASICINPUTS]);
  //Create VarLabels for GeoModel internal state variables (ISVs)
  int nx;
  char* namea[5000];
  char* keya[5000];
  double rdim[700];
  int iadvct[100];
  int itype[100];

  DMMRXV( UI, UI, UI, nx, namea, keya, rinit, rdim, iadvct, itype );

  d_NINSV=nx;
  //  cout << "d_NINSV = " << d_NINSV << endl;

  initializeLocalMPMLabels();
}

Diamm::Diamm(const Diamm* cm) : ConstitutiveModel(cm)
{
  for(int i=0;i<d_NDMMPROP;i++){
    UI[i] = cm->UI[i];
  }

  //Create VarLabels for Diamm internal state variables (ISVs)
  initializeLocalMPMLabels();
}

Diamm::~Diamm()
{
   for (unsigned int i = 0; i< ISVLabels.size();i++){
     VarLabel::destroy(ISVLabels[i]);
   }
}

void Diamm::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","diamm");
  }

  cm_ps->appendElement("B0",UI[0]);   // initial bulk modulus (stress)
  cm_ps->appendElement("B1",UI[1]);   // initial bulk modulus (stress)
  cm_ps->appendElement("B2",UI[2]);   // initial bulk modulus (stress)

  cm_ps->appendElement("G0",UI[3]);   // initial shear modulus (stress)
  cm_ps->appendElement("G1",UI[4]);   // nonlinear shear mod param
  cm_ps->appendElement("G2",UI[5]);   // nonlinear shear mod param
  cm_ps->appendElement("G3",UI[6]);   // nonlinear shear mod param

  cm_ps->appendElement("A1",UI[7]);  // meridional yld prof param
  cm_ps->appendElement("A2",UI[8]);  // meridional yld prof param
  cm_ps->appendElement("A3",UI[9]);  // meridional yld prof param
  cm_ps->appendElement("A4",UI[10]);  // meridional yld prof param
  cm_ps->appendElement("A5",UI[11]);  // meridional yld prof param
  cm_ps->appendElement("A6",UI[12]);  // meridional yld prof param

  cm_ps->appendElement("AN",UI[13]);  //

  cm_ps->appendElement("R0",UI[14]);  //
  cm_ps->appendElement("T0",UI[15]);  //
  cm_ps->appendElement("C0",UI[16]);  //

  cm_ps->appendElement("S1",UI[17]);  //
  cm_ps->appendElement("GP",UI[18]);  //
  cm_ps->appendElement("CV",UI[19]);  //
  cm_ps->appendElement("TM",UI[20]);  //

  cm_ps->appendElement("T1",UI[21]);  //
  cm_ps->appendElement("T2",UI[22]);  //
  cm_ps->appendElement("T3",UI[23]);  //
  cm_ps->appendElement("T4",UI[24]);  //

  cm_ps->appendElement("XP",UI[25]);//
  cm_ps->appendElement("SC",UI[26]);//

  cm_ps->appendElement("IDK",UI[27]);//
  cm_ps->appendElement("IDG",UI[28]);//

  cm_ps->appendElement("A4PF",UI[29]);//

  cm_ps->appendElement("TQC",UI[30]);//
  cm_ps->appendElement("F1",UI[31]);//

  cm_ps->appendElement("TEST",UI[32]);//
  cm_ps->appendElement("DEJAVU",UI[33]);//

  int dcprop=d_NBASICINPUTS-1;
  cm_ps->appendElement("DC1",UI[dcprop+1]);//
  cm_ps->appendElement("DC2",UI[dcprop+2]);//
  cm_ps->appendElement("DC3",UI[dcprop+3]);//
  cm_ps->appendElement("DC4",UI[dcprop+4]);//
  cm_ps->appendElement("DC5",UI[dcprop+5]);//
  cm_ps->appendElement("DC6",UI[dcprop+6]);//
  cm_ps->appendElement("DC7",UI[dcprop+7]);//
  cm_ps->appendElement("DC8",UI[dcprop+8]);//
  cm_ps->appendElement("DC9",UI[dcprop+9]);//
  cm_ps->appendElement("DC10",UI[dcprop+10]);//
  cm_ps->appendElement("DC11",UI[dcprop+11]);//
  cm_ps->appendElement("DC12",UI[dcprop+12]);//
  cm_ps->appendElement("DC13",UI[dcprop+13]);//

}

Diamm* Diamm::clone()
{
  return scinew Diamm(*this);
}

void Diamm::initializeCMData(const Patch* patch,
                               const MPMMaterial* matl,
                               DataWarehouse* new_dw)
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  initSharedDataForExplicit(patch, matl, new_dw);

  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  StaticArray<ParticleVariable<double> > ISVs(d_NINSV+1);

  cout << "In initializeCMData" << endl;
  for(int i=0;i<d_NINSV;i++){
    new_dw->allocateAndPut(ISVs[i],ISVLabels[i], pset);
    ParticleSubset::iterator iter = pset->begin();
    for(;iter != pset->end(); iter++){
      ISVs[i][*iter] = rinit[i];
    }
  }

  computeStableTimestep(patch, matl, new_dw);
}

void Diamm::addParticleState(std::vector<const VarLabel*>& from,
                               std::vector<const VarLabel*>& to)
{
  // Add the local particle state data for this constitutive model.
  for(int i=0;i<d_NINSV;i++){
    from.push_back(ISVLabels[i]);
    to.push_back(ISVLabels_preReloc[i]);
  }
}

void Diamm::computeStableTimestep(const Patch* patch,
                                    const MPMMaterial* matl,
                                    DataWarehouse* new_dw)
{
  // This is only called for the initial timestep - all other timesteps
  // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int dwi = matl->getDWIndex();
  ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
  constParticleVariable<double> pmass, pvolume;
  constParticleVariable<Vector> pvelocity;

  new_dw->get(pmass,     lb->pMassLabel,     pset);
  new_dw->get(pvolume,   lb->pVolumeLabel,   pset);
  new_dw->get(pvelocity, lb->pVelocityLabel, pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double bulk = UI[0];
  double G = UI[3];
  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
     particleIndex idx = *iter;

     // Compute wave speed at each particle, store the maximum
     c_dil = sqrt((bulk + 4.*G/3.)*pvolume[idx]/pmass[idx]);
     WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                      Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                      Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
  }
  //UI[14]=matl->getInitialDensity();
  //UI[15]=matl->getRoomTemperature();
  //UI[14]=bulk/matl->getInitialDensity();  ??tim
  //UI[19]=matl->getInitialCv();
  WaveSpeed = dx/WaveSpeed;
  double delT_new = WaveSpeed.minComponent();
  new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
}

void Diamm::computeStressTensor(const PatchSubset* patches,
                                  const MPMMaterial* matl,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw)
{
  double rho_orig = matl->getInitialDensity();
  for(int p=0;p<patches->size();p++){
    double se = 0.0;
    const Patch* patch = patches->get(p);

    Matrix3 Identity; Identity.Identity();
    double c_dil=0.0;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
    Vector dx = patch->dCell();

    int dwi = matl->getDWIndex();
    // Create array for the particle position
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    constParticleVariable<Matrix3> deformationGradient, pstress;
    ParticleVariable<Matrix3> pstress_new;
    constParticleVariable<Matrix3> deformationGradient_new, velGrad;
    constParticleVariable<double> pmass, pvolume, ptemperature;
    constParticleVariable<double> pvolume_new;
    constParticleVariable<Vector> pvelocity;
    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    old_dw->get(pstress,             lb->pStressLabel,             pset);
    old_dw->get(pmass,               lb->pMassLabel,               pset);
    old_dw->get(pvolume,             lb->pVolumeLabel,             pset);
    old_dw->get(pvelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(ptemperature,        lb->pTemperatureLabel,        pset);
    old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);

    StaticArray<constParticleVariable<double> > ISVs(d_NINSV+1);
    for(int i=0;i<d_NINSV;i++){
      old_dw->get(ISVs[i],           ISVLabels[i],                 pset);
    }

    ParticleVariable<double> pdTdt,p_q;

    new_dw->allocateAndPut(pstress_new,     lb->pStressLabel_preReloc,   pset);
    new_dw->allocateAndPut(pdTdt,           lb->pdTdtLabel_preReloc,     pset);
    new_dw->allocateAndPut(p_q,             lb->p_qLabel_preReloc,       pset);
    new_dw->get(deformationGradient_new,
                                 lb->pDeformationMeasureLabel_preReloc,  pset);
    new_dw->get(pvolume_new,     lb->pVolumeLabel_preReloc,              pset);
    new_dw->get(velGrad,         lb->pVelGradLabel_preReloc,             pset);

    StaticArray<ParticleVariable<double> > ISVs_new(d_NINSV+1);
    for(int i=0;i<d_NINSV;i++){
      new_dw->allocateAndPut(ISVs_new[i],ISVLabels_preReloc[i], pset);
    }

    for(ParticleSubset::iterator iter = pset->begin();
                                        iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Assign zero internal heating by default - modify if necessary.
      pdTdt[idx] = 0.0;

      // Calculate rate of deformation D, and deviatoric rate DPrime,
      Matrix3 D = (velGrad[idx] + velGrad[idx].Transpose())*.5;

      // get the volumetric part of the deformation
      double J = deformationGradient_new[idx].Determinant();
      // Check 1: Look at Jacobian
      if (!(J > 0.0)) {
        cerr << getpid() ;
        constParticleVariable<long64> pParticleID;
        old_dw->get(pParticleID, lb->pParticleIDLabel, pset);
        cerr << "**ERROR** Negative Jacobian of deformation gradient"
             << " in particle " << pParticleID[idx] << endl;
        cerr << "l = " << velGrad[idx] << endl;
        cerr << "F_old = " << deformationGradient[idx] << endl;
        cerr << "F_new = " << deformationGradient_new[idx] << endl;
        cerr << "J = " << J << endl;
        throw InternalError("Negative Jacobian",__FILE__,__LINE__);
      }

      // Compute the local sound speed
      double rho_cur = rho_orig/J;

      // NEED TO FIND R
      Matrix3 tensorR, tensorU;

      // Look into using Rebecca's PD algorithm
      deformationGradient_new[idx].polarDecompositionRMB(tensorU, tensorR);

      // This is the previous timestep Cauchy stress
      // unrotated tensorSig=R^T*pstress*R
      Matrix3 tensorSig = (tensorR.Transpose())*(pstress[idx]*tensorR);

      // Load into 1-D array for the fortran code
      double sigarg[6];
      sigarg[0]=tensorSig(0,0);
      sigarg[1]=tensorSig(1,1);
      sigarg[2]=tensorSig(2,2);
      sigarg[3]=tensorSig(0,1);
      sigarg[4]=tensorSig(1,2);
      sigarg[5]=tensorSig(2,0);

      // UNROTATE D: S=R^T*D*R
      D=(tensorR.Transpose())*(D*tensorR);

      // Load into 1-D array for the fortran code
      double Darray[6];
      Darray[0]=D(0,0);
      Darray[1]=D(1,1);
      Darray[2]=D(2,2);
      Darray[3]=D(0,1);
      Darray[4]=D(1,2);
      Darray[5]=D(2,0);
      double svarg[d_NINSV];
      double USM=9e99;
      double dt = delT;
      int nblk = 1;

      // Load ISVs into a 1D array for fortran code
      for(int i=0;i<d_NINSV;i++){
        svarg[i]=ISVs[i][idx];
      }

      DIAMM_CALC(nblk, d_NINSV, dt, UI, sigarg, Darray, svarg, USM);

      // Unload ISVs from 1D array into ISVs_new
      for(int i=0;i<d_NINSV;i++){
        ISVs_new[i][idx]=svarg[i];
      }

      // This is the Cauchy stress, still unrotated
      tensorSig(0,0) = sigarg[0];
      tensorSig(1,1) = sigarg[1];
      tensorSig(2,2) = sigarg[2];
      tensorSig(0,1) = sigarg[3];
      tensorSig(1,0) = sigarg[3];
      tensorSig(2,1) = sigarg[4];
      tensorSig(1,2) = sigarg[4];
      tensorSig(2,0) = sigarg[5];
      tensorSig(0,2) = sigarg[5];

      // ROTATE pstress_new: S=R*tensorSig*R^T
      pstress_new[idx] = (tensorR*tensorSig)*(tensorR.Transpose());

      c_dil = sqrt(USM/rho_cur);

      // Compute the strain energy for all the particles
      Matrix3 AvgStress = (pstress_new[idx] + pstress[idx])*.5;

      double e = (D(0,0)*AvgStress(0,0) +
                  D(1,1)*AvgStress(1,1) +
                  D(2,2)*AvgStress(2,2) +
              2.*(D(0,1)*AvgStress(0,1) +
                  D(0,2)*AvgStress(0,2) +
                  D(1,2)*AvgStress(1,2))) * pvolume_new[idx]*delT;

      se += e;

      // Compute wave speed at each particle, store the maximum
      Vector pvelocity_idx = pvelocity[idx];
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity_idx.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pvelocity_idx.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pvelocity_idx.z()),WaveSpeed.z()));

      // Compute artificial viscosity term
      if (flag->d_artificial_viscosity) {
        double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;
        double c_bulk = sqrt(UI[0]/rho_cur);
        p_q[idx] = artificialBulkViscosity(D.Trace(), c_bulk, rho_cur, dx_ave);
      } else {
        p_q[idx] = 0.;
      }
    }  // end loop over particles

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(se),     lb->StrainEnergyLabel);
    }
  }
}

void Diamm::carryForward(const PatchSubset* patches,
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
    StaticArray<constParticleVariable<double> > ISVs(d_NINSV+1);
    StaticArray<ParticleVariable<double> > ISVs_new(d_NINSV+1);

    for(int i=0;i<d_NINSV;i++){
      old_dw->get(ISVs[i],ISVLabels[i], pset);
      new_dw->allocateAndPut(ISVs_new[i],ISVLabels_preReloc[i], pset);
      ISVs_new[i].copyData(ISVs[i]);
  }

    // Don't affect the strain energy or timestep size
    new_dw->put(delt_vartype(1.e10), lb->delTLabel, patch->getLevel());
    
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(0.),     lb->StrainEnergyLabel);
    }
  }

}

void Diamm::addInitialComputesAndRequires(Task* task,
                                            const MPMMaterial* matl,
                                            const PatchSet* ) const
{
  // Add the computes and requires that are common to all explicit
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();

  cout << "In add InitialComputesAnd" << endl;

  // Other constitutive model and input dependent computes and requires
  for(int i=0;i<d_NINSV;i++){
    task->computes(ISVLabels[i], matlset);
  }
}

void Diamm::addComputesAndRequires(Task* task,
                                     const MPMMaterial* matl,
                                     const PatchSet* patches) const
{
  // Add the computes and requires that are common to all explicit
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForHypoExplicit(task, matlset, patches);

  // Computes and requires for internal state data
  for(int i=0;i<d_NINSV;i++){
    task->requires(Task::OldDW, ISVLabels[i],          matlset, Ghost::None);
    task->computes(             ISVLabels_preReloc[i], matlset);
  }
}

void Diamm::addComputesAndRequires(Task*,
                                     const MPMMaterial*,
                                     const PatchSet*,
                                     const bool ) const
{
}

double Diamm::computeRhoMicroCM(double pressure,
                                  const double p_ref,
                                  const MPMMaterial* matl, 
                                  double temperature,
                                  double rho_guess)
{
  double rho_orig = matl->getInitialDensity();
  double p_gauge = pressure - p_ref;
  double rho_cur;
  double bulk = UI[0];

  rho_cur = rho_orig/(1-p_gauge/bulk);

  return rho_cur;

#if 1
  cout << "NO VERSION OF computeRhoMicroCM EXISTS YET FOR Diamm" << endl;
#endif
}

void Diamm::computePressEOSCM(double rho_cur, double& pressure,
                                double p_ref,
                                double& dp_drho,      double& tmp,
                                const MPMMaterial* matl, 
                                double temperature)
{

  double bulk = UI[0];
  double rho_orig = matl->getInitialDensity();

  double p_g = bulk*(1.0 - rho_orig/rho_cur);
  pressure = p_ref + p_g;
  dp_drho  = bulk*rho_orig/(rho_cur*rho_cur);
  tmp = bulk/rho_cur;  // speed of sound squared

#if 1
  cout << "NO VERSION OF computePressEOSCM EXISTS YET FOR Diamm" << endl;
#endif
}

double Diamm::getCompressibility()
{
  return 1.0/UI[0];
}

void
Diamm::getInputParameters(ProblemSpecP& ps)
{
  ps->getWithDefault("B0",UI[0],0.0);              // initial bulk modulus (stress)
  ps->getWithDefault("B1",UI[1],0.0);              // initial bulk modulus (stress)
  ps->getWithDefault("B2",UI[2],0.0);              // initial bulk modulus (stress)
  ps->getWithDefault("G0",UI[3],0.0);              // initial shear modulus (stress)
  ps->getWithDefault("G1",UI[4],0.0);   // nonlinear shear mod param (dim. less)
  ps->getWithDefault("G2",UI[5],0.0);   // nonlinear shear mod param (1/stress)
  ps->getWithDefault("G3",UI[6],0.0);   // nonlinear shear mod param (stress)
  ps->getWithDefault("A1",UI[7],0.0);  // meridional yld prof param (stress)
  ps->getWithDefault("A2",UI[8],0.0);  // meridional yld prof param (1/stress)
  ps->getWithDefault("A3",UI[9],0.0);  // meridional yld prof param (stress)
  ps->getWithDefault("A4",UI[10],0.0);  // meridional yld prof param (dim. less)
  ps->getWithDefault("A5",UI[11],0.0);  // meridional yld prof param (dim. less)
  ps->getWithDefault("A6",UI[12],0.0);  // meridional yld prof param (dim. less)
  ps->getWithDefault("AN",UI[13],0.0);  //
  ps->getWithDefault("R0",UI[14],0.0);  //
  ps->getWithDefault("T0",UI[15],0.0);  //
  ps->getWithDefault("C0",UI[16],0.0);  //
  ps->getWithDefault("S1",UI[17],0.0);  //
  ps->getWithDefault("GP",UI[18],0.0);  //
  ps->getWithDefault("CV",UI[19],0.0);  //
  ps->getWithDefault("TM",UI[20],0.0);  //
  ps->getWithDefault("T1",UI[21],0.0);  //
  ps->getWithDefault("T2",UI[22],0.0);  //
  ps->getWithDefault("T3",UI[23],0.0);  //
  ps->getWithDefault("T4",UI[24],0.0);  //
  ps->getWithDefault("XP",UI[25],0.0);//
  ps->getWithDefault("SC",UI[26],0.0);//
  ps->getWithDefault("IDK",UI[27],0.0);//
  ps->getWithDefault("IDG",UI[28],0.0);//
  ps->getWithDefault("A2PF",UI[29],0.0);//
  ps->getWithDefault("TQC",UI[30],0.0);//
  ps->getWithDefault("F1",UI[31],0.0);//
  ps->getWithDefault("TEST",UI[32],0.0);//
  ps->getWithDefault("DEJAVU",UI[33],0.0);//

  ps->getWithDefault("DC1",UI[34],0.0);//
  ps->getWithDefault("DC2",UI[35],0.0);//
  ps->getWithDefault("DC3",UI[36],0.0);//
  ps->getWithDefault("DC4",UI[37],0.0);//
  ps->getWithDefault("DC5",UI[38],0.0);//
  ps->getWithDefault("DC6",UI[39],0.0);//
  ps->getWithDefault("DC7",UI[40],0.0);//
  ps->getWithDefault("DC8",UI[41],0.0);//
  ps->getWithDefault("DC9",UI[42],0.0);//
  ps->getWithDefault("DC10",UI[43],0.0);//
  ps->getWithDefault("DC11",UI[44],0.0);//
  ps->getWithDefault("DC12",UI[45],0.0);//
  ps->getWithDefault("DC13",UI[46],0.0);//
}

void
Diamm::initializeLocalMPMLabels()
{
  vector<string> ISVNames;

  ISVNames.push_back("EQDOT");
  ISVNames.push_back("I1");
  ISVNames.push_back("ROOTJ2");
  ISVNames.push_back("EQPS");
  ISVNames.push_back("EVOL");
  ISVNames.push_back("T");
  ISVNames.push_back("CS");
  ISVNames.push_back("R");
  ISVNames.push_back("EU");
  ISVNames.push_back("RJ");
  ISVNames.push_back("AM");
  ISVNames.push_back("EQPV");
  ISVNames.push_back("F4");
  ISVNames.push_back("QSSIGXX");
  ISVNames.push_back("QSSIGYY");
  ISVNames.push_back("QSSIGZZ");
  ISVNames.push_back("QSSIGXY");
  ISVNames.push_back("QSSIGYZ");
  ISVNames.push_back("QSSIGXZ");
  ISVNames.push_back("EXX");
  ISVNames.push_back("EYY");
  ISVNames.push_back("EZZ");
  ISVNames.push_back("EXY");
  ISVNames.push_back("EYZ");
  ISVNames.push_back("EXZ");
  ISVNames.push_back("EJ2");


  for(int i=0;i<d_NINSV;i++){
    ISVLabels.push_back(VarLabel::create(ISVNames[i],
                          ParticleVariable<double>::getTypeDescription()));
    ISVLabels_preReloc.push_back(VarLabel::create(ISVNames[i]+"+",
                          ParticleVariable<double>::getTypeDescription()));
  }
}
