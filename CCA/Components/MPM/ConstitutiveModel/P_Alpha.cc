/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
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


#include <CCA/Components/MPM/ConstitutiveModel/P_Alpha.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Grid/Patch.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Math/Matrix3.h>
#include <Core/Grid/Variables/NodeIterator.h> 
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Math/MinMax.h>
#include <Core/Malloc/Allocator.h>
#include <fstream>
#include <iostream>

using std::cerr;
using namespace Uintah;

P_Alpha::P_Alpha(ProblemSpecP& ps,MPMFlags* Mflag)
  : ConstitutiveModel(Mflag)
{
  ps->require("Ps",    d_initialData.Ps);
  ps->require("Pe",    d_initialData.Pe);
  ps->require("rhoS",  d_initialData.rhoS);
  ps->require("alpha0",d_initialData.alpha0);
  ps->require("K0",    d_initialData.K0);
  ps->require("Ks",    d_initialData.Ks);

  alphaLabel           = VarLabel::create("p.alpha",
                            ParticleVariable<double>::getTypeDescription());
  alphaLabel_preReloc  = VarLabel::create("p.alpha+",
                            ParticleVariable<double>::getTypeDescription());

}

P_Alpha::P_Alpha(const P_Alpha* cm) : ConstitutiveModel(cm)
{
  d_initialData.Ps     = cm->d_initialData.Ps;
  d_initialData.Pe     = cm->d_initialData.Pe;
  d_initialData.rhoS   = cm->d_initialData.rhoS;
  d_initialData.alpha0 = cm->d_initialData.alpha0;
  d_initialData.K0     = cm->d_initialData.K0;
  d_initialData.Ks     = cm->d_initialData.Ks;
}

P_Alpha::~P_Alpha()
{
  VarLabel::destroy(alphaLabel);
  VarLabel::destroy(alphaLabel_preReloc);
}

void P_Alpha::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","p_alpha");
  }

  cm_ps->appendElement("Ps",    d_initialData.Ps);
  cm_ps->appendElement("Pe",    d_initialData.Pe);
  cm_ps->appendElement("rhoS",  d_initialData.rhoS);
  cm_ps->appendElement("alpha0",d_initialData.alpha0);
  cm_ps->appendElement("K0",    d_initialData.K0);
  cm_ps->appendElement("Ks",    d_initialData.Ks);
}

P_Alpha* P_Alpha::clone()
{
  return scinew P_Alpha(*this);
}

void P_Alpha::addInitialComputesAndRequires(Task* task,
                                         const MPMMaterial* matl,
                                         const PatchSet*) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(alphaLabel,matlset);
}

void P_Alpha::initializeCMData(const Patch* patch,
                                        const MPMMaterial* matl,
                                        DataWarehouse* new_dw)
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  initSharedDataForExplicit(patch, matl, new_dw);

  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  ParticleVariable<double>      alpha_min;
  new_dw->allocateAndPut(alpha_min, alphaLabel, pset);

  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
        alpha_min[*iter]      = d_initialData.alpha0;
  }

  computeStableTimestep(patch, matl, new_dw);
}

void P_Alpha::allocateCMDataAddRequires(Task* task,
                                           const MPMMaterial* matl ,
                                           const PatchSet* patches,
                                           MPMLabel* ) const
{
  const MaterialSubset* matlset = matl->thisMaterial();

  // Allocate the variables shared by all constitutive models
  // for the particle convert operation
  // This method is defined in the ConstitutiveModel base class.
  addSharedRForConvertExplicit(task, matlset, patches);
}

void P_Alpha::allocateCMDataAdd(DataWarehouse* new_dw,
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
}


void P_Alpha::addParticleState(std::vector<const VarLabel*>& from,
                                std::vector<const VarLabel*>& to)
{
  // Add the local particle state data for this constitutive model.
  from.push_back(alphaLabel);
  to.push_back(alphaLabel_preReloc);
}

void P_Alpha::computeStableTimestep(const Patch* patch,
                                    const MPMMaterial* matl,
                                    DataWarehouse* new_dw)
{
  // This is only called for the initial timestep - all other timesteps
  // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int dwi = matl->getDWIndex();
  // Retrieve the array of constitutive parameters
  ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
  constParticleVariable<double> pmass, pvolume;
  constParticleVariable<Vector> pvelocity;

  new_dw->get(pmass,     lb->pMassLabel,        pset);
  new_dw->get(pvolume,   lb->pVolumeLabel,      pset);
  new_dw->get(pvelocity, lb->pVelocityLabel,    pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double K0 = d_initialData.K0;

  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
     particleIndex idx = *iter;

     double rhoM = pmass[idx]/pvolume[idx];

     double tmp = K0/rhoM;

     // Compute wave speed at each particle, store the maximum
     c_dil = sqrt(tmp);
     WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                      Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                      Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
  }
  WaveSpeed = dx/WaveSpeed;
  double delT_new = WaveSpeed.minComponent();
  new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
}

void P_Alpha::computeStressTensor(const PatchSubset* patches,
                                      const MPMMaterial* matl,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw)
{
  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);
    Matrix3 deformationGradientInc;
    double se=0.;
    double c_dil=0.0;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
    Matrix3 Identity;

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<Vector> d_S(interpolator->size());
    vector<double> S(interpolator->size());

    Identity.Identity();

    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    constParticleVariable<Point> px;
    ParticleVariable<Matrix3> deformationGradient_new;
    constParticleVariable<Matrix3> deformationGradient;
    ParticleVariable<Matrix3> pstress;
    constParticleVariable<double> pmass;
    constParticleVariable<double> alpha_min_old, ptemperature;
    ParticleVariable<double> pvolume;
    ParticleVariable<double> alpha_min_new;
    constParticleVariable<Vector> pvelocity, psize;
    constNCVariable<Vector> gvelocity;
    ParticleVariable<double> pdTdt,p_q;
    ParticleVariable<Matrix3>      velGrad;
    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    Ghost::GhostType  gac   = Ghost::AroundCells;

    new_dw->get(gvelocity, lb->gVelocityStarLabel, dwi,patch, gac, NGN);

    old_dw->get(px,                          lb->pXLabel,                 pset);
    old_dw->get(pmass,                       lb->pMassLabel,              pset);
    old_dw->get(psize,                       lb->pSizeLabel,              pset);
    old_dw->get(pvelocity,                   lb->pVelocityLabel,          pset);
    old_dw->get(ptemperature,                lb->pTemperatureLabel,       pset);
    old_dw->get(deformationGradient,         lb->pDeformationMeasureLabel,pset);
    old_dw->get(alpha_min_old,               alphaLabel,                  pset);

    new_dw->allocateAndPut(pstress,          lb->pStressLabel_preReloc,   pset);
    new_dw->allocateAndPut(pvolume,          lb->pVolumeLabel_preReloc,   pset);
    new_dw->allocateAndPut(pdTdt,            lb->pdTdtLabel_preReloc,     pset);
    new_dw->allocateAndPut(p_q,              lb->p_qLabel_preReloc,       pset);
    new_dw->allocateAndPut(deformationGradient_new,
                                   lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->allocateAndPut(alpha_min_new,     alphaLabel_preReloc,        pset);

    // Temporary Allocations
    new_dw->allocateTemporary(velGrad,                                    pset);

    double Ps = d_initialData.Ps;
    double Pe = d_initialData.Pe;
    double alpha0 = d_initialData.alpha0;
    double K0 = d_initialData.K0;
    double Ks = d_initialData.Ks;
    double rhoS = d_initialData.rhoS;

    double rho_orig = matl->getInitialDensity();
    double rhoP     = rho_orig/(1.-Pe/K0);
    double alphaP   = rhoS/rhoP;

    for(ParticleSubset::iterator iter = pset->begin();
        iter != pset->end(); iter++){
       particleIndex idx = *iter;

      Matrix3 velGrad_new(0.0);
      if(!flag->d_axisymmetric){
        // Get the node indices that surround the cell
        interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],
                                                  deformationGradient[idx]);

        computeVelocityGradient(velGrad_new,ni,d_S, oodx, gvelocity);
      } else {  // axi-symmetric kinematics
        // Get the node indices that surround the cell
        interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,
                                           psize[idx],deformationGradient[idx]);
        // x -> r, y -> z, z -> theta
        computeAxiSymVelocityGradient(velGrad_new,ni,d_S,S,oodx,gvelocity,px[idx]);
      }

      // Compute the deformation gradient increment using the time_step
      // velocity gradient
      // F_n^np1 = dudx * dt + Identity
      deformationGradientInc = velGrad_new * delT + Identity;

      // Update the deformation gradient tensor to its time n+1 value.
      deformationGradient_new[idx] = deformationGradientInc *
                                     deformationGradient[idx];

      velGrad[idx] = velGrad_new;
    }

    // The following is used only for pressure stabilization
    CCVariable<double> J_CC;
    new_dw->allocateTemporary(J_CC,       patch);
    J_CC.initialize(0.);
    if(flag->d_doPressureStabilization) {
      CCVariable<double> vol_0_CC;
      CCVariable<double> vol_CC;
      new_dw->allocateTemporary(vol_0_CC, patch);
      new_dw->allocateTemporary(vol_CC,   patch);

      vol_0_CC.initialize(0.);
      vol_CC.initialize(0.);
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // get the volumetric part of the deformation
        double J = deformationGradient_new[idx].Determinant();

        // Get the deformed volume
        pvolume[idx]=(pmass[idx]/rho_orig)*J;

        IntVector cell_index;
        patch->findCell(px[idx],cell_index);

        vol_CC[cell_index]  +=pvolume[idx];
        vol_0_CC[cell_index]+=pmass[idx]/rho_orig;
      }

      for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++){
        IntVector c = *iter;
        J_CC[c]=vol_CC[c]/vol_0_CC[c];
      }
    } //end of pressureStabilization loop  at the patch level

    for(ParticleSubset::iterator iter = pset->begin();
        iter != pset->end(); iter++){
       particleIndex idx = *iter;

      // More Pressure Stabilization
      if(flag->d_doPressureStabilization) {
        IntVector cell_index;
        patch->findCell(px[idx],cell_index);

        // get the original volumetric part of the deformation
        double J = deformationGradient_new[idx].Determinant();

        // Change F such that the determinant is equal to the average for
        // the cell
        deformationGradient_new[idx]*=cbrt(J_CC[cell_index])/cbrt(J);
      }

      double Jold = deformationGradient[idx].Determinant();
      double Jnew = deformationGradient_new[idx].Determinant();
      double Jinc = Jnew/Jold;
      double rhoM = rho_orig/Jnew;
      pvolume[idx]=pmass[idx]/rhoM;

      double alpha = rhoS/rhoM;

      double p=0.;
      double dAel_dp=0.;
      double c = sqrt(Ks/rhoS);
      double cs=sqrt(Ks/rhoS);
      double ce=sqrt(K0/rho_orig);

      if(alpha < alpha0 && alpha >= 1.0){
       if(alpha <= alpha_min_old[idx]){  // loading
        if(alpha <= alpha0 && alpha > alphaP){
          // elastic response
          p = K0*(1.-rho_orig/rhoM);
          c = sqrt(K0/rhoM);
        }
        else if(alpha <= alphaP && alpha > 1.0){
          p= Ps - (Ps-Pe)*sqrt((alpha - 1.)/(alphaP - 1.0));
          c = cs + (ce - cs)*((alpha - 1.)/(alpha0 - 1.));
        }
       } else { // alpha < alpha_min, unloading
        if(alpha < alpha0 && alpha >= alphaP && alpha_min_old[idx] >= alphaP){
          // still in initial elastic response
          p = K0*(1.-rho_orig/rhoM);
          c = sqrt(K0/rhoM);
        }
        else if((alpha < alphaP && alpha > 1.0) || alpha_min_old[idx] < alphaP){
          // First, get plastic pressure
          p= Ps - (Ps-Pe)*sqrt((alpha - 1.)/(alphaP - 1.0));
          double h = 1. + (ce - cs)*(alpha - 1.0)/(cs*(alpha0-1.));
          dAel_dp = ((alpha*alpha)/Ks)*(1. - 1./(h*h));
          double dPel = (alpha - alpha_min_old[idx])/dAel_dp;
          p += dPel;
          c = cs + (ce - cs)*((alpha - 1.)/(alpha0 - 1.));
        }
       }
      }
      else if(alpha<1.0){
#if 1
        p = Ps+Ks*(1.-alpha);
        c = cs;
#endif
#if 0
       // Get the state data
       double rho = rhoM;
       double T_0 = 300.;
       double Gamma_0 = 1.54;
       double C_0 = 4029.;
       double S_alpha = 1.237;

       // Calc. zeta
       double zeta = (rho/rhoS - 1.0);

       // Calculate internal energy E
       double cv = matl->getSpecificHeat();
       double E = (cv)*(ptemperature[idx] - T_0)*rhoS;

       // Calculate the pressure
       double p = Gamma_0*E;
       if (rho != rhoS) {
         double numer = rhoS*(C_0*C_0)*(1.0/zeta+
                              (1.0-0.5*Gamma_0));
         double denom = 1.0/zeta - (S_alpha-1.0);
         if (denom == 0.0) {
           cout << "rh0_0 = " << rhoS << " zeta = " << zeta
                << " numer = " << numer << endl;
           denom = 1.0e-5;
         }
          p += numer/(denom*denom);
        }
      double etime = d_sharedState->getElapsedTime();
      cout << "678 " << " " << etime << " " << alpha << " " << p << endl;
        p = Ps + -1.*p;
#endif
      }

      alpha_min_new[idx]=min(alpha,alpha_min_old[idx]);

      if(alpha > alpha0 || p < 0.){
          double rho_max = rhoS/alpha_min_new[idx];
          p = .5*K0*(1.-rho_max/rhoM);
      }

//      p=max(p,0.0);

//      double etime = d_sharedState->getElapsedTime();
//      cout << "12345 " << " " << etime << " " << alpha << " " << p << " " << 1./dAel_dp << endl;

      // Compute artificial viscosity term
      if (flag->d_artificial_viscosity) {
        double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;
        double c_bulk = c;
        Matrix3 D=(velGrad[idx] + velGrad[idx].Transpose())*0.5;
        p_q[idx] = artificialBulkViscosity(D.Trace(), c_bulk, rhoM, dx_ave);
      } else {
        p_q[idx] = 0.;
      }

      pstress[idx] = Identity*(-p);

      // Temp increase due to P*dV work
      // FIX?
      double cv = matl->getSpecificHeat();
      pdTdt[idx] = (-p)*(Jinc-1.)*(1./(rhoM*cv))/delT;

      Vector pvelocity_idx = pvelocity[idx];
      c_dil = sqrt(K0/rhoM);
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity_idx.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pvelocity_idx.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pvelocity_idx.z()),WaveSpeed.z()));
    }

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
    
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(se),        lb->StrainEnergyLabel);
    }

    delete interpolator;
  }
}

void P_Alpha::addComputesAndRequires(Task* task,
                                     const MPMMaterial* matl,
                                     const PatchSet* patches) const
{
  // Add the computes and requires that are common to all explicit 
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForExplicit(task, matlset, patches);
  Ghost::GhostType  gnone = Ghost::None;

  task->requires(Task::OldDW, alphaLabel,  matlset, gnone);
  task->computes(alphaLabel_preReloc,      matlset);
}

void 
P_Alpha::addComputesAndRequires(Task* ,
                               const MPMMaterial* ,
                               const PatchSet* ,
                               const bool ) const
{
}

// The "CM" versions use the pressure-volume relationship of the CNH model
double P_Alpha::computeRhoMicroCM(double press, 
                                      const double Temp,
                                      const MPMMaterial* matl,
                                      double temperature,
                                      double rho_guess)
{
  cerr << "NO VERSION OF computeRhoMicroCM EXISTS YET FOR P_Alpha" << endl;

  return matl->getInitialDensity();
}

void P_Alpha::computePressEOSCM(double rhoM,double& pressure, 
                                   double Temp,
                                   double& dp_drho, double& tmp,
                                   const MPMMaterial*, 
                                   double temperature)
{
  cerr << "NO VERSION OF computePressEOSCM EXISTS YET FOR P_Alpha" << endl;
  pressure = 101325.;
}

double P_Alpha::getCompressibility()
{
  cerr << "NO VERSION OF getCompressibility EXISTS YET FOR P_Alpha" << endl;
  return 1.0/d_initialData.K0;
}

namespace Uintah {

} // End namespace Uintah
