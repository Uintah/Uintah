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


#include <CCA/Components/MPM/ConstitutiveModel/ProgramBurn.h>
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
using namespace SCIRun;

ProgramBurn::ProgramBurn(ProblemSpecP& ps, MPMFlags* Mflag)
  : ConstitutiveModel(Mflag)
{

  d_useModifiedEOS = false;
  ps->require("A",    d_initialData.d_A);
  ps->require("B",    d_initialData.d_B);
  ps->require("R1",   d_initialData.d_R1);
  ps->require("R2",   d_initialData.d_R2);
  ps->require("om",   d_initialData.d_om);
  ps->require("rho0", d_initialData.d_rho0);
}

ProgramBurn::ProgramBurn(const ProgramBurn* cm) : ConstitutiveModel(cm)
{
  d_useModifiedEOS = cm->d_useModifiedEOS ;
  d_initialData.d_A = cm->d_initialData.d_A;
  d_initialData.d_B = cm->d_initialData.d_B;
  d_initialData.d_R1 = cm->d_initialData.d_R1;
  d_initialData.d_R2 = cm->d_initialData.d_R2;
  d_initialData.d_om = cm->d_initialData.d_om;
  d_initialData.d_rho0 = cm->d_initialData.d_rho0;
}

ProgramBurn::~ProgramBurn()
{
}

void ProgramBurn::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","murnahanMPM");
  }
  
  cm_ps->appendElement("A",    d_initialData.d_A);
  cm_ps->appendElement("B",    d_initialData.d_B);
  cm_ps->appendElement("R1",   d_initialData.d_R1);
  cm_ps->appendElement("R2",   d_initialData.d_R2);
  cm_ps->appendElement("om",   d_initialData.d_om);
  cm_ps->appendElement("rho0", d_initialData.d_rho0);
}

ProgramBurn* ProgramBurn::clone()
{
  return scinew ProgramBurn(*this);
}

void ProgramBurn::initializeCMData(const Patch* patch,
                             const MPMMaterial* matl,
                             DataWarehouse* new_dw)
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  initSharedDataForExplicit(patch, matl, new_dw);

  computeStableTimestep(patch, matl, new_dw);
}

void ProgramBurn::allocateCMDataAddRequires(Task* task,
                                            const MPMMaterial* matl,
                                            const PatchSet* patches,
                                            MPMLabel* ) const
{
  const MaterialSubset* matlset = matl->thisMaterial();

  // Allocate the variables shared by all constitutive models
  // for the particle convert operation
  // This method is defined in the ConstitutiveModel base class.
  addSharedRForConvertExplicit(task, matlset, patches);
}


void ProgramBurn::allocateCMDataAdd(DataWarehouse* new_dw,
                                    ParticleSubset* addset,
                                    map<const VarLabel*, ParticleVariableBase*>* newState,
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

void ProgramBurn::addParticleState(std::vector<const VarLabel*>& ,
                                   std::vector<const VarLabel*>& )
{
  // Add the local particle state data for this constitutive model.
}

void ProgramBurn::computeStableTimestep(const Patch* patch,
                                        const MPMMaterial* matl,
                                        DataWarehouse* new_dw)
{
  // This is only called for the initial timestep - all other timesteps
  // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int dwi = matl->getDWIndex();
  // Retrieve the array of constitutive parameters
  ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
  constParticleVariable<double> pmass, pvolume,ptemperature;
  constParticleVariable<Vector> pvelocity;

  new_dw->get(pmass,        lb->pMassLabel,        pset);
  new_dw->get(pvolume,      lb->pVolumeLabel,      pset);
  new_dw->get(pvelocity,    lb->pVelocityLabel,    pset);
  new_dw->get(ptemperature, lb->pTemperatureLabel, pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double A = d_initialData.d_A;
  double B = d_initialData.d_B;
  double R1 = d_initialData.d_R1;
  double R2 = d_initialData.d_R2;
  double om = d_initialData.d_om;
  double rho0 = d_initialData.d_rho0;
  double cv = matl->getSpecificHeat();
  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
     particleIndex idx = *iter;
     // Compute wave speed at each particle, store the maximum
     double rhoM = pmass[idx]/pvolume[idx];
     double V = rho0/rhoM;
     double P1 = A*exp(-R1*V);
     double P2 = B*exp(-R2*V);
     double dp_drho = (R1*rho0*P1 + R2*rho0*P2)/(rhoM*rhoM) + om*cv*ptemperature[idx];
     double c_2 = dp_drho;
//     c_2 = dp_drho + dp_de * press_CC[c]/(rho_micro[indx][c] * rho_micro[indx][c]);
     c_dil = sqrt(c_2);
     WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                      Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                      Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
  }
  WaveSpeed = dx/WaveSpeed;
  double delT_new = WaveSpeed.minComponent();
  new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
}

void ProgramBurn::computeStressTensor(const PatchSubset* patches,
                                      const MPMMaterial* matl,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw)
{
    for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);
    Matrix3 velGrad,Shear;
    double /*p,*/se=0.;
//    double c_dil=0.0;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
//    double onethird = (1.0/3.0);
    Matrix3 Identity;
    Identity.Identity();

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<Vector> d_S(interpolator->size());
    vector<double> S(interpolator->size());

    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    constParticleVariable<Point> px;
    ParticleVariable<Matrix3> deformationGradient_new;
    constParticleVariable<Matrix3> deformationGradient;
    ParticleVariable<Matrix3> pstress;
    constParticleVariable<double> pmass;
    ParticleVariable<double> pvolume;
    constParticleVariable<Vector> pvelocity;
    constParticleVariable<Vector> psize;
    ParticleVariable<double> pdTdt,p_q;

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    Ghost::GhostType  gac   = Ghost::AroundCells;
    old_dw->get(px,                  lb->pXLabel,                  pset);
    old_dw->get(pmass,               lb->pMassLabel,               pset);
    old_dw->get(pvelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);
    old_dw->get(psize,               lb->pSizeLabel,               pset);
    
    new_dw->allocateAndPut(pstress,          lb->pStressLabel_preReloc,  pset);
    new_dw->allocateAndPut(pvolume,          lb->pVolumeLabel_preReloc,  pset);
    new_dw->allocateAndPut(pdTdt,            lb->pdTdtLabel_preReloc,    pset);
    new_dw->allocateAndPut(p_q,              lb->p_qLabel_preReloc,      pset);
    new_dw->allocateAndPut(deformationGradient_new,
                                  lb->pDeformationMeasureLabel_preReloc, pset);

//    double A = d_initialData.d_A;
//    double B = d_initialData.d_B;
//    double R1 = d_initialData.d_R1;
//    double R2 = d_initialData.d_R2;
//    double om = d_initialData.d_om;
//    double rho0 = d_initialData.d_rho0; // matl->getInitialDensity();

    constNCVariable<Vector> gvelocity;
    new_dw->get(gvelocity, lb->gVelocityStarLabel,dwi,patch,gac,NGN);

//  if(!flag->d_doGridReset){
//    cerr << "The water model doesn't work without resetting the grid" << endl;
//  }

    for(ParticleSubset::iterator iter = pset->begin();
        iter != pset->end(); iter++){
      particleIndex idx = *iter;
      
      // Assign zero internal heating by default - modify if necessary.
      pdTdt[idx] = 0.0;

      velGrad.set(0.0);
      if(!flag->d_axisymmetric){
        // Get the node indices that surround the cell
        interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],deformationGradient[idx]);

        computeVelocityGradient(velGrad,ni,d_S, oodx, gvelocity);
      } else {  // axi-symmetric kinematics
        // Get the node indices that surround the cell
        interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,
                                                            psize[idx],
                                                   deformationGradient[idx]);
        // x -> r, y -> z, z -> theta
        computeAxiSymVelocityGradient(velGrad,ni,d_S,S,oodx,gvelocity,px[idx]);
      }

      deformationGradient_new[idx]=(velGrad*delT+Identity)
                                    *deformationGradient[idx];

#if 0
      double J = deformationGradient_new[idx].Determinant();

      // Calculate rate of deformation D, and deviatoric rate DPrime,
      Matrix3 D = (velGrad + velGrad.Transpose())*0.5;
      Matrix3 DPrime = D - Identity*onethird*D.Trace();

      // Get the deformed volume and current density
      double rho_cur = rho_orig/J;
      pvolume[idx] = pmass[idx]/rho_cur;

      // Viscous part of the stress
      Shear = DPrime*(2.*viscosity);

      // get the hydrostatic part of the stress
      double jtotheminusgamma = pow(J,-gamma);
      p = bulk*(jtotheminusgamma - 1.0);

      // compute the total stress (volumetric + deviatoric)
      pstress[idx] = Identity*(-p) + Shear;

      Vector pvelocity_idx = pvelocity[idx];
      c_dil = sqrt((gamma*jtotheminusgamma*bulk)/rho_cur);
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity_idx.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pvelocity_idx.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pvelocity_idx.z()),WaveSpeed.z()));
                                                                                
      // Compute artificial viscosity term
      if (flag->d_artificial_viscosity) {
        double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;
        double c_bulk = sqrt(bulk/rho_cur);
        Matrix3 D=(velGrad + velGrad.Transpose())*0.5;
        p_q[idx] = artificialBulkViscosity(D.Trace(), c_bulk, rho_cur, dx_ave);
      } else {
        p_q[idx] = 0.;
      }
#endif
    }  // end loop over particles

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());

    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(se),      lb->StrainEnergyLabel);
    }
    delete interpolator;
  }
}

void ProgramBurn::carryForward(const PatchSubset* patches,
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
    new_dw->put(delt_vartype(1.e10), lb->delTLabel, patch->getLevel());
    
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(0.),   lb->StrainEnergyLabel);  
    }
  }
}

void ProgramBurn::addComputesAndRequires(Task* task,
                                          const MPMMaterial* matl,
                                          const PatchSet* patches) const
{
  // Add the computes and requires that are common to all explicit 
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForExplicit(task, matlset, patches);

}

void 
ProgramBurn::addComputesAndRequires(Task* ,
                                   const MPMMaterial* ,
                                   const PatchSet* ,
                                   const bool ) const
{
}


// This is not yet implemented - JG- 7/26/10
double ProgramBurn::computeRhoMicroCM(double pressure, 
                                      const double p_ref,
                                      const MPMMaterial* matl,
                                      double temperature,
                                      double rho_guess)
{
    cout << "NO VERSION OF computeRhoMicroCM EXISTS YET FOR ProgramBurn"
       << endl;
    double rho_orig = d_initialData.d_rho0; //matl->getInitialDensity();

    return rho_orig;
}

void ProgramBurn::computePressEOSCM(const double rhoM,double& pressure, 
                                    const double p_ref,
                                    double& dp_drho, double& tmp,
                                    const MPMMaterial* matl,
                                    double temperature)
{
  double A = d_initialData.d_A;
  double B = d_initialData.d_B;
  double R1 = d_initialData.d_R1;
  double R2 = d_initialData.d_R2;
  double om = d_initialData.d_om;
  double rho0 = d_initialData.d_rho0;
  double cv = matl->getSpecificHeat();
  double V = rho0/rhoM;
  double P1 = A*exp(-R1*V);
  double P2 = B*exp(-R2*V);
  double P3 = om*cv*tmp*rhoM;

  pressure = P1 + P2 + P3;

  dp_drho = (R1*rho0*P1 + R2*rho0*P2)/(rhoM*rhoM) + om*cv*tmp;
}

// This is not yet implemented - JG- 7/26/10
double ProgramBurn::getCompressibility()
{
   cout << "NO VERSION OF getCompressibility EXISTS YET FOR ProgramBurn"<< endl;
  return 1.0;
}

namespace Uintah {
} // End namespace Uintah
