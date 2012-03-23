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


#include <CCA/Components/MPM/ConstitutiveModel/Biswajit/JWLppSolid.h>
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

using namespace std;
using namespace Uintah;
using namespace SCIRun;

JWLppSolid::JWLppSolid(ProblemSpecP& ps, MPMFlags* Mflag)
  : ConstitutiveModel(Mflag)
{
  d_useModifiedEOS = false;

  // These two parameters are used for the unburned Murnahan EOS
  ps->require("K",    d_initialData.d_K);
  ps->require("n",    d_initialData.d_n);

  // These parameters are used for the product JWL EOS
  ps->require("A",    d_initialData.d_A);
  ps->require("B",    d_initialData.d_B);
  ps->require("C",    d_initialData.d_C);
  ps->require("R1",   d_initialData.d_R1);
  ps->require("R2",   d_initialData.d_R2);
  ps->require("om",   d_initialData.d_om);
  ps->require("rho0", d_initialData.d_rho0);

  // These parameters are needed for the reaction model
  ps->require("G",    d_initialData.d_G); // Rate coefficient
  ps->require("b",    d_initialData.d_b); // Pressure exponent

  // Initial stress
  // Fix: Need to make it more general.  Add gravity turn-on option and 
  //      read from file option etc.
  ps->getWithDefault("useInitialStress", d_useInitialStress, false);
  d_init_pressure = 0.0;
  if (d_useInitialStress) {
    ps->getWithDefault("initial_pressure", d_init_pressure, 0.0);
  } 

  pProgressFLabel          = VarLabel::create("p.progressF",
                               ParticleVariable<double>::getTypeDescription());
  pProgressFLabel_preReloc = VarLabel::create("p.progressF+",
                               ParticleVariable<double>::getTypeDescription());
  pProgressdelFLabel          = VarLabel::create("p.progressdelF",
                               ParticleVariable<double>::getTypeDescription());
  pProgressdelFLabel_preReloc = VarLabel::create("p.progressdelF+",
                               ParticleVariable<double>::getTypeDescription());
}

JWLppSolid::JWLppSolid(const JWLppSolid* cm) : ConstitutiveModel(cm)
{
  d_useModifiedEOS = cm->d_useModifiedEOS ;

  d_initialData.d_K = cm->d_initialData.d_K;
  d_initialData.d_n = cm->d_initialData.d_n;

  d_initialData.d_A = cm->d_initialData.d_A;
  d_initialData.d_B = cm->d_initialData.d_B;
  d_initialData.d_C = cm->d_initialData.d_C;
  d_initialData.d_R1 = cm->d_initialData.d_R1;
  d_initialData.d_R2 = cm->d_initialData.d_R2;
  d_initialData.d_om = cm->d_initialData.d_om;
  d_initialData.d_rho0 = cm->d_initialData.d_rho0;

  d_initialData.d_G    = cm->d_initialData.d_G;
  d_initialData.d_b    = cm->d_initialData.d_b;

  // Initial stress
  d_useInitialStress = cm->d_useInitialStress;
  d_init_pressure = cm->d_init_pressure;

  pProgressFLabel          = VarLabel::create("p.progressF",
                               ParticleVariable<double>::getTypeDescription());
  pProgressFLabel_preReloc = VarLabel::create("p.progressF+",
                               ParticleVariable<double>::getTypeDescription());
  pProgressdelFLabel          = VarLabel::create("p.progressdelF",
                               ParticleVariable<double>::getTypeDescription());
  pProgressdelFLabel_preReloc = VarLabel::create("p.progressdelF+",
                               ParticleVariable<double>::getTypeDescription());
}

JWLppSolid::~JWLppSolid()
{
  VarLabel::destroy(pProgressFLabel);
  VarLabel::destroy(pProgressFLabel_preReloc);
  VarLabel::destroy(pProgressdelFLabel);
  VarLabel::destroy(pProgressdelFLabel_preReloc);
}

void JWLppSolid::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","jwlpp_mpm");
  }
  
  cm_ps->appendElement("K",    d_initialData.d_K);
  cm_ps->appendElement("n",    d_initialData.d_n);

  cm_ps->appendElement("A",    d_initialData.d_A);
  cm_ps->appendElement("B",    d_initialData.d_B);
  cm_ps->appendElement("C",    d_initialData.d_C);
  cm_ps->appendElement("R1",   d_initialData.d_R1);
  cm_ps->appendElement("R2",   d_initialData.d_R2);
  cm_ps->appendElement("om",   d_initialData.d_om);
  cm_ps->appendElement("rho0", d_initialData.d_rho0);

  cm_ps->appendElement("b", d_initialData.d_b);
  cm_ps->appendElement("G", d_initialData.d_G);

  cm_ps->appendElement("useInitialStress", d_useInitialStress);
  if (d_useInitialStress) {
    cm_ps->appendElement("initial_pressure", d_init_pressure);
  }
}

JWLppSolid* JWLppSolid::clone()
{
  return scinew JWLppSolid(*this);
}

void JWLppSolid::initializeCMData(const Patch* patch,
                             const MPMMaterial* matl,
                             DataWarehouse* new_dw)
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  //initSharedDataForExplicit(patch, matl, new_dw);

  Matrix3 Identity;
  Identity.Identity();
  Matrix3 zero(0.0);
  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  ParticleVariable<double>  pdTdt;
  ParticleVariable<Matrix3> pDefGrad;
  ParticleVariable<Matrix3> pStress;

  new_dw->allocateAndPut(pdTdt,       lb->pdTdtLabel,               pset);
  new_dw->allocateAndPut(pDefGrad,    lb->pDeformationMeasureLabel, pset);
  new_dw->allocateAndPut(pStress,     lb->pStressLabel,             pset);

  ParticleSubset::iterator iter = pset->begin();
  // Initial stress option 
  if (!d_useInitialStress) {
    for(; iter != pset->end(); iter++){
      particleIndex idx = *iter;
      pdTdt[idx] = 0.0;
      pDefGrad[idx] = Identity;
      pStress[idx] = zero;
    }
  } else {
    double p = d_init_pressure;
    Matrix3 sigInit(p, 0.0, 0.0, 0.0, p, 0.0, 0.0, 0.0, p);
    for(;iter != pset->end(); iter++){
      particleIndex idx = *iter;
      pdTdt[idx] = 0.0;
      pDefGrad[idx] = Identity;
      pStress[idx] = sigInit;
    }
  }
  
  ParticleVariable<double> pProgress;
  new_dw->allocateAndPut(pProgress,pProgressFLabel,pset);
  ParticleVariable<double> pProgressdelF;
  new_dw->allocateAndPut(pProgressdelF,pProgressdelFLabel,pset);
  for(ParticleSubset::iterator iter=pset->begin(); iter != pset->end(); iter++){
    pProgress[*iter]     = 0.0;
    pProgressdelF[*iter] = 0.0;
  }

  computeStableTimestep(patch, matl, new_dw);
}

void JWLppSolid::allocateCMDataAddRequires(Task* task,
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


void JWLppSolid::allocateCMDataAdd(DataWarehouse* new_dw,
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

void JWLppSolid::addParticleState(std::vector<const VarLabel*>& from,
                                   std::vector<const VarLabel*>& to)
{
  // Add the local particle state data for this constitutive model.
  from.push_back(pProgressFLabel);
  to.push_back(pProgressFLabel_preReloc);
  from.push_back(pProgressdelFLabel);
  to.push_back(pProgressdelFLabel_preReloc);
}

void JWLppSolid::computeStableTimestep(const Patch* patch,
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

  double K    = d_initialData.d_K;
  double n    = d_initialData.d_n;
  double rho0 = d_initialData.d_rho0;
  for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++){
     particleIndex idx = *iter;
     // Compute wave speed at each particle, store the maximum
     double rhoM = pmass[idx]/pvolume[idx];
     double dp_drho = (1./(K*rho0))*pow((rhoM/rho0),n-1.);
     c_dil = sqrt(dp_drho);
     WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                      Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                      Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
  }
  WaveSpeed = dx/WaveSpeed;
  double delT_new = WaveSpeed.minComponent();
  new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
}

void JWLppSolid::computeStressTensor(const PatchSubset* patches,
                                      const MPMMaterial* matl,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw)
{
  for(int pp=0; pp<patches->size(); pp++){
    const Patch* patch = patches->get(pp);
    double p,se  = 0.0;
    double c_dil = 0.0;
    Vector WaveSpeed(1.e-12, 1.e-12, 1.e-12);
    Matrix3 Identity;
    Identity.Identity();

    Vector dx      = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<Vector>    d_S(interpolator->size());
    vector<double>    S(interpolator->size());

    int dwi              = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    // variables to hold this timestep's values
    constParticleVariable<double>  pmass, pProgressF, pProgressdelF, pvolume_old;
    ParticleVariable<double>       pvolume;
    ParticleVariable<double>       pdTdt, p_q, pProgressF_new, pProgressdelF_new;
    constParticleVariable<Vector>  pvelocity;
    constParticleVariable<Vector>  psize;
    constParticleVariable<Point>   px;
    constParticleVariable<Matrix3> deformationGradient, pstress;
    ParticleVariable<Matrix3>      velGrad;
    ParticleVariable<Matrix3>      deformationGradient_new;
    ParticleVariable<Matrix3>      pstress_new;


    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    Ghost::GhostType  gac   = Ghost::AroundCells;
    old_dw->get(px,                  lb->pXLabel,                  pset);
    old_dw->get(pmass,               lb->pMassLabel,               pset);
    old_dw->get(pvelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);
    old_dw->get(psize,               lb->pSizeLabel,               pset);
    old_dw->get(pvolume_old,         lb->pVolumeLabel,             pset);
    old_dw->get(pstress,             lb->pStressLabel,            pset);
    old_dw->get(pProgressF,          pProgressFLabel,              pset);
    old_dw->get(pProgressdelF,       pProgressdelFLabel,           pset);
    
    new_dw->allocateAndPut(pstress_new,      lb->pStressLabel_preReloc,    pset);
    new_dw->allocateAndPut(pvolume,          lb->pVolumeLabel_preReloc,    pset);
    new_dw->allocateAndPut(pdTdt,            lb->pdTdtLabel_preReloc,      pset);
    new_dw->allocateAndPut(p_q,              lb->p_qLabel_preReloc,        pset);
    new_dw->allocateAndPut(deformationGradient_new,
                                  lb->pDeformationMeasureLabel_preReloc,   pset);

    new_dw->allocateAndPut(pProgressF_new,    pProgressFLabel_preReloc,    pset);
    new_dw->allocateAndPut(pProgressdelF_new, pProgressdelFLabel_preReloc, pset);

    new_dw->allocateTemporary(velGrad,                                     pset);

    constNCVariable<Vector> gvelocity;
    new_dw->get(gvelocity, lb->gVelocityStarLabel, dwi, patch, gac, NGN);

    double K = d_initialData.d_K;
    double n = d_initialData.d_n;
    double A = d_initialData.d_A;
    double B = d_initialData.d_B;
    double C = d_initialData.d_C;
    double R1 = d_initialData.d_R1;
    double R2 = d_initialData.d_R2;
    double om = d_initialData.d_om;
    double rho0 = d_initialData.d_rho0; // matl->getInitialDensity();

    double b = d_initialData.d_b;
    double G = d_initialData.d_G;

    // These are used in the reaction model
    double rctRho;
    double prodRho;
    double F;
    double pressure;

    if(!flag->d_doGridReset){
      cerr << "The jwlpp_mpm model doesn't work without resetting the grid"
           << endl;
    }

    for(ParticleSubset::iterator iter = pset->begin();
        iter != pset->end(); iter++){
      particleIndex idx = *iter;

      Point p = px[idx];
       
      // This is the burn logic
      pressure = 1.0/3.0 * pstress[idx].Trace();
      if( fabs(pressure) > 2.0e8)  // hard coded pressure threshold
      {
         // compute the burned fraction
         rctRho  = pmass[idx]/pvolume[idx];
         prodRho = rho0 - rctRho;  // this is suspect... Joseph 04/06/11

         F = prodRho / (rctRho + prodRho);
         if( F >= 0.0 && F < 0.99)
         {
            pProgressdelF_new[idx] = G * pow( pressure, b) * (1.0-F);
         }

         pProgressdelF_new[idx] *= delT;
         pProgressF_new[idx]     = F;
      }

      // Assign zero internal heating by default - modify if necessary.
      pdTdt[idx] = 0.0;

      Matrix3 velGrad_new(0.0);
      if(!flag->d_axisymmetric){
        // Get the node indices that surround the cell
        interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],
                                                      deformationGradient[idx]);

        computeVelocityGradient(velGrad_new,ni,d_S, oodx, gvelocity);
      } else {  // axi-symmetric kinematics
        // Get the node indices that surround the cell
        interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,
                                                            psize[idx],
                                                   deformationGradient[idx]);
        // x -> r, y -> z, z -> theta
        computeAxiSymVelocityGradient(velGrad_new,ni,d_S,S,oodx,gvelocity,
                                                                  px[idx]);
      }

      deformationGradient_new[idx]=(velGrad_new*delT+Identity)
                                    *deformationGradient[idx];
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
        double rho_cur = rho0/J;
        pvolume[idx] = pmass[idx]/rho_cur;

        IntVector cell_index;
        patch->findCell(px[idx],cell_index);

        vol_CC[cell_index]  +=pvolume[idx];
        vol_0_CC[cell_index]+=pmass[idx]/rho0;
      }

      for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++){
        IntVector c = *iter;
        J_CC[c]=vol_CC[c]/vol_0_CC[c];
      }
    } //end of pressureStabilization loop  at the patch level

    for(ParticleSubset::iterator iter = pset->begin();
        iter != pset->end(); iter++){
      particleIndex idx = *iter;

      double J = deformationGradient_new[idx].Determinant();
      if (!(J > 0.0)) {
        cerr << "**ERROR in JWL++MPM** Negative Jacobian of deformation gradient" << endl;
        cerr << "idx = " << idx << " J = " << J << " matl = " << matl << endl;
        cerr << "F_old = " << deformationGradient[idx]     << endl;
        cerr << "F_new = " << deformationGradient_new[idx] << endl;
        cerr << "VelGrad = " << velGrad[idx] << endl;
      }

      // More Pressure Stabilization
      if(flag->d_doPressureStabilization) {
        IntVector cell_index;
        patch->findCell(px[idx],cell_index);

        // Change F such that the determinant is equal to the average for
        // the cell
        deformationGradient_new[idx]*=cbrt(J_CC[cell_index])/cbrt(J);
        J=J_CC[cell_index];
      }

      //  The following computes a pressure for partially burned particles
      //  as a mixture of Murnahan and JWL pressures, based on pProgressF
      //  This is as described in Eq. 5 of "JWL++: ..." by Souers, et al.
      double pM = (1./(n*K))*(pow(J,-n)-1.);
      double pJWL=pM;
      if(pProgressF_new[idx] > 0.0){
        double one_plus_omega = 1.+om;
        double inv_rho_rat=J; //rho0/rhoM;
        double rho_rat=1./J;  //rhoM/rho0;
        double A_e_to_the_R1_rho0_over_rhoM=A*exp(-R1*inv_rho_rat);
        double B_e_to_the_R2_rho0_over_rhoM=B*exp(-R2*inv_rho_rat);
        double C_rho_rat_tothe_one_plus_omega=C*pow(rho_rat,one_plus_omega);

        pJWL  = A_e_to_the_R1_rho0_over_rhoM +
                B_e_to_the_R2_rho0_over_rhoM +
                C_rho_rat_tothe_one_plus_omega;
      }

      p = pM*(1.0-pProgressF_new[idx]) + pJWL*pProgressF_new[idx];

      // Get the deformed volume and current density
      double rho_cur = rho0/J;
      pvolume[idx] = pmass[idx]/rho_cur;

      // compute the total stress
      pstress_new[idx] = Identity*(-p);

      Vector pvelocity_idx = pvelocity[idx];

      // Compute wave speed at each particle, store the maximum
      double dp_drho = (1./(K*rho0))*pow((rho_cur/rho0),n-1.);
      c_dil = sqrt(dp_drho);
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity_idx.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pvelocity_idx.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pvelocity_idx.z()),WaveSpeed.z()));
                                                                                
      // Compute artificial viscosity term
      if (flag->d_artificial_viscosity) {
        double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;
        double c_bulk = sqrt(1./(K*rho_cur));
        Matrix3 D=(velGrad[idx] + velGrad[idx].Transpose())*0.5;
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
      new_dw->put(sum_vartype(se),      lb->StrainEnergyLabel);
    }
    delete interpolator;
  }
}

void JWLppSolid::carryForward(const PatchSubset* patches,
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

void JWLppSolid::addComputesAndRequires(Task* task,
                                          const MPMMaterial* matl,
                                          const PatchSet* patches) const
{
  // Add the computes and requires that are common to all explicit 
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForHypoExplicit(task, matlset, patches);

  task->requires(Task::OldDW, pProgressFLabel,    matlset, Ghost::None);
  task->computes(pProgressFLabel_preReloc,        matlset);
  task->requires(Task::OldDW, pProgressdelFLabel, matlset, Ghost::None);
  task->computes(pProgressdelFLabel_preReloc,     matlset);
}

void JWLppSolid::addInitialComputesAndRequires(Task* task,
                                         const MPMMaterial* matl,
                                         const PatchSet*) const
{ 
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(pProgressFLabel,       matlset);
  task->computes(pProgressdelFLabel,    matlset);
}

void 
JWLppSolid::addComputesAndRequires(Task* ,
                                   const MPMMaterial* ,
                                   const PatchSet* ,
                                   const bool ) const
{
}


// This is not yet implemented - JG- 7/26/10
double JWLppSolid::computeRhoMicroCM(double pressure, 
                                      const double p_ref,
                                      const MPMMaterial* matl,
                                      double temperature,
                                      double rho_guess)
{
    cout << "NO VERSION OF computeRhoMicroCM EXISTS YET FOR JWLppSolid"
       << endl;
    double rho_orig = d_initialData.d_rho0; //matl->getInitialDensity();

    return rho_orig;
}

void JWLppSolid::computePressEOSCM(const double rhoM,double& pressure, 
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
double JWLppSolid::getCompressibility()
{
   cout << "NO VERSION OF getCompressibility EXISTS YET FOR JWLppSolid"<< endl;
  return 1.0;
}

namespace Uintah {
} // End namespace Uintah
