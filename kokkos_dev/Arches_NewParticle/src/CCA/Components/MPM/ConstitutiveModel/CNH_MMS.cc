#include <CCA/Components/MPM/ConstitutiveModel/CNH_MMS.h>
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

CNH_MMS::CNH_MMS(ProblemSpecP& ps, MPMFlags* Mflag)
  : ConstitutiveModel(Mflag)
{
  d_useModifiedEOS = false;
  ps->require("bulk_modulus", d_initialData.Bulk);
  ps->require("shear_modulus",d_initialData.Shear);
  ps->get("useModifiedEOS",d_useModifiedEOS); 
}

CNH_MMS::CNH_MMS(const CNH_MMS* cm) : ConstitutiveModel(cm)
{
  d_useModifiedEOS = cm->d_useModifiedEOS ;
  d_initialData.Bulk = cm->d_initialData.Bulk;
  d_initialData.Shear = cm->d_initialData.Shear;
}

CNH_MMS::~CNH_MMS()
{
}

void CNH_MMS::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","cnh_mms");
  }
  
  cm_ps->appendElement("bulk_modulus",d_initialData.Bulk);
  cm_ps->appendElement("shear_modulus",d_initialData.Shear);
  cm_ps->appendElement("useModifiedEOS",d_useModifiedEOS);
}

CNH_MMS* CNH_MMS::clone()
{
  return scinew CNH_MMS(*this);
}

void CNH_MMS::initializeCMData(const Patch* patch,
                               const MPMMaterial* matl,
                               DataWarehouse* new_dw)
{
  Matrix3 I; I.Identity();
  Matrix3 zero(0.);
  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);
                                                                                
  ParticleVariable<double>  pdTdt;
  ParticleVariable<Matrix3> pDefGrad, pStress;
  constParticleVariable<Point>  px;
  constParticleVariable<Vector>  pdisp;
  double mu = d_initialData.Shear;
  double bulk = d_initialData.Bulk;
  double lambda = (3.*bulk-2.*mu)/3.;
//  double E = 9.*bulk*mu/(3.*bulk+mu);
//  double rho0 = matl->getInitialDensity();
//  double c = E/rho0;
  double A=.1;
                                                                                
  new_dw->allocateAndPut(pdTdt,       lb->pdTdtLabel,               pset);
  new_dw->allocateAndPut(pDefGrad,    lb->pDeformationMeasureLabel, pset);
  new_dw->allocateAndPut(pStress,     lb->pStressLabel,             pset);
  new_dw->get(px,                     lb->pXLabel,                  pset);
  new_dw->get(pdisp,                  lb->pDispLabel,               pset);
                                                                                
  // To fix : For a material that is initially stressed we need to
  // modify the stress tensors to comply with the initial stress state
  ParticleSubset::iterator iter = pset->begin();
  for(; iter != pset->end(); iter++){
    particleIndex idx = *iter;
    pdTdt[idx] = 0.0;
    Point X=px[idx]-pdisp[idx];
    double Fxx=1.;
    double Fyy=1.+A*M_PI*cos(M_PI*X.y())*sin(2./3.*M_PI);
    double Fzz=1.+A*M_PI*cos(M_PI*X.z())*sin(4./3.*M_PI);
    pDefGrad[idx] = Matrix3(Fxx,0.,0.,0.,Fyy,0.,0.,0.,Fzz);

    double J=pDefGrad[idx].Determinant();
    Matrix3 Shear= (pDefGrad[idx]*pDefGrad[idx].Transpose() - I)*mu;

    double p = lambda*log(J);
//    pStress[idx] = Matrix3(2.*mu*(J-1./J),0.,0.,0.,0.,0.,0.,0.,0.);
    pStress[idx] = (I*p + Shear)/J;
  }

  computeStableTimestep(patch, matl, new_dw);
}

void CNH_MMS::allocateCMDataAddRequires(Task* task,
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


void CNH_MMS::allocateCMDataAdd(DataWarehouse* new_dw,
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

void CNH_MMS::addParticleState(std::vector<const VarLabel*>& ,
                                   std::vector<const VarLabel*>& )
{
  // Add the local particle state data for this constitutive model.
}

void CNH_MMS::computeStableTimestep(const Patch* patch,
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

  new_dw->get(pmass,     lb->pMassLabel, pset);
  new_dw->get(pvolume,   lb->pVolumeLabel, pset);
  new_dw->get(pvelocity, lb->pVelocityLabel, pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double mu = d_initialData.Shear;
  double bulk = d_initialData.Bulk;
  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
     particleIndex idx = *iter;

     // Compute wave speed at each particle, store the maximum
     c_dil = sqrt((bulk + 4.*mu/3.)*pvolume[idx]/pmass[idx]);
     WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                      Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                      Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
  }
  WaveSpeed = dx/WaveSpeed;
  double delT_new = WaveSpeed.minComponent();
  new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
}

void CNH_MMS::computeStressTensor(const PatchSubset* patches,
                                      const MPMMaterial* matl,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw)
{
  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
    Matrix3 Identity;
    Identity.Identity();

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);

    Vector dx = patch->dCell();

    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    constParticleVariable<Point> px;
    ParticleVariable<Matrix3> deformationGradient_new;
    constParticleVariable<Matrix3> deformationGradient;
    ParticleVariable<Matrix3> pstress;
    constParticleVariable<double> pmass,pcolor;
    ParticleVariable<double> pvolume_new;
    constParticleVariable<Vector> pvelocity;
    constParticleVariable<Vector> psize;
    ParticleVariable<double> pdTdt;
    delt_vartype delT;

    Ghost::GhostType  gac   = Ghost::AroundCells;
    old_dw->get(px,                  lb->pXLabel,                  pset);
    old_dw->get(pmass,               lb->pMassLabel,               pset);
    old_dw->get(pvelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);
    old_dw->get(psize,               lb->pSizeLabel,               pset);
    
    new_dw->allocateAndPut(pstress,     lb->pStressLabel_preReloc, pset);
    new_dw->allocateAndPut(pvolume_new, lb->pVolumeLabel_preReloc, pset);
    new_dw->allocateAndPut(pdTdt,       lb->pdTdtLabel_preReloc,   pset);
    if(flag->d_with_color) {
      old_dw->get(pcolor,      lb->pColorLabel,  pset);
    }

    new_dw->allocateAndPut(deformationGradient_new,
                                  lb->pDeformationMeasureLabel_preReloc, pset);

    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    double shear = d_initialData.Shear;
    double bulk  = d_initialData.Bulk;

    double lambda = (3.*bulk-2.*shear)/3.;
    double mu =   shear;

    double rho_orig = matl->getInitialDensity();

    if(flag->d_doGridReset){
      constNCVariable<Vector> gvelocity;
      new_dw->get(gvelocity, lb->gVelocityStarLabel,dwi,patch,gac,NGN);
      computeDeformationGradientFromVelocity(gvelocity,
                                             pset, px, psize,
                                             deformationGradient,
                                             deformationGradient_new,
                                             dx, interpolator, delT);
    }
    else if(!flag->d_doGridReset){
      constNCVariable<Vector> gdisplacement;
      new_dw->get(gdisplacement, lb->gDisplacementLabel,dwi,patch,gac,NGN);
      computeDeformationGradientFromDisplacement(gdisplacement,
                                                 pset, px, psize,
                                                 deformationGradient_new,
                                                 deformationGradient,
                                                 dx, interpolator);
    }

    for(ParticleSubset::iterator iter = pset->begin();
        iter != pset->end(); iter++){
      particleIndex idx = *iter;
      
      // Assign zero internal heating by default - modify if necessary.
      pdTdt[idx] = 0.0;

      // get the volumetric part of the deformation
      double J = deformationGradient_new[idx].Determinant();

      // Get the deformed volume
      pvolume_new[idx]=(pmass[idx]/rho_orig)*J;

      // Compute local wave speed
      double rho_cur = rho_orig/J;
      double c_dil = sqrt((bulk + 4.*shear/3.)/rho_cur);

      Matrix3 Shear = 
          (deformationGradient_new[idx]*deformationGradient_new[idx].Transpose()             - Identity)*mu;

      // get the hydrostatic part of the stress (times J)
      double p = lambda*log(J);

      // compute the total stress (volumetric + deviatoric)
      pstress[idx] = (Identity*p + Shear)/J;

      Vector pvelocity_idx = pvelocity[idx];
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity_idx.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pvelocity_idx.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pvelocity_idx.z()),WaveSpeed.z()));
    }

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    double se = 0;
    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
    
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(se),        lb->StrainEnergyLabel);
    }
    delete interpolator;
  }
}

void CNH_MMS::carryForward(const PatchSubset* patches,
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
      new_dw->put(sum_vartype(0.),     lb->StrainEnergyLabel);
    }
  }
}

void CNH_MMS::addComputesAndRequires(Task* task,
                                          const MPMMaterial* matl,
                                          const PatchSet* patches) const
{
  // Add the computes and requires that are common to all explicit 
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForExplicit(task, matlset, patches);
  if(flag->d_with_color) {
    task->requires(Task::OldDW, lb->pColorLabel,  Ghost::None);
  }
}

void 
CNH_MMS::addComputesAndRequires(Task* ,
                                   const MPMMaterial* ,
                                   const PatchSet* ,
                                   const bool ) const
{
}

// The "CM" versions use the pressure-volume relationship of the CNH model
double CNH_MMS::computeRhoMicroCM(double pressure, 
                                      const double p_ref,
                                      const MPMMaterial* matl,
                                      double temperature,
                                      double rho_guess)
{
  double rho_orig = matl->getInitialDensity();
  double bulk = d_initialData.Bulk;
  
  double p_gauge = pressure - p_ref;
  double rho_cur;
 
  if(d_useModifiedEOS && p_gauge < 0.0) {
    double A = p_ref;           // MODIFIED EOS
    double n = p_ref/bulk;
    rho_cur = rho_orig*pow(pressure/A,n);
  } else {                      // STANDARD EOS
    double p_g_over_bulk = p_gauge/bulk;
    rho_cur=rho_orig*(p_g_over_bulk + sqrt(p_g_over_bulk*p_g_over_bulk +1.));
  }
  return rho_cur;
}

void CNH_MMS::computePressEOSCM(const double rho_cur,double& pressure, 
                                    const double p_ref,
                                    double& dp_drho, double& tmp,
                                    const MPMMaterial* matl,
                                    double temperature)
{
  double bulk = d_initialData.Bulk;
  double rho_orig = matl->getInitialDensity();

  if(d_useModifiedEOS && rho_cur < rho_orig){
    double A = p_ref;           // MODIFIED EOS
    double n = bulk/p_ref;
    double rho_rat_to_the_n = pow(rho_cur/rho_orig,n);
    pressure = A*rho_rat_to_the_n;
    dp_drho  = (bulk/rho_cur)*rho_rat_to_the_n;
    tmp      = dp_drho;         // speed of sound squared
  } else {                      // STANDARD EOS            
    double p_g = .5*bulk*(rho_cur/rho_orig - rho_orig/rho_cur);
    pressure   = p_ref + p_g;
    dp_drho    = .5*bulk*(rho_orig/(rho_cur*rho_cur) + 1./rho_orig);
    tmp        = bulk/rho_cur;  // speed of sound squared
  }
}

double CNH_MMS::getCompressibility()
{
  return 1.0/d_initialData.Bulk;
}


namespace Uintah {
  
#if 0
  static MPI_Datatype makeMPI_CMData()
  {
    ASSERTEQ(sizeof(CNH_MMS::StateData), sizeof(double)*0);
    MPI_Datatype mpitype;
    MPI_Type_vector(1, 0, 0, MPI_DOUBLE, &mpitype);
    MPI_Type_commit(&mpitype);
    return mpitype;
  }
  
  const TypeDescription* fun_getTypeDescription(CNH_MMS::StateData*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
                                  "CNH_MMS::StateData", 
                                  true, &makeMPI_CMData);
    }
    return td;
  }
#endif
} // End namespace Uintah
