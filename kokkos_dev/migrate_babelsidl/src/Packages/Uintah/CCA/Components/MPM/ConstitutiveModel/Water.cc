#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/Water.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Core/Math/MinMax.h>
#include <Core/Malloc/Allocator.h>
#include <sgi_stl_warnings_off.h>
#include <fstream>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using std::cerr;
using namespace Uintah;
using namespace SCIRun;

Water::Water(ProblemSpecP& ps, MPMFlags* Mflag)
  : ConstitutiveModel(Mflag)
{

  d_useModifiedEOS = false;
  ps->require("bulk_modulus", d_initialData.d_Bulk);
  ps->require("viscosity",    d_initialData.d_Viscosity);
  ps->require("gamma",        d_initialData.d_Gamma);
}

Water::Water(const Water* cm) : ConstitutiveModel(cm)
{
  d_useModifiedEOS = cm->d_useModifiedEOS ;
  d_initialData.d_Bulk = cm->d_initialData.d_Bulk;
  d_initialData.d_Viscosity = cm->d_initialData.d_Viscosity;
  d_initialData.d_Gamma = cm->d_initialData.d_Gamma;
}

Water::~Water()
{
}

void Water::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","water");
  }
  
  cm_ps->appendElement("bulk_modulus",d_initialData.d_Bulk);
  cm_ps->appendElement("viscosity",   d_initialData.d_Viscosity);
  cm_ps->appendElement("gamma",       d_initialData.d_Gamma);
}

Water* Water::clone()
{
  return scinew Water(*this);
}

void Water::initializeCMData(const Patch* patch,
                             const MPMMaterial* matl,
                             DataWarehouse* new_dw)
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  initSharedDataForExplicit(patch, matl, new_dw);

  computeStableTimestep(patch, matl, new_dw);
}

void Water::allocateCMDataAddRequires(Task* task,
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


void Water::allocateCMDataAdd(DataWarehouse* new_dw,
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

void Water::addParticleState(std::vector<const VarLabel*>& ,
                                   std::vector<const VarLabel*>& )
{
  // Add the local particle state data for this constitutive model.
}

void Water::computeStableTimestep(const Patch* patch,
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

  double bulk = d_initialData.d_Bulk;
  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
     particleIndex idx = *iter;
     // Compute wave speed at each particle, store the maximum
     c_dil = sqrt((bulk)*pvolume[idx]/pmass[idx]);
     WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                      Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                      Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
  }
  WaveSpeed = dx/WaveSpeed;
  double delT_new = WaveSpeed.minComponent();
  new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
}

void Water::computeStressTensor(const PatchSubset* patches,
                                      const MPMMaterial* matl,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw)
{
  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);
    Matrix3 velGrad,Shear;
    double p,se=0.;
    double c_dil=0.0;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
    double onethird = (1.0/3.0);
    Matrix3 Identity;
    Identity.Identity();

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<Vector> d_S(interpolator->size());

    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    constParticleVariable<Point> px;
    ParticleVariable<Matrix3> deformationGradient_new;
    constParticleVariable<Matrix3> deformationGradient;
    ParticleVariable<Matrix3> pstress;
    constParticleVariable<double> pmass,pvolume;
    ParticleVariable<double> pvolume_deformed;
    constParticleVariable<Vector> pvelocity;
    constParticleVariable<Vector> psize;
    ParticleVariable<double> pdTdt;
    delt_vartype delT;

    Ghost::GhostType  gac   = Ghost::AroundCells;
    old_dw->get(px,                  lb->pXLabel,                  pset);
    old_dw->get(pmass,               lb->pMassLabel,               pset);
    old_dw->get(pvelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(pvolume,             lb->pVolumeLabel,             pset);
    old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);
    old_dw->get(psize,               lb->pSizeLabel,               pset);
    
    new_dw->allocateAndPut(pstress,          lb->pStressLabel_preReloc,  pset);
    new_dw->allocateAndPut(pvolume_deformed, lb->pVolumeDeformedLabel,   pset);
    new_dw->allocateAndPut(pdTdt,            lb->pdTdtLabel_preReloc,    pset);

    new_dw->allocateAndPut(deformationGradient_new,
                                  lb->pDeformationMeasureLabel_preReloc, pset);

    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    double viscosity = d_initialData.d_Viscosity;
    double bulk  = d_initialData.d_Bulk;
    double gamma = d_initialData.d_Gamma;

    double rho_orig = matl->getInitialDensity();

    constNCVariable<Vector> gvelocity;
    new_dw->get(gvelocity, lb->gVelocityLabel,dwi,patch,gac,NGN);

    if(flag->d_doGridReset){
      computeDeformationGradientFromVelocity(gvelocity,
                                             pset, px, psize,
                                             deformationGradient,
                                             deformationGradient_new,
                                             dx, interpolator, delT);
    }
    else{
      cerr << "The water model doesn't work without resetting the grid" << endl;
    }

    for(ParticleSubset::iterator iter = pset->begin();
        iter != pset->end(); iter++){
      particleIndex idx = *iter;
      
      // Assign zero internal heating by default - modify if necessary.
      pdTdt[idx] = 0.0;

      interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx]);

      computeGrad(velGrad, ni, d_S, oodx, gvelocity);
                                                                                
      double Jinc = (velGrad * delT + Identity).Determinant();

      // Calculate rate of deformation D, and deviatoric rate DPrime,
      Matrix3 D = (velGrad + velGrad.Transpose())*0.5;
      Matrix3 DPrime = D - Identity*onethird*D.Trace();

      // Get the deformed volume and current density
      pvolume_deformed[idx]=pvolume[idx]*Jinc;
      double rho_cur = pmass[idx]/pvolume_deformed[idx];
      double vol_orig = pmass[idx]/rho_orig;

      // Viscous part of the stress
      Shear = DPrime*(2.*viscosity);

      // get the hydrostatic part of the stress
      p = bulk*(pow(vol_orig/pvolume_deformed[idx],gamma) - 1.0);

      // compute the total stress (volumetric + deviatoric)
      pstress[idx] = Identity*(-p) + Shear;

      Vector pvelocity_idx = pvelocity[idx];
      c_dil = sqrt((bulk)/rho_cur);
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity_idx.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pvelocity_idx.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pvelocity_idx.z()),WaveSpeed.z()));
    }

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
    new_dw->put(sum_vartype(se),        lb->StrainEnergyLabel);

    delete interpolator;
  }
}


void Water::carryForward(const PatchSubset* patches,
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
    new_dw->put(sum_vartype(0.),     lb->StrainEnergyLabel);
  }
}

void Water::addComputesAndRequires(Task* task,
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
Water::addComputesAndRequires(Task* ,
                                   const MPMMaterial* ,
                                   const PatchSet* ,
                                   const bool ) const
{
}


// The "CM" versions use the pressure-volume relationship of the CNH model
double Water::computeRhoMicroCM(double pressure, 
                                      const double p_ref,
                                      const MPMMaterial* matl)
{
  double rho_orig = matl->getInitialDensity();
  double bulk = d_initialData.d_Bulk;
  
  double p_gauge = pressure - p_ref;
  double rho_cur;
 
  double p_g_over_bulk = p_gauge/bulk;
  rho_cur=rho_orig*(p_g_over_bulk + sqrt(p_g_over_bulk*p_g_over_bulk +1.));

  return rho_cur;
}

void Water::computePressEOSCM(const double rho_cur,double& pressure, 
                                    const double p_ref,
                                    double& dp_drho, double& tmp,
                                    const MPMMaterial* matl)
{
  double bulk = d_initialData.d_Bulk;
  double rho_orig = matl->getInitialDensity();

  double p_g = .5*bulk*(rho_cur/rho_orig - rho_orig/rho_cur);
  pressure   = p_ref + p_g;
  dp_drho    = .5*bulk*(rho_orig/(rho_cur*rho_cur) + 1./rho_orig);
  tmp        = bulk/rho_cur;  // speed of sound squared
}

double Water::getCompressibility()
{
  return 1.0/d_initialData.d_Bulk;
}

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1209
#endif

namespace Uintah {
  
#if 0
  static MPI_Datatype makeMPI_CMData()
  {
    ASSERTEQ(sizeof(Water::StateData), sizeof(double)*0);
    MPI_Datatype mpitype;
    MPI_Type_vector(1, 0, 0, MPI_DOUBLE, &mpitype);
    MPI_Type_commit(&mpitype);
    return mpitype;
  }
  
  const TypeDescription* fun_getTypeDescription(Water::StateData*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
                                  "Water::StateData", 
                                  true, &makeMPI_CMData);
    }
    return td;
  }
#endif
} // End namespace Uintah
