#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/CompNeoHookPlas.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
#include <Core/Math/MinMax.h>
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/Short27.h> //for Fracture
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h> 
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Core/Malloc/Allocator.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

using std::cerr;
using namespace Uintah;
using namespace SCIRun;

CompNeoHookPlas::CompNeoHookPlas(ProblemSpecP& ps, MPMLabel* Mlb, 
                                 MPMFlags* Mflag)
  : ConstitutiveModel(Mlb,Mflag)
{
  
  d_useModifiedEOS = false;
  ps->require("bulk_modulus",d_initialData.Bulk);
  ps->require("shear_modulus",d_initialData.Shear);
  ps->require("yield_stress",d_initialData.FlowStress);
  ps->require("hardening_modulus",d_initialData.K);
  ps->require("alpha",d_initialData.Alpha);
  ps->get("useModifiedEOS",d_useModifiedEOS);
  
  p_statedata_label = VarLabel::create("p.statedata_cnhp",
                             ParticleVariable<StateData>::getTypeDescription());
  p_statedata_label_preReloc = VarLabel::create("p.statedata_cnhp+",
                             ParticleVariable<StateData>::getTypeDescription());
 
  bElBarLabel = VarLabel::create("p.bElBar",
                ParticleVariable<Matrix3>::getTypeDescription());
  bElBarLabel_preReloc = VarLabel::create("p.bElBar+",
                ParticleVariable<Matrix3>::getTypeDescription());

}

CompNeoHookPlas::CompNeoHookPlas(const CompNeoHookPlas* cm)
{
  lb = cm->lb;
  flag = cm->flag;
  NGN = cm->NGN;
  d_useModifiedEOS = cm->d_useModifiedEOS ;
  d_initialData.Bulk = cm->d_initialData.Bulk;
  d_initialData.Shear = cm->d_initialData.Shear;
  d_initialData.FlowStress = cm->d_initialData.FlowStress;
  d_initialData.K = cm->d_initialData.K;
  d_initialData.Alpha = cm->d_initialData.Alpha;
  
  p_statedata_label = VarLabel::create("p.statedata_cnhp",
                             ParticleVariable<StateData>::getTypeDescription());
  p_statedata_label_preReloc = VarLabel::create("p.statedata_cnhp+",
                             ParticleVariable<StateData>::getTypeDescription());
 
  bElBarLabel = VarLabel::create("p.bElBar",
                ParticleVariable<Matrix3>::getTypeDescription());
  bElBarLabel_preReloc = VarLabel::create("p.bElBar+",
                ParticleVariable<Matrix3>::getTypeDescription());
}

void CompNeoHookPlas::addParticleState(std::vector<const VarLabel*>& from,
                                       std::vector<const VarLabel*>& to)
{
  // Add the local particle state data for this constitutive model.
  from.push_back(p_statedata_label);
  from.push_back(bElBarLabel);

  to.push_back(p_statedata_label_preReloc);
  to.push_back(bElBarLabel_preReloc);
}

CompNeoHookPlas::~CompNeoHookPlas()
{
  // Destructor 
  VarLabel::destroy(p_statedata_label);
  VarLabel::destroy(p_statedata_label_preReloc);
  VarLabel::destroy(bElBarLabel);
  VarLabel::destroy(bElBarLabel_preReloc);
  
}

void CompNeoHookPlas::initializeCMData(const Patch* patch,
                                        const MPMMaterial* matl,
                                        DataWarehouse* new_dw)
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  initSharedDataForExplicit(patch, matl, new_dw);

  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  Matrix3 Identity, zero(0.);
  Identity.Identity();

  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);
  ParticleVariable<StateData> statedata;
  ParticleVariable<Matrix3> bElBar;

  new_dw->allocateAndPut(statedata, p_statedata_label,  pset);
  new_dw->allocateAndPut(bElBar,    bElBarLabel,        pset);

  ParticleSubset::iterator iter = pset->begin();
  for(;iter != pset->end(); iter++){
    statedata[*iter].Alpha = d_initialData.Alpha;
    bElBar[*iter] = Identity;
  }

  computeStableTimestep(patch, matl, new_dw);
}

void CompNeoHookPlas::allocateCMDataAddRequires(Task* task,
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
  task->requires(Task::NewDW,p_statedata_label_preReloc, 
                 matlset, Ghost::None);
  task->requires(Task::NewDW,bElBarLabel_preReloc, 
                 matlset, Ghost::None);
}


void CompNeoHookPlas::allocateCMDataAdd(DataWarehouse* new_dw,
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
  ParticleVariable<StateData> statedata;
  ParticleVariable<Matrix3>   bElBar;

  constParticleVariable<StateData> o_statedata;
  constParticleVariable<Matrix3>   o_bElBar;

  new_dw->allocateTemporary(statedata,addset);
  new_dw->allocateTemporary(bElBar,addset);

  new_dw->get(o_statedata,p_statedata_label_preReloc,delset);
  new_dw->get(o_bElBar,bElBarLabel_preReloc,delset);

  ParticleSubset::iterator o,n = addset->begin();
  for (o=delset->begin(); o != delset->end(); o++, n++) {
    statedata[*n].Alpha = o_statedata[*o].Alpha;
    bElBar[*n] = o_bElBar[*o];
  }

  (*newState)[p_statedata_label]=statedata.clone();
  (*newState)[ bElBarLabel]=bElBar.clone();
}

void CompNeoHookPlas::computeStableTimestep(const Patch* patch,
                                             const MPMMaterial* matl,
                                             DataWarehouse* new_dw)
{
   // This is only called for the initial timestep - all other timesteps
   // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int matlindex = matl->getDWIndex();
  // Retrieve the array of constitutive parameters
  ParticleSubset* pset = new_dw->getParticleSubset(matlindex, patch);
  constParticleVariable<StateData> statedata;
  constParticleVariable<double> pmass, pvolume;
  constParticleVariable<Vector> pvelocity;

  new_dw->get(statedata, p_statedata_label,  pset);
  new_dw->get(pmass,     lb->pMassLabel,     pset);
  new_dw->get(pvolume,   lb->pVolumeLabel,   pset);
  new_dw->get(pvelocity, lb->pVelocityLabel, pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double mu = d_initialData.Shear;
  double bulk = d_initialData.Bulk;
  for(ParticleSubset::iterator iter = pset->begin();
      iter != pset->end(); iter++){
     particleIndex idx = *iter;

     // Compute wave speed at each particle, store the maximum
     Vector pvelocity_idx = pvelocity[idx];
     if(pmass[idx] > 0){
       c_dil = sqrt((bulk + 4.*mu/3.)*pvolume[idx]/pmass[idx]);
     }
     else{
       c_dil = 0.0;
       pvelocity_idx = Vector(0.0,0.0,0.0);
     }
     WaveSpeed=Vector(Max(c_dil+fabs(pvelocity_idx.x()),WaveSpeed.x()),
                      Max(c_dil+fabs(pvelocity_idx.y()),WaveSpeed.y()),
                      Max(c_dil+fabs(pvelocity_idx.z()),WaveSpeed.z()));
  }

  WaveSpeed = dx/WaveSpeed;
  double delT_new = WaveSpeed.minComponent();
  new_dw->put(delt_vartype(patch->getLevel()->adjustDelt(delT_new)), 
              lb->delTLabel);
}

void CompNeoHookPlas::computeStressTensor(const PatchSubset* patches,
                                          const MPMMaterial* matl,
                                          DataWarehouse* old_dw,
                                          DataWarehouse* new_dw)
{
  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);

    Matrix3 bElBarTrial,deformationGradientInc,Identity;
    Matrix3 shearTrial,Shear,normal,fbar,velGrad;
    double J,p,fTrial,IEl,muBar,delgamma,sTnorm,Jinc,U,W;
    double onethird = (1.0/3.0),sqtwthds = sqrt(2.0/3.0), c_dil = 0.0,se = 0.;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni;
    ni.reserve(interpolator->size());
    vector<Vector> d_S;
    d_S.reserve(interpolator->size());



    Identity.Identity();

    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};
    //double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;

    int matlindex = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch);
    constParticleVariable<Point> px;
    ParticleVariable<Matrix3> deformationGradient_new, bElBar_new;
    constParticleVariable<Matrix3> deformationGradient, bElBar;
    ParticleVariable<Matrix3> pstress;
    constParticleVariable<StateData> statedata_old;
    ParticleVariable<StateData> statedata;
    constParticleVariable<double> pmass;
    ParticleVariable<double> pvolume_deformed;
    constParticleVariable<Vector> pvelocity;
    constNCVariable<Vector> gvelocity;
    delt_vartype delT;
    constParticleVariable<Vector> psize;

    Ghost::GhostType  gac   = Ghost::AroundCells;

    old_dw->get(psize,               lb->pSizeLabel,                   pset);
    
    old_dw->get(px,                    lb->pXLabel,                      pset);
    old_dw->get(bElBar,                bElBarLabel,                      pset);
    old_dw->get(statedata_old,         p_statedata_label,                pset);
    old_dw->get(pmass,                 lb->pMassLabel,                   pset);
    old_dw->get(pvelocity,             lb->pVelocityLabel,               pset);
    old_dw->get(deformationGradient,   lb->pDeformationMeasureLabel,     pset);
    new_dw->allocateAndPut(pstress,    lb->pStressLabel_preReloc,        pset);
    new_dw->allocateAndPut(bElBar_new, bElBarLabel_preReloc,             pset);
    new_dw->allocateAndPut(statedata,  p_statedata_label_preReloc,       pset);
    new_dw->allocateAndPut(pvolume_deformed, lb->pVolumeDeformedLabel,   pset);
    new_dw->allocateAndPut(deformationGradient_new,
                                  lb->pDeformationMeasureLabel_preReloc, pset);
    statedata.copyData(statedata_old);

    new_dw->get(gvelocity, lb->gVelocityLabel, matlindex,patch, gac, NGN);
    old_dw->get(delT, lb->delTLabel,getLevel(patches));

    constParticleVariable<Short27> pgCode;
    constNCVariable<Vector> Gvelocity;
    if (flag->d_fracture) {
      new_dw->get(pgCode, lb->pgCodeLabel, pset);
      new_dw->get(Gvelocity,lb->GVelocityLabel, matlindex, patch, gac, NGN);
    }

    constParticleVariable<int> pConnectivity;
    ParticleVariable<Vector> pRotationRate;
    ParticleVariable<double> pStrainEnergy;

    // Allocate variable to store internal heating rate
    ParticleVariable<double> pIntHeatRate;
    new_dw->allocateAndPut(pIntHeatRate, lb->pInternalHeatRateLabel_preReloc, 
                           pset);

    double shear = d_initialData.Shear;
    double bulk  = d_initialData.Bulk;
    double flow  = d_initialData.FlowStress;
    double K     = d_initialData.K;

    double rho_orig = matl->getInitialDensity();

    for(ParticleSubset::iterator iter = pset->begin();
        iter != pset->end(); iter++){
       particleIndex idx = *iter;

      // Assign zero internal heating by default - modify if necessary.
      pIntHeatRate[idx] = 0.0;

       // Get the node indices that surround the cell
      interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx]);

       Vector gvel;
       velGrad.set(0.0);
       for(int k = 0; k < flag->d_8or27; k++) {
         if (flag->d_fracture) {
           if(pgCode[idx][k]==1) gvel = gvelocity[ni[k]];
           if(pgCode[idx][k]==2) gvel = Gvelocity[ni[k]];
         } else
           gvel = gvelocity[ni[k]];
         for (int j = 0; j<3; j++){
           double d_SXoodx = d_S[k][j] * oodx[j];
           for (int i = 0; i<3; i++) {
             velGrad(i,j) += gvel[i] * d_SXoodx;
           }
         }
       }
       
      // Calculate the stress Tensor (symmetric 3 x 3 Matrix) given the
      // time step and the velocity gradient and the material constants
      double alpha = statedata[idx].Alpha;

      // Compute the deformation gradient increment using the time_step
      // velocity gradient
      // F_n^np1 = dudx * dt + Identity
      deformationGradientInc = velGrad * delT + Identity;

      Jinc = deformationGradientInc.Determinant();

      // Update the deformation gradient tensor to its time n+1 value.
      deformationGradient_new[idx] = deformationGradientInc *
                                     deformationGradient[idx];

      // get the volume preserving part of the deformation gradient increment
      fbar = deformationGradientInc * pow(Jinc,-onethird);

      // predict the elastic part of the volume preserving part of the left
      // Cauchy-Green deformation tensor
      bElBarTrial = fbar*bElBar[idx]*fbar.Transpose();
      IEl = onethird*bElBarTrial.Trace();

      // shearTrial is equal to the shear modulus times dev(bElBar)
      shearTrial = (bElBarTrial - Identity*IEl)*shear;

      // get the volumetric part of the deformation
      J = deformationGradient_new[idx].Determinant();

      // Compute the deformed volume and the local sound speed
      double rho_cur = rho_orig/J;
      pvolume_deformed[idx]=pmass[idx]/rho_cur;
      c_dil = sqrt((bulk + 4.*shear/3.)/rho_cur);

      // get the hydrostatic part of the stress
      p = 0.5*bulk*(J - 1.0/J);

      // compute bulk viscosity
      /*
      if (flag->d_artificial_viscosity) {
        Matrix3 tensorD = (velGrad + velGrad.Transpose())*0.5;
        double Dkk = tensorD.Trace();
        double c_bulk = sqrt(bulk/rho_cur);
        double q = artificialBulkViscosity(Dkk, c_bulk, rho_cur, dx_ave);
        p -= q;
      }
      */

      // Compute ||shearTrial||
      sTnorm = shearTrial.Norm();

      muBar = IEl * shear;

      // Check for plastic loading
      fTrial = sTnorm - sqtwthds*(K*alpha + flow);

      if(fTrial > 0.0){
        // plastic

        delgamma = (fTrial/(2.0*muBar)) / (1.0 + (K/(3.0*muBar)));

        normal = shearTrial/sTnorm;

        // The actual elastic shear stress
        Shear = shearTrial - normal*2.0*muBar*delgamma;

        // Deal with history variables
        statedata[idx].Alpha = alpha + sqtwthds*delgamma;
        bElBar_new[idx] = Shear/shear + Identity*IEl;
      }
      else {
        // not plastic

        bElBar_new[idx] = bElBarTrial;
        Shear = shearTrial;
      }

      // compute the total stress (volumetric + deviatoric)
      pstress[idx] = Identity*p + Shear/J;

      // Compute the strain energy for all the particles
      U = .5*bulk*(.5*(J*J - 1.0) - log(J));
      W = .5*shear*(bElBar_new[idx].Trace() - 3.0);
      double e = (U + W)*pvolume_deformed[idx]/J;
      se += e;

      // Compute wave speed at each particle, store the maximum

      Vector pvelocity_idx = pvelocity[idx];
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity_idx.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pvelocity_idx.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pvelocity_idx.z()),WaveSpeed.z()));

    }

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();

    new_dw->put(delt_vartype(patch->getLevel()->adjustDelt(delT_new)), 
                lb->delTLabel);
    new_dw->put(sum_vartype(se),        lb->StrainEnergyLabel);

    delete interpolator;
  }
}

void CompNeoHookPlas::carryForward(const PatchSubset* patches,
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
    ParticleVariable<Matrix3> bElBar_new;
    constParticleVariable<Matrix3> bElBar;
    ParticleVariable<StateData> statedata;
    constParticleVariable<StateData> statedata_old;
    old_dw->get(statedata_old,         p_statedata_label,              pset);
    old_dw->get(bElBar,                bElBarLabel,                    pset);
    new_dw->allocateAndPut(statedata,  p_statedata_label_preReloc,     pset);
    new_dw->allocateAndPut(bElBar_new, bElBarLabel_preReloc,           pset);
    statedata.copyData(statedata_old);

    for(ParticleSubset::iterator iter = pset->begin();
                                 iter != pset->end(); iter++){
      particleIndex idx = *iter;
      bElBar_new[idx] = bElBar[idx];
    }
    new_dw->put(delt_vartype(patch->getLevel()->adjustDelt(1.e10)), 
                lb->delTLabel);
    new_dw->put(sum_vartype(0.),     lb->StrainEnergyLabel);
  }
}
 
void CompNeoHookPlas::addInitialComputesAndRequires(Task* task,
                                                    const MPMMaterial* matl,
                                                    const PatchSet*) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(p_statedata_label, matlset);
  task->computes(bElBarLabel,       matlset);
}

void CompNeoHookPlas::addComputesAndRequires(Task* task,
                                             const MPMMaterial* matl,
                                             const PatchSet* patches) const
{
  // Add the computes and requires that are common to all explicit 
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForExplicit(task, matlset, patches);

  // Other constitutive model and input dependent computes and requires
  Ghost::GhostType  gnone = Ghost::None;

  task->requires(Task::OldDW, p_statedata_label, matlset,gnone);
  task->requires(Task::OldDW, bElBarLabel,       matlset,gnone);

  task->computes(bElBarLabel_preReloc,       matlset);
  task->computes(p_statedata_label_preReloc, matlset);
}

void 
CompNeoHookPlas::addComputesAndRequires(Task* ,
                                   const MPMMaterial* ,
                                   const PatchSet* ,
                                   const bool ) const
{
}

double CompNeoHookPlas::computeRhoMicroCM(double pressure,
                                     const double p_ref,
                                          const MPMMaterial* matl)
{
  double rho_orig = matl->getInitialDensity();
  double bulk = d_initialData.Bulk;
  
  double p_gauge = pressure - p_ref;
  double rho_cur;

  if(d_useModifiedEOS && p_gauge < 0.0) {
    double A = p_ref;  // modified EOS
    double n = p_ref/bulk;
    rho_cur  = rho_orig*pow(pressure/A,n);
  } else {             // Standard EOS
    double p_g_over_bulk = p_gauge/bulk;
    rho_cur=rho_orig*(p_g_over_bulk + sqrt(p_g_over_bulk*p_g_over_bulk +1.));
  }
  return rho_cur;
}

void CompNeoHookPlas::computePressEOSCM(double rho_cur,double& pressure,
                                        double p_ref,  
                                        double& dp_drho, double& tmp,
                                        const MPMMaterial* matl)
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

double CompNeoHookPlas::getCompressibility()
{
  return 1.0/d_initialData.Bulk;
}

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1209
#endif

namespace Uintah {

static MPI_Datatype makeMPI_CMData()
{
   ASSERTEQ(sizeof(CompNeoHookPlas::StateData), sizeof(double)*1);
   MPI_Datatype mpitype;
   MPI_Type_vector(1, 1, 1, MPI_DOUBLE, &mpitype);
   MPI_Type_commit(&mpitype);
   return mpitype;
}

const TypeDescription* fun_getTypeDescription(CompNeoHookPlas::StateData*)
{
   static TypeDescription* td = 0;
   if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
                          "CompNeoHookPlas::StateData", true, &makeMPI_CMData);
   }
   return td;   
}
} // End namespace Uintah
