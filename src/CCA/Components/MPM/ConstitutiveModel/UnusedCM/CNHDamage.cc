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


#include "CNHDamage.h"
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
#include <Core/Malloc/Allocator.h>
#include <fstream>
#include <iostream>
#include <Core/Math/MinMax.h>
#include <Core/Math/Gaussian.h>
#include <Core/Math/Weibull.h>
#include <Core/Math/Short27.h> //for Fracture

using std::cerr;
using namespace Uintah;


CNHDamage::CNHDamage(ProblemSpecP& ps, MPMFlags* Mflag)
  : CompNeoHook(ps, Mflag), ImplicitCM()
{
  // Initialize local VarLabels
  initializeLocalMPMLabels();

  // Get the failure stress/strain data
  getFailureStrainData(ps);

  // Set the erosion algorithm
  setErosionAlgorithm();
}

CNHDamage::CNHDamage(const CNHDamage* cm):CompNeoHook(cm)
{
  // Initialize local VarLabels
  initializeLocalMPMLabels();

  // Set the failure strain data
  setFailureStrainData(cm);

  // Set the erosion algorithm
  setErosionAlgorithm(cm);
}

CNHDamage::~CNHDamage()
{
  VarLabel::destroy(bElBarLabel);
  VarLabel::destroy(bElBarLabel_preReloc);

  VarLabel::destroy(pFailureStrainLabel);
  VarLabel::destroy(pLocalizedLabel);
  VarLabel::destroy(pDeformRateLabel);
  VarLabel::destroy(pFailureStrainLabel_preReloc);
  VarLabel::destroy(pLocalizedLabel_preReloc);
  VarLabel::destroy(pDeformRateLabel_preReloc);
}

void CNHDamage::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","cnh_damage");
  }

  CompNeoHook::outputProblemSpec(cm_ps,false);

  cm_ps->appendElement("failure_strain_mean",    d_epsf.mean);
  cm_ps->appendElement("failure_strain_std",     d_epsf.std);
  cm_ps->appendElement("failure_strain_scale",   d_epsf.scale);
  cm_ps->appendElement("failure_strain_seed" ,   d_epsf.seed);
  cm_ps->appendElement("failure_strain_distrib", d_epsf.dist);
  cm_ps->appendElement("failure_by_stress",      d_epsf.failureByStress);

}

CNHDamage* CNHDamage::clone()
{
  return scinew CNHDamage(*this);
}

void 
CNHDamage::initializeLocalMPMLabels()
{
  bElBarLabel =         VarLabel::create("p.beBar",
                        ParticleVariable<Matrix3>::getTypeDescription());
  pFailureStrainLabel = VarLabel::create("p.epsf",
                        ParticleVariable<double>::getTypeDescription());
  pLocalizedLabel     = VarLabel::create("p.localized",
                        ParticleVariable<int>::getTypeDescription());
  pDeformRateLabel    = VarLabel::create("p.deformRate",
                        ParticleVariable<Matrix3>::getTypeDescription());

  bElBarLabel_preReloc =         VarLabel::create("p.beBar+",
                         ParticleVariable<Matrix3>::getTypeDescription());
  pFailureStrainLabel_preReloc = VarLabel::create("p.epsf+",
                         ParticleVariable<double>::getTypeDescription());
  pLocalizedLabel_preReloc     = VarLabel::create("p.localized+",
                         ParticleVariable<int>::getTypeDescription());
  pDeformRateLabel_preReloc    = VarLabel::create("p.deformRate+",
                         ParticleVariable<Matrix3>::getTypeDescription());
}

void 
CNHDamage::getFailureStrainData(ProblemSpecP& ps)
{
  d_epsf.mean   = 10.0; // Mean failure strain
  d_epsf.std    = 0.0;  // STD failure strain
  d_epsf.scale  = 1.0; // Scale Parameter for Weibull Distribution
  d_epsf.seed   = 0; // seed for weibull distribution generator
  d_epsf.dist   = "constant";
  d_epsf.failureByStress = false; // failure by strain default
  ps->get("failure_strain_mean",    d_epsf.mean);
  ps->get("failure_strain_std",     d_epsf.std);
  ps->get("failure_strain_scale",   d_epsf.scale);
  ps->get("failure_strain_seed",    d_epsf.seed);
  ps->get("failure_strain_distrib", d_epsf.dist);
  ps->get("failure_by_stress", d_epsf.failureByStress);
}

void 
CNHDamage::setFailureStrainData(const CNHDamage* cm)
{
  d_epsf.mean = cm->d_epsf.mean;
  d_epsf.std = cm->d_epsf.std;
  d_epsf.scale = cm->d_epsf.scale;
  d_epsf.seed  = cm->d_epsf.seed;
  d_epsf.dist = cm->d_epsf.dist;
  d_epsf.failureByStress = cm->d_epsf.failureByStress;
}

void 
CNHDamage::setErosionAlgorithm()
{
  d_setStressToZero = false;
  d_allowNoTension = false;
  d_removeMass = false;
  d_allowNoShear = false;
  if (flag->d_doErosion) {
    if (flag->d_erosionAlgorithm == "RemoveMass") 
      d_removeMass = true;
    else if (flag->d_erosionAlgorithm == "AllowNoTension") 
      d_allowNoTension = true;
    else if (flag->d_erosionAlgorithm == "ZeroStress") 
      d_setStressToZero = true;
    else if (flag->d_erosionAlgorithm == "AllowNoShear") 
      d_allowNoShear = true;
  }
}

void 
CNHDamage::setErosionAlgorithm(const CNHDamage* cm)
{
  d_setStressToZero = cm->d_setStressToZero;
  d_allowNoTension = cm->d_allowNoTension;
  d_removeMass = cm->d_removeMass;
  d_allowNoShear = cm->d_allowNoShear;
}

void 
CNHDamage::addInitialComputesAndRequires(Task* task,
                                         const MPMMaterial* matl,
                                         const PatchSet* ) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(bElBarLabel,         matlset);
  task->computes(pFailureStrainLabel, matlset);
  task->computes(pLocalizedLabel,     matlset);
  if (flag->d_integrator != MPMFlags::Implicit) {
    task->computes(pDeformRateLabel,  matlset);
  }
}

void 
CNHDamage::initializeCMData(const Patch* patch,
                            const MPMMaterial* matl,
                            DataWarehouse* new_dw)
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  if (flag->d_integrator == MPMFlags::Implicit) 
    initSharedDataForImplicit(patch, matl, new_dw);
  else {
    initSharedDataForExplicit(patch, matl, new_dw);
    computeStableTimestep(patch, matl, new_dw);
  }

  // Local stuff
  Matrix3 Id; Id.Identity();
  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  ParticleVariable<Matrix3> pBeBar;
  ParticleVariable<double>  pFailureStrain;
  ParticleVariable<int>     pLocalized;
  constParticleVariable<double> pVolume;

  new_dw->get(pVolume, lb->pVolumeLabel, pset);
  new_dw->allocateAndPut(pBeBar,         bElBarLabel,         pset);
  new_dw->allocateAndPut(pFailureStrain, pFailureStrainLabel, pset);
  new_dw->allocateAndPut(pLocalized,     pLocalizedLabel,     pset);

  ParticleSubset::iterator iter = pset->begin();

  if (d_epsf.dist == "gauss"){
    // Initialize a gaussian random number generator
    SCIRun::Gaussian gaussGen(d_epsf.mean, d_epsf.std, 0);

    for(;iter != pset->end();iter++){
      pBeBar[*iter] = Id;
      pFailureStrain[*iter] = fabs(gaussGen.rand());
      pLocalized[*iter] = 0;
    }
  } else if (d_epsf.dist == "weibull"){
    // Initialize a weibull random number generator
    SCIRun::Weibull weibGen(d_epsf.mean, d_epsf.std, d_epsf.scale, d_epsf.seed);

    for(;iter != pset->end();iter++){
      pBeBar[*iter] = Id;
      pFailureStrain[*iter] = weibGen.rand(pVolume[*iter]);
      pLocalized[*iter] = 0;
    }
  } else if (d_epsf.dist == "constant") {
    for(;iter != pset->end();iter++){
      pBeBar[*iter] = Id;
      pFailureStrain[*iter] = d_epsf.mean;
      pLocalized[*iter] = 0;
    }
  }

  if (flag->d_integrator != MPMFlags::Implicit) {
    Matrix3 zero(0.0);
    ParticleVariable<Matrix3> pDeformRate;
    new_dw->allocateAndPut(pDeformRate, pDeformRateLabel, pset);
    for(iter = pset->begin(); iter != pset->end(); iter++){
      pDeformRate[*iter] = zero;
    }
  }
}

void 
CNHDamage::addComputesAndRequires(Task* task,
                                  const MPMMaterial* matl,
                                  const PatchSet* patches) const
{
  Ghost::GhostType  gnone = Ghost::None;
  const MaterialSubset* matlset = matl->thisMaterial();
  if (flag->d_integrator == MPMFlags::Implicit) {
    addSharedCRForImplicit(task, matlset, patches);
  } else {
    addSharedCRForExplicit(task, matlset, patches);
    task->requires(Task::OldDW, lb->pErosionLabel,     matlset, gnone);
    task->computes(pDeformRateLabel_preReloc,          matlset);
  }

  //for pParticleID
  task->requires(Task::OldDW, lb->pParticleIDLabel,  matlset, gnone);

  // Other constitutive model and input dependent computes and requires
  task->requires(Task::OldDW, bElBarLabel,           matlset, gnone);
  task->requires(Task::OldDW, pFailureStrainLabel,   matlset, gnone);
  task->requires(Task::OldDW, pLocalizedLabel,       matlset, gnone);

  task->computes(bElBarLabel_preReloc,               matlset);
  task->computes(pFailureStrainLabel_preReloc,       matlset);
  task->computes(pLocalizedLabel_preReloc,           matlset);
}

void 
CNHDamage::computeStressTensor(const PatchSubset* patches,
                               const MPMMaterial* matl,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw)
{
  // Local variables 
  double  J, p, IEl, U, W, c_dil;
  Matrix3 Shear, pBBar_new, pDefGradInc;
  Matrix3 pDispGrad, FF;

  // Constants
  double onethird = (1.0/3.0);
  Matrix3 Identity;
  Identity.Identity();
  double shear = d_initialData.Shear;
  double bulk  = d_initialData.Bulk;
  double rho_orig = matl->getInitialDensity();

  // Loop thru patches
  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);

    // Get patch info
    Vector dx = patch->dCell();

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<Vector> d_S(interpolator->size());
    vector<double> S(interpolator->size());

    // Initialize patch variables
    double se = 0.0;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

    // Get particle info
    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    // Get delT
    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));
    Ghost::GhostType gac = Ghost::AroundCells;

    // Particle and grid data
    constParticleVariable<Short27> pgCode;
    constParticleVariable<int>     pLocalized;
    constParticleVariable<double>  pmass, pFailureStrain, pErosion;
    constParticleVariable<Point>   pX;
    constParticleVariable<Vector>  pSize, pVelocity;
    constParticleVariable<Matrix3> pDefGrad, pBeBar;
    constNCVariable<Vector>        gDisp;
    constNCVariable<Vector>        gVelocity;
    constNCVariable<Vector>        GVelocity; 
    ParticleVariable<int>          pLocalized_new;
    ParticleVariable<double>       pvolume_new, pdTdt, pFailureStrain_new,p_q;
    ParticleVariable<Matrix3>      pDefGrad_new, pBeBar_new, pStress_new;
    ParticleVariable<Matrix3>      pDeformRate;
    constParticleVariable<long64>  pParticleID;


    old_dw->get(pmass,                    lb->pMassLabel,               pset);
    old_dw->get(pX,                       lb->pXLabel,                  pset);
    old_dw->get(pSize,                    lb->pSizeLabel,               pset);
    old_dw->get(pVelocity,                lb->pVelocityLabel,           pset);
    old_dw->get(pDefGrad,                 lb->pDeformationMeasureLabel, pset);
    old_dw->get(pBeBar,                   bElBarLabel,                  pset);
    old_dw->get(pLocalized,               pLocalizedLabel,              pset);
    old_dw->get(pFailureStrain,           pFailureStrainLabel,          pset);
    old_dw->get(pErosion,                 lb->pErosionLabel,            pset);
    old_dw->get(pParticleID,              lb->pParticleIDLabel,         pset);

    // Get Grid info
    new_dw->get(gVelocity,   lb->gVelocityStarLabel, dwi, patch, gac, NGN);
    if (flag->d_fracture) {
      new_dw->get(pgCode,    lb->pgCodeLabel, pset);
      new_dw->get(GVelocity, lb->GVelocityStarLabel, dwi, patch, gac, NGN);
    }
    
    // Allocate space for updated particle variables
    new_dw->allocateAndPut(pvolume_new, 
                           lb->pVolumeLabel_preReloc,             pset);
    new_dw->allocateAndPut(pdTdt, 
                           lb->pdTdtLabel_preReloc,               pset);
    new_dw->allocateAndPut(pDefGrad_new,
                           lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->allocateAndPut(pBeBar_new, 
                           bElBarLabel_preReloc,                  pset);
    new_dw->allocateAndPut(pStress_new,        
                           lb->pStressLabel_preReloc,             pset);
    new_dw->allocateAndPut(pLocalized_new, 
                           pLocalizedLabel_preReloc,              pset);
    new_dw->allocateAndPut(pFailureStrain_new, 
                           pFailureStrainLabel_preReloc,          pset);
    new_dw->allocateAndPut(pDeformRate, 
                           pDeformRateLabel_preReloc,             pset);
    new_dw->allocateAndPut(p_q,    lb->p_qLabel_preReloc,         pset);

    // Copy failure strains to new dw
    pFailureStrain_new.copyData(pFailureStrain);

    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};
    // Loop thru particle set in patch
    ParticleSubset::iterator iter = pset->begin();
    for (; iter != pset->end(); iter++) {
      particleIndex idx = *iter;
      
      // Assign zero internal heating by default - modify if necessary.
      pdTdt[idx] = 0.0;
      // Initialize velocity gradient
      Matrix3 velGrad(0.0);

      if(!flag->d_axisymmetric){
        // Get the node indices that surround the cell
        interpolator->findCellAndShapeDerivatives(pX[idx],ni,d_S,pSize[idx],pDefGrad[idx]);

        short pgFld[27];
        if (flag->d_fracture) {
         for(int k=0; k<27; k++){
           pgFld[k]=pgCode[idx][k];
         }
         computeVelocityGradient(velGrad,ni,d_S,oodx,pgFld,gVelocity,GVelocity);
        } else {
        double erosion = pErosion[idx];
        computeVelocityGradient(velGrad,ni,d_S, oodx, gVelocity, erosion);
        }
      } else {  // axi-symmetric kinematics
        // Get the node indices that surround the cell
        interpolator->findCellAndWeightsAndShapeDerivatives(pX[idx],ni,S,d_S,
                                                                    pSize[idx],pDefGrad[idx]);
        // x -> r, y -> z, z -> theta
        computeAxiSymVelocityGradient(velGrad,ni,d_S,S,oodx,gVelocity,pX[idx]);
      }

      // Compute the rate of defomation tensor
      Matrix3 D = (velGrad + velGrad.Transpose())*0.5;
      pDeformRate[idx] = D;

      // Compute the deformation gradient increment using the time_step
      // velocity gradient ( F_n^np1 = dudx * dt + Identity)
      pDefGradInc = velGrad*delT + Identity;
      
//       double Jinc = pDefGradInc.Determinant();

      // Update the deformation gradient tensor to its time n+1 value.
      FF = pDefGradInc*pDefGrad[idx];

      // if already failed, use previous FF
      if(d_setStressToZero && pLocalized[idx]){
        FF = pDefGrad[idx];
      }

      J = FF.Determinant();

      if (!(J > 0.0)) {
        cerr << getpid() ;
        cerr << "**ERROR** Negative Jacobian of deformation gradient"
             << " in particle " << pParticleID[idx] << endl;
        cerr << "l = " << velGrad << endl;
        cerr << "F_old = " << pDefGrad[idx] << endl;
        cerr << "F_inc = " << pDefGradInc << endl;
        cerr << "F_new = " << FF << endl;
        cerr << "J = " << J << endl;
        throw ParameterNotFound("**ERROR**:CNHDamage", __FILE__, __LINE__);
      }

      pDefGrad_new[idx] = FF;
      
    } // end of the particle loop
      
    // The following is used only for pressure stabilization
    CCVariable<double> J_CC;
    new_dw->allocateTemporary(J_CC,     patch);
    J_CC.initialize(0.);
    if(flag->d_doPressureStabilization) {
      CCVariable<double> vol_0_CC;
      CCVariable<double> vol_CC;
      new_dw->allocateTemporary(vol_0_CC, patch);
      new_dw->allocateTemporary(vol_CC, patch);

      vol_0_CC.initialize(0.);
      vol_CC.initialize(0.);
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;
  
        // get the volumetric part of the deformation
        J = pDefGrad_new[idx].Determinant();
  
        // Get the deformed volume
        pvolume_new[idx]=(pmass[idx]/rho_orig)*J;
  
        IntVector cell_index;
        patch->findCell(pX[idx],cell_index);
  
        vol_CC[cell_index]+=pvolume_new[idx];
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
        
        if(flag->d_doPressureStabilization) {
        IntVector cell_index;
        patch->findCell(pX[idx],cell_index);

        // get the original volumetric part of the deformation
        J = pDefGrad_new[idx].Determinant();

        // Change F such that the determinant is equal to the average for
        // the cell
        pDefGrad_new[idx]*=cbrt(J_CC[cell_index])/cbrt(J);
      }

      J = pDefGrad_new[idx].Determinant();

      // Get the deformed volume
      pvolume_new[idx]=(pmass[idx]/rho_orig)*J;

      // Compute Bbar
//      Matrix3 pRelDefGradBar = pDefGradInc*pow(Jinc, -onethird);
//       Matrix3 pRelDefGradBar = pDefGradInc/cbrt(Jinc);

//       pBBar_new = pRelDefGradBar*pBeBar[idx]*pRelDefGradBar.Transpose();

      double cubeRootJ=cbrt(J);
      double Jtothetwothirds=cubeRootJ*cubeRootJ;
      pBBar_new = pDefGrad_new[idx]* pDefGrad_new[idx].Transpose()/Jtothetwothirds;

      IEl = onethird*pBBar_new.Trace();
      pBeBar_new[idx] = pBBar_new;

      // Shear is equal to the shear modulus times dev(bElBar)
      Shear = (pBBar_new - Identity*IEl)*shear;

      // get the hydrostatic part of the stress
      p = 0.5*bulk*(J - 1.0/J);

      // compute the total stress (volumetric + deviatoric)
      pStress_new[idx] = Identity*p + Shear/J;

      // Modify the stress if particle has failed
      updateFailedParticlesAndModifyStress(pDefGrad_new[idx],
               pFailureStrain[idx], pLocalized[idx], pLocalized_new[idx], 
               pStress_new[idx], pParticleID[idx]);

      // Compute the strain energy for non-localized particles
      if(pLocalized_new[idx] == 0){
        U = .5*bulk*(.5*(J*J - 1.0) - log(J));
        W = .5*shear*(pBBar_new.Trace() - 3.0);
        double e = (U + W)*pvolume_new[idx]/J;
        se += e;
      }
      
      // Compute local wave speed
      double rho_cur = rho_orig/J;
      c_dil = sqrt((bulk + 4.*shear/3.)/rho_cur);
      Vector pVel = pVelocity[idx];
      WaveSpeed=Vector(Max(c_dil+fabs(pVel.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pVel.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pVel.z()),WaveSpeed.z()));

      // Compute artificial viscosity term
      if (flag->d_artificial_viscosity) {
        double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;
        double c_bulk = sqrt(bulk/rho_cur);
        p_q[idx] = artificialBulkViscosity(pDeformRate[idx].Trace(),
                                           c_bulk, rho_cur, dx_ave);
      } else {
        p_q[idx] = 0.;
      }
    }  // end loop over particles

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
    
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(se), lb->StrainEnergyLabel);
    }

    delete interpolator;
  }
}


// Modify the stress if particle has failed
void
CNHDamage::updateFailedParticlesAndModifyStress(const Matrix3& FF, 
                                                const double& pFailureStrain, 
                                                const int& pLocalized,
                                                int& pLocalized_new, 
                                                Matrix3& pStress_new,
                                                const long64 particleID)
{
  Matrix3 Identity, zero(0.0); Identity.Identity();

  // Compute Finger tensor (left Cauchy-Green) 
  Matrix3 bb = FF*FF.Transpose();

  // Compute pressure
  double pressure = (1.0/3.0)*pStress_new.Trace();

  // Compute Eulerian strain tensor
  Matrix3 ee = (Identity - bb.Inverse())*0.5;      

  // Compute the maximum principal strain or stress
  Vector  eigval(0.0, 0.0, 0.0);
  Matrix3 eigvec(0.0);

  if (d_epsf.failureByStress) {
      pStress_new.eigen(eigval, eigvec);
  } else {                      //failure by strain
      ee.eigen(eigval, eigvec);
  }

//change to maximum principal stress or strain;
//now it distinguishes tension from compression
//let ffjhl know if this causes a problem.
//  double epsMax = Max(fabs(eigval[0]),fabs(eigval[2]));
//  double epsMax = Max(Max(eigval[0],eigval[1]), eigval[2]);
  //The first eigenvalue returned by "eigen" is always the largest 
  double epsMax = eigval[0];

  // Find if the particle has failed
  pLocalized_new = pLocalized;
  if (epsMax > pFailureStrain) pLocalized_new = 1;
  if (pLocalized != pLocalized_new) {
     cout << "Particle " << particleID << " has failed : eps = " << epsMax 
          << " eps_f = " << pFailureStrain << endl;
  }

  // If the particle has failed, apply various erosion algorithms
  if (flag->d_doErosion) {
    if (pLocalized || pLocalized_new) {
      if (d_allowNoTension) {
        if (pressure > 0.0) pStress_new = zero;
        else pStress_new = Identity*pressure;
      } else if (d_allowNoShear) pStress_new = Identity*pressure;
      else if (d_setStressToZero) pStress_new = zero;
    }
  }
}

void 
CNHDamage::computeStressTensorImplicit(const PatchSubset* patches,
                                       const MPMMaterial* matl,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw)
{
  // Constants
  Ghost::GhostType gac = Ghost::AroundCells;
  double onethird = (1.0/3.0);
  Matrix3 Identity; Identity.Identity();
  int dwi = matl->getDWIndex();
  double shear = d_initialData.Shear;
  double bulk  = d_initialData.Bulk;
  double rho_orig = matl->getInitialDensity();

  // Particle and grid data
  constParticleVariable<int>     pLocalized;
  constParticleVariable<double>  pmass, pFailureStrain;
  constParticleVariable<Point>   pX;
  constParticleVariable<Vector>  pSize;
  constParticleVariable<Matrix3> pDefGrad, pBeBar;
  constNCVariable<Vector>        gDisp;
  ParticleVariable<int>          pLocalized_new;
  ParticleVariable<double>       pVol_new, pdTdt, pFailureStrain_new;
  ParticleVariable<Matrix3>      pDefGrad_new, pBeBar_new, pStress_new;

  constParticleVariable<long64>   pParticleID;

  // Local variables 
  double  J = 0.0, p = 0.0, IEl = 0.0, U = 0.0, W = 0.0;
  Matrix3 Shear(0.0), pBBar_new(0.0), pDefGradInc(0.0);
  Matrix3 pDispGrad(0.0), FF(0.0);

  // Loop thru patches
  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<Vector> d_S(interpolator->size());

    // Initialize patch variables
    double se = 0.0;


    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    // Get particle info
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    old_dw->get(pmass,                    lb->pMassLabel,               pset);
    old_dw->get(pX,                       lb->pXLabel,                  pset);
    old_dw->get(pSize,                    lb->pSizeLabel,               pset);
    old_dw->get(pDefGrad,                 lb->pDeformationMeasureLabel, pset);
    old_dw->get(pBeBar,                   bElBarLabel,                  pset);
    old_dw->get(pLocalized,               pLocalizedLabel,              pset);
    old_dw->get(pFailureStrain,           pFailureStrainLabel,          pset);
    old_dw->get(pParticleID,              lb->pParticleIDLabel,         pset);

    // Get Grid info
    new_dw->get(gDisp, lb->dispNewLabel, dwi, patch, gac, 1);
    
    // Allocate space for updated particle variables
    new_dw->allocateAndPut(pVol_new, 
                           lb->pVolumeDeformedLabel,              pset);
    new_dw->allocateAndPut(pdTdt, 
                           lb->pdTdtLabel_preReloc,   pset);
    new_dw->allocateAndPut(pDefGrad_new,
                           lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->allocateAndPut(pBeBar_new, 
                           bElBarLabel_preReloc,                  pset);
    new_dw->allocateAndPut(pStress_new,        
                           lb->pStressLabel_preReloc,             pset);

    new_dw->allocateAndPut(pLocalized_new, 
                           pLocalizedLabel_preReloc,              pset);
    new_dw->allocateAndPut(pFailureStrain_new, 
                           pFailureStrainLabel_preReloc,          pset);

    // Copy failure strains to new dw
    pFailureStrain_new.copyData(pFailureStrain);

    // Loop thru particle set in patch
    ParticleSubset::iterator iter = pset->begin();
    for (; iter != pset->end(); iter++) {
      particleIndex idx = *iter;
      
      // Assign zero internal heating by default - modify if necessary.
      pdTdt[idx] = 0.0;

      interpolator->findCellAndShapeDerivatives(pX[idx], ni, d_S, pSize[idx],pDefGrad[idx]);
      // Compute the displacement gradient and the deformation gradient
      computeGrad(pDispGrad,ni,d_S, oodx, gDisp);
      pDefGradInc = pDispGrad + Identity;         
      double Jinc = pDefGradInc.Determinant();

      // Update the deformation gradient tensor to its time n+1 value.
      FF = pDefGradInc*pDefGrad[idx];
      J = FF.Determinant();
      if (!(J > 0.0)) {
        cerr << getpid() 
             << "**ERROR** Negative Jacobian of deformation gradient" << endl;
        throw ParameterNotFound("**ERROR**:CNHDamage", __FILE__, __LINE__);
      }
      pDefGrad_new[idx] = FF;

      // Get the deformed volume
      pVol_new[idx]=(pmass[idx]/rho_orig)*J;

      // Compute Bbar
//      Matrix3 pRelDefGradBar = pDefGradInc*pow(Jinc, -onethird);
      Matrix3 pRelDefGradBar = pDefGradInc/cbrt(Jinc);
      pBBar_new = pRelDefGradBar*pBeBar[idx]*pRelDefGradBar.Transpose();
      pBeBar_new[idx] = pBBar_new;
      IEl = onethird*pBBar_new.Trace();

      // Shear is equal to the shear modulus times dev(bElBar)
      Shear = (pBBar_new - Identity*IEl)*shear;

      // get the hydrostatic part of the stress
      p = bulk*log(J)/J;

      // compute the total stress (volumetric + deviatoric)
      pStress_new[idx] = Identity*p + Shear/J;
      //cout << "Last:p = " << p << " J = " << J << " tdev = " << Shear << endl;

      // Modify the stress if particle has failed
      updateFailedParticlesAndModifyStress(FF, pFailureStrain[idx], 
                                           pLocalized[idx],pLocalized_new[idx], 
                                           pStress_new[idx], pParticleID[idx]);

      // Compute the strain energy for all the particles
      U = .5*bulk*(.5*(J*J - 1.0) - log(J));
      W = .5*shear*(pBBar_new.Trace() - 3.0);
      double e = (U + W)*pVol_new[idx]/J;
      se += e;
    }
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(se), lb->StrainEnergyLabel);
    }

    delete interpolator;

  }
}

void 
CNHDamage::addComputesAndRequires(Task* task,
                                  const MPMMaterial* matl,
                                  const PatchSet* patches,
                                  const bool recurse,
                                  const bool SchedParent) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForImplicit(task, matlset, patches, recurse,SchedParent);

  // Local stuff
  task->requires(Task::ParentOldDW, bElBarLabel, matlset, Ghost::None);
}

void 
CNHDamage::computeStressTensor(const PatchSubset* patches,
                               const MPMMaterial* matl,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw,
                               Solver* solver,
                               const bool )

{
  // Constants
  int dwi = matl->getDWIndex();
  double onethird = (1.0/3.0);
  double shear = d_initialData.Shear;
  double bulk  = d_initialData.Bulk;
  double rho_orig = matl->getInitialDensity();

  Ghost::GhostType gac = Ghost::AroundCells;
  Matrix3 Identity; Identity.Identity();
  DataWarehouse* parent_old_dw = 
    new_dw->getOtherDataWarehouse(Task::ParentOldDW);

  // Particle and grid variables
  constParticleVariable<double>  pVol,pmass;
  constParticleVariable<Point>   pX;
  constParticleVariable<Vector>  pSize;
  constParticleVariable<Matrix3> pDefGrad, pBeBar;
  constNCVariable<Vector>        gDisp;
  ParticleVariable<double>       pVol_new;
  ParticleVariable<Matrix3>      pDefGrad_new, pBeBar_new, pStress;

  // Local variables
  Matrix3 Shear(0.0), pDefGradInc(0.0), pDispGrad(0.0), pRelDefGradBar(0.0);
  double D[6][6];
  double B[6][24];
  double Bnl[3][24];
  double Kmatrix[24][24];
  int dof[24];
  double v[576];

  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<Vector> d_S(interpolator->size());

    IntVector lowIndex = patch->getNodeLowIndex();
    IntVector highIndex = patch->getNodeHighIndex()+IntVector(1,1,1);
    Array3<int> l2g(lowIndex,highIndex);
    solver->copyL2G(l2g,patch);

    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};
    
    ParticleSubset* pset = parent_old_dw->getParticleSubset(dwi, patch);
    parent_old_dw->get(pX,       lb->pXLabel,                  pset);
    parent_old_dw->get(pSize,    lb->pSizeLabel,               pset);
    parent_old_dw->get(pmass,    lb->pMassLabel,               pset);
    parent_old_dw->get(pDefGrad, lb->pDeformationMeasureLabel, pset);
    parent_old_dw->get(pBeBar,   bElBarLabel,                  pset);
    old_dw->get(gDisp,           lb->dispNewLabel, dwi, patch, gac, 1);
  
    new_dw->allocateAndPut(pStress,  lb->pStressLabel_preReloc, pset);
    new_dw->allocateAndPut(pVol_new, lb->pVolumeDeformedLabel,  pset);
    new_dw->allocateTemporary(pDefGrad_new, pset);
    new_dw->allocateTemporary(pBeBar_new,   pset);

    ParticleSubset::iterator iter = pset->begin();
    for(; iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Compute the displacement gradient and B matrices


      interpolator->findCellAndShapeDerivatives(pX[idx], ni, d_S, pSize[idx],pDefGrad[idx]);

      computeGradAndBmats(pDispGrad,ni,d_S, oodx, gDisp, l2g,B, Bnl, dof);

      // Compute the deformation gradient increment using the pDispGrad
      // Update the deformation gradient tensor to its time n+1 value.
      pDefGradInc = pDispGrad + Identity;
      pDefGrad_new[idx] = pDefGradInc*pDefGrad[idx];
      double J = pDefGrad_new[idx].Determinant();

      // Updat the particle volume
      double volold = (pmass[idx]/rho_orig);
      double volnew = volold*J;
      pVol_new[idx] = volnew;

      // Compute BeBar
      pRelDefGradBar = pDefGradInc/cbrt(pDefGradInc.Determinant());
      pBeBar_new[idx] = pRelDefGradBar*pBeBar[idx]*pRelDefGradBar.Transpose();

      // Shear is equal to the shear modulus times dev(bElBar)
      double mubar = onethird*pBeBar_new[idx].Trace()*shear;
      Matrix3 shrTrl = (pBeBar_new[idx]*shear - Identity*mubar);

      // get the hydrostatic part of the stress
      double p = bulk*log(J)/J;

      // compute the total stress (volumetric + deviatoric)
      pStress[idx] = Identity*p + shrTrl/J;
      //cout << "p = " << p << " J = " << J << " tdev = " << shrTrl << endl;

      // Compute the tangent stiffness matrix
      computeTangentStiffnessMatrix(shrTrl, mubar, J, bulk, D);

      // Print out stuff
      /*
      cout.setf(ios::scientific,ios::floatfield);
      cout.precision(10);
      cout << "B = " << endl;
      for(int kk = 0; kk < 24; kk++) {
        for (int ll = 0; ll < 6; ++ll) {
          cout << B[ll][kk] << " " ;
        }
        cout << endl;
      }
      cout << "Bnl = " << endl;
      for(int kk = 0; kk < 24; kk++) {
        for (int ll = 0; ll < 3; ++ll) {
          cout << Bnl[ll][kk] << " " ;
        }
        cout << endl;
      }
      cout << "D = " << endl;
      for(int kk = 0; kk < 6; kk++) {
        for (int ll = 0; ll < 6; ++ll) {
          cout << D[ll][kk] << " " ;
        }
        cout << endl;
      }
      */

      // Compute K matrix = Kmat + Kgeo
      computeStiffnessMatrix(B, Bnl, D, pStress[idx], volold, volnew,
                             Kmatrix);

      // Assemble into global K matrix
      for (int I = 0; I < 24; I++){
        for (int J = 0; J < 24; J++){
          v[24*I+J] = Kmatrix[I][J];
        }
      }
      solver->fillMatrix(24,dof,24,dof,v);

    }  // end of loop over particles
    delete interpolator;
  }
  solver->flushMatrix();
}

/*! Compute tangent stiffness matrix */
void 
CNHDamage::computeTangentStiffnessMatrix(const Matrix3& sigdev, 
                                         const double&  mubar,
                                         const double&  J,
                                         const double&  bulk,
                                         double D[6][6])
{
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
void 
CNHDamage::computeStiffnessMatrix(const double B[6][24],
                                  const double Bnl[3][24],
                                  const double D[6][6],
                                  const Matrix3& sig,
                                  const double& vol_old,
                                  const double& vol_new,
                                  double Kmatrix[24][24])
{

  // Kmat = B.transpose()*D*B*volold
  double Kmat[24][24];
  BtDB(B, D, Kmat);

  // Kgeo = Bnl.transpose*sig*Bnl*volnew;
  double Kgeo[24][24];
  BnlTSigBnl(sig, Bnl, Kgeo);

  /*
  cout.setf(ios::scientific,ios::floatfield);
  cout.precision(10);
  cout << "Kmat = " << endl;
  for(int kk = 0; kk < 24; kk++) {
    for (int ll = 0; ll < 24; ++ll) {
      cout << Kmat[ll][kk] << " " ;
    }
    cout << endl;
  }
  cout << "Kgeo = " << endl;
  for(int kk = 0; kk < 24; kk++) {
    for (int ll = 0; ll < 24; ++ll) {
      cout << Kgeo[ll][kk] << " " ;
    }
    cout << endl;
  }
  */

  for(int ii = 0;ii<24;ii++){
    for(int jj = 0;jj<24;jj++){
      Kmatrix[ii][jj] =  Kmat[ii][jj]*vol_old + Kgeo[ii][jj]*vol_new;
    }
  }
}

void 
CNHDamage::carryForward(const PatchSubset* patches,
                        const MPMMaterial* matl,
                        DataWarehouse* old_dw,
                        DataWarehouse* new_dw)
{
  CompNeoHook::carryForward(patches, matl, old_dw, new_dw);

  // Carry forward the data local to this constitutive model 
  int dwi = matl->getDWIndex();

  constParticleVariable<Matrix3> pBeBar;
  constParticleVariable<double>  pFailureStrain;
  constParticleVariable<int>     pLocalized;
  ParticleVariable<Matrix3>      pBeBar_new;
  ParticleVariable<double>       pFailureStrain_new;
  ParticleVariable<int>          pLocalized_new;

  // Loop thru patches
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    old_dw->get(pBeBar,         bElBarLabel,             pset);
    old_dw->get(pFailureStrain, pFailureStrainLabel,     pset);
    old_dw->get(pLocalized,        pLocalizedLabel,         pset);

    new_dw->allocateAndPut(pBeBar_new, 
                           bElBarLabel_preReloc,         pset);
    new_dw->allocateAndPut(pFailureStrain_new,    
                           pFailureStrainLabel_preReloc, pset);
    new_dw->allocateAndPut(pLocalized_new,      
                           pLocalizedLabel_preReloc,     pset);

    pBeBar_new.copyData(pBeBar);
    pFailureStrain_new.copyData(pFailureStrain);
    pLocalized_new.copyData(pLocalized);

    if (flag->d_integrator != MPMFlags::Implicit) {
      ParticleVariable<Matrix3> pDeformRate;
      new_dw->allocateAndPut(pDeformRate,      
                             pDeformRateLabel_preReloc, pset);
      pDeformRate.copyData(pBeBar);
    }
  }
}

void 
CNHDamage::addRequiresDamageParameter(Task* task,
                                      const MPMMaterial* matl,
                                      const PatchSet* ) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::NewDW, pLocalizedLabel_preReloc, matlset, Ghost::None);
}

void 
CNHDamage::getDamageParameter(const Patch* patch,
                              ParticleVariable<int>& damage,
                              int dwi,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw)
{
  ParticleSubset* pset = old_dw->getParticleSubset(dwi,patch);
  constParticleVariable<int> pLocalized;
  new_dw->get(pLocalized, pLocalizedLabel_preReloc, pset);

  ParticleSubset::iterator iter;
  for (iter = pset->begin(); iter != pset->end(); iter++) {
    damage[*iter] = pLocalized[*iter];
  }
}
         
void 
CNHDamage::allocateCMDataAddRequires(Task* task,
                                     const MPMMaterial* matl,
                                     const PatchSet* patches,
                                     MPMLabel* lb) const
{
  CompNeoHook::allocateCMDataAddRequires(task, matl, patches, lb);

  // Add requires local to this model
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::NewDW, bElBarLabel_preReloc,         matlset, 
                 Ghost::None);
  task->requires(Task::NewDW, pFailureStrainLabel_preReloc, matlset, 
                 Ghost::None);
  task->requires(Task::NewDW, pLocalizedLabel_preReloc,     matlset, 
                 Ghost::None);
  if (flag->d_integrator != MPMFlags::Implicit) {
    task->requires(Task::NewDW, pDeformRateLabel_preReloc,  matlset, 
                   Ghost::None);
  }
}


void 
CNHDamage::allocateCMDataAdd(DataWarehouse* new_dw,
                             ParticleSubset* addset,
                             map<const VarLabel*, 
                             ParticleVariableBase*>* newState,
                             ParticleSubset* delset,
                             DataWarehouse* old_dw)
{
  CompNeoHook::allocateCMDataAdd(new_dw, addset, newState, delset, old_dw);

  // Copy the data local to this constitutive model from the particles to 
  // be deleted to the particles to be added
  constParticleVariable<Matrix3> o_pBeBar;
  constParticleVariable<double>  o_pFailureStrain;
  constParticleVariable<int>     o_pLocalized;
  new_dw->get(o_pBeBar,         bElBarLabel_preReloc,         delset);
  new_dw->get(o_pFailureStrain, pFailureStrainLabel_preReloc, delset);
  new_dw->get(o_pLocalized,        pLocalizedLabel_preReloc,     delset);

  ParticleVariable<Matrix3> pBeBar;
  ParticleVariable<double>  pFailureStrain;
  ParticleVariable<int>     pLocalized;
  new_dw->allocateTemporary(pBeBar,         addset);
  new_dw->allocateTemporary(pFailureStrain, addset);
  new_dw->allocateTemporary(pLocalized,        addset);

  ParticleSubset::iterator o,n = addset->begin();
  for (o=delset->begin(); o != delset->end(); o++, n++) {
    pBeBar[*n] = o_pBeBar[*o];
    pFailureStrain[*n] = o_pFailureStrain[*o];
    pLocalized[*n] = o_pLocalized[*o];
  }
  (*newState)[bElBarLabel] = pBeBar.clone();
  (*newState)[pFailureStrainLabel] = pFailureStrain.clone();
  (*newState)[pLocalizedLabel] = pLocalized.clone();

  if (flag->d_integrator != MPMFlags::Implicit) {
    constParticleVariable<Matrix3> o_pDeformRate;
    new_dw->get(o_pDeformRate, pDeformRateLabel_preReloc, delset);
    ParticleVariable<Matrix3> pDeformRate;
    new_dw->allocateTemporary(pDeformRate, addset);
    ParticleSubset::iterator o,n = addset->begin();
    for (o=delset->begin(); o != delset->end(); o++, n++) {
      pDeformRate[*n] = o_pDeformRate[*o];
    }
    (*newState)[pDeformRateLabel] = pDeformRate.clone();
  }
}


void 
CNHDamage::addParticleState(std::vector<const VarLabel*>& from,
                            std::vector<const VarLabel*>& to)
{
  CompNeoHook::addParticleState(from, to);

  // Add the local particle state data for this constitutive model.
  from.push_back(bElBarLabel);
  from.push_back(pFailureStrainLabel);
  from.push_back(pLocalizedLabel);
  to.push_back(bElBarLabel_preReloc);
  to.push_back(pFailureStrainLabel_preReloc);
  to.push_back(pLocalizedLabel_preReloc);
  if (flag->d_integrator != MPMFlags::Implicit) {
    from.push_back(pDeformRateLabel);
    to.push_back(pDeformRateLabel_preReloc);
  }
}

void 
CNHDamage::BnlTSigBnl(const Matrix3& sig, const double Bnl[3][24],
                      double Kgeo[24][24]) const
{
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

  t1 = Bnl[0][0]*sig(0,0);
  t4 = Bnl[0][0]*sig(0,0);
  t2 = Bnl[0][0]*sig(0,1);
  t3 = Bnl[0][0]*sig(0,2);
  t5 = Bnl[1][1]*sig(1,1);
  t8 = Bnl[1][1]*sig(1,1);
  t6 = Bnl[1][1]*sig(1,2);
  t7 = Bnl[1][1]*sig(0,1);
  t9 = Bnl[2][2]*sig(2,2);
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

  Kgeo[0][0] = t1*Bnl[0][0];
  Kgeo[0][1] = t2*Bnl[1][1];
  Kgeo[0][2] = t3*Bnl[2][2];
  Kgeo[0][3] = t4*Bnl[0][3];
  Kgeo[0][4] = t2*Bnl[1][4];
  Kgeo[0][5] = t3*Bnl[2][5];
  Kgeo[0][6] = t4*Bnl[0][6];
  Kgeo[0][7] = t2*Bnl[1][7];
  Kgeo[0][8] = t3*Bnl[2][8];
  Kgeo[0][9] = t4*Bnl[0][9];
  Kgeo[0][10] = t2*Bnl[1][10];
  Kgeo[0][11] = t3*Bnl[2][11];
  Kgeo[0][12] = t4*Bnl[0][12];
  Kgeo[0][13] = t2*Bnl[1][13];
  Kgeo[0][14] = t3*Bnl[2][14];
  Kgeo[0][15] = t4*Bnl[0][15];
  Kgeo[0][16] = t2*Bnl[1][16];
  Kgeo[0][17] = t3*Bnl[2][17];
  Kgeo[0][18] = t4*Bnl[0][18];
  Kgeo[0][19] = t2*Bnl[1][19];
  Kgeo[0][20] = t3*Bnl[2][20];
  Kgeo[0][21] = t4*Bnl[0][21];
  Kgeo[0][22] = t2*Bnl[1][22];
  Kgeo[0][23] = t3*Bnl[2][23];
  Kgeo[1][0] = Kgeo[0][1];
  Kgeo[1][1] = t5*Bnl[1][1];
  Kgeo[1][2] = t6*Bnl[2][2];
  Kgeo[1][3] = t7*Bnl[0][3];
  Kgeo[1][4] = Bnl[1][4]*t8;
  Kgeo[1][5] = t6*Bnl[2][5];
  Kgeo[1][6] = t7*Bnl[0][6];
  Kgeo[1][7] = Bnl[1][7]*t8;
  Kgeo[1][8] = t6*Bnl[2][8];
  Kgeo[1][9] = t7*Bnl[0][9];
  Kgeo[1][10] = Bnl[1][10]*t8;
  Kgeo[1][11] = t6*Bnl[2][11];
  Kgeo[1][12] = t7*Bnl[0][12];
  Kgeo[1][13] = Bnl[1][13]*t8;
  Kgeo[1][14] = t6*Bnl[2][14];
  Kgeo[1][15] = t7*Bnl[0][15];
  Kgeo[1][16] = Bnl[1][16]*t8;
  Kgeo[1][17] = t6*Bnl[2][17];
  Kgeo[1][18] = t7*Bnl[0][18];
  Kgeo[1][19] = Bnl[1][19]*t8;
  Kgeo[1][20] = t6*Bnl[2][20];
  Kgeo[1][21] = t7*Bnl[0][21];
  Kgeo[1][22] = Bnl[1][22]*t8;
  Kgeo[1][23] = t6*Bnl[2][23];
  Kgeo[2][0] = Kgeo[0][2];
  Kgeo[2][1] = Kgeo[1][2];
  Kgeo[2][2] = t9*Bnl[2][2];
  Kgeo[2][3] = t10*Bnl[0][3];
  Kgeo[2][4] = Bnl[1][4]*t11;
  Kgeo[2][5] = t12*Bnl[2][5];
  Kgeo[2][6] = t10*Bnl[0][6];
  Kgeo[2][7] = Bnl[1][7]*t11;
  Kgeo[2][8] = t12*Bnl[2][8];
  Kgeo[2][9] = t10*Bnl[0][9];
  Kgeo[2][10] = Bnl[1][10]*t11;
  Kgeo[2][11] = t12*Bnl[2][11];
  Kgeo[2][12] = t10*Bnl[0][12];
  Kgeo[2][13] = Bnl[1][13]*t11;
  Kgeo[2][14] = t12*Bnl[2][14];
  Kgeo[2][15] = t10*Bnl[0][15];
  Kgeo[2][16] = Bnl[1][16]*t11;
  Kgeo[2][17] = t12*Bnl[2][17];
  Kgeo[2][18] = t10*Bnl[0][18];
  Kgeo[2][19] = t11*Bnl[1][19];
  Kgeo[2][20] = t12*Bnl[2][20];
  Kgeo[2][21] = t10*Bnl[0][21];
  Kgeo[2][22] = t11*Bnl[1][22];
  Kgeo[2][23] = t12*Bnl[2][23];
  Kgeo[3][0] = Kgeo[0][3];
  Kgeo[3][1] = Kgeo[1][3];
  Kgeo[3][2] = Kgeo[2][3];
  Kgeo[3][3] = t13*Bnl[0][3];
  Kgeo[3][4] = t14*Bnl[1][4];
  Kgeo[3][5] = Bnl[2][5]*t15;
  Kgeo[3][6] = t16*Bnl[0][6];
  Kgeo[3][7] = t14*Bnl[1][7];
  Kgeo[3][8] = Bnl[2][8]*t15;
  Kgeo[3][9] = t16*Bnl[0][9];
  Kgeo[3][10] = t14*Bnl[1][10];
  Kgeo[3][11] = Bnl[2][11]*t15;
  Kgeo[3][12] = t16*Bnl[0][12];
  Kgeo[3][13] = t14*Bnl[1][13];
  Kgeo[3][14] = Bnl[2][14]*t15;
  Kgeo[3][15] = t16*Bnl[0][15];
  Kgeo[3][16] = t14*Bnl[1][16];
  Kgeo[3][17] = Bnl[2][17]*t15;
  Kgeo[3][18] = t16*Bnl[0][18];
  Kgeo[3][19] = t14*Bnl[1][19];
  Kgeo[3][20] = Bnl[2][20]*t15;
  Kgeo[3][21] = t16*Bnl[0][21];
  Kgeo[3][22] = t14*Bnl[1][22];
  Kgeo[3][23] = Bnl[2][23]*t15;
  Kgeo[4][0] = Kgeo[0][4];
  Kgeo[4][1] = Kgeo[1][4];
  Kgeo[4][2] = Kgeo[2][4];
  Kgeo[4][3] = Kgeo[3][4];
  Kgeo[4][4] = t17*Bnl[1][4];
  Kgeo[4][5] = t18*Bnl[2][5];
  Kgeo[4][6] = t19*Bnl[0][6];
  Kgeo[4][7] = t20*Bnl[1][7];
  Kgeo[4][8] = t18*Bnl[2][8];
  Kgeo[4][9] = t19*Bnl[0][9];
  Kgeo[4][10] = t20*Bnl[1][10];
  Kgeo[4][11] = t18*Bnl[2][11];
  Kgeo[4][12] = t19*Bnl[0][12];
  Kgeo[4][13] = t20*Bnl[1][13];
  Kgeo[4][14] = t18*Bnl[2][14];
  Kgeo[4][15] = t19*Bnl[0][15];
  Kgeo[4][16] = t20*Bnl[1][16];
  Kgeo[4][17] = t18*Bnl[2][17];
  Kgeo[4][18] = t19*Bnl[0][18];
  Kgeo[4][19] = t20*Bnl[1][19];
  Kgeo[4][20] = t18*Bnl[2][20];
  Kgeo[4][21] = t19*Bnl[0][21];
  Kgeo[4][22] = t20*Bnl[1][22];
  Kgeo[4][23] = t18*Bnl[2][23];
  Kgeo[5][0] = Kgeo[0][5];
  Kgeo[5][1] = Kgeo[1][5];
  Kgeo[5][2] = Kgeo[2][5];
  Kgeo[5][3] = Kgeo[3][5];
  Kgeo[5][4] = Kgeo[4][5];
  Kgeo[5][5] = t21*Bnl[2][5];
  Kgeo[5][6] = t22*Bnl[0][6];
  Kgeo[5][7] = t23*Bnl[1][7];
  Kgeo[5][8] = t24*Bnl[2][8];
  Kgeo[5][9] = t22*Bnl[0][9];
  Kgeo[5][10] = t23*Bnl[1][10];
  Kgeo[5][11] = t24*Bnl[2][11];
  Kgeo[5][12] = t22*Bnl[0][12];
  Kgeo[5][13] = t23*Bnl[1][13];
  Kgeo[5][14] = t24*Bnl[2][14];
  Kgeo[5][15] = t22*Bnl[0][15];
  Kgeo[5][16] = t23*Bnl[1][16];
  Kgeo[5][17] = t24*Bnl[2][17];
  Kgeo[5][18] = t22*Bnl[0][18];
  Kgeo[5][19] = t23*Bnl[1][19];
  Kgeo[5][20] = t24*Bnl[2][20];
  Kgeo[5][21] = t22*Bnl[0][21];
  Kgeo[5][22] = t23*Bnl[1][22];
  Kgeo[5][23] = t24*Bnl[2][23];
  Kgeo[6][0] = Kgeo[0][6];
  Kgeo[6][1] = Kgeo[1][6];
  Kgeo[6][2] = Kgeo[2][6];
  Kgeo[6][3] = Kgeo[3][6];
  Kgeo[6][4] = Kgeo[4][6];
  Kgeo[6][5] = Kgeo[5][6];
  Kgeo[6][6] = t25*Bnl[0][6];
  Kgeo[6][7] = t26*Bnl[1][7];
  Kgeo[6][8] = t27*Bnl[2][8];
  Kgeo[6][9] = t28*Bnl[0][9];
  Kgeo[6][10] = t26*Bnl[1][10];
  Kgeo[6][11] = t27*Bnl[2][11];
  Kgeo[6][12] = t28*Bnl[0][12];
  Kgeo[6][13] = t26*Bnl[1][13];
  Kgeo[6][14] = t27*Bnl[2][14];
  Kgeo[6][15] = t28*Bnl[0][15];
  Kgeo[6][16] = t26*Bnl[1][16];
  Kgeo[6][17] = t27*Bnl[2][17];
  Kgeo[6][18] = t28*Bnl[0][18];
  Kgeo[6][19] = t26*Bnl[1][19];
  Kgeo[6][20] = t27*Bnl[2][20];
  Kgeo[6][21] = t28*Bnl[0][21];
  Kgeo[6][22] = t26*Bnl[1][22];
  Kgeo[6][23] = t27*Bnl[2][23];
  Kgeo[7][0] = Kgeo[0][7];
  Kgeo[7][1] = Kgeo[1][7];
  Kgeo[7][2] = Kgeo[2][7];
  Kgeo[7][3] = Kgeo[3][7];
  Kgeo[7][4] = Kgeo[4][7];
  Kgeo[7][5] = Kgeo[5][7];
  Kgeo[7][6] = Kgeo[6][7];
  Kgeo[7][7] = t29*Bnl[1][7];
  Kgeo[7][8] = t30*Bnl[2][8];
  Kgeo[7][9] = t31*Bnl[0][9];
  Kgeo[7][10] = t32*Bnl[1][10];
  Kgeo[7][11] = t30*Bnl[2][11];
  Kgeo[7][12] = t31*Bnl[0][12];
  Kgeo[7][13] = t32*Bnl[1][13];
  Kgeo[7][14] = t30*Bnl[2][14];
  Kgeo[7][15] = t31*Bnl[0][15];
  Kgeo[7][16] = t32*Bnl[1][16];
  Kgeo[7][17] = t30*Bnl[2][17];
  Kgeo[7][18] = t31*Bnl[0][18];
  Kgeo[7][19] = t32*Bnl[1][19];
  Kgeo[7][20] = t30*Bnl[2][20];
  Kgeo[7][21] = t31*Bnl[0][21];
  Kgeo[7][22] = t32*Bnl[1][22];
  Kgeo[7][23] = t30*Bnl[2][23];
  Kgeo[8][0] = Kgeo[0][8];
  Kgeo[8][1] = Kgeo[1][8];
  Kgeo[8][2] = Kgeo[2][8];
  Kgeo[8][3] = Kgeo[3][8];
  Kgeo[8][4] = Kgeo[4][8];
  Kgeo[8][5] = Kgeo[5][8];
  Kgeo[8][6] = Kgeo[6][8];
  Kgeo[8][7] = Kgeo[7][8];
  Kgeo[8][8] = t33*Bnl[2][8];
  Kgeo[8][9] = t34*Bnl[0][9];
  Kgeo[8][10] = t35*Bnl[1][10];
  Kgeo[8][11] = t36*Bnl[2][11];
  Kgeo[8][12] = t34*Bnl[0][12];
  Kgeo[8][13] = t35*Bnl[1][13];
  Kgeo[8][14] = t36*Bnl[2][14];
  Kgeo[8][15] = t34*Bnl[0][15];
  Kgeo[8][16] = t35*Bnl[1][16];
  Kgeo[8][17] = t36*Bnl[2][17];
  Kgeo[8][18] = t34*Bnl[0][18];
  Kgeo[8][19] = t35*Bnl[1][19];
  Kgeo[8][20] = t36*Bnl[2][20];
  Kgeo[8][21] = t34*Bnl[0][21];
  Kgeo[8][22] = t35*Bnl[1][22];
  Kgeo[8][23] = t36*Bnl[2][23];
  Kgeo[9][0] = Kgeo[0][9];
  Kgeo[9][1] = Kgeo[1][9];
  Kgeo[9][2] = Kgeo[2][9];
  Kgeo[9][3] = Kgeo[3][9];
  Kgeo[9][4] = Kgeo[4][9];
  Kgeo[9][5] = Kgeo[5][9];
  Kgeo[9][6] = Kgeo[6][9];
  Kgeo[9][7] = Kgeo[7][9];
  Kgeo[9][8] = Kgeo[8][9];
  Kgeo[9][9] = t37*Bnl[0][9];
  Kgeo[9][10] = t38*Bnl[1][10];
  Kgeo[9][11] = t39*Bnl[2][11];
  Kgeo[9][12] = t40*Bnl[0][12];
  Kgeo[9][13] = t38*Bnl[1][13];
  Kgeo[9][14] = t39*Bnl[2][14];
  Kgeo[9][15] = t40*Bnl[0][15];
  Kgeo[9][16] = t38*Bnl[1][16];
  Kgeo[9][17] = t39*Bnl[2][17];
  Kgeo[9][18] = t40*Bnl[0][18];
  Kgeo[9][19] = t38*Bnl[1][19];
  Kgeo[9][20] = t39*Bnl[2][20];
  Kgeo[9][21] = t40*Bnl[0][21];
  Kgeo[9][22] = t38*Bnl[1][22];
  Kgeo[9][23] = t39*Bnl[2][23];
  Kgeo[10][0] = Kgeo[0][10];
  Kgeo[10][1] = Kgeo[1][10];
  Kgeo[10][2] = Kgeo[2][10];
  Kgeo[10][3] = Kgeo[3][10];
  Kgeo[10][4] = Kgeo[4][10];
  Kgeo[10][5] = Kgeo[5][10];
  Kgeo[10][6] = Kgeo[6][10];
  Kgeo[10][7] = Kgeo[7][10];
  Kgeo[10][8] = Kgeo[8][10];
  Kgeo[10][9] = Kgeo[9][10];
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
  Kgeo[11][0] = Kgeo[0][11];
  Kgeo[11][1] = Kgeo[1][11];
  Kgeo[11][2] = Kgeo[2][11];
  Kgeo[11][3] = Kgeo[3][11];
  Kgeo[11][4] = Kgeo[4][11];
  Kgeo[11][5] = Kgeo[5][11];
  Kgeo[11][6] = Kgeo[6][11];
  Kgeo[11][7] = Kgeo[7][11];
  Kgeo[11][8] = Kgeo[8][11];
  Kgeo[11][9] = Kgeo[9][11];
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
  Kgeo[12][0] = Kgeo[0][12];
  Kgeo[12][1] = Kgeo[1][12];
  Kgeo[12][2] = Kgeo[2][12];
  Kgeo[12][3] = Kgeo[3][12];
  Kgeo[12][4] = Kgeo[4][12];
  Kgeo[12][5] = Kgeo[5][12];
  Kgeo[12][6] = Kgeo[6][12];
  Kgeo[12][7] = Kgeo[7][12];
  Kgeo[12][8] = Kgeo[8][12];
  Kgeo[12][9] = Kgeo[9][12];
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
  Kgeo[13][0] = Kgeo[0][13];
  Kgeo[13][1] = Kgeo[1][13];
  Kgeo[13][2] = Kgeo[2][13];
  Kgeo[13][3] = Kgeo[3][13];
  Kgeo[13][4] = Kgeo[4][13];
  Kgeo[13][5] = Kgeo[5][13];
  Kgeo[13][6] = Kgeo[6][13];
  Kgeo[13][7] = Kgeo[7][13];
  Kgeo[13][8] = Kgeo[8][13];
  Kgeo[13][9] = Kgeo[9][13];
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
  Kgeo[14][0] = Kgeo[0][14];
  Kgeo[14][1] = Kgeo[1][14];
  Kgeo[14][2] = Kgeo[2][14];
  Kgeo[14][3] = Kgeo[3][14];
  Kgeo[14][4] = Kgeo[4][14];
  Kgeo[14][5] = Kgeo[5][14];
  Kgeo[14][6] = Kgeo[6][14];
  Kgeo[14][7] = Kgeo[7][14];
  Kgeo[14][8] = Kgeo[8][14];
  Kgeo[14][9] = Kgeo[9][14];
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
  Kgeo[15][0] = Kgeo[0][15];
  Kgeo[15][1] = Kgeo[1][15];
  Kgeo[15][2] = Kgeo[2][15];
  Kgeo[15][3] = Kgeo[3][15];
  Kgeo[15][4] = Kgeo[4][15];
  Kgeo[15][5] = Kgeo[5][15];
  Kgeo[15][6] = Kgeo[6][15];
  Kgeo[15][7] = Kgeo[7][15];
  Kgeo[15][8] = Kgeo[8][15];
  Kgeo[15][9] = Kgeo[9][15];
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
  Kgeo[16][0] = Kgeo[0][16];
  Kgeo[16][1] = Kgeo[1][16];
  Kgeo[16][2] = Kgeo[2][16];
  Kgeo[16][3] = Kgeo[3][16];
  Kgeo[16][4] = Kgeo[4][16];
  Kgeo[16][5] = Kgeo[5][16];
  Kgeo[16][6] = Kgeo[6][16];
  Kgeo[16][7] = Kgeo[7][16];
  Kgeo[16][8] = Kgeo[8][16];
  Kgeo[16][9] = Kgeo[9][16];
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
  Kgeo[17][0] = Kgeo[0][17];
  Kgeo[17][1] = Kgeo[1][17];
  Kgeo[17][2] = Kgeo[2][17];
  Kgeo[17][3] = Kgeo[3][17];
  Kgeo[17][4] = Kgeo[4][17];
  Kgeo[17][5] = Kgeo[5][17];
  Kgeo[17][6] = Kgeo[6][17];
  Kgeo[17][7] = Kgeo[7][17];
  Kgeo[17][8] = Kgeo[8][17];
  Kgeo[17][9] = Kgeo[9][17];
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
  Kgeo[18][0] = Kgeo[0][18];
  Kgeo[18][1] = Kgeo[1][18];
  Kgeo[18][2] = Kgeo[2][18];
  Kgeo[18][3] = Kgeo[3][18];
  Kgeo[18][4] = Kgeo[4][18];
  Kgeo[18][5] = Kgeo[5][18];
  Kgeo[18][6] = Kgeo[6][18];
  Kgeo[18][7] = Kgeo[7][18];
  Kgeo[18][8] = Kgeo[8][18];
  Kgeo[18][9] = Kgeo[9][18];
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
  Kgeo[19][0] = Kgeo[0][19];
  Kgeo[19][1] = Kgeo[1][19];
  Kgeo[19][2] = Kgeo[2][19];
  Kgeo[19][3] = Kgeo[3][19];
  Kgeo[19][4] = Kgeo[4][19];
  Kgeo[19][5] = Kgeo[5][19];
  Kgeo[19][6] = Kgeo[6][19];
  Kgeo[19][7] = Kgeo[7][19];
  Kgeo[19][8] = Kgeo[8][19];
  Kgeo[19][9] = Kgeo[9][19];
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
  Kgeo[20][0] = Kgeo[0][20];
  Kgeo[20][1] = Kgeo[1][20];
  Kgeo[20][2] = Kgeo[2][20];
  Kgeo[20][3] = Kgeo[3][20];
  Kgeo[20][4] = Kgeo[4][20];
  Kgeo[20][5] = Kgeo[5][20];
  Kgeo[20][6] = Kgeo[6][20];
  Kgeo[20][7] = Kgeo[7][20];
  Kgeo[20][8] = Kgeo[8][20];
  Kgeo[20][9] = Kgeo[9][20];
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
  Kgeo[21][0] = Kgeo[0][21];
  Kgeo[21][1] = Kgeo[1][21];
  Kgeo[21][2] = Kgeo[2][21];
  Kgeo[21][3] = Kgeo[3][21];
  Kgeo[21][4] = Kgeo[4][21];
  Kgeo[21][5] = Kgeo[5][21];
  Kgeo[21][6] = Kgeo[6][21];
  Kgeo[21][7] = Kgeo[7][21];
  Kgeo[21][8] = Kgeo[8][21];
  Kgeo[21][9] = Kgeo[9][21];
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
  Kgeo[22][0] = Kgeo[0][22];
  Kgeo[22][1] = Kgeo[1][22];
  Kgeo[22][2] = Kgeo[2][22];
  Kgeo[22][3] = Kgeo[3][22];
  Kgeo[22][4] = Kgeo[4][22];
  Kgeo[22][5] = Kgeo[5][22];
  Kgeo[22][6] = Kgeo[6][22];
  Kgeo[22][7] = Kgeo[7][22];
  Kgeo[22][8] = Kgeo[8][22];
  Kgeo[22][9] = Kgeo[9][22];
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
  Kgeo[23][0] = Kgeo[0][23];
  Kgeo[23][1] = Kgeo[1][23];
  Kgeo[23][2] = Kgeo[2][23];
  Kgeo[23][3] = Kgeo[3][23];
  Kgeo[23][4] = Kgeo[4][23];
  Kgeo[23][5] = Kgeo[5][23];
  Kgeo[23][6] = Kgeo[6][23];
  Kgeo[23][7] = Kgeo[7][23];
  Kgeo[23][8] = Kgeo[8][23];
  Kgeo[23][9] = Kgeo[9][23];
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
