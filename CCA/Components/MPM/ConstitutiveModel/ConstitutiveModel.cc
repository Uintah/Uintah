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


#include <CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/MPMFlags.h>
#include <Core/Math/Matrix3.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Math/FastMatrix.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Malloc/Allocator.h>
#include <cmath>
#include <iostream>

using namespace Uintah;

#ifndef M_PI
# define M_PI           3.14159265358979323846  /* pi */
#endif

ConstitutiveModel::ConstitutiveModel(MPMFlags* Mflag)
{
  lb = scinew MPMLabel();
  flag = Mflag;
  if(flag->d_8or27==8){
    NGN=1;
  } else if(flag->d_8or27==27 || flag->d_8or27==64){ 
    NGN=2;
  }
}

ConstitutiveModel::ConstitutiveModel(const ConstitutiveModel* cm)
{
  lb = scinew MPMLabel();
  flag = cm->flag;
  NGN = cm->NGN;
  NGP = cm->NGP;
  d_sharedState = cm->d_sharedState;
}

ConstitutiveModel::~ConstitutiveModel()
{
  delete lb;
}

void 
ConstitutiveModel::addInitialComputesAndRequires(Task* ,
                                                 const MPMMaterial* ,
                                                 const PatchSet*) const
{
}

///////////////////////////////////////////////////////////////////////
/*! Initialize the common quantities that all the explicit constituive
 *  models compute */
///////////////////////////////////////////////////////////////////////
void 
ConstitutiveModel::initSharedDataForExplicit(const Patch* patch,
                                             const MPMMaterial* matl,
                                             DataWarehouse* new_dw)
{
  Matrix3 I; I.Identity();
  Matrix3 zero(0.);
  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  ParticleVariable<double>  pdTdt;
  ParticleVariable<Matrix3> pDefGrad, pStress;

  new_dw->allocateAndPut(pdTdt,       lb->pdTdtLabel,               pset);
  new_dw->allocateAndPut(pDefGrad,    lb->pDeformationMeasureLabel, pset);
  new_dw->allocateAndPut(pStress,     lb->pStressLabel,             pset);

  // To fix : For a material that is initially stressed we need to
  // modify the stress tensors to comply with the initial stress state
  ParticleSubset::iterator iter = pset->begin();
  for(; iter != pset->end(); iter++){
    particleIndex idx = *iter;
    pdTdt[idx] = 0.0;
    pDefGrad[idx] = I;
    pStress[idx] = zero;
  }
}

void 
ConstitutiveModel::addComputesAndRequires(Task*, 
                                          const MPMMaterial*,
                                          const PatchSet*) const
{
  throw InternalError("Stub Task: ConstitutiveModel::addComputesAndRequires ", __FILE__, __LINE__);
}

void 
ConstitutiveModel::addComputesAndRequires(Task*, 
                                          const MPMMaterial*,
                                          const PatchSet*,
                                          const bool ,
                                          const bool) const
{
  throw InternalError("Stub Task: ConstitutiveModel::addComputesAndRequires ", __FILE__, __LINE__);  
}

void ConstitutiveModel::scheduleCheckNeedAddMPMMaterial(Task* task, 
                                                        const MPMMaterial*,
                                                        const PatchSet*) const
{
  task->computes(lb->NeedAddMPMMaterialLabel);
}

void 
ConstitutiveModel::addSharedCRForHypoExplicit(Task* task,
                                              const MaterialSubset* matlset,
                                              const PatchSet* p) const
{
  Ghost::GhostType  gnone = Ghost::None;
  addSharedCRForExplicit(task, matlset, p);
  task->requires(Task::OldDW, lb->pStressLabel,             matlset, gnone);
}

void 
ConstitutiveModel::addSharedCRForExplicit(Task* task,
                                          const MaterialSubset* matlset,
                                          const PatchSet* ) const
{
  Ghost::GhostType  gnone = Ghost::None;
  Ghost::GhostType  gac   = Ghost::AroundCells;

  task->requires(Task::OldDW, lb->delTLabel);
  task->requires(Task::OldDW, lb->pXLabel,                  matlset, gnone);
  task->requires(Task::OldDW, lb->pMassLabel,               matlset, gnone);
  task->requires(Task::OldDW, lb->pVolumeLabel,             matlset, gnone);
  task->requires(Task::OldDW, lb->pTemperatureLabel,        matlset, gnone);
  task->requires(Task::OldDW, lb->pVelocityLabel,           matlset, gnone);
  task->requires(Task::OldDW, lb->pDeformationMeasureLabel, matlset, gnone);
  task->requires(Task::NewDW, lb->gVelocityStarLabel,       matlset, gac, NGN);
  if(!flag->d_doGridReset){
    task->requires(Task::NewDW, lb->gDisplacementLabel,     matlset, gac, NGN);
  }
  task->requires(Task::OldDW, lb->pSizeLabel,               matlset, gnone);
  task->requires(Task::OldDW, lb->pTempPreviousLabel,       matlset, gnone);
  if (flag->d_fracture) {
    task->requires(Task::NewDW, lb->pgCodeLabel,            matlset, gnone); 
    task->requires(Task::NewDW, lb->GVelocityStarLabel,     matlset, gac, NGN);
  }

  task->computes(lb->pStressLabel_preReloc,             matlset);
  task->computes(lb->pDeformationMeasureLabel_preReloc, matlset);
  task->computes(lb->pVolumeLabel_preReloc,             matlset);
  task->computes(lb->pdTdtLabel_preReloc,               matlset);
  //task->computes(lb->p_qLabel_preReloc,                 matlset);
}

void 
ConstitutiveModel::computeStressTensor(const PatchSubset*,
                                       const MPMMaterial*,
                                       DataWarehouse*,
                                       DataWarehouse*)
{
  throw InternalError("Stub Task: ConstitutiveModel::computeStressTensor ", __FILE__, __LINE__);
}

void 
ConstitutiveModel::computeStressTensorImplicit(const PatchSubset*,
                                               const MPMMaterial*,
                                               DataWarehouse*,
                                               DataWarehouse*)
{
  throw InternalError("Stub Task: ConstitutiveModel::computeStressTensorImplicit ", __FILE__, __LINE__);
}

void ConstitutiveModel::checkNeedAddMPMMaterial(const PatchSubset*,
                                                const MPMMaterial*,
                                                DataWarehouse* new_dw,
                                                DataWarehouse*)
{
  double need_add=0.;
                                                                                
  new_dw->put(sum_vartype(need_add),     lb->NeedAddMPMMaterialLabel);
}


void 
ConstitutiveModel::carryForward(const PatchSubset*,
                                const MPMMaterial*,
                                DataWarehouse*,
                                DataWarehouse*)
{
  throw InternalError("Stub Task: ConstitutiveModel::carryForward ", __FILE__, __LINE__);
}

void
ConstitutiveModel::carryForwardSharedData(ParticleSubset* pset,
                                          DataWarehouse*  old_dw,
                                          DataWarehouse*  new_dw,
                                          const MPMMaterial* matl)
{
  double rho_orig = matl->getInitialDensity();
  Matrix3 Id, Zero(0.0); Id.Identity();

  constParticleVariable<double>  pMass;
  constParticleVariable<Matrix3> pDefGrad_old;
  old_dw->get(pMass,            lb->pMassLabel,               pset);
  old_dw->get(pDefGrad_old,     lb->pDeformationMeasureLabel, pset);

  ParticleVariable<double>  pVol_new, pIntHeatRate_new,p_q;
  ParticleVariable<Matrix3> pDefGrad_new, pStress_new;
  new_dw->allocateAndPut(pVol_new,         lb->pVolumeLabel_preReloc,  pset);
  new_dw->allocateAndPut(pIntHeatRate_new, lb->pdTdtLabel_preReloc,    pset);
  new_dw->allocateAndPut(pDefGrad_new,  lb->pDeformationMeasureLabel_preReloc, 
                                                                       pset);
  new_dw->allocateAndPut(pStress_new,   lb->pStressLabel_preReloc,     pset);
  new_dw->allocateAndPut(p_q,           lb->p_qLabel_preReloc,         pset);

  ParticleSubset::iterator iter = pset->begin();
  for(; iter != pset->end(); iter++){
    particleIndex idx = *iter;
    pVol_new[idx] = (pMass[idx]/rho_orig);
    pIntHeatRate_new[idx] = 0.0;
    pDefGrad_new[idx] = pDefGrad_old[idx];
    //pDefGrad_new[idx] = Id;
    pStress_new[idx] = Zero;
    p_q[idx]=0.;
  }
}

void 
ConstitutiveModel::allocateCMDataAddRequires(Task*, const MPMMaterial*,
                                             const PatchSet*,
                                             MPMLabel*) const
{
  throw InternalError("Stub Task: ConstitutiveModel::allocateCMDataAddRequires ", __FILE__, __LINE__);
}

void 
ConstitutiveModel::addSharedRForConvertExplicit(Task* task,
                                                const MaterialSubset* mset,
                                                const PatchSet*) const
{
  Ghost::GhostType  gnone = Ghost::None;
  task->requires(Task::NewDW,lb->pdTdtLabel_preReloc,              mset,gnone);
  task->requires(Task::NewDW,lb->pDeformationMeasureLabel_preReloc,mset,gnone);
  task->requires(Task::NewDW,lb->pStressLabel_preReloc,            mset,gnone);
}

void
ConstitutiveModel::copyDelToAddSetForConvertExplicit(DataWarehouse* new_dw,
                                                     ParticleSubset* delset,
                                                     ParticleSubset* addset,
                                                     map<const VarLabel*, ParticleVariableBase*>* newState)
{
  constParticleVariable<double>  pIntHeatRate_del;
  constParticleVariable<Matrix3> pDefGrad_del;
  constParticleVariable<Matrix3> pStress_del;

  new_dw->get(pIntHeatRate_del, lb->pdTdtLabel_preReloc,               delset);
  new_dw->get(pDefGrad_del,     lb->pDeformationMeasureLabel_preReloc, delset);
  new_dw->get(pStress_del,      lb->pStressLabel_preReloc,             delset);

  ParticleVariable<double>  pIntHeatRate_add;
  ParticleVariable<Matrix3> pDefGrad_add;
  ParticleVariable<Matrix3> pStress_add;

  new_dw->allocateTemporary(pIntHeatRate_add, addset);
  new_dw->allocateTemporary(pDefGrad_add,     addset);
  new_dw->allocateTemporary(pStress_add,      addset);

  ParticleSubset::iterator del = delset->begin();
  ParticleSubset::iterator add = addset->begin();
  for (; del != delset->end(); del++, add++) {
    pIntHeatRate_add[*add] = pIntHeatRate_del[*del];
    pDefGrad_add[*add] = pDefGrad_del[*del];
    pStress_add[*add]  = pStress_del[*del];
  }

  (*newState)[lb->pdTdtLabel] = pIntHeatRate_add.clone();
  (*newState)[lb->pDeformationMeasureLabel] = pDefGrad_add.clone();
  (*newState)[lb->pStressLabel] = pStress_add.clone();
}

void 
ConstitutiveModel::addRequiresDamageParameter(Task*, 
                                              const MPMMaterial*,
                                              const PatchSet*) const
{
}

void 
ConstitutiveModel::getDamageParameter(const Patch* ,
                                      ParticleVariable<int>& ,int ,
                                      DataWarehouse* ,
                                      DataWarehouse* )
{
}

Vector 
ConstitutiveModel::getInitialFiberDir()
{
  return Vector(0.,0.,1);
}

//______________________________________________________________________
//______________________________________________________________________
//          HARDWIRE FOR AN IDEAL GAS -Todd 
double 
ConstitutiveModel::computeRhoMicro(double press, double gamma,
                                   double cv, double Temp, double rho_guess)
{
  // Pointwise computation of microscopic density
  return  press/((gamma - 1.0)*cv*Temp);
}

void 
ConstitutiveModel::computePressEOS(double rhoM, double gamma,
                                   double cv, double Temp, double& press, 
                                   double& dp_drho, double& dp_de)
{
  // Pointwise computation of thermodynamic quantities
  press   = (gamma - 1.0)*rhoM*cv*Temp;
  dp_drho = (gamma - 1.0)*cv*Temp;
  dp_de   = (gamma - 1.0)*rhoM;
}
//______________________________________________________________________


// Convert J-integral into stress intensity (for FRACTURE)
void 
ConstitutiveModel::ConvertJToK(const MPMMaterial*,
                               const string&,
                               const Vector&,
                               const double&,
                               const Vector&,
                               Vector& SIF)
{
  SIF=Vector(-9999.,-9999.,-9999.);
}

// Detect if crack propagtes and the propagation direction (for FRACTURE)
short
ConstitutiveModel::CrackPropagates(const double& , const double& , 
                                   const double& , double& theta)
{
  enum {NO=0, YES};
  theta=0.0;
  return NO;
}

double 
ConstitutiveModel::artificialBulkViscosity(double Dkk, 
                                           double c_bulk, 
                                           double rho,
                                           double dx) const 
{
  double q = 0.0;
  if (Dkk < 0.0) {
    double A1 = flag->d_artificialViscCoeff1;
    double A2 = flag->d_artificialViscCoeff2;
    //double c_bulk = sqrt(K/rho);
    q = (A1*fabs(c_bulk*Dkk*dx) + A2*(Dkk*Dkk*dx*dx))*rho;
  }
  return q;
}

void
ConstitutiveModel::computeDeformationGradientFromDisplacement(
                                           constNCVariable<Vector> gDisp,
                                           ParticleSubset* pset,
                                           constParticleVariable<Point> px,
                                           constParticleVariable<Vector> psize,
                                           ParticleVariable<Matrix3> &Fnew,
                                           constParticleVariable<Matrix3> &Fold,
                                           Vector dx,
                                           ParticleInterpolator* interp) 
{
  Matrix3 dispGrad,Identity;
  Identity.Identity();
  vector<IntVector> ni(interp->size());
  vector<Vector> d_S(interp->size());
  double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};
                                                                            
  for(ParticleSubset::iterator iter = pset->begin();
       iter != pset->end(); iter++){
    particleIndex idx = *iter;
                                                                            
    // Get the node indices that surround the cell
    interp->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],Fold[idx]);
                                                                            
    computeGrad(dispGrad, ni, d_S, oodx, gDisp);

    // Update the deformation gradient tensor to its time n+1 value.
    // Compute the deformation gradient from the displacement gradient
    Fnew[idx] = Identity + dispGrad;

    double J = Fnew[idx].Determinant();
    if (!(J > 0)) {
      ostringstream warn;
      warn << "**ERROR** : ConstitutiveModel::computeDeformationGradientFromDisplacement" << endl << "Negative or zero determinant of Jacobian." << endl;
      warn << "     Particle = " << idx << " J = " << J << " position = " << px[idx] << endl;
      warn << "     Disp Grad = " << dispGrad << endl; 
      warn << "     F_new = " << Fnew[idx] << endl; 
      throw InvalidValue(warn.str(), __FILE__, __LINE__);
    }
  }
}

void 
ConstitutiveModel::computeDeformationGradientFromVelocity(
                                           constNCVariable<Vector> gVel,
                                           ParticleSubset* pset,
                                           constParticleVariable<Point> px,
                                           constParticleVariable<Vector> psize,
                                           constParticleVariable<Matrix3> Fold,
                                           ParticleVariable<Matrix3> &Fnew,
                                           Vector dx,
                                           ParticleInterpolator* interp,
                                           const double& delT)
{
    Matrix3 velGrad,deformationGradientInc, Identity;
    Identity.Identity();
    vector<IntVector> ni(interp->size());
    vector<Vector> d_S(interp->size());
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    for(ParticleSubset::iterator iter = pset->begin();
      iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Get the node indices that surround the cell
      interp->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],Fold[idx]);

      computeGrad(velGrad, ni, d_S, oodx, gVel);

      // Compute the deformation gradient increment using the time_step
      // velocity gradient
      // F_n^np1 = dudx * dt + Identity
      deformationGradientInc = velGrad * delT + Identity;
                                                                              
      // Update the deformation gradient tensor to its time n+1 value.
      Fnew[idx] = deformationGradientInc * Fold[idx];

      double J = Fnew[idx].Determinant();
      if (!(J > 0)) {
        ostringstream warn;
        warn << "**ERROR** CompNeoHook: Negative or zero determinant of Jacobian."
             << " Particle has inverted." << endl;
        warn << "     Particle = " << idx << ", J = " << J << ", position = " << px[idx]<<endl;
        warn << "          Vel Grad = \n" << velGrad << endl; 
        warn << "          F_inc = \n" << deformationGradientInc << endl; 
        warn << "          F_old = \n" << Fold[idx] << endl; 
        warn << "          F_new = \n" << Fnew[idx] << endl; 
        warn << "          gVelocity:" << endl;
        for(int k = 0; k < flag->d_8or27; k++) {
          warn<< "             node: " << ni[k] << " vel: " << gVel[ni[k]] << endl;
        }
        
        throw InvalidValue(warn.str(), __FILE__, __LINE__);
      }

    }
}

void
ConstitutiveModel::computeDeformationGradientFromTotalDisplacement(
                                           constNCVariable<Vector> gDisp,
                                           ParticleSubset* pset,
                                           constParticleVariable<Point> px,
                                           ParticleVariable<Matrix3> &Fnew,
                                           constParticleVariable<Matrix3>& Fold,
                                           Vector dx,
                                           constParticleVariable<Vector> psize,
                                           ParticleInterpolator* interp)
{
  Matrix3 dispGrad,Identity;
  Identity.Identity();
  vector<IntVector> ni(interp->size());
  vector<double> S(interp->size());
  vector<Vector> d_S(interp->size());
  double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};
                                                                                
  for(ParticleSubset::iterator iter = pset->begin();
       iter != pset->end(); iter++){
    particleIndex idx = *iter;
                                                                                
    // Get the node indices that surround the cell
    interp->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],Fold[idx]);
                                                                                
    computeGrad(dispGrad, ni, d_S, oodx, gDisp);
                                                                                
    // Update the deformation gradient tensor to its time n+1 value.
    // Compute the deformation gradient from the displacement gradient
    Fnew[idx] = Identity + dispGrad;
  }
}
                                                                                
void
ConstitutiveModel::computeDeformationGradientFromIncrementalDisplacement(
                                           constNCVariable<Vector> gDisp,
                                           ParticleSubset* pset,
                                           constParticleVariable<Point> px,
                                           constParticleVariable<Matrix3> Fold,
                                           ParticleVariable<Matrix3> &Fnew,
                                           Vector dx,
                                           constParticleVariable<Vector> psize,
                                           ParticleInterpolator* interp)
{
    Matrix3 IncDispGrad,deformationGradientInc, Identity;
    Identity.Identity();
    vector<IntVector> ni(interp->size());
    vector<double> S(interp->size());
    vector<Vector> d_S(interp->size());

    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};
                                                                                
    for(ParticleSubset::iterator iter = pset->begin();
      iter != pset->end(); iter++){
      particleIndex idx = *iter;
                                                                                
      // Get the node indices that surround the cell
      interp->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],Fold[idx]);
                                                                                
      computeGrad(IncDispGrad, ni, d_S, oodx, gDisp);
                                                                                
      // Compute the deformation gradient increment
      deformationGradientInc = IncDispGrad + Identity;
                                                                                
      // Update the deformation gradient tensor to its time n+1 value.
      Fnew[idx] = deformationGradientInc * Fold[idx];
    }
}
