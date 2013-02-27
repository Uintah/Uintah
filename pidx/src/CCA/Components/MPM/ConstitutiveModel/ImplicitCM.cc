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

#include <CCA/Components/MPM/ConstitutiveModel/ImplicitCM.h>
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
#include <Core/Malloc/Allocator.h>
#include <cmath>
#include <iostream>

using namespace Uintah;


ImplicitCM::ImplicitCM()
{
  d_lb = scinew MPMLabel();
}

ImplicitCM::ImplicitCM(const ImplicitCM* cm)
{
  d_lb = scinew MPMLabel();
}

ImplicitCM::~ImplicitCM()
{
  delete d_lb;
}

///////////////////////////////////////////////////////////////////////
/*! Initialize the common quantities that all the implicit constituive
 *  models compute */
///////////////////////////////////////////////////////////////////////
void 
ImplicitCM::initSharedDataForImplicit(const Patch* patch,
                                      const MPMMaterial* matl,
                                      DataWarehouse* new_dw)
{
  Matrix3 I; I.Identity();
  Matrix3 zero(0.);
  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

//  ParticleVariable<double>  pdTdt;
  ParticleVariable<Matrix3> pDefGrad, pStress;

//  new_dw->allocateAndPut(pdTdt,       d_lb->pdTdtLabel,               pset);
  new_dw->allocateAndPut(pDefGrad,    d_lb->pDeformationMeasureLabel, pset);
  new_dw->allocateAndPut(pStress,     d_lb->pStressLabel,             pset);

  // To fix : For a material that is initially stressed we need to
  // modify the stress tensors to comply with the initial stress state
  ParticleSubset::iterator iter = pset->begin();
  for(; iter != pset->end(); iter++){
    particleIndex idx = *iter;
//    pdTdt[idx] = 0.0;
    pDefGrad[idx] = I;
    pStress[idx] = zero;
  }
}

void 
ImplicitCM::addComputesAndRequires(Task*, 
                                   const MPMMaterial*,
                                   const PatchSet*,
                                   const bool) const
{
}


void 
ImplicitCM::addSharedCRForImplicit(Task* task,
                                          const MaterialSubset* matlset,
                                          const bool reset) const
{
  Ghost::GhostType  gnone = Ghost::None;
  Ghost::GhostType  gac   = Ghost::AroundCells;

  task->requires(Task::OldDW, d_lb->delTLabel);
  task->requires(Task::OldDW, d_lb->pXLabel,           matlset, gnone);
  task->requires(Task::OldDW, d_lb->pSizeLabel,        matlset, gnone);
  task->requires(Task::OldDW, d_lb->pMassLabel,        matlset, gnone);
  task->requires(Task::OldDW, d_lb->pVolumeLabel,      matlset, gnone);
  task->requires(Task::OldDW, d_lb->pTemperatureLabel, matlset, gnone);
  task->requires(Task::OldDW, d_lb->pDeformationMeasureLabel,
                                                       matlset, gnone);
  task->requires(Task::OldDW, d_lb->pStressLabel,      matlset, gnone);
  if(reset){
    task->requires(Task::NewDW, d_lb->dispNewLabel,      matlset,gac,1);
  } else {
    task->requires(Task::NewDW, d_lb->gDisplacementLabel,matlset,gac,1);
  }

  task->computes(d_lb->pStressLabel_preReloc,             matlset);  
  task->computes(d_lb->pDeformationMeasureLabel_preReloc, matlset);
  task->computes(d_lb->pVolumeDeformedLabel,              matlset);
  task->computes(d_lb->pdTdtLabel_preReloc,               matlset);
}

void
ImplicitCM::addSharedCRForImplicitHypo(Task* task,
                                       const MaterialSubset* matlset,
                                       const bool reset) const
{

  addSharedCRForImplicit(task,matlset,reset);
  task->requires(Task::OldDW, d_lb->pStressLabel,      matlset, Ghost::None);
}

void 
ImplicitCM::addSharedCRForImplicit(Task* task,
                                          const MaterialSubset* matlset,
                                          const bool reset,
                                          const bool /*recurse*/, 
                                          const bool SchedParent) const
{
  Ghost::GhostType  gnone = Ghost::None;
  Ghost::GhostType  gac   = Ghost::AroundCells;

  if(SchedParent){
    // For subscheduler
    task->requires(Task::ParentOldDW, d_lb->pXLabel,           matlset, gnone);
    task->requires(Task::ParentOldDW, d_lb->pSizeLabel,        matlset, gnone);
    task->requires(Task::ParentOldDW, d_lb->pMassLabel,        matlset, gnone);
    task->requires(Task::ParentOldDW, d_lb->pVolumeLabel,      matlset, gnone);
    task->requires(Task::ParentOldDW, d_lb->pTemperatureLabel, matlset, gnone);
    task->requires(Task::ParentOldDW, d_lb->pDeformationMeasureLabel,
                                                               matlset, gnone);

    task->computes(d_lb->pStressLabel_preReloc,                 matlset);  
    task->computes(d_lb->pDeformationMeasureLabel_preReloc,     matlset);
    task->computes(d_lb->pVolumeDeformedLabel,                  matlset);
    task->computes(d_lb->pdTdtLabel_preReloc,                   matlset);
    if(reset){
      task->requires(Task::OldDW,     d_lb->dispNewLabel,      matlset, gac,1);
    }else {
      task->requires(Task::OldDW,     d_lb->gDisplacementLabel,matlset, gac,1);
    }
  }
  else{
    // For scheduleIterate
    task->requires(Task::OldDW, d_lb->pXLabel,                  matlset, gnone);
    task->requires(Task::OldDW, d_lb->pSizeLabel,               matlset, gnone);
    task->requires(Task::OldDW, d_lb->pMassLabel,               matlset, gnone);
    task->requires(Task::OldDW, d_lb->pVolumeLabel,             matlset, gnone);
    task->requires(Task::OldDW, d_lb->pTemperatureLabel,        matlset, gnone);
    task->requires(Task::OldDW, d_lb->pDeformationMeasureLabel, matlset, gnone);
  }

}

void
ImplicitCM::addSharedCRForImplicitHypo(Task* task,
                                       const MaterialSubset* matlset,
                                       const bool reset,
                                       const bool /*recurse*/,
                                       const bool SchedParent) const
{
  addSharedCRForImplicit(task,matlset,reset,true,SchedParent);
  if(SchedParent){
    // For subscheduler
    task->requires(Task::ParentOldDW, d_lb->pStressLabel, matlset, Ghost::None);
  }else{
    task->requires(Task::OldDW,       d_lb->pStressLabel, matlset, Ghost::None);
  }
}

void
ImplicitCM::carryForwardSharedDataImplicit(ParticleSubset* pset,
                                           DataWarehouse*  old_dw,
                                           DataWarehouse*  new_dw,
                                           const MPMMaterial* matl)
{
  double rho_orig = matl->getInitialDensity();
  Matrix3 Id, Zero(0.0); Id.Identity();
                                                                                
  constParticleVariable<double>  pMass;
  constParticleVariable<Matrix3> pDefGrad_old;
  old_dw->get(pMass,            d_lb->pMassLabel,               pset);
  old_dw->get(pDefGrad_old,     d_lb->pDeformationMeasureLabel, pset);
                                                                                
  ParticleVariable<double>  pIntHeatRate_new, pVol_Def_new;
  ParticleVariable<Matrix3> pDefGrad_new, pStress_new;
  new_dw->allocateAndPut(pVol_Def_new,     d_lb->pVolumeDeformedLabel,   pset);
  new_dw->allocateAndPut(pIntHeatRate_new, d_lb->pdTdtLabel_preReloc,    pset);
  new_dw->allocateAndPut(pDefGrad_new,  d_lb->pDeformationMeasureLabel_preReloc,
                         pset);
  new_dw->allocateAndPut(pStress_new,   d_lb->pStressLabel_preReloc, pset);
                                                                                
  ParticleSubset::iterator iter = pset->begin();
  for(; iter != pset->end(); iter++){
    particleIndex idx = *iter;
    pVol_Def_new[idx] = (pMass[idx]/rho_orig);
    pIntHeatRate_new[idx] = 0.0;
    pDefGrad_new[idx] = pDefGrad_old[idx];
    //pDefGrad_new[idx] = Id;
    pStress_new[idx] = Zero;
  }
}

void 
ImplicitCM::computeStressTensorImplicit(const PatchSubset*,
                                       const MPMMaterial*,
                                       DataWarehouse*,
                                       DataWarehouse*,
                                       Solver* ,
                                       const bool)
{
  throw InternalError("Stub Task: ImplicitCM::computeStressTensorImplicit ", __FILE__, __LINE__);
}

void ImplicitCM::loadBMatsGIMP(Array3<int> l2g,
                               int dof[81],
                               double B[6][81],
                               double Bnl[3][81],
                               vector<Vector> d_S,
                               vector<IntVector> ni,
                               double* oodx) const
{
    for(int k = 0; k < 27; k++) {
      // Need to loop over the neighboring patches l2g to get the right
      // dof number.

      int l2g_node_num = l2g[ni[k]];
      dof[3*k]  =l2g_node_num;
      dof[3*k+1]=l2g_node_num+1;
      dof[3*k+2]=l2g_node_num+2;
                                                                                
      B[0][3*k] = d_S[k][0]*oodx[0];
      B[3][3*k] = d_S[k][1]*oodx[1];
      B[5][3*k] = d_S[k][2]*oodx[2];
      B[1][3*k] = 0.;
      B[2][3*k] = 0.;
      B[4][3*k] = 0.;
                                                                                
      B[1][3*k+1] = d_S[k][1]*oodx[1];
      B[3][3*k+1] = d_S[k][0]*oodx[0];
      B[4][3*k+1] = d_S[k][2]*oodx[2];
      B[0][3*k+1] = 0.;
      B[2][3*k+1] = 0.;
      B[5][3*k+1] = 0.;
                                                                                
      B[2][3*k+2] = d_S[k][2]*oodx[2];
      B[4][3*k+2] = d_S[k][1]*oodx[1];
      B[5][3*k+2] = d_S[k][0]*oodx[0];
      B[0][3*k+2] = 0.;
      B[1][3*k+2] = 0.;
      B[3][3*k+2] = 0.;
                                                                                
      Bnl[0][3*k] = d_S[k][0]*oodx[0];
      Bnl[1][3*k] = 0.;
      Bnl[2][3*k] = 0.;
      Bnl[0][3*k+1] = 0.;
      Bnl[1][3*k+1] = d_S[k][1]*oodx[1];
      Bnl[2][3*k+1] = 0.;
      Bnl[0][3*k+2] = 0.;
      Bnl[1][3*k+2] = 0.;
      Bnl[2][3*k+2] = d_S[k][2]*oodx[2];
    }
}

void ImplicitCM::loadBMats(Array3<int> l2g,
                           int dof[24],
                           double B[6][24],
                           double Bnl[3][24],
                           vector<Vector> d_S,
                           vector<IntVector> ni,
                           double* oodx) const
{
    for(int k = 0; k < 8; k++) {
      // Need to loop over the neighboring patches l2g to get the right
      // dof number.
      int l2g_node_num = l2g[ni[k]];
      dof[3*k]  =l2g_node_num;
      dof[3*k+1]=l2g_node_num+1;
      dof[3*k+2]=l2g_node_num+2;
                                                                            
      B[0][3*k] = d_S[k][0]*oodx[0];
      B[3][3*k] = d_S[k][1]*oodx[1];
      B[5][3*k] = d_S[k][2]*oodx[2];
      B[1][3*k] = 0.;
      B[2][3*k] = 0.;
      B[4][3*k] = 0.;
                                                                            
      B[1][3*k+1] = d_S[k][1]*oodx[1];
      B[3][3*k+1] = d_S[k][0]*oodx[0];
      B[4][3*k+1] = d_S[k][2]*oodx[2];
      B[0][3*k+1] = 0.;
      B[2][3*k+1] = 0.;
      B[5][3*k+1] = 0.;
                                                                            
      B[2][3*k+2] = d_S[k][2]*oodx[2];
      B[4][3*k+2] = d_S[k][1]*oodx[1];
      B[5][3*k+2] = d_S[k][0]*oodx[0];
      B[0][3*k+2] = 0.;
      B[1][3*k+2] = 0.;
      B[3][3*k+2] = 0.;
                                                                            
      Bnl[0][3*k] = d_S[k][0]*oodx[0];
      Bnl[1][3*k] = 0.;
      Bnl[2][3*k] = 0.;
      Bnl[0][3*k+1] = 0.;
      Bnl[1][3*k+1] = d_S[k][1]*oodx[1];
      Bnl[2][3*k+1] = 0.;
      Bnl[0][3*k+2] = 0.;
      Bnl[1][3*k+2] = 0.;
      Bnl[2][3*k+2] = d_S[k][2]*oodx[2];
    }
}

void
ImplicitCM::BnltDBnlGIMP(double Bnl[3][81], 
                         double sig[3][3],
                         double BnTsigBn[81][81]) const
{
    for (int i=0;i<81;i++){
      for (int j=0;j<81;j++){
        BnTsigBn[i][j] = 0.;
        for (int k=0;k<3;k++){
          for (int l=0;l<3;l++){
            BnTsigBn[i][j]+=Bnl[l][i]*sig[l][k]*Bnl[k][j];
          }
        }
      }
    }
}

void
ImplicitCM::BnltDBnl(double Bnl[3][24], 
                     double sig[3][3],
                     double BnTsigBn[24][24]) const
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

  t1 = Bnl[0][0]*Bnl[0][0];
  BnTsigBn[0][0] = t1*sig[0][0];
  t2 = Bnl[0][0]*sig[0][1];
  BnTsigBn[0][1] = t2*Bnl[1][1];
  t3 = Bnl[0][0]*sig[0][2];
  BnTsigBn[0][2] = t3*Bnl[2][2];
  t4 = Bnl[0][0]*sig[0][0];
  BnTsigBn[0][3] = t4*Bnl[0][3];
  BnTsigBn[0][4] = t2*Bnl[1][4];
  BnTsigBn[0][5] = t3*Bnl[2][5];
  BnTsigBn[0][6] = t4*Bnl[0][6];
  BnTsigBn[0][7] = t2*Bnl[1][7];
  BnTsigBn[0][8] = t3*Bnl[2][8];
  BnTsigBn[0][9] = t4*Bnl[0][9];
  BnTsigBn[0][10] = t2*Bnl[1][10];
  BnTsigBn[0][11] = t3*Bnl[2][11];
  BnTsigBn[0][12] = t4*Bnl[0][12];
  BnTsigBn[0][13] = t2*Bnl[1][13];
  BnTsigBn[0][14] = t3*Bnl[2][14];
  BnTsigBn[0][15] = t4*Bnl[0][15];
  BnTsigBn[0][16] = t2*Bnl[1][16];
  BnTsigBn[0][17] = t3*Bnl[2][17];
  BnTsigBn[0][18] = t4*Bnl[0][18];
  BnTsigBn[0][19] = t2*Bnl[1][19];
  BnTsigBn[0][20] = t3*Bnl[2][20];
  BnTsigBn[0][21] = t4*Bnl[0][21];
  BnTsigBn[0][22] = t2*Bnl[1][22];
  BnTsigBn[0][23] = t3*Bnl[2][23];
  BnTsigBn[1][0] = BnTsigBn[0][1];
  t5 = Bnl[1][1]*Bnl[1][1];
  BnTsigBn[1][1] = t5*sig[1][1];
  t6 = Bnl[1][1]*sig[1][2];
  BnTsigBn[1][2] = t6*Bnl[2][2];
  t7 = Bnl[1][1]*sig[0][1];
  BnTsigBn[1][3] = t7*Bnl[0][3];
  t8 = Bnl[1][1]*sig[1][1];
  BnTsigBn[1][4] = Bnl[1][4]*t8;
  BnTsigBn[1][5] = t6*Bnl[2][5];
  BnTsigBn[1][6] = t7*Bnl[0][6];
  BnTsigBn[1][7] = Bnl[1][7]*t8;
  BnTsigBn[1][8] = t6*Bnl[2][8];
  BnTsigBn[1][9] = t7*Bnl[0][9];
  BnTsigBn[1][10] = Bnl[1][10]*t8;
  BnTsigBn[1][11] = t6*Bnl[2][11];
  BnTsigBn[1][12] = t7*Bnl[0][12];
  BnTsigBn[1][13] = Bnl[1][13]*t8;
  BnTsigBn[1][14] = t6*Bnl[2][14];
  BnTsigBn[1][15] = t7*Bnl[0][15];
  BnTsigBn[1][16] = Bnl[1][16]*t8;
  BnTsigBn[1][17] = t6*Bnl[2][17];
  BnTsigBn[1][18] = t7*Bnl[0][18];
  BnTsigBn[1][19] = Bnl[1][19]*t8;
  BnTsigBn[1][20] = t6*Bnl[2][20];
  BnTsigBn[1][21] = t7*Bnl[0][21];
  BnTsigBn[1][22] = Bnl[1][22]*t8;
  BnTsigBn[1][23] = t6*Bnl[2][23];
  BnTsigBn[2][0] = BnTsigBn[0][2];
  BnTsigBn[2][1] = BnTsigBn[1][2];
  t9 = Bnl[2][2]*Bnl[2][2];
  BnTsigBn[2][2] = t9*sig[2][2];
  t10 = Bnl[2][2]*sig[0][2];
  BnTsigBn[2][3] = t10*Bnl[0][3];
  t11 = Bnl[2][2]*sig[1][2];
  BnTsigBn[2][4] = Bnl[1][4]*t11;
  t12 = Bnl[2][2]*sig[2][2];
  BnTsigBn[2][5] = t12*Bnl[2][5];
  BnTsigBn[2][6] = t10*Bnl[0][6];
  BnTsigBn[2][7] = Bnl[1][7]*t11;
  BnTsigBn[2][8] = t12*Bnl[2][8];
  BnTsigBn[2][9] = t10*Bnl[0][9];
  BnTsigBn[2][10] = Bnl[1][10]*t11;
  BnTsigBn[2][11] = t12*Bnl[2][11];
  BnTsigBn[2][12] = t10*Bnl[0][12];
  BnTsigBn[2][13] = Bnl[1][13]*t11;
  BnTsigBn[2][14] = t12*Bnl[2][14];
  BnTsigBn[2][15] = t10*Bnl[0][15];
  BnTsigBn[2][16] = Bnl[1][16]*t11;
  BnTsigBn[2][17] = t12*Bnl[2][17];
  BnTsigBn[2][18] = t10*Bnl[0][18];
  BnTsigBn[2][19] = t11*Bnl[1][19];
  BnTsigBn[2][20] = t12*Bnl[2][20];
  BnTsigBn[2][21] = t10*Bnl[0][21];
  BnTsigBn[2][22] = t11*Bnl[1][22];
  BnTsigBn[2][23] = t12*Bnl[2][23];
  BnTsigBn[3][0] = BnTsigBn[0][3];
  BnTsigBn[3][1] = BnTsigBn[1][3];
  BnTsigBn[3][2] = BnTsigBn[2][3];
  t13 = Bnl[0][3]*Bnl[0][3];
  BnTsigBn[3][3] = t13*sig[0][0];
  t14 = Bnl[0][3]*sig[0][1];
  BnTsigBn[3][4] = t14*Bnl[1][4];
  t15 = Bnl[0][3]*sig[0][2];
  BnTsigBn[3][5] = Bnl[2][5]*t15;
  t16 = Bnl[0][3]*sig[0][0];
  BnTsigBn[3][6] = t16*Bnl[0][6];
  BnTsigBn[3][7] = t14*Bnl[1][7];
  BnTsigBn[3][8] = Bnl[2][8]*t15;
  BnTsigBn[3][9] = t16*Bnl[0][9];
  BnTsigBn[3][10] = t14*Bnl[1][10];
  BnTsigBn[3][11] = Bnl[2][11]*t15;
  BnTsigBn[3][12] = t16*Bnl[0][12];
  BnTsigBn[3][13] = t14*Bnl[1][13];
  BnTsigBn[3][14] = Bnl[2][14]*t15;
  BnTsigBn[3][15] = t16*Bnl[0][15];
  BnTsigBn[3][16] = t14*Bnl[1][16];
  BnTsigBn[3][17] = Bnl[2][17]*t15;
  BnTsigBn[3][18] = t16*Bnl[0][18];
  BnTsigBn[3][19] = t14*Bnl[1][19];
  BnTsigBn[3][20] = Bnl[2][20]*t15;
  BnTsigBn[3][21] = t16*Bnl[0][21];
  BnTsigBn[3][22] = t14*Bnl[1][22];
  BnTsigBn[3][23] = Bnl[2][23]*t15;
  BnTsigBn[4][0] = BnTsigBn[0][4];
  BnTsigBn[4][1] = BnTsigBn[1][4];
  BnTsigBn[4][2] = BnTsigBn[2][4];
  BnTsigBn[4][3] = BnTsigBn[3][4];
  t17 = Bnl[1][4]*Bnl[1][4];
  BnTsigBn[4][4] = t17*sig[1][1];
  t18 = Bnl[1][4]*sig[1][2];
  BnTsigBn[4][5] = t18*Bnl[2][5];
  t19 = Bnl[1][4]*sig[0][1];
  BnTsigBn[4][6] = t19*Bnl[0][6];
  t20 = Bnl[1][4]*sig[1][1];
  BnTsigBn[4][7] = t20*Bnl[1][7];
  BnTsigBn[4][8] = t18*Bnl[2][8];
  BnTsigBn[4][9] = t19*Bnl[0][9];
  BnTsigBn[4][10] = t20*Bnl[1][10];
  BnTsigBn[4][11] = t18*Bnl[2][11];
  BnTsigBn[4][12] = t19*Bnl[0][12];
  BnTsigBn[4][13] = t20*Bnl[1][13];
  BnTsigBn[4][14] = t18*Bnl[2][14];
  BnTsigBn[4][15] = t19*Bnl[0][15];
  BnTsigBn[4][16] = t20*Bnl[1][16];
  BnTsigBn[4][17] = t18*Bnl[2][17];
  BnTsigBn[4][18] = t19*Bnl[0][18];
  BnTsigBn[4][19] = t20*Bnl[1][19];
  BnTsigBn[4][20] = t18*Bnl[2][20];
  BnTsigBn[4][21] = t19*Bnl[0][21];
  BnTsigBn[4][22] = t20*Bnl[1][22];
  BnTsigBn[4][23] = t18*Bnl[2][23];
  BnTsigBn[5][0] = BnTsigBn[0][5];
  BnTsigBn[5][1] = BnTsigBn[1][5];
  BnTsigBn[5][2] = BnTsigBn[2][5];
  BnTsigBn[5][3] = BnTsigBn[3][5];
  BnTsigBn[5][4] = BnTsigBn[4][5];
  t21 = Bnl[2][5]*Bnl[2][5];
  BnTsigBn[5][5] = t21*sig[2][2];
  t22 = Bnl[2][5]*sig[0][2];
  BnTsigBn[5][6] = t22*Bnl[0][6];
  t23 = Bnl[2][5]*sig[1][2];
  BnTsigBn[5][7] = t23*Bnl[1][7];
  t24 = Bnl[2][5]*sig[2][2];
  BnTsigBn[5][8] = t24*Bnl[2][8];
  BnTsigBn[5][9] = t22*Bnl[0][9];
  BnTsigBn[5][10] = t23*Bnl[1][10];
  BnTsigBn[5][11] = t24*Bnl[2][11];
  BnTsigBn[5][12] = t22*Bnl[0][12];
  BnTsigBn[5][13] = t23*Bnl[1][13];
  BnTsigBn[5][14] = t24*Bnl[2][14];
  BnTsigBn[5][15] = t22*Bnl[0][15];
  BnTsigBn[5][16] = t23*Bnl[1][16];
  BnTsigBn[5][17] = t24*Bnl[2][17];
  BnTsigBn[5][18] = t22*Bnl[0][18];
  BnTsigBn[5][19] = t23*Bnl[1][19];
  BnTsigBn[5][20] = t24*Bnl[2][20];
  BnTsigBn[5][21] = t22*Bnl[0][21];
  BnTsigBn[5][22] = t23*Bnl[1][22];
  BnTsigBn[5][23] = t24*Bnl[2][23];
  BnTsigBn[6][0] = BnTsigBn[0][6];
  BnTsigBn[6][1] = BnTsigBn[1][6];
  BnTsigBn[6][2] = BnTsigBn[2][6];
  BnTsigBn[6][3] = BnTsigBn[3][6];
  BnTsigBn[6][4] = BnTsigBn[4][6];
  BnTsigBn[6][5] = BnTsigBn[5][6];
  t25 = Bnl[0][6]*Bnl[0][6];
  BnTsigBn[6][6] = t25*sig[0][0];
  t26 = Bnl[0][6]*sig[0][1];
  BnTsigBn[6][7] = t26*Bnl[1][7];
  t27 = Bnl[0][6]*sig[0][2];
  BnTsigBn[6][8] = t27*Bnl[2][8];
  t28 = Bnl[0][6]*sig[0][0];
  BnTsigBn[6][9] = t28*Bnl[0][9];
  BnTsigBn[6][10] = t26*Bnl[1][10];
  BnTsigBn[6][11] = t27*Bnl[2][11];
  BnTsigBn[6][12] = t28*Bnl[0][12];
  BnTsigBn[6][13] = t26*Bnl[1][13];
  BnTsigBn[6][14] = t27*Bnl[2][14];
  BnTsigBn[6][15] = t28*Bnl[0][15];
  BnTsigBn[6][16] = t26*Bnl[1][16];
  BnTsigBn[6][17] = t27*Bnl[2][17];
  BnTsigBn[6][18] = t28*Bnl[0][18];
  BnTsigBn[6][19] = t26*Bnl[1][19];
  BnTsigBn[6][20] = t27*Bnl[2][20];
  BnTsigBn[6][21] = t28*Bnl[0][21];
  BnTsigBn[6][22] = t26*Bnl[1][22];
  BnTsigBn[6][23] = t27*Bnl[2][23];
  BnTsigBn[7][0] = BnTsigBn[0][7];
  BnTsigBn[7][1] = BnTsigBn[1][7];
  BnTsigBn[7][2] = BnTsigBn[2][7];
  BnTsigBn[7][3] = BnTsigBn[3][7];
  BnTsigBn[7][4] = BnTsigBn[4][7];
  BnTsigBn[7][5] = BnTsigBn[5][7];
  BnTsigBn[7][6] = BnTsigBn[6][7];
  t29 = Bnl[1][7]*Bnl[1][7];
  BnTsigBn[7][7] = t29*sig[1][1];
  t30 = Bnl[1][7]*sig[1][2];
  BnTsigBn[7][8] = t30*Bnl[2][8];
  t31 = Bnl[1][7]*sig[0][1];
  BnTsigBn[7][9] = t31*Bnl[0][9];
  t32 = Bnl[1][7]*sig[1][1];
  BnTsigBn[7][10] = t32*Bnl[1][10];
  BnTsigBn[7][11] = t30*Bnl[2][11];
  BnTsigBn[7][12] = t31*Bnl[0][12];
  BnTsigBn[7][13] = t32*Bnl[1][13];
  BnTsigBn[7][14] = t30*Bnl[2][14];
  BnTsigBn[7][15] = t31*Bnl[0][15];
  BnTsigBn[7][16] = t32*Bnl[1][16];
  BnTsigBn[7][17] = t30*Bnl[2][17];
  BnTsigBn[7][18] = t31*Bnl[0][18];
  BnTsigBn[7][19] = t32*Bnl[1][19];
  BnTsigBn[7][20] = t30*Bnl[2][20];
  BnTsigBn[7][21] = t31*Bnl[0][21];
  BnTsigBn[7][22] = t32*Bnl[1][22];
  BnTsigBn[7][23] = t30*Bnl[2][23];
  BnTsigBn[8][0] = BnTsigBn[0][8];
  BnTsigBn[8][1] = BnTsigBn[1][8];
  BnTsigBn[8][2] = BnTsigBn[2][8];
  BnTsigBn[8][3] = BnTsigBn[3][8];
  BnTsigBn[8][4] = BnTsigBn[4][8];
  BnTsigBn[8][5] = BnTsigBn[5][8];
  BnTsigBn[8][6] = BnTsigBn[6][8];
  BnTsigBn[8][7] = BnTsigBn[7][8];
  t33 = Bnl[2][8]*Bnl[2][8];
  BnTsigBn[8][8] = t33*sig[2][2];
  t34 = Bnl[2][8]*sig[0][2];
  BnTsigBn[8][9] = t34*Bnl[0][9];
  t35 = Bnl[2][8]*sig[1][2];
  BnTsigBn[8][10] = t35*Bnl[1][10];
  t36 = Bnl[2][8]*sig[2][2];
  BnTsigBn[8][11] = t36*Bnl[2][11];
  BnTsigBn[8][12] = t34*Bnl[0][12];
  BnTsigBn[8][13] = t35*Bnl[1][13];
  BnTsigBn[8][14] = t36*Bnl[2][14];
  BnTsigBn[8][15] = t34*Bnl[0][15];
  BnTsigBn[8][16] = t35*Bnl[1][16];
  BnTsigBn[8][17] = t36*Bnl[2][17];
  BnTsigBn[8][18] = t34*Bnl[0][18];
  BnTsigBn[8][19] = t35*Bnl[1][19];
  BnTsigBn[8][20] = t36*Bnl[2][20];
  BnTsigBn[8][21] = t34*Bnl[0][21];
  BnTsigBn[8][22] = t35*Bnl[1][22];
  BnTsigBn[8][23] = t36*Bnl[2][23];
  BnTsigBn[9][0] = BnTsigBn[0][9];
  BnTsigBn[9][1] = BnTsigBn[1][9];
  BnTsigBn[9][2] = BnTsigBn[2][9];
  BnTsigBn[9][3] = BnTsigBn[3][9];
  BnTsigBn[9][4] = BnTsigBn[4][9];
  BnTsigBn[9][5] = BnTsigBn[5][9];
  BnTsigBn[9][6] = BnTsigBn[6][9];
  BnTsigBn[9][7] = BnTsigBn[7][9];
  BnTsigBn[9][8] = BnTsigBn[8][9];
  t37 = Bnl[0][9]*Bnl[0][9];
  BnTsigBn[9][9] = t37*sig[0][0];
  t38 = Bnl[0][9]*sig[0][1];
  BnTsigBn[9][10] = t38*Bnl[1][10];
  t39 = Bnl[0][9]*sig[0][2];
  BnTsigBn[9][11] = t39*Bnl[2][11];
  t40 = Bnl[0][9]*sig[0][0];
  BnTsigBn[9][12] = t40*Bnl[0][12];
  BnTsigBn[9][13] = t38*Bnl[1][13];
  BnTsigBn[9][14] = t39*Bnl[2][14];
  BnTsigBn[9][15] = t40*Bnl[0][15];
  BnTsigBn[9][16] = t38*Bnl[1][16];
  BnTsigBn[9][17] = t39*Bnl[2][17];
  BnTsigBn[9][18] = t40*Bnl[0][18];
  BnTsigBn[9][19] = t38*Bnl[1][19];
  BnTsigBn[9][20] = t39*Bnl[2][20];
  BnTsigBn[9][21] = t40*Bnl[0][21];
  BnTsigBn[9][22] = t38*Bnl[1][22];
  BnTsigBn[9][23] = t39*Bnl[2][23];
  BnTsigBn[10][0] = BnTsigBn[0][10];
  BnTsigBn[10][1] = BnTsigBn[1][10];
  BnTsigBn[10][2] = BnTsigBn[2][10];
  BnTsigBn[10][3] = BnTsigBn[3][10];
  BnTsigBn[10][4] = BnTsigBn[4][10];
  BnTsigBn[10][5] = BnTsigBn[5][10];
  BnTsigBn[10][6] = BnTsigBn[6][10];
  BnTsigBn[10][7] = BnTsigBn[7][10];
  BnTsigBn[10][8] = BnTsigBn[8][10];
  BnTsigBn[10][9] = BnTsigBn[9][10];
  t41 = Bnl[1][10]*Bnl[1][10];
  BnTsigBn[10][10] = t41*sig[1][1];
  t42 = Bnl[1][10]*sig[1][2];
  BnTsigBn[10][11] = t42*Bnl[2][11];
  t43 = Bnl[1][10]*sig[0][1];
  BnTsigBn[10][12] = t43*Bnl[0][12];
  t44 = Bnl[1][10]*sig[1][1];
  BnTsigBn[10][13] = t44*Bnl[1][13];
  BnTsigBn[10][14] = t42*Bnl[2][14];
  BnTsigBn[10][15] = t43*Bnl[0][15];
  BnTsigBn[10][16] = t44*Bnl[1][16];
  BnTsigBn[10][17] = t42*Bnl[2][17];
  BnTsigBn[10][18] = t43*Bnl[0][18];
  BnTsigBn[10][19] = t44*Bnl[1][19];
  BnTsigBn[10][20] = t42*Bnl[2][20];
  BnTsigBn[10][21] = t43*Bnl[0][21];
  BnTsigBn[10][22] = t44*Bnl[1][22];
  BnTsigBn[10][23] = t42*Bnl[2][23];
  BnTsigBn[11][0] = BnTsigBn[0][11];
  BnTsigBn[11][1] = BnTsigBn[1][11];
  BnTsigBn[11][2] = BnTsigBn[2][11];
  BnTsigBn[11][3] = BnTsigBn[3][11];
  BnTsigBn[11][4] = BnTsigBn[4][11];
  BnTsigBn[11][5] = BnTsigBn[5][11];
  BnTsigBn[11][6] = BnTsigBn[6][11];
  BnTsigBn[11][7] = BnTsigBn[7][11];
  BnTsigBn[11][8] = BnTsigBn[8][11];
  BnTsigBn[11][9] = BnTsigBn[9][11];
  BnTsigBn[11][10] = BnTsigBn[10][11];
  t45 = Bnl[2][11]*Bnl[2][11];
  BnTsigBn[11][11] = t45*sig[2][2];
  t46 = Bnl[2][11]*sig[0][2];
  BnTsigBn[11][12] = t46*Bnl[0][12];
  t47 = Bnl[2][11]*sig[1][2];
  BnTsigBn[11][13] = t47*Bnl[1][13];
  t48 = Bnl[2][11]*sig[2][2];
  BnTsigBn[11][14] = t48*Bnl[2][14];
  BnTsigBn[11][15] = t46*Bnl[0][15];
  BnTsigBn[11][16] = t47*Bnl[1][16];
  BnTsigBn[11][17] = t48*Bnl[2][17];
  BnTsigBn[11][18] = t46*Bnl[0][18];
  BnTsigBn[11][19] = t47*Bnl[1][19];
  BnTsigBn[11][20] = t48*Bnl[2][20];
  BnTsigBn[11][21] = t46*Bnl[0][21];
  BnTsigBn[11][22] = t47*Bnl[1][22];
  BnTsigBn[11][23] = t48*Bnl[2][23];
  BnTsigBn[12][0] = BnTsigBn[0][12];
  BnTsigBn[12][1] = BnTsigBn[1][12];
  BnTsigBn[12][2] = BnTsigBn[2][12];
  BnTsigBn[12][3] = BnTsigBn[3][12];
  BnTsigBn[12][4] = BnTsigBn[4][12];
  BnTsigBn[12][5] = BnTsigBn[5][12];
  BnTsigBn[12][6] = BnTsigBn[6][12];
  BnTsigBn[12][7] = BnTsigBn[7][12];
  BnTsigBn[12][8] = BnTsigBn[8][12];
  BnTsigBn[12][9] = BnTsigBn[9][12];
  BnTsigBn[12][10] = BnTsigBn[10][12];
  BnTsigBn[12][11] = BnTsigBn[11][12];
  t49 = Bnl[0][12]*Bnl[0][12];
  BnTsigBn[12][12] = t49*sig[0][0];
  t50 = Bnl[0][12]*sig[0][1];
  BnTsigBn[12][13] = t50*Bnl[1][13];
  t51 = Bnl[0][12]*sig[0][2];
  BnTsigBn[12][14] = t51*Bnl[2][14];
  t52 = Bnl[0][12]*sig[0][0];
  BnTsigBn[12][15] = t52*Bnl[0][15];
  BnTsigBn[12][16] = t50*Bnl[1][16];
  BnTsigBn[12][17] = t51*Bnl[2][17];
  BnTsigBn[12][18] = t52*Bnl[0][18];
  BnTsigBn[12][19] = t50*Bnl[1][19];
  BnTsigBn[12][20] = t51*Bnl[2][20];
  BnTsigBn[12][21] = t52*Bnl[0][21];
  BnTsigBn[12][22] = t50*Bnl[1][22];
  BnTsigBn[12][23] = t51*Bnl[2][23];
  BnTsigBn[13][0] = BnTsigBn[0][13];
  BnTsigBn[13][1] = BnTsigBn[1][13];
  BnTsigBn[13][2] = BnTsigBn[2][13];
  BnTsigBn[13][3] = BnTsigBn[3][13];
  BnTsigBn[13][4] = BnTsigBn[4][13];
  BnTsigBn[13][5] = BnTsigBn[5][13];
  BnTsigBn[13][6] = BnTsigBn[6][13];
  BnTsigBn[13][7] = BnTsigBn[7][13];
  BnTsigBn[13][8] = BnTsigBn[8][13];
  BnTsigBn[13][9] = BnTsigBn[9][13];
  BnTsigBn[13][10] = BnTsigBn[10][13];
  BnTsigBn[13][11] = BnTsigBn[11][13];
  BnTsigBn[13][12] = BnTsigBn[12][13];
  t53 = Bnl[1][13]*Bnl[1][13];
  BnTsigBn[13][13] = t53*sig[1][1];
  t54 = Bnl[1][13]*sig[1][2];
  BnTsigBn[13][14] = t54*Bnl[2][14];
  t55 = Bnl[1][13]*sig[0][1];
  BnTsigBn[13][15] = t55*Bnl[0][15];
  t56 = Bnl[1][13]*sig[1][1];
  BnTsigBn[13][16] = t56*Bnl[1][16];
  BnTsigBn[13][17] = t54*Bnl[2][17];
  BnTsigBn[13][18] = t55*Bnl[0][18];
  BnTsigBn[13][19] = t56*Bnl[1][19];
  BnTsigBn[13][20] = t54*Bnl[2][20];
  BnTsigBn[13][21] = t55*Bnl[0][21];
  BnTsigBn[13][22] = t56*Bnl[1][22];
  BnTsigBn[13][23] = t54*Bnl[2][23];
  BnTsigBn[14][0] = BnTsigBn[0][14];
  BnTsigBn[14][1] = BnTsigBn[1][14];
  BnTsigBn[14][2] = BnTsigBn[2][14];
  BnTsigBn[14][3] = BnTsigBn[3][14];
  BnTsigBn[14][4] = BnTsigBn[4][14];
  BnTsigBn[14][5] = BnTsigBn[5][14];
  BnTsigBn[14][6] = BnTsigBn[6][14];
  BnTsigBn[14][7] = BnTsigBn[7][14];
  BnTsigBn[14][8] = BnTsigBn[8][14];
  BnTsigBn[14][9] = BnTsigBn[9][14];
  BnTsigBn[14][10] = BnTsigBn[10][14];
  BnTsigBn[14][11] = BnTsigBn[11][14];
  BnTsigBn[14][12] = BnTsigBn[12][14];
  BnTsigBn[14][13] = BnTsigBn[13][14];
  t57 = Bnl[2][14]*Bnl[2][14];
  BnTsigBn[14][14] = t57*sig[2][2];
  t58 = Bnl[2][14]*sig[0][2];
  BnTsigBn[14][15] = t58*Bnl[0][15];
  t59 = Bnl[2][14]*sig[1][2];
  BnTsigBn[14][16] = t59*Bnl[1][16];
  t60 = Bnl[2][14]*sig[2][2];
  BnTsigBn[14][17] = t60*Bnl[2][17];
  BnTsigBn[14][18] = t58*Bnl[0][18];
  BnTsigBn[14][19] = t59*Bnl[1][19];
  BnTsigBn[14][20] = t60*Bnl[2][20];
  BnTsigBn[14][21] = t58*Bnl[0][21];
  BnTsigBn[14][22] = t59*Bnl[1][22];
  BnTsigBn[14][23] = t60*Bnl[2][23];
  BnTsigBn[15][0] = BnTsigBn[0][15];
  BnTsigBn[15][1] = BnTsigBn[1][15];
  BnTsigBn[15][2] = BnTsigBn[2][15];
  BnTsigBn[15][3] = BnTsigBn[3][15];
  BnTsigBn[15][4] = BnTsigBn[4][15];
  BnTsigBn[15][5] = BnTsigBn[5][15];
  BnTsigBn[15][6] = BnTsigBn[6][15];
  BnTsigBn[15][7] = BnTsigBn[7][15];
  BnTsigBn[15][8] = BnTsigBn[8][15];
  BnTsigBn[15][9] = BnTsigBn[9][15];
  BnTsigBn[15][10] = BnTsigBn[10][15];
  BnTsigBn[15][11] = BnTsigBn[11][15];
  BnTsigBn[15][12] = BnTsigBn[12][15];
  BnTsigBn[15][13] = BnTsigBn[13][15];
  BnTsigBn[15][14] = BnTsigBn[14][15];
  t61 = Bnl[0][15]*Bnl[0][15];
  BnTsigBn[15][15] = t61*sig[0][0];
  t62 = Bnl[0][15]*sig[0][1];
  BnTsigBn[15][16] = t62*Bnl[1][16];
  t63 = Bnl[0][15]*sig[0][2];
  BnTsigBn[15][17] = t63*Bnl[2][17];
  t64 = Bnl[0][15]*sig[0][0];
  BnTsigBn[15][18] = t64*Bnl[0][18];
  BnTsigBn[15][19] = t62*Bnl[1][19];
  BnTsigBn[15][20] = t63*Bnl[2][20];
  BnTsigBn[15][21] = t64*Bnl[0][21];
  BnTsigBn[15][22] = t62*Bnl[1][22];
  BnTsigBn[15][23] = t63*Bnl[2][23];
  BnTsigBn[16][0] = BnTsigBn[0][16];
  BnTsigBn[16][1] = BnTsigBn[1][16];
  BnTsigBn[16][2] = BnTsigBn[2][16];
  BnTsigBn[16][3] = BnTsigBn[3][16];
  BnTsigBn[16][4] = BnTsigBn[4][16];
  BnTsigBn[16][5] = BnTsigBn[5][16];
  BnTsigBn[16][6] = BnTsigBn[6][16];
  BnTsigBn[16][7] = BnTsigBn[7][16];
  BnTsigBn[16][8] = BnTsigBn[8][16];
  BnTsigBn[16][9] = BnTsigBn[9][16];
  BnTsigBn[16][10] = BnTsigBn[10][16];
  BnTsigBn[16][11] = BnTsigBn[11][16];
  BnTsigBn[16][12] = BnTsigBn[12][16];
  BnTsigBn[16][13] = BnTsigBn[13][16];
  BnTsigBn[16][14] = BnTsigBn[14][16];
  BnTsigBn[16][15] = BnTsigBn[15][16];
  t65 = Bnl[1][16]*Bnl[1][16];
  BnTsigBn[16][16] = t65*sig[1][1];
  t66 = Bnl[1][16]*sig[1][2];
  BnTsigBn[16][17] = t66*Bnl[2][17];
  t67 = Bnl[1][16]*sig[0][1];
  BnTsigBn[16][18] = t67*Bnl[0][18];
  t68 = Bnl[1][16]*sig[1][1];
  BnTsigBn[16][19] = t68*Bnl[1][19];
  BnTsigBn[16][20] = t66*Bnl[2][20];
  BnTsigBn[16][21] = t67*Bnl[0][21];
  BnTsigBn[16][22] = t68*Bnl[1][22];
  BnTsigBn[16][23] = t66*Bnl[2][23];
  BnTsigBn[17][0] = BnTsigBn[0][17];
  BnTsigBn[17][1] = BnTsigBn[1][17];
  BnTsigBn[17][2] = BnTsigBn[2][17];
  BnTsigBn[17][3] = BnTsigBn[3][17];
  BnTsigBn[17][4] = BnTsigBn[4][17];
  BnTsigBn[17][5] = BnTsigBn[5][17];
  BnTsigBn[17][6] = BnTsigBn[6][17];
  BnTsigBn[17][7] = BnTsigBn[7][17];
  BnTsigBn[17][8] = BnTsigBn[8][17];
  BnTsigBn[17][9] = BnTsigBn[9][17];
  BnTsigBn[17][10] = BnTsigBn[10][17];
  BnTsigBn[17][11] = BnTsigBn[11][17];
  BnTsigBn[17][12] = BnTsigBn[12][17];
  BnTsigBn[17][13] = BnTsigBn[13][17];
  BnTsigBn[17][14] = BnTsigBn[14][17];
  BnTsigBn[17][15] = BnTsigBn[15][17];
  BnTsigBn[17][16] = BnTsigBn[16][17];
  t69 = Bnl[2][17]*Bnl[2][17];
  BnTsigBn[17][17] = t69*sig[2][2];
  t70 = Bnl[2][17]*sig[0][2];
  BnTsigBn[17][18] = t70*Bnl[0][18];
  t71 = Bnl[2][17]*sig[1][2];
  BnTsigBn[17][19] = t71*Bnl[1][19];
  t72 = Bnl[2][17]*sig[2][2];
  BnTsigBn[17][20] = t72*Bnl[2][20];
  BnTsigBn[17][21] = t70*Bnl[0][21];
  BnTsigBn[17][22] = t71*Bnl[1][22];
  BnTsigBn[17][23] = t72*Bnl[2][23];
  BnTsigBn[18][0] = BnTsigBn[0][18];
  BnTsigBn[18][1] = BnTsigBn[1][18];
  BnTsigBn[18][2] = BnTsigBn[2][18];
  BnTsigBn[18][3] = BnTsigBn[3][18];
  BnTsigBn[18][4] = BnTsigBn[4][18];
  BnTsigBn[18][5] = BnTsigBn[5][18];
  BnTsigBn[18][6] = BnTsigBn[6][18];
  BnTsigBn[18][7] = BnTsigBn[7][18];
  BnTsigBn[18][8] = BnTsigBn[8][18];
  BnTsigBn[18][9] = BnTsigBn[9][18];
  BnTsigBn[18][10] = BnTsigBn[10][18];
  BnTsigBn[18][11] = BnTsigBn[11][18];
  BnTsigBn[18][12] = BnTsigBn[12][18];
  BnTsigBn[18][13] = BnTsigBn[13][18];
  BnTsigBn[18][14] = BnTsigBn[14][18];
  BnTsigBn[18][15] = BnTsigBn[15][18];
  BnTsigBn[18][16] = BnTsigBn[16][18];
  BnTsigBn[18][17] = BnTsigBn[17][18];
  t73 = Bnl[0][18]*Bnl[0][18];
  BnTsigBn[18][18] = t73*sig[0][0];
  t74 = Bnl[0][18]*sig[0][1];
  BnTsigBn[18][19] = t74*Bnl[1][19];
  t75 = Bnl[0][18]*sig[0][2];
  BnTsigBn[18][20] = t75*Bnl[2][20];
  BnTsigBn[18][21] = Bnl[0][18]*sig[0][0]*Bnl[0][21];
  BnTsigBn[18][22] = t74*Bnl[1][22];
  BnTsigBn[18][23] = t75*Bnl[2][23];
  BnTsigBn[19][0] = BnTsigBn[0][19];
  BnTsigBn[19][1] = BnTsigBn[1][19];
  BnTsigBn[19][2] = BnTsigBn[2][19];
  BnTsigBn[19][3] = BnTsigBn[3][19];
  BnTsigBn[19][4] = BnTsigBn[4][19];
  BnTsigBn[19][5] = BnTsigBn[5][19];
  BnTsigBn[19][6] = BnTsigBn[6][19];
  BnTsigBn[19][7] = BnTsigBn[7][19];
  BnTsigBn[19][8] = BnTsigBn[8][19];
  BnTsigBn[19][9] = BnTsigBn[9][19];
  BnTsigBn[19][10] = BnTsigBn[10][19];
  BnTsigBn[19][11] = BnTsigBn[11][19];
  BnTsigBn[19][12] = BnTsigBn[12][19];
  BnTsigBn[19][13] = BnTsigBn[13][19];
  BnTsigBn[19][14] = BnTsigBn[14][19];
  BnTsigBn[19][15] = BnTsigBn[15][19];
  BnTsigBn[19][16] = BnTsigBn[16][19];
  BnTsigBn[19][17] = BnTsigBn[17][19];
  BnTsigBn[19][18] = BnTsigBn[18][19];
  t77 = Bnl[1][19]*Bnl[1][19];
  BnTsigBn[19][19] = t77*sig[1][1];
  t78 = Bnl[1][19]*sig[1][2];
  BnTsigBn[19][20] = t78*Bnl[2][20];
  BnTsigBn[19][21] = Bnl[1][19]*sig[0][1]*Bnl[0][21];
  BnTsigBn[19][22] = Bnl[1][19]*sig[1][1]*Bnl[1][22];
  BnTsigBn[19][23] = t78*Bnl[2][23];
  BnTsigBn[20][0] = BnTsigBn[0][20];
  BnTsigBn[20][1] = BnTsigBn[1][20];
  BnTsigBn[20][2] = BnTsigBn[2][20];
  BnTsigBn[20][3] = BnTsigBn[3][20];
  BnTsigBn[20][4] = BnTsigBn[4][20];
  BnTsigBn[20][5] = BnTsigBn[5][20];
  BnTsigBn[20][6] = BnTsigBn[6][20];
  BnTsigBn[20][7] = BnTsigBn[7][20];
  BnTsigBn[20][8] = BnTsigBn[8][20];
  BnTsigBn[20][9] = BnTsigBn[9][20];
  BnTsigBn[20][10] = BnTsigBn[10][20];
  BnTsigBn[20][11] = BnTsigBn[11][20];
  BnTsigBn[20][12] = BnTsigBn[12][20];
  BnTsigBn[20][13] = BnTsigBn[13][20];
  BnTsigBn[20][14] = BnTsigBn[14][20];
  BnTsigBn[20][15] = BnTsigBn[15][20];
  BnTsigBn[20][16] = BnTsigBn[16][20];
  BnTsigBn[20][17] = BnTsigBn[17][20];
  BnTsigBn[20][18] = BnTsigBn[18][20];
  BnTsigBn[20][19] = BnTsigBn[19][20];
  t81 = Bnl[2][20]*Bnl[2][20];
  BnTsigBn[20][20] = t81*sig[2][2];
  BnTsigBn[20][21] = Bnl[2][20]*sig[0][2]*Bnl[0][21];
  BnTsigBn[20][22] = Bnl[2][20]*sig[1][2]*Bnl[1][22];
  BnTsigBn[20][23] = Bnl[2][20]*sig[2][2]*Bnl[2][23];
  BnTsigBn[21][0] = BnTsigBn[0][21];
  BnTsigBn[21][1] = BnTsigBn[1][21];
  BnTsigBn[21][2] = BnTsigBn[2][21];
  BnTsigBn[21][3] = BnTsigBn[3][21];
  BnTsigBn[21][4] = BnTsigBn[4][21];
  BnTsigBn[21][5] = BnTsigBn[5][21];
  BnTsigBn[21][6] = BnTsigBn[6][21];
  BnTsigBn[21][7] = BnTsigBn[7][21];
  BnTsigBn[21][8] = BnTsigBn[8][21];
  BnTsigBn[21][9] = BnTsigBn[9][21];
  BnTsigBn[21][10] = BnTsigBn[10][21];
  BnTsigBn[21][11] = BnTsigBn[11][21];
  BnTsigBn[21][12] = BnTsigBn[12][21];
  BnTsigBn[21][13] = BnTsigBn[13][21];
  BnTsigBn[21][14] = BnTsigBn[14][21];
  BnTsigBn[21][15] = BnTsigBn[15][21];
  BnTsigBn[21][16] = BnTsigBn[16][21];
  BnTsigBn[21][17] = BnTsigBn[17][21];
  BnTsigBn[21][18] = BnTsigBn[18][21];
  BnTsigBn[21][19] = BnTsigBn[19][21];
  BnTsigBn[21][20] = BnTsigBn[20][21];
  t85 = Bnl[0][21]*Bnl[0][21];
  BnTsigBn[21][21] = t85*sig[0][0];
  BnTsigBn[21][22] = Bnl[0][21]*sig[0][1]*Bnl[1][22];
  BnTsigBn[21][23] = Bnl[0][21]*sig[0][2]*Bnl[2][23];
  BnTsigBn[22][0] = BnTsigBn[0][22];
  BnTsigBn[22][1] = BnTsigBn[1][22];
  BnTsigBn[22][2] = BnTsigBn[2][22];
  BnTsigBn[22][3] = BnTsigBn[3][22];
  BnTsigBn[22][4] = BnTsigBn[4][22];
  BnTsigBn[22][5] = BnTsigBn[5][22];
  BnTsigBn[22][6] = BnTsigBn[6][22];
  BnTsigBn[22][7] = BnTsigBn[7][22];
  BnTsigBn[22][8] = BnTsigBn[8][22];
  BnTsigBn[22][9] = BnTsigBn[9][22];
  BnTsigBn[22][10] = BnTsigBn[10][22];
  BnTsigBn[22][11] = BnTsigBn[11][22];
  BnTsigBn[22][12] = BnTsigBn[12][22];
  BnTsigBn[22][13] = BnTsigBn[13][22];
  BnTsigBn[22][14] = BnTsigBn[14][22];
  BnTsigBn[22][15] = BnTsigBn[15][22];
  BnTsigBn[22][16] = BnTsigBn[16][22];
  BnTsigBn[22][17] = BnTsigBn[17][22];
  BnTsigBn[22][18] = BnTsigBn[18][22];
  BnTsigBn[22][19] = BnTsigBn[19][22];
  BnTsigBn[22][20] = BnTsigBn[20][22];
  BnTsigBn[22][21] = BnTsigBn[21][22];
  t88 = Bnl[1][22]*Bnl[1][22];
  BnTsigBn[22][22] = t88*sig[1][1];
  BnTsigBn[22][23] = Bnl[1][22]*sig[1][2]*Bnl[2][23];
  BnTsigBn[23][0] = BnTsigBn[0][23];
  BnTsigBn[23][1] = BnTsigBn[1][23];
  BnTsigBn[23][2] = BnTsigBn[2][23];
  BnTsigBn[23][3] = BnTsigBn[3][23];
  BnTsigBn[23][4] = BnTsigBn[4][23];
  BnTsigBn[23][5] = BnTsigBn[5][23];
  BnTsigBn[23][6] = BnTsigBn[6][23];
  BnTsigBn[23][7] = BnTsigBn[7][23];
  BnTsigBn[23][8] = BnTsigBn[8][23];
  BnTsigBn[23][9] = BnTsigBn[9][23];
  BnTsigBn[23][10] = BnTsigBn[10][23];
  BnTsigBn[23][11] = BnTsigBn[11][23];
  BnTsigBn[23][12] = BnTsigBn[12][23];
  BnTsigBn[23][13] = BnTsigBn[13][23];
  BnTsigBn[23][14] = BnTsigBn[14][23];
  BnTsigBn[23][15] = BnTsigBn[15][23];
  BnTsigBn[23][16] = BnTsigBn[16][23];
  BnTsigBn[23][17] = BnTsigBn[17][23];
  BnTsigBn[23][18] = BnTsigBn[18][23];
  BnTsigBn[23][19] = BnTsigBn[19][23];
  BnTsigBn[23][20] = BnTsigBn[20][23];
  BnTsigBn[23][21] = BnTsigBn[21][23];
  BnTsigBn[23][22] = BnTsigBn[22][23];
  t90 = Bnl[2][23]*Bnl[2][23];
  BnTsigBn[23][23] = t90*sig[2][2];
}

void ImplicitCM::BtDBGIMP(const double B[6][81],
                          const double D[6][6],
                          double Kmat[81][81]) const
{
    for (int i=0;i<81;i++){
      for (int j=0;j<81;j++){
        Kmat[i][j] = 0.;
        for (int k=0;k<6;k++){
          for (int l=0;l<6;l++){
            Kmat[i][j]+=B[l][i]*D[l][k]*B[k][j];
          }
        }
      }
    }
}

void
ImplicitCM::BtDB(const double B[6][24], 
                 const double D[6][6],
                 double Kmat[24][24]) const
{
#if 0
    for (int i=0;i<24;i++){
      for (int j=0;j<24;j++){
        Kmat[i][j] = 0.;
        for (int k=0;k<6;k++){
          for (int l=0;l<6;l++){
            Kmat[i][j]+=B[l][i]*D[l][k]*B[k][j];
          }
        }
      }
    }

#endif

  double t100, t105, t1060, t1065, t1070, t1075, t1081, t1086, t110, t115,t1156;
  double t1161, t1166, t1171, t1177, t1182, t121, t1252, t1257,t126,t1262,t1267;
  double t1273, t1278, t1348, t1353, t1358, t1363, t1369, t1374,t14,t1444,t1449;
  double t1454, t1459, t1465, t1470, t1540, t1545,t1550,t1555,t1561,t1566,t1636;
  double t1641, t1646, t1651, t1657, t1662, t1732,t1737,t1742,t1747,t1753,t1758;
  double t1828, t1833, t1838, t1843, t1849, t1854, t19, t1924,t1929,t1934,t1939;
  double t1945, t1950, t196, t201, t2020, t2025, t2030, t2035,t2041,t2046, t206;
  double t211, t2121, t2126;
  double t2131, t2137, t2142, t217;
  //double t2212, t2217;
  double t222;
  double t2222;
  //double t2227;
  double t2233, t2238, t25, t292, t297, t30, t302, t307, t313;
  double t318, t388, t393, t398, t4, t403, t409, t414, t484, t489, t494;
  double t499, t505, t510, t580, t585, t590, t595, t601, t606, t676, t681;
  double t686, t691, t697, t702, t772, t777, t782, t787, t793, t798, t868;
  double t873, t878, t883, t889, t894, t9, t964, t969, t974, t979, t985, t990;

  t4 = B[0][0]*D[0][0]+B[3][0]*D[0][3]+B[5][0]*D[0][5];
  t9 = B[0][0]*D[0][3]+B[3][0]*D[3][3]+B[5][0]*D[3][5];
  t14 = B[0][0]*D[0][5]+B[3][0]*D[3][5]+B[5][0]*D[5][5];
  Kmat[0][0] = t4*B[0][0]+t9*B[3][0]+t14*B[5][0];
  t19 = B[0][0]*D[0][1]+B[3][0]*D[1][3]+B[5][0]*D[1][5];
  t25 = B[0][0]*D[0][4]+B[3][0]*D[3][4]+B[5][0]*D[4][5];
  Kmat[0][1] = t19*B[1][1]+t9*B[3][1]+t25*B[4][1];
  t30 = B[0][0]*D[0][2]+B[3][0]*D[2][3]+B[5][0]*D[2][5];
  Kmat[0][2] = t30*B[2][2]+t25*B[4][2]+t14*B[5][2];
  Kmat[0][3] = t4*B[0][3]+t9*B[3][3]+t14*B[5][3];
  Kmat[0][4] = t19*B[1][4]+t9*B[3][4]+t25*B[4][4];
  Kmat[0][5] = t30*B[2][5]+t25*B[4][5]+t14*B[5][5];
  Kmat[0][6] = t4*B[0][6]+t9*B[3][6]+t14*B[5][6];
  Kmat[0][7] = t19*B[1][7]+t9*B[3][7]+t25*B[4][7];
  Kmat[0][8] = t30*B[2][8]+t25*B[4][8]+t14*B[5][8];
  Kmat[0][9] = t4*B[0][9]+t9*B[3][9]+t14*B[5][9];
  Kmat[0][10] = t19*B[1][10]+t9*B[3][10]+t25*B[4][10];
  Kmat[0][11] = t30*B[2][11]+t25*B[4][11]+t14*B[5][11];
  Kmat[0][12] = t4*B[0][12]+t9*B[3][12]+t14*B[5][12];
  Kmat[0][13] = t19*B[1][13]+t9*B[3][13]+t25*B[4][13];
  Kmat[0][14] = t30*B[2][14]+t25*B[4][14]+t14*B[5][14];
  Kmat[0][15] = t4*B[0][15]+t9*B[3][15]+t14*B[5][15];
  Kmat[0][16] = t19*B[1][16]+t9*B[3][16]+t25*B[4][16];
  Kmat[0][17] = t30*B[2][17]+t25*B[4][17]+t14*B[5][17];
  Kmat[0][18] = t4*B[0][18]+t9*B[3][18]+t14*B[5][18];
  Kmat[0][19] = t19*B[1][19]+t9*B[3][19]+t25*B[4][19];
  Kmat[0][20] = t30*B[2][20]+t25*B[4][20]+t14*B[5][20];
  Kmat[0][21] = t4*B[0][21]+t9*B[3][21]+t14*B[5][21];
  Kmat[0][22] = t19*B[1][22]+t9*B[3][22]+t25*B[4][22];
  Kmat[0][23] = t30*B[2][23]+t25*B[4][23]+t14*B[5][23];
  t100 = B[1][1]*D[0][1]+B[3][1]*D[0][3]+B[4][1]*D[0][4];
  t105 = B[1][1]*D[1][3]+B[3][1]*D[3][3]+B[4][1]*D[3][4];
  t110 = B[1][1]*D[1][5]+B[3][1]*D[3][5]+B[4][1]*D[4][5];
  Kmat[1][0] = Kmat[0][1];
  t115 = B[1][1]*D[1][1]+B[3][1]*D[1][3]+B[4][1]*D[1][4];
  t121 = B[1][1]*D[1][4]+B[3][1]*D[3][4]+B[4][1]*D[4][4];
  Kmat[1][1] = t115*B[1][1]+t105*B[3][1]+t121*B[4][1];
  t126 = B[1][1]*D[1][2]+B[3][1]*D[2][3]+B[4][1]*D[2][4];
  Kmat[1][2] = t126*B[2][2]+t121*B[4][2]+t110*B[5][2];
  Kmat[1][3] = t100*B[0][3]+t105*B[3][3]+t110*B[5][3];
  Kmat[1][4] = t115*B[1][4]+t105*B[3][4]+t121*B[4][4];
  Kmat[1][5] = t126*B[2][5]+t121*B[4][5]+t110*B[5][5];
  Kmat[1][6] = t100*B[0][6]+t105*B[3][6]+t110*B[5][6];
  Kmat[1][7] = t115*B[1][7]+t105*B[3][7]+t121*B[4][7];
  Kmat[1][8] = t126*B[2][8]+t121*B[4][8]+t110*B[5][8];
  Kmat[1][9] = t100*B[0][9]+t105*B[3][9]+t110*B[5][9];
  Kmat[1][10] = t115*B[1][10]+t105*B[3][10]+t121*B[4][10];
  Kmat[1][11] = t126*B[2][11]+t121*B[4][11]+t110*B[5][11];
  Kmat[1][12] = t100*B[0][12]+t105*B[3][12]+t110*B[5][12];
  Kmat[1][13] = t115*B[1][13]+t105*B[3][13]+t121*B[4][13];
  Kmat[1][14] = t126*B[2][14]+t121*B[4][14]+t110*B[5][14];
  Kmat[1][15] = t100*B[0][15]+t105*B[3][15]+t110*B[5][15];
  Kmat[1][16] = t115*B[1][16]+t105*B[3][16]+t121*B[4][16];
  Kmat[1][17] = t126*B[2][17]+t121*B[4][17]+t110*B[5][17];
  Kmat[1][18] = t100*B[0][18]+t105*B[3][18]+t110*B[5][18];
  Kmat[1][19] = t115*B[1][19]+t105*B[3][19]+t121*B[4][19];
  Kmat[1][20] = t126*B[2][20]+t121*B[4][20]+t110*B[5][20];
  Kmat[1][21] = t100*B[0][21]+t105*B[3][21]+t110*B[5][21];
  Kmat[1][22] = t115*B[1][22]+t105*B[3][22]+t121*B[4][22];
  Kmat[1][23] = t126*B[2][23]+t121*B[4][23]+t110*B[5][23];
  t196 = B[2][2]*D[0][2]+B[4][2]*D[0][4]+B[5][2]*D[0][5];
  t201 = B[2][2]*D[2][3]+B[4][2]*D[3][4]+B[5][2]*D[3][5];
  t206 = B[2][2]*D[2][5]+B[4][2]*D[4][5]+B[5][2]*D[5][5];
  Kmat[2][0] = Kmat[0][2];
  t211 = B[2][2]*D[1][2]+B[4][2]*D[1][4]+B[5][2]*D[1][5];
  t217 = B[2][2]*D[2][4]+B[4][2]*D[4][4]+B[5][2]*D[4][5];
  Kmat[2][1] = Kmat[1][2];
  t222 = B[2][2]*D[2][2]+B[4][2]*D[2][4]+B[5][2]*D[2][5];
  Kmat[2][2] = t222*B[2][2]+t217*B[4][2]+t206*B[5][2];
  Kmat[2][3] = t196*B[0][3]+t201*B[3][3]+t206*B[5][3];
  Kmat[2][4] = t211*B[1][4]+t201*B[3][4]+t217*B[4][4];
  Kmat[2][5] = t222*B[2][5]+t217*B[4][5]+t206*B[5][5];
  Kmat[2][6] = t196*B[0][6]+t201*B[3][6]+t206*B[5][6];
  Kmat[2][7] = t211*B[1][7]+t201*B[3][7]+t217*B[4][7];
  Kmat[2][8] = t222*B[2][8]+t217*B[4][8]+t206*B[5][8];
  Kmat[2][9] = t196*B[0][9]+t201*B[3][9]+t206*B[5][9];
  Kmat[2][10] = t211*B[1][10]+t201*B[3][10]+t217*B[4][10];
  Kmat[2][11] = t222*B[2][11]+t217*B[4][11]+t206*B[5][11];
  Kmat[2][12] = t196*B[0][12]+t201*B[3][12]+t206*B[5][12];
  Kmat[2][13] = t211*B[1][13]+t201*B[3][13]+t217*B[4][13];
  Kmat[2][14] = t222*B[2][14]+t217*B[4][14]+t206*B[5][14];
  Kmat[2][15] = t196*B[0][15]+t201*B[3][15]+t206*B[5][15];
  Kmat[2][16] = t211*B[1][16]+t201*B[3][16]+t217*B[4][16];
  Kmat[2][17] = t222*B[2][17]+t217*B[4][17]+t206*B[5][17];
  Kmat[2][18] = t196*B[0][18]+t201*B[3][18]+t206*B[5][18];
  Kmat[2][19] = t211*B[1][19]+t201*B[3][19]+t217*B[4][19];
  Kmat[2][20] = t222*B[2][20]+t217*B[4][20]+t206*B[5][20];
  Kmat[2][21] = t196*B[0][21]+t201*B[3][21]+t206*B[5][21];
  Kmat[2][22] = t211*B[1][22]+t201*B[3][22]+t217*B[4][22];
  Kmat[2][23] = t222*B[2][23]+t217*B[4][23]+t206*B[5][23];
  t292 = B[0][3]*D[0][0]+B[3][3]*D[0][3]+B[5][3]*D[0][5];
  t297 = B[0][3]*D[0][3]+B[3][3]*D[3][3]+B[5][3]*D[3][5];
  t302 = B[0][3]*D[0][5]+B[3][3]*D[3][5]+B[5][3]*D[5][5];
  Kmat[3][0] = Kmat[0][3];
  t307 = B[0][3]*D[0][1]+B[3][3]*D[1][3]+B[5][3]*D[1][5];
  t313 = B[0][3]*D[0][4]+B[3][3]*D[3][4]+B[5][3]*D[4][5];
  Kmat[3][1] = Kmat[1][3];
  t318 = B[0][3]*D[0][2]+B[3][3]*D[2][3]+B[5][3]*D[2][5];
  Kmat[3][2] = Kmat[2][3];
  Kmat[3][3] = t292*B[0][3]+t297*B[3][3]+t302*B[5][3];
  Kmat[3][4] = t307*B[1][4]+t297*B[3][4]+t313*B[4][4];
  Kmat[3][5] = t318*B[2][5]+t313*B[4][5]+t302*B[5][5];
  Kmat[3][6] = t292*B[0][6]+t297*B[3][6]+t302*B[5][6];
  Kmat[3][7] = t307*B[1][7]+t297*B[3][7]+t313*B[4][7];
  Kmat[3][8] = t318*B[2][8]+t313*B[4][8]+t302*B[5][8];
  Kmat[3][9] = t292*B[0][9]+t297*B[3][9]+t302*B[5][9];
  Kmat[3][10] = t307*B[1][10]+t297*B[3][10]+t313*B[4][10];
  Kmat[3][11] = t318*B[2][11]+t313*B[4][11]+t302*B[5][11];
  Kmat[3][12] = t292*B[0][12]+t297*B[3][12]+t302*B[5][12];
  Kmat[3][13] = t307*B[1][13]+t297*B[3][13]+t313*B[4][13];
  Kmat[3][14] = t318*B[2][14]+t313*B[4][14]+t302*B[5][14];
  Kmat[3][15] = t292*B[0][15]+t297*B[3][15]+t302*B[5][15];
  Kmat[3][16] = t307*B[1][16]+t297*B[3][16]+t313*B[4][16];
  Kmat[3][17] = t318*B[2][17]+t313*B[4][17]+t302*B[5][17];
  Kmat[3][18] = t292*B[0][18]+t297*B[3][18]+t302*B[5][18];
  Kmat[3][19] = t307*B[1][19]+t297*B[3][19]+t313*B[4][19];
  Kmat[3][20] = t318*B[2][20]+t313*B[4][20]+t302*B[5][20];
  Kmat[3][21] = t292*B[0][21]+t297*B[3][21]+t302*B[5][21];
  Kmat[3][22] = t307*B[1][22]+t297*B[3][22]+t313*B[4][22];
  Kmat[3][23] = t318*B[2][23]+t313*B[4][23]+t302*B[5][23];
  t388 = B[1][4]*D[0][1]+B[3][4]*D[0][3]+B[4][4]*D[0][4];
  t393 = B[1][4]*D[1][3]+B[3][4]*D[3][3]+B[4][4]*D[3][4];
  t398 = B[1][4]*D[1][5]+B[3][4]*D[3][5]+B[4][4]*D[4][5];
  Kmat[4][0] = Kmat[0][4];
  t403 = B[1][4]*D[1][1]+B[3][4]*D[1][3]+B[4][4]*D[1][4];
  t409 = B[1][4]*D[1][4]+B[3][4]*D[3][4]+B[4][4]*D[4][4];
  Kmat[4][1] = Kmat[1][4];
  t414 = B[1][4]*D[1][2]+B[3][4]*D[2][3]+B[4][4]*D[2][4];
  Kmat[4][2] = Kmat[2][4];
  Kmat[4][3] = Kmat[3][4];
  Kmat[4][4] = t403*B[1][4]+t393*B[3][4]+t409*B[4][4];
  Kmat[4][5] = t414*B[2][5]+t409*B[4][5]+t398*B[5][5];
  Kmat[4][6] = t388*B[0][6]+t393*B[3][6]+t398*B[5][6];
  Kmat[4][7] = t403*B[1][7]+t393*B[3][7]+t409*B[4][7];
  Kmat[4][8] = t414*B[2][8]+t409*B[4][8]+t398*B[5][8];
  Kmat[4][9] = t388*B[0][9]+t393*B[3][9]+t398*B[5][9];
  Kmat[4][10] = t403*B[1][10]+t393*B[3][10]+t409*B[4][10];
  Kmat[4][11] = t414*B[2][11]+t409*B[4][11]+t398*B[5][11];
  Kmat[4][12] = t388*B[0][12]+t393*B[3][12]+t398*B[5][12];
  Kmat[4][13] = t403*B[1][13]+t393*B[3][13]+t409*B[4][13];
  Kmat[4][14] = t414*B[2][14]+t409*B[4][14]+t398*B[5][14];
  Kmat[4][15] = t388*B[0][15]+t393*B[3][15]+t398*B[5][15];
  Kmat[4][16] = t403*B[1][16]+t393*B[3][16]+t409*B[4][16];
  Kmat[4][17] = t414*B[2][17]+t409*B[4][17]+t398*B[5][17];
  Kmat[4][18] = t388*B[0][18]+t393*B[3][18]+t398*B[5][18];
  Kmat[4][19] = t403*B[1][19]+t393*B[3][19]+t409*B[4][19];
  Kmat[4][20] = t414*B[2][20]+t409*B[4][20]+t398*B[5][20];
  Kmat[4][21] = t388*B[0][21]+t393*B[3][21]+t398*B[5][21];
  Kmat[4][22] = t403*B[1][22]+t393*B[3][22]+t409*B[4][22];
  Kmat[4][23] = t414*B[2][23]+t409*B[4][23]+t398*B[5][23];
  t484 = B[2][5]*D[0][2]+B[4][5]*D[0][4]+B[5][5]*D[0][5];
  t489 = B[2][5]*D[2][3]+B[4][5]*D[3][4]+B[5][5]*D[3][5];
  t494 = B[2][5]*D[2][5]+B[4][5]*D[4][5]+B[5][5]*D[5][5];
  Kmat[5][0] = Kmat[0][5];
  t499 = B[2][5]*D[1][2]+B[4][5]*D[1][4]+B[5][5]*D[1][5];
  t505 = B[2][5]*D[2][4]+B[4][5]*D[4][4]+B[5][5]*D[4][5];
  Kmat[5][1] = Kmat[1][5];
  t510 = B[2][5]*D[2][2]+B[4][5]*D[2][4]+B[5][5]*D[2][5];
  Kmat[5][2] = Kmat[2][5];
  Kmat[5][3] = Kmat[3][5];
  Kmat[5][4] = Kmat[4][5];
  Kmat[5][5] = t510*B[2][5]+t505*B[4][5]+t494*B[5][5];
  Kmat[5][6] = t484*B[0][6]+t489*B[3][6]+t494*B[5][6];
  Kmat[5][7] = t499*B[1][7]+t489*B[3][7]+t505*B[4][7];
  Kmat[5][8] = t510*B[2][8]+t505*B[4][8]+t494*B[5][8];
  Kmat[5][9] = t484*B[0][9]+t489*B[3][9]+t494*B[5][9];
  Kmat[5][10] = t499*B[1][10]+t489*B[3][10]+t505*B[4][10];
  Kmat[5][11] = t510*B[2][11]+t505*B[4][11]+t494*B[5][11];
  Kmat[5][12] = t484*B[0][12]+t489*B[3][12]+t494*B[5][12];
  Kmat[5][13] = t499*B[1][13]+t489*B[3][13]+t505*B[4][13];
  Kmat[5][14] = t510*B[2][14]+t505*B[4][14]+t494*B[5][14];
  Kmat[5][15] = t484*B[0][15]+t489*B[3][15]+t494*B[5][15];
  Kmat[5][16] = t499*B[1][16]+t489*B[3][16]+t505*B[4][16];
  Kmat[5][17] = t510*B[2][17]+t505*B[4][17]+t494*B[5][17];
  Kmat[5][18] = t484*B[0][18]+t489*B[3][18]+t494*B[5][18];
  Kmat[5][19] = t499*B[1][19]+t489*B[3][19]+t505*B[4][19];
  Kmat[5][20] = t510*B[2][20]+t505*B[4][20]+t494*B[5][20];
  Kmat[5][21] = t484*B[0][21]+t489*B[3][21]+t494*B[5][21];
  Kmat[5][22] = t499*B[1][22]+t489*B[3][22]+t505*B[4][22];
  Kmat[5][23] = t510*B[2][23]+t505*B[4][23]+t494*B[5][23];
  t580 = B[0][6]*D[0][0]+B[3][6]*D[0][3]+B[5][6]*D[0][5];
  t585 = B[0][6]*D[0][3]+B[3][6]*D[3][3]+B[5][6]*D[3][5];
  t590 = B[0][6]*D[0][5]+B[3][6]*D[3][5]+B[5][6]*D[5][5];
  Kmat[6][0] = Kmat[0][6];
  t595 = B[0][6]*D[0][1]+B[3][6]*D[1][3]+B[5][6]*D[1][5];
  t601 = B[0][6]*D[0][4]+B[3][6]*D[3][4]+B[5][6]*D[4][5];
  Kmat[6][1] = Kmat[1][6];
  t606 = B[0][6]*D[0][2]+B[3][6]*D[2][3]+B[5][6]*D[2][5];
  Kmat[6][2] = Kmat[2][6];
  Kmat[6][3] = Kmat[3][6];
  Kmat[6][4] = Kmat[4][6];
  Kmat[6][5] = Kmat[5][6];
  Kmat[6][6] = t580*B[0][6]+t585*B[3][6]+t590*B[5][6];
  Kmat[6][7] = t595*B[1][7]+t585*B[3][7]+t601*B[4][7];
  Kmat[6][8] = t606*B[2][8]+t601*B[4][8]+t590*B[5][8];
  Kmat[6][9] = t580*B[0][9]+t585*B[3][9]+t590*B[5][9];
  Kmat[6][10] = t595*B[1][10]+t585*B[3][10]+t601*B[4][10];
  Kmat[6][11] = t606*B[2][11]+t601*B[4][11]+t590*B[5][11];
  Kmat[6][12] = t580*B[0][12]+t585*B[3][12]+t590*B[5][12];
  Kmat[6][13] = t595*B[1][13]+t585*B[3][13]+t601*B[4][13];
  Kmat[6][14] = t606*B[2][14]+t601*B[4][14]+t590*B[5][14];
  Kmat[6][15] = t580*B[0][15]+t585*B[3][15]+t590*B[5][15];
  Kmat[6][16] = t595*B[1][16]+t585*B[3][16]+t601*B[4][16];
  Kmat[6][17] = t606*B[2][17]+t601*B[4][17]+t590*B[5][17];
  Kmat[6][18] = t580*B[0][18]+t585*B[3][18]+t590*B[5][18];
  Kmat[6][19] = t595*B[1][19]+t585*B[3][19]+t601*B[4][19];
  Kmat[6][20] = t606*B[2][20]+t601*B[4][20]+t590*B[5][20];
  Kmat[6][21] = t580*B[0][21]+t585*B[3][21]+t590*B[5][21];
  Kmat[6][22] = t595*B[1][22]+t585*B[3][22]+t601*B[4][22];
  Kmat[6][23] = t606*B[2][23]+t601*B[4][23]+t590*B[5][23];
  t676 = B[1][7]*D[0][1]+B[3][7]*D[0][3]+B[4][7]*D[0][4];
  t681 = B[1][7]*D[1][3]+B[3][7]*D[3][3]+B[4][7]*D[3][4];
  t686 = B[1][7]*D[1][5]+B[3][7]*D[3][5]+B[4][7]*D[4][5];
  Kmat[7][0] = Kmat[0][7];
  t691 = B[1][7]*D[1][1]+B[3][7]*D[1][3]+B[4][7]*D[1][4];
  t697 = B[1][7]*D[1][4]+B[3][7]*D[3][4]+B[4][7]*D[4][4];
  Kmat[7][1] = Kmat[1][7];
  t702 = B[1][7]*D[1][2]+B[3][7]*D[2][3]+B[4][7]*D[2][4];
  Kmat[7][2] = Kmat[2][7];
  Kmat[7][3] = Kmat[3][7];
  Kmat[7][4] = Kmat[4][7];
  Kmat[7][5] = Kmat[5][7];
  Kmat[7][6] = Kmat[6][7];
  Kmat[7][7] = t691*B[1][7]+t681*B[3][7]+t697*B[4][7];
  Kmat[7][8] = t702*B[2][8]+t697*B[4][8]+t686*B[5][8];
  Kmat[7][9] = t676*B[0][9]+t681*B[3][9]+t686*B[5][9];
  Kmat[7][10] = t691*B[1][10]+t681*B[3][10]+t697*B[4][10];
  Kmat[7][11] = t702*B[2][11]+t697*B[4][11]+t686*B[5][11];
  Kmat[7][12] = t676*B[0][12]+t681*B[3][12]+t686*B[5][12];
  Kmat[7][13] = t691*B[1][13]+t681*B[3][13]+t697*B[4][13];
  Kmat[7][14] = t702*B[2][14]+t697*B[4][14]+t686*B[5][14];
  Kmat[7][15] = t676*B[0][15]+t681*B[3][15]+t686*B[5][15];
  Kmat[7][16] = t691*B[1][16]+t681*B[3][16]+t697*B[4][16];
  Kmat[7][17] = t702*B[2][17]+t697*B[4][17]+t686*B[5][17];
  Kmat[7][18] = t676*B[0][18]+t681*B[3][18]+t686*B[5][18];
  Kmat[7][19] = t691*B[1][19]+t681*B[3][19]+t697*B[4][19];
  Kmat[7][20] = t702*B[2][20]+t697*B[4][20]+t686*B[5][20];
  Kmat[7][21] = t676*B[0][21]+t681*B[3][21]+t686*B[5][21];
  Kmat[7][22] = t691*B[1][22]+t681*B[3][22]+t697*B[4][22];
  Kmat[7][23] = t702*B[2][23]+t697*B[4][23]+t686*B[5][23];
  t772 = B[2][8]*D[0][2]+B[4][8]*D[0][4]+B[5][8]*D[0][5];
  t777 = B[2][8]*D[2][3]+B[4][8]*D[3][4]+B[5][8]*D[3][5];
  t782 = B[2][8]*D[2][5]+B[4][8]*D[4][5]+B[5][8]*D[5][5];
  Kmat[8][0] = Kmat[0][8];
  t787 = B[2][8]*D[1][2]+B[4][8]*D[1][4]+B[5][8]*D[1][5];
  t793 = B[2][8]*D[2][4]+B[4][8]*D[4][4]+B[5][8]*D[4][5];
  Kmat[8][1] = Kmat[1][8];
  t798 = B[2][8]*D[2][2]+B[4][8]*D[2][4]+B[5][8]*D[2][5];
  Kmat[8][2] = Kmat[2][8];
  Kmat[8][3] = Kmat[3][8];
  Kmat[8][4] = Kmat[4][8];
  Kmat[8][5] = Kmat[5][8];
  Kmat[8][6] = Kmat[6][8];
  Kmat[8][7] = Kmat[7][8];
  Kmat[8][8] = t798*B[2][8]+t793*B[4][8]+t782*B[5][8];
  Kmat[8][9] = t772*B[0][9]+t777*B[3][9]+t782*B[5][9];
  Kmat[8][10] = t787*B[1][10]+t777*B[3][10]+t793*B[4][10];
  Kmat[8][11] = t798*B[2][11]+t793*B[4][11]+t782*B[5][11];
  Kmat[8][12] = t772*B[0][12]+t777*B[3][12]+t782*B[5][12];
  Kmat[8][13] = t787*B[1][13]+t777*B[3][13]+t793*B[4][13];
  Kmat[8][14] = t798*B[2][14]+t793*B[4][14]+t782*B[5][14];
  Kmat[8][15] = t772*B[0][15]+t777*B[3][15]+t782*B[5][15];
  Kmat[8][16] = t787*B[1][16]+t777*B[3][16]+t793*B[4][16];
  Kmat[8][17] = t798*B[2][17]+t793*B[4][17]+t782*B[5][17];
  Kmat[8][18] = t772*B[0][18]+t777*B[3][18]+t782*B[5][18];
  Kmat[8][19] = t787*B[1][19]+t777*B[3][19]+t793*B[4][19];
  Kmat[8][20] = t798*B[2][20]+t793*B[4][20]+t782*B[5][20];
  Kmat[8][21] = t772*B[0][21]+t777*B[3][21]+t782*B[5][21];
  Kmat[8][22] = t787*B[1][22]+t777*B[3][22]+t793*B[4][22];
  Kmat[8][23] = t798*B[2][23]+t793*B[4][23]+t782*B[5][23];
  t868 = B[0][9]*D[0][0]+B[3][9]*D[0][3]+B[5][9]*D[0][5];
  t873 = B[0][9]*D[0][3]+B[3][9]*D[3][3]+B[5][9]*D[3][5];
  t878 = B[0][9]*D[0][5]+B[3][9]*D[3][5]+B[5][9]*D[5][5];
  Kmat[9][0] = Kmat[0][9];
  t883 = B[0][9]*D[0][1]+B[3][9]*D[1][3]+B[5][9]*D[1][5];
  t889 = B[0][9]*D[0][4]+B[3][9]*D[3][4]+B[5][9]*D[4][5];
  Kmat[9][1] = Kmat[1][9];
  t894 = B[0][9]*D[0][2]+B[3][9]*D[2][3]+B[5][9]*D[2][5];
  Kmat[9][2] = Kmat[2][9];
  Kmat[9][3] = Kmat[3][9];
  Kmat[9][4] = Kmat[4][9];
  Kmat[9][5] = Kmat[5][9];
  Kmat[9][6] = Kmat[6][9];
  Kmat[9][7] = Kmat[7][9];
  Kmat[9][8] = Kmat[8][9];
  Kmat[9][9] = t868*B[0][9]+t873*B[3][9]+t878*B[5][9];
  Kmat[9][10] = t883*B[1][10]+t873*B[3][10]+t889*B[4][10];
  Kmat[9][11] = t894*B[2][11]+t889*B[4][11]+t878*B[5][11];
  Kmat[9][12] = t868*B[0][12]+t873*B[3][12]+t878*B[5][12];
  Kmat[9][13] = t883*B[1][13]+t873*B[3][13]+t889*B[4][13];
  Kmat[9][14] = t894*B[2][14]+t889*B[4][14]+t878*B[5][14];
  Kmat[9][15] = t868*B[0][15]+t873*B[3][15]+t878*B[5][15];
  Kmat[9][16] = t883*B[1][16]+t873*B[3][16]+t889*B[4][16];
  Kmat[9][17] = t894*B[2][17]+t889*B[4][17]+t878*B[5][17];
  Kmat[9][18] = t868*B[0][18]+t873*B[3][18]+t878*B[5][18];
  Kmat[9][19] = t883*B[1][19]+t873*B[3][19]+t889*B[4][19];
  Kmat[9][20] = t894*B[2][20]+t889*B[4][20]+t878*B[5][20];
  Kmat[9][21] = t868*B[0][21]+t873*B[3][21]+t878*B[5][21];
  Kmat[9][22] = t883*B[1][22]+t873*B[3][22]+t889*B[4][22];
  Kmat[9][23] = t894*B[2][23]+t889*B[4][23]+t878*B[5][23];
  t964 = B[1][10]*D[0][1]+B[3][10]*D[0][3]+B[4][10]*D[0][4];
  t969 = B[1][10]*D[1][3]+B[3][10]*D[3][3]+B[4][10]*D[3][4];
  t974 = B[1][10]*D[1][5]+B[3][10]*D[3][5]+B[4][10]*D[4][5];
  Kmat[10][0] = Kmat[0][10];
  t979 = B[1][10]*D[1][1]+B[3][10]*D[1][3]+B[4][10]*D[1][4];
  t985 = B[1][10]*D[1][4]+B[3][10]*D[3][4]+B[4][10]*D[4][4];
  Kmat[10][1] = Kmat[1][10];
  t990 = B[1][10]*D[1][2]+B[3][10]*D[2][3]+B[4][10]*D[2][4];
  Kmat[10][2] = Kmat[2][10];
  Kmat[10][3] = Kmat[3][10];
  Kmat[10][4] = Kmat[4][10];
  Kmat[10][5] = Kmat[5][10];
  Kmat[10][6] = Kmat[6][10];
  Kmat[10][7] = Kmat[7][10];
  Kmat[10][8] = Kmat[8][10];
  Kmat[10][9] = Kmat[9][10];
  Kmat[10][10] = t979*B[1][10]+t969*B[3][10]+t985*B[4][10];
  Kmat[10][11] = t990*B[2][11]+t985*B[4][11]+t974*B[5][11];
  Kmat[10][12] = t964*B[0][12]+t969*B[3][12]+t974*B[5][12];
  Kmat[10][13] = t979*B[1][13]+t969*B[3][13]+t985*B[4][13];
  Kmat[10][14] = t990*B[2][14]+t985*B[4][14]+t974*B[5][14];
  Kmat[10][15] = t964*B[0][15]+t969*B[3][15]+t974*B[5][15];
  Kmat[10][16] = t979*B[1][16]+t969*B[3][16]+t985*B[4][16];
  Kmat[10][17] = t990*B[2][17]+t985*B[4][17]+t974*B[5][17];
  Kmat[10][18] = t964*B[0][18]+t969*B[3][18]+t974*B[5][18];
  Kmat[10][19] = t979*B[1][19]+t969*B[3][19]+t985*B[4][19];
  Kmat[10][20] = t990*B[2][20]+t985*B[4][20]+t974*B[5][20];
  Kmat[10][21] = t964*B[0][21]+t969*B[3][21]+t974*B[5][21];
  Kmat[10][22] = t979*B[1][22]+t969*B[3][22]+t985*B[4][22];
  Kmat[10][23] = t990*B[2][23]+t985*B[4][23]+t974*B[5][23];
  t1060 = B[2][11]*D[0][2]+B[4][11]*D[0][4]+B[5][11]*D[0][5];
  t1065 = B[2][11]*D[2][3]+B[4][11]*D[3][4]+B[5][11]*D[3][5];
  t1070 = B[2][11]*D[2][5]+B[4][11]*D[4][5]+B[5][11]*D[5][5];
  Kmat[11][0] = Kmat[0][11];
  t1075 = B[2][11]*D[1][2]+B[4][11]*D[1][4]+B[5][11]*D[1][5];
  t1081 = B[2][11]*D[2][4]+B[4][11]*D[4][4]+B[5][11]*D[4][5];
  Kmat[11][1] = Kmat[1][11];
  t1086 = B[2][11]*D[2][2]+B[4][11]*D[2][4]+B[5][11]*D[2][5];
  Kmat[11][2] = Kmat[2][11];
  Kmat[11][3] = Kmat[3][11];
  Kmat[11][4] = Kmat[4][11];
  Kmat[11][5] = Kmat[5][11];
  Kmat[11][6] = Kmat[6][11];
  Kmat[11][7] = Kmat[7][11];
  Kmat[11][8] = Kmat[8][11];
  Kmat[11][9] = Kmat[9][11];
  Kmat[11][10] = Kmat[10][11];
  Kmat[11][11] = t1086*B[2][11]+t1081*B[4][11]+t1070*B[5][11];
  Kmat[11][12] = t1060*B[0][12]+t1065*B[3][12]+t1070*B[5][12];
  Kmat[11][13] = t1075*B[1][13]+t1065*B[3][13]+t1081*B[4][13];
  Kmat[11][14] = t1086*B[2][14]+t1081*B[4][14]+t1070*B[5][14];
  Kmat[11][15] = t1060*B[0][15]+t1065*B[3][15]+t1070*B[5][15];
  Kmat[11][16] = t1075*B[1][16]+t1065*B[3][16]+t1081*B[4][16];
  Kmat[11][17] = t1086*B[2][17]+t1081*B[4][17]+t1070*B[5][17];
  Kmat[11][18] = t1060*B[0][18]+t1065*B[3][18]+t1070*B[5][18];
  Kmat[11][19] = t1075*B[1][19]+t1065*B[3][19]+t1081*B[4][19];
  Kmat[11][20] = t1086*B[2][20]+t1081*B[4][20]+t1070*B[5][20];
  Kmat[11][21] = t1060*B[0][21]+t1065*B[3][21]+t1070*B[5][21];
  Kmat[11][22] = t1075*B[1][22]+t1065*B[3][22]+t1081*B[4][22];
  Kmat[11][23] = t1086*B[2][23]+t1081*B[4][23]+t1070*B[5][23];
  t1156 = B[0][12]*D[0][0]+B[3][12]*D[0][3]+B[5][12]*D[0][5];
  t1161 = B[0][12]*D[0][3]+B[3][12]*D[3][3]+B[5][12]*D[3][5];
  t1166 = B[0][12]*D[0][5]+B[3][12]*D[3][5]+B[5][12]*D[5][5];
  Kmat[12][0] = Kmat[0][12];
  t1171 = B[0][12]*D[0][1]+B[3][12]*D[1][3]+B[5][12]*D[1][5];
  t1177 = B[0][12]*D[0][4]+B[3][12]*D[3][4]+B[5][12]*D[4][5];
  Kmat[12][1] = Kmat[1][12];
  t1182 = B[0][12]*D[0][2]+B[3][12]*D[2][3]+B[5][12]*D[2][5];
  Kmat[12][2] = Kmat[2][12];
  Kmat[12][3] = Kmat[3][12];
  Kmat[12][4] = Kmat[4][12];
  Kmat[12][5] = Kmat[5][12];
  Kmat[12][6] = Kmat[6][12];
  Kmat[12][7] = Kmat[7][12];
  Kmat[12][8] = Kmat[8][12];
  Kmat[12][9] = Kmat[9][12];
  Kmat[12][10] = Kmat[10][12];
  Kmat[12][11] = Kmat[11][12];
  Kmat[12][12] = t1156*B[0][12]+t1161*B[3][12]+t1166*B[5][12];
  Kmat[12][13] = t1171*B[1][13]+t1161*B[3][13]+t1177*B[4][13];
  Kmat[12][14] = t1182*B[2][14]+t1177*B[4][14]+t1166*B[5][14];
  Kmat[12][15] = t1156*B[0][15]+t1161*B[3][15]+t1166*B[5][15];
  Kmat[12][16] = t1171*B[1][16]+t1161*B[3][16]+t1177*B[4][16];
  Kmat[12][17] = t1182*B[2][17]+t1177*B[4][17]+t1166*B[5][17];
  Kmat[12][18] = t1156*B[0][18]+t1161*B[3][18]+t1166*B[5][18];
  Kmat[12][19] = t1171*B[1][19]+t1161*B[3][19]+t1177*B[4][19];
  Kmat[12][20] = t1182*B[2][20]+t1177*B[4][20]+t1166*B[5][20];
  Kmat[12][21] = t1156*B[0][21]+t1161*B[3][21]+t1166*B[5][21];
  Kmat[12][22] = t1171*B[1][22]+t1161*B[3][22]+t1177*B[4][22];
  Kmat[12][23] = t1182*B[2][23]+t1177*B[4][23]+t1166*B[5][23];
  t1252 = B[1][13]*D[0][1]+B[3][13]*D[0][3]+B[4][13]*D[0][4];
  t1257 = B[1][13]*D[1][3]+B[3][13]*D[3][3]+B[4][13]*D[3][4];
  t1262 = B[1][13]*D[1][5]+B[3][13]*D[3][5]+B[4][13]*D[4][5];
  Kmat[13][0] = Kmat[0][13];
  t1267 = B[1][13]*D[1][1]+B[3][13]*D[1][3]+B[4][13]*D[1][4];
  t1273 = B[1][13]*D[1][4]+B[3][13]*D[3][4]+B[4][13]*D[4][4];
  Kmat[13][1] = Kmat[1][13];
  t1278 = B[1][13]*D[1][2]+B[3][13]*D[2][3]+B[4][13]*D[2][4];
  Kmat[13][2] = Kmat[2][13];
  Kmat[13][3] = Kmat[3][13];
  Kmat[13][4] = Kmat[4][13];
  Kmat[13][5] = Kmat[5][13];
  Kmat[13][6] = Kmat[6][13];
  Kmat[13][7] = Kmat[7][13];
  Kmat[13][8] = Kmat[8][13];
  Kmat[13][9] = Kmat[9][13];
  Kmat[13][10] = Kmat[10][13];
  Kmat[13][11] = Kmat[11][13];
  Kmat[13][12] = Kmat[12][13];
  Kmat[13][13] = t1267*B[1][13]+t1257*B[3][13]+t1273*B[4][13];
  Kmat[13][14] = t1278*B[2][14]+t1273*B[4][14]+t1262*B[5][14];
  Kmat[13][15] = t1252*B[0][15]+t1257*B[3][15]+t1262*B[5][15];
  Kmat[13][16] = t1267*B[1][16]+t1257*B[3][16]+t1273*B[4][16];
  Kmat[13][17] = t1278*B[2][17]+t1273*B[4][17]+t1262*B[5][17];
  Kmat[13][18] = t1252*B[0][18]+t1257*B[3][18]+t1262*B[5][18];
  Kmat[13][19] = t1267*B[1][19]+t1257*B[3][19]+t1273*B[4][19];
  Kmat[13][20] = t1278*B[2][20]+t1273*B[4][20]+t1262*B[5][20];
  Kmat[13][21] = t1252*B[0][21]+t1257*B[3][21]+t1262*B[5][21];
  Kmat[13][22] = t1267*B[1][22]+t1257*B[3][22]+t1273*B[4][22];
  Kmat[13][23] = t1278*B[2][23]+t1273*B[4][23]+t1262*B[5][23];
  t1348 = B[2][14]*D[0][2]+B[4][14]*D[0][4]+B[5][14]*D[0][5];
  t1353 = B[2][14]*D[2][3]+B[4][14]*D[3][4]+B[5][14]*D[3][5];
  t1358 = B[2][14]*D[2][5]+B[4][14]*D[4][5]+B[5][14]*D[5][5];
  Kmat[14][0] = Kmat[0][14];
  t1363 = B[2][14]*D[1][2]+B[4][14]*D[1][4]+B[5][14]*D[1][5];
  t1369 = B[2][14]*D[2][4]+B[4][14]*D[4][4]+B[5][14]*D[4][5];
  Kmat[14][1] = Kmat[1][14];
  t1374 = B[2][14]*D[2][2]+B[4][14]*D[2][4]+B[5][14]*D[2][5];
  Kmat[14][2] = Kmat[2][14];
  Kmat[14][3] = Kmat[3][14];
  Kmat[14][4] = Kmat[4][14];
  Kmat[14][5] = Kmat[5][14];
  Kmat[14][6] = Kmat[6][14];
  Kmat[14][7] = Kmat[7][14];
  Kmat[14][8] = Kmat[8][14];
  Kmat[14][9] = Kmat[9][14];
  Kmat[14][10] = Kmat[10][14];
  Kmat[14][11] = Kmat[11][14];
  Kmat[14][12] = Kmat[12][14];
  Kmat[14][13] = Kmat[13][14];
  Kmat[14][14] = t1374*B[2][14]+t1369*B[4][14]+t1358*B[5][14];
  Kmat[14][15] = t1348*B[0][15]+t1353*B[3][15]+t1358*B[5][15];
  Kmat[14][16] = t1363*B[1][16]+t1353*B[3][16]+t1369*B[4][16];
  Kmat[14][17] = t1374*B[2][17]+t1369*B[4][17]+t1358*B[5][17];
  Kmat[14][18] = t1348*B[0][18]+t1353*B[3][18]+t1358*B[5][18];
  Kmat[14][19] = t1363*B[1][19]+t1353*B[3][19]+t1369*B[4][19];
  Kmat[14][20] = t1374*B[2][20]+t1369*B[4][20]+t1358*B[5][20];
  Kmat[14][21] = t1348*B[0][21]+t1353*B[3][21]+t1358*B[5][21];
  Kmat[14][22] = t1363*B[1][22]+t1353*B[3][22]+t1369*B[4][22];
  Kmat[14][23] = t1374*B[2][23]+t1369*B[4][23]+t1358*B[5][23];
  t1444 = B[0][15]*D[0][0]+B[3][15]*D[0][3]+B[5][15]*D[0][5];
  t1449 = B[0][15]*D[0][3]+B[3][15]*D[3][3]+B[5][15]*D[3][5];
  t1454 = B[0][15]*D[0][5]+B[3][15]*D[3][5]+B[5][15]*D[5][5];
  Kmat[15][0] = Kmat[0][15];
  t1459 = B[0][15]*D[0][1]+B[3][15]*D[1][3]+B[5][15]*D[1][5];
  t1465 = B[0][15]*D[0][4]+B[3][15]*D[3][4]+B[5][15]*D[4][5];
  Kmat[15][1] = Kmat[1][15];
  t1470 = B[0][15]*D[0][2]+B[3][15]*D[2][3]+B[5][15]*D[2][5];
  Kmat[15][2] = Kmat[2][15];
  Kmat[15][3] = Kmat[3][15];
  Kmat[15][4] = Kmat[4][15];
  Kmat[15][5] = Kmat[5][15];
  Kmat[15][6] = Kmat[6][15];
  Kmat[15][7] = Kmat[7][15];
  Kmat[15][8] = Kmat[8][15];
  Kmat[15][9] = Kmat[9][15];
  Kmat[15][10] = Kmat[10][15];
  Kmat[15][11] = Kmat[11][15];
  Kmat[15][12] = Kmat[12][15];
  Kmat[15][13] = Kmat[13][15];
  Kmat[15][14] = Kmat[14][15];
  Kmat[15][15] = t1444*B[0][15]+t1449*B[3][15]+t1454*B[5][15];
  Kmat[15][16] = t1459*B[1][16]+t1449*B[3][16]+t1465*B[4][16];
  Kmat[15][17] = t1470*B[2][17]+t1465*B[4][17]+t1454*B[5][17];
  Kmat[15][18] = t1444*B[0][18]+t1449*B[3][18]+t1454*B[5][18];
  Kmat[15][19] = t1459*B[1][19]+t1449*B[3][19]+t1465*B[4][19];
  Kmat[15][20] = t1470*B[2][20]+t1465*B[4][20]+t1454*B[5][20];
  Kmat[15][21] = t1444*B[0][21]+t1449*B[3][21]+t1454*B[5][21];
  Kmat[15][22] = t1459*B[1][22]+t1449*B[3][22]+t1465*B[4][22];
  Kmat[15][23] = t1470*B[2][23]+t1465*B[4][23]+t1454*B[5][23];
  t1540 = B[1][16]*D[0][1]+B[3][16]*D[0][3]+B[4][16]*D[0][4];
  t1545 = B[1][16]*D[1][3]+B[3][16]*D[3][3]+B[4][16]*D[3][4];
  t1550 = B[1][16]*D[1][5]+B[3][16]*D[3][5]+B[4][16]*D[4][5];
  Kmat[16][0] = Kmat[0][16];
  t1555 = B[1][16]*D[1][1]+B[3][16]*D[1][3]+B[4][16]*D[1][4];
  t1561 = B[1][16]*D[1][4]+B[3][16]*D[3][4]+B[4][16]*D[4][4];
  Kmat[16][1] = Kmat[1][16];
  t1566 = B[1][16]*D[1][2]+B[3][16]*D[2][3]+B[4][16]*D[2][4];
  Kmat[16][2] = Kmat[2][16];
  Kmat[16][3] = Kmat[3][16];
  Kmat[16][4] = Kmat[4][16];
  Kmat[16][5] = Kmat[5][16];
  Kmat[16][6] = Kmat[6][16];
  Kmat[16][7] = Kmat[7][16];
  Kmat[16][8] = Kmat[8][16];
  Kmat[16][9] = Kmat[9][16];
  Kmat[16][10] = Kmat[10][16];
  Kmat[16][11] = Kmat[11][16];
  Kmat[16][12] = Kmat[12][16];
  Kmat[16][13] = Kmat[13][16];
  Kmat[16][14] = Kmat[14][16];
  Kmat[16][15] = Kmat[15][16];
  Kmat[16][16] = t1555*B[1][16]+t1545*B[3][16]+t1561*B[4][16];
  Kmat[16][17] = t1566*B[2][17]+t1561*B[4][17]+t1550*B[5][17];
  Kmat[16][18] = t1540*B[0][18]+t1545*B[3][18]+t1550*B[5][18];
  Kmat[16][19] = t1555*B[1][19]+t1545*B[3][19]+t1561*B[4][19];
  Kmat[16][20] = t1566*B[2][20]+t1561*B[4][20]+t1550*B[5][20];
  Kmat[16][21] = t1540*B[0][21]+t1545*B[3][21]+t1550*B[5][21];
  Kmat[16][22] = t1555*B[1][22]+t1545*B[3][22]+t1561*B[4][22];
  Kmat[16][23] = t1566*B[2][23]+t1561*B[4][23]+t1550*B[5][23];
  t1636 = B[2][17]*D[0][2]+B[4][17]*D[0][4]+B[5][17]*D[0][5];
  t1641 = B[2][17]*D[2][3]+B[4][17]*D[3][4]+B[5][17]*D[3][5];
  t1646 = B[2][17]*D[2][5]+B[4][17]*D[4][5]+B[5][17]*D[5][5];
  Kmat[17][0] = Kmat[0][17];
  t1651 = B[2][17]*D[1][2]+B[4][17]*D[1][4]+B[5][17]*D[1][5];
  t1657 = B[2][17]*D[2][4]+B[4][17]*D[4][4]+B[5][17]*D[4][5];
  Kmat[17][1] = Kmat[1][17];
  t1662 = B[2][17]*D[2][2]+B[4][17]*D[2][4]+B[5][17]*D[2][5];
  Kmat[17][2] = Kmat[2][17];
  Kmat[17][3] = Kmat[3][17];
  Kmat[17][4] = Kmat[4][17];
  Kmat[17][5] = Kmat[5][17];
  Kmat[17][6] = Kmat[6][17];
  Kmat[17][7] = Kmat[7][17];
  Kmat[17][8] = Kmat[8][17];
  Kmat[17][9] = Kmat[9][17];
  Kmat[17][10] = Kmat[10][17];
  Kmat[17][11] = Kmat[11][17];
  Kmat[17][12] = Kmat[12][17];
  Kmat[17][13] = Kmat[13][17];
  Kmat[17][14] = Kmat[14][17];
  Kmat[17][15] = Kmat[15][17];
  Kmat[17][16] = Kmat[16][17];
  Kmat[17][17] = t1662*B[2][17]+t1657*B[4][17]+t1646*B[5][17];
  Kmat[17][18] = t1636*B[0][18]+t1641*B[3][18]+t1646*B[5][18];
  Kmat[17][19] = t1651*B[1][19]+t1641*B[3][19]+t1657*B[4][19];
  Kmat[17][20] = t1662*B[2][20]+t1657*B[4][20]+t1646*B[5][20];
  Kmat[17][21] = t1636*B[0][21]+t1641*B[3][21]+t1646*B[5][21];
  Kmat[17][22] = t1651*B[1][22]+t1641*B[3][22]+t1657*B[4][22];
  Kmat[17][23] = t1662*B[2][23]+t1657*B[4][23]+t1646*B[5][23];
  t1732 = B[0][18]*D[0][0]+B[3][18]*D[0][3]+B[5][18]*D[0][5];
  t1737 = B[0][18]*D[0][3]+B[3][18]*D[3][3]+B[5][18]*D[3][5];
  t1742 = B[0][18]*D[0][5]+B[3][18]*D[3][5]+B[5][18]*D[5][5];
  Kmat[18][0] = Kmat[0][18];
  t1747 = B[0][18]*D[0][1]+B[3][18]*D[1][3]+B[5][18]*D[1][5];
  t1753 = B[0][18]*D[0][4]+B[3][18]*D[3][4]+B[5][18]*D[4][5];
  Kmat[18][1] = Kmat[1][18];
  t1758 = B[0][18]*D[0][2]+B[3][18]*D[2][3]+B[5][18]*D[2][5];
  Kmat[18][2] = Kmat[2][18];
  Kmat[18][3] = Kmat[3][18];
  Kmat[18][4] = Kmat[4][18];
  Kmat[18][5] = Kmat[5][18];
  Kmat[18][6] = Kmat[6][18];
  Kmat[18][7] = Kmat[7][18];
  Kmat[18][8] = Kmat[8][18];
  Kmat[18][9] = Kmat[9][18];
  Kmat[18][10] = Kmat[10][18];
  Kmat[18][11] = Kmat[11][18];
  Kmat[18][12] = Kmat[12][18];
  Kmat[18][13] = Kmat[13][18];
  Kmat[18][14] = Kmat[14][18];
  Kmat[18][15] = Kmat[15][18];
  Kmat[18][16] = Kmat[16][18];
  Kmat[18][17] = Kmat[17][18];
  Kmat[18][18] = t1732*B[0][18]+t1737*B[3][18]+t1742*B[5][18];
  Kmat[18][19] = t1747*B[1][19]+t1737*B[3][19]+t1753*B[4][19];
  Kmat[18][20] = t1758*B[2][20]+t1753*B[4][20]+t1742*B[5][20];
  Kmat[18][21] = t1732*B[0][21]+t1737*B[3][21]+t1742*B[5][21];
  Kmat[18][22] = t1747*B[1][22]+t1737*B[3][22]+t1753*B[4][22];
  Kmat[18][23] = t1758*B[2][23]+t1753*B[4][23]+t1742*B[5][23];
  t1828 = B[1][19]*D[0][1]+B[3][19]*D[0][3]+B[4][19]*D[0][4];
  t1833 = B[1][19]*D[1][3]+B[3][19]*D[3][3]+B[4][19]*D[3][4];
  t1838 = B[1][19]*D[1][5]+B[3][19]*D[3][5]+B[4][19]*D[4][5];
  Kmat[19][0] = Kmat[0][19];
  t1843 = B[1][19]*D[1][1]+B[3][19]*D[1][3]+B[4][19]*D[1][4];
  t1849 = B[1][19]*D[1][4]+B[3][19]*D[3][4]+B[4][19]*D[4][4];
  Kmat[19][1] = Kmat[1][19];
  t1854 = B[1][19]*D[1][2]+B[3][19]*D[2][3]+B[4][19]*D[2][4];
  Kmat[19][2] = Kmat[2][19];
  Kmat[19][3] = Kmat[3][19];
  Kmat[19][4] = Kmat[4][19];
  Kmat[19][5] = Kmat[5][19];
  Kmat[19][6] = Kmat[6][19];
  Kmat[19][7] = Kmat[7][19];
  Kmat[19][8] = Kmat[8][19];
  Kmat[19][9] = Kmat[9][19];
  Kmat[19][10] = Kmat[10][19];
  Kmat[19][11] = Kmat[11][19];
  Kmat[19][12] = Kmat[12][19];
  Kmat[19][13] = Kmat[13][19];
  Kmat[19][14] = Kmat[14][19];
  Kmat[19][15] = Kmat[15][19];
  Kmat[19][16] = Kmat[16][19];
  Kmat[19][17] = Kmat[17][19];
  Kmat[19][18] = Kmat[18][19];
  Kmat[19][19] = t1843*B[1][19]+t1833*B[3][19]+t1849*B[4][19];
  Kmat[19][20] = t1854*B[2][20]+t1849*B[4][20]+t1838*B[5][20];
  Kmat[19][21] = t1828*B[0][21]+t1833*B[3][21]+t1838*B[5][21];
  Kmat[19][22] = t1843*B[1][22]+t1833*B[3][22]+t1849*B[4][22];
  Kmat[19][23] = t1854*B[2][23]+t1849*B[4][23]+t1838*B[5][23];
  t1924 = B[2][20]*D[0][2]+B[4][20]*D[0][4]+B[5][20]*D[0][5];
  t1929 = B[2][20]*D[2][3]+B[4][20]*D[3][4]+B[5][20]*D[3][5];
  t1934 = B[2][20]*D[2][5]+B[4][20]*D[4][5]+B[5][20]*D[5][5];
  Kmat[20][0] = Kmat[0][20];
  t1939 = B[2][20]*D[1][2]+B[4][20]*D[1][4]+B[5][20]*D[1][5];
  t1945 = B[2][20]*D[2][4]+B[4][20]*D[4][4]+B[5][20]*D[4][5];
  Kmat[20][1] = Kmat[1][20];
  t1950 = B[2][20]*D[2][2]+B[4][20]*D[2][4]+B[5][20]*D[2][5];
  Kmat[20][2] = Kmat[2][20];
  Kmat[20][3] = Kmat[3][20];
  Kmat[20][4] = Kmat[4][20];
  Kmat[20][5] = Kmat[5][20];
  Kmat[20][6] = Kmat[6][20];
  Kmat[20][7] = Kmat[7][20];
  Kmat[20][8] = Kmat[8][20];
  Kmat[20][9] = Kmat[9][20];
  Kmat[20][10] = Kmat[10][20];
  Kmat[20][11] = Kmat[11][20];
  Kmat[20][12] = Kmat[12][20];
  Kmat[20][13] = Kmat[13][20];
  Kmat[20][14] = Kmat[14][20];
  Kmat[20][15] = Kmat[15][20];
  Kmat[20][16] = Kmat[16][20];
  Kmat[20][17] = Kmat[17][20];
  Kmat[20][18] = Kmat[18][20];
  Kmat[20][19] = Kmat[19][20];
  Kmat[20][20] = t1950*B[2][20]+t1945*B[4][20]+t1934*B[5][20];
  Kmat[20][21] = t1924*B[0][21]+t1929*B[3][21]+t1934*B[5][21];
  Kmat[20][22] = t1939*B[1][22]+t1929*B[3][22]+t1945*B[4][22];
  Kmat[20][23] = t1950*B[2][23]+t1945*B[4][23]+t1934*B[5][23];
  t2020 = B[0][21]*D[0][0]+B[3][21]*D[0][3]+B[5][21]*D[0][5];
  t2025 = B[0][21]*D[0][3]+B[3][21]*D[3][3]+B[5][21]*D[3][5];
  t2030 = B[0][21]*D[0][5]+B[3][21]*D[3][5]+B[5][21]*D[5][5];
  Kmat[21][0] = Kmat[0][21];
  t2035 = B[0][21]*D[0][1]+B[3][21]*D[1][3]+B[5][21]*D[1][5];
  t2041 = B[0][21]*D[0][4]+B[3][21]*D[3][4]+B[5][21]*D[4][5];
  Kmat[21][1] = Kmat[1][21];
  t2046 = B[0][21]*D[0][2]+B[3][21]*D[2][3]+B[5][21]*D[2][5];
  Kmat[21][2] = Kmat[2][21];
  Kmat[21][3] = Kmat[3][21];
  Kmat[21][4] = Kmat[4][21];
  Kmat[21][5] = Kmat[5][21];
  Kmat[21][6] = Kmat[6][21];
  Kmat[21][7] = Kmat[7][21];
  Kmat[21][8] = Kmat[8][21];
  Kmat[21][9] = Kmat[9][21];
  Kmat[21][10] = Kmat[10][21];
  Kmat[21][11] = Kmat[11][21];
  Kmat[21][12] = Kmat[12][21];
  Kmat[21][13] = Kmat[13][21];
  Kmat[21][14] = Kmat[14][21];
  Kmat[21][15] = Kmat[15][21];
  Kmat[21][16] = Kmat[16][21];
  Kmat[21][17] = Kmat[17][21];
  Kmat[21][18] = Kmat[18][21];
  Kmat[21][19] = Kmat[19][21];
  Kmat[21][20] = Kmat[20][21];
  Kmat[21][21] = t2020*B[0][21]+t2025*B[3][21]+t2030*B[5][21];
  Kmat[21][22] = t2035*B[1][22]+t2025*B[3][22]+t2041*B[4][22];
  Kmat[21][23] = t2046*B[2][23]+t2041*B[4][23]+t2030*B[5][23];
  //t2116 = B[1][22]*D[0][1]+B[3][22]*D[0][3]+B[4][22]*D[0][4];
  t2121 = B[1][22]*D[1][3]+B[3][22]*D[3][3]+B[4][22]*D[3][4];
  t2126 = B[1][22]*D[1][5]+B[3][22]*D[3][5]+B[4][22]*D[4][5];
  Kmat[22][0] = Kmat[0][22];
  t2131 = B[1][22]*D[1][1]+B[3][22]*D[1][3]+B[4][22]*D[1][4];
  t2137 = B[1][22]*D[1][4]+B[3][22]*D[3][4]+B[4][22]*D[4][4];
  Kmat[22][1] = Kmat[1][22];
  t2142 = B[1][22]*D[1][2]+B[3][22]*D[2][3]+B[4][22]*D[2][4];
  Kmat[22][2] = Kmat[2][22];
  Kmat[22][3] = Kmat[3][22];
  Kmat[22][4] = Kmat[4][22];
  Kmat[22][5] = Kmat[5][22];
  Kmat[22][6] = Kmat[6][22];
  Kmat[22][7] = Kmat[7][22];
  Kmat[22][8] = Kmat[8][22];
  Kmat[22][9] = Kmat[9][22];
  Kmat[22][10] = Kmat[10][22];
  Kmat[22][11] = Kmat[11][22];
  Kmat[22][12] = Kmat[12][22];
  Kmat[22][13] = Kmat[13][22];
  Kmat[22][14] = Kmat[14][22];
  Kmat[22][15] = Kmat[15][22];
  Kmat[22][16] = Kmat[16][22];
  Kmat[22][17] = Kmat[17][22];
  Kmat[22][18] = Kmat[18][22];
  Kmat[22][19] = Kmat[19][22];
  Kmat[22][20] = Kmat[20][22];
  Kmat[22][21] = Kmat[21][22];
  Kmat[22][22] = t2131*B[1][22]+t2121*B[3][22]+t2137*B[4][22];
  Kmat[22][23] = t2142*B[2][23]+t2137*B[4][23]+t2126*B[5][23];
  //t2212 = B[2][23]*D[0][2]+B[4][23]*D[0][4]+B[5][23]*D[0][5];
  //t2217 = B[2][23]*D[2][3]+B[4][23]*D[3][4]+B[5][23]*D[3][5];
  t2222 = B[2][23]*D[2][5]+B[4][23]*D[4][5]+B[5][23]*D[5][5];
  Kmat[23][0] = Kmat[0][23];
  //t2227 = B[2][23]*D[1][2]+B[4][23]*D[1][4]+B[5][23]*D[1][5];
  t2233 = B[2][23]*D[2][4]+B[4][23]*D[4][4]+B[5][23]*D[4][5];
  Kmat[23][1] = Kmat[1][23];
  t2238 = B[2][23]*D[2][2]+B[4][23]*D[2][4]+B[5][23]*D[2][5];
  Kmat[23][2] = Kmat[2][23];
  Kmat[23][3] = Kmat[3][23];
  Kmat[23][4] = Kmat[4][23];
  Kmat[23][5] = Kmat[5][23];
  Kmat[23][6] = Kmat[6][23];
  Kmat[23][7] = Kmat[7][23];
  Kmat[23][8] = Kmat[8][23];
  Kmat[23][9] = Kmat[9][23];
  Kmat[23][10] = Kmat[10][23];
  Kmat[23][11] = Kmat[11][23];
  Kmat[23][12] = Kmat[12][23];
  Kmat[23][13] = Kmat[13][23];
  Kmat[23][14] = Kmat[14][23];
  Kmat[23][15] = Kmat[15][23];
  Kmat[23][16] = Kmat[16][23];
  Kmat[23][17] = Kmat[17][23];
  Kmat[23][18] = Kmat[18][23];
  Kmat[23][19] = Kmat[19][23];
  Kmat[23][20] = Kmat[20][23];
  Kmat[23][21] = Kmat[21][23];
  Kmat[23][22] = Kmat[22][23];
  Kmat[23][23] = t2238*B[2][23]+t2233*B[4][23]+t2222*B[5][23];
}
