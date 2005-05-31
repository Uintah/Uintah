#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ImplicitCM.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMFlags.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Packages/Uintah/Core/Math/FastMatrix.h>
#include <Core/Malloc/Allocator.h>
#include <math.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;


ImplicitCM::ImplicitCM()
{
}

ImplicitCM::ImplicitCM(MPMLabel* Mlb) : d_lb(Mlb)
{
}

ImplicitCM::~ImplicitCM()
{
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

  ParticleVariable<double>  pIntHeatRate;
  ParticleVariable<Matrix3> pDefGrad, pStress;

  new_dw->allocateAndPut(pIntHeatRate,d_lb->pInternalHeatRateLabel,   pset);
  new_dw->allocateAndPut(pDefGrad,    d_lb->pDeformationMeasureLabel, pset);
  new_dw->allocateAndPut(pStress,     d_lb->pStressLabel,             pset);

  // To fix : For a material that is initially stressed we need to
  // modify the stress tensors to comply with the initial stress state
  ParticleSubset::iterator iter = pset->begin();
  for(; iter != pset->end(); iter++){
    particleIndex idx = *iter;
    pIntHeatRate[idx] = 0.0;
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
                                          const PatchSet* ) const
{
  Ghost::GhostType  gnone = Ghost::None;
  Ghost::GhostType  gac   = Ghost::AroundCells;

  task->requires(Task::OldDW, d_lb->pXLabel,           matlset, gnone);
  task->requires(Task::OldDW, d_lb->pMassLabel,        matlset, gnone);
  task->requires(Task::OldDW, d_lb->pVolumeLabel,      matlset, gnone);
  task->requires(Task::OldDW, d_lb->pTemperatureLabel, matlset, gnone);
  task->requires(Task::OldDW, d_lb->pDeformationMeasureLabel,
                                                     matlset, gnone);
  task->requires(Task::OldDW, d_lb->pStressLabel,      matlset, gnone);
  task->requires(Task::OldDW, d_lb->dispNewLabel,      matlset, gac, 1);

  task->computes(d_lb->pStressLabel_preReloc,             matlset);  
  task->computes(d_lb->pDeformationMeasureLabel_preReloc, matlset);
  task->computes(d_lb->pVolumeDeformedLabel,              matlset);
  task->computes(d_lb->pInternalHeatRateLabel_preReloc,   matlset);
}

void 
ImplicitCM::addSharedCRForImplicit(Task* task,
                                          const MaterialSubset* matlset,
                                          const PatchSet* ,
                                          const bool ) const
{
  Ghost::GhostType  gnone = Ghost::None;
  Ghost::GhostType  gac   = Ghost::AroundCells;

  task->requires(Task::ParentOldDW, d_lb->pXLabel,           matlset, gnone);
  task->requires(Task::ParentOldDW, d_lb->pMassLabel,        matlset, gnone);
  task->requires(Task::ParentOldDW, d_lb->pVolumeLabel,      matlset, gnone);
  task->requires(Task::ParentOldDW, d_lb->pTemperatureLabel, matlset, gnone);
  task->requires(Task::ParentOldDW, d_lb->pDeformationMeasureLabel,
                                                           matlset, gnone);
  task->requires(Task::ParentOldDW, d_lb->pStressLabel,      matlset, gnone);
  task->requires(Task::OldDW,       d_lb->dispNewLabel,      matlset, gac, 1);

  task->computes(d_lb->pStressLabel_preReloc,             matlset);  
  task->computes(d_lb->pDeformationMeasureLabel_preReloc, matlset);
  task->computes(d_lb->pVolumeDeformedLabel,              matlset);
  task->computes(d_lb->pInternalHeatRateLabel_preReloc,   matlset);
}



void 
ImplicitCM::computeStressTensor(const PatchSubset*,
                                       const MPMMaterial*,
                                       DataWarehouse*,
                                       DataWarehouse*,
#ifdef HAVE_PETSC
                                       MPMPetscSolver* ,
#else
                                       SimpleSolver* ,
#endif
                                       const bool)
{
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

