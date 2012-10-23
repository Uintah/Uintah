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

#include <CCA/Components/MPM/ConstitutiveModel/TransIsoHyperImplicit.h>
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
#include <Core/Math/TangentModulusTensor.h> //added this for stiffness
#include <Core/Math/Short27.h> //for Fracture
#include <Core/Grid/Variables/NodeIterator.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Math/MinMax.h>
#include <Core/Malloc/Allocator.h>
#include <fstream>
#include <iostream>

using namespace std;
using namespace Uintah;

// _________________this is a transversely isotropic hyperelastic material [JW]
//__________________see Material 18 in LSDYNA manual
//__________________with strain-based failure criteria
//__________________implicit MPM

TransIsoHyperImplicit::TransIsoHyperImplicit(ProblemSpecP& ps,MPMFlags* Mflag)
  : ConstitutiveModel(Mflag),ImplicitCM()
{
  d_useModifiedEOS = false;
//______________________material properties
  ps->require("bulk_modulus", d_initialData.Bulk);
  ps->require("c1", d_initialData.c1);//Mooney Rivlin constant 1
  ps->require("c2", d_initialData.c2);//Mooney Rivlin constant 2
  ps->require("c3", d_initialData.c3);//scales exponential stresses
  ps->require("c4", d_initialData.c4);//controls uncrimping of fibers
  ps->require("c5", d_initialData.c5);//straightened fibers modulus
  ps->require("fiber_stretch", d_initialData.lambda_star);
  ps->require("direction_of_symm", d_initialData.a0);
  ps->require("failure_option",d_initialData.failure);//failure flag True/False
  ps->require("max_fiber_strain",d_initialData.crit_stretch);
  ps->require("max_matrix_strain",d_initialData.crit_shear);
  ps->get("useModifiedEOS",d_useModifiedEOS);//no negative pressure for solids

  pStretchLabel = VarLabel::create("p.stretch",
     ParticleVariable<double>::getTypeDescription());
  pStretchLabel_preReloc = VarLabel::create("p.stretch+",
     ParticleVariable<double>::getTypeDescription());

  pFailureLabel = VarLabel::create("p.fail",
     ParticleVariable<double>::getTypeDescription());
  pFailureLabel_preReloc = VarLabel::create("p.fail+",
     ParticleVariable<double>::getTypeDescription());
}

TransIsoHyperImplicit::TransIsoHyperImplicit(const TransIsoHyperImplicit* cm)
  : ConstitutiveModel(cm), ImplicitCM(cm)
{
  d_useModifiedEOS = cm->d_useModifiedEOS ;

  d_initialData.Bulk = cm->d_initialData.Bulk;
  d_initialData.c1 = cm->d_initialData.c1;
  d_initialData.c2 = cm->d_initialData.c2;
  d_initialData.c3 = cm->d_initialData.c3;
  d_initialData.c4 = cm->d_initialData.c4;
  d_initialData.c5 = cm->d_initialData.c5;
  d_initialData.lambda_star = cm->d_initialData.lambda_star;
  d_initialData.a0 = cm->d_initialData.a0;
  d_initialData.failure = cm->d_initialData.failure;
  d_initialData.crit_stretch = cm->d_initialData.crit_stretch;
  d_initialData.crit_shear = cm->d_initialData.crit_shear;
}

TransIsoHyperImplicit::~TransIsoHyperImplicit()
// _______________________DESTRUCTOR
{
  VarLabel::destroy(pStretchLabel);
  VarLabel::destroy(pStretchLabel_preReloc);
  VarLabel::destroy(pFailureLabel);
  VarLabel::destroy(pFailureLabel_preReloc);
}


void TransIsoHyperImplicit::outputProblemSpec(ProblemSpecP& ps,
                                              bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","trans_iso_hyper");
  }

  cm_ps->appendElement("bulk_modulus", d_initialData.Bulk);
  cm_ps->appendElement("c1", d_initialData.c1);
  cm_ps->appendElement("c2", d_initialData.c2);
  cm_ps->appendElement("c3", d_initialData.c3);
  cm_ps->appendElement("c4", d_initialData.c4);
  cm_ps->appendElement("c5", d_initialData.c5);
  cm_ps->appendElement("fiber_stretch", d_initialData.lambda_star);
  cm_ps->appendElement("direction_of_symm", d_initialData.a0);
  cm_ps->appendElement("failure_option",d_initialData.failure);
  cm_ps->appendElement("max_fiber_strain",d_initialData.crit_stretch);
  cm_ps->appendElement("max_matrix_strain",d_initialData.crit_shear);
  cm_ps->appendElement("useModifiedEOS",d_useModifiedEOS);
}

TransIsoHyperImplicit* TransIsoHyperImplicit::clone()
{
  return scinew TransIsoHyperImplicit(*this);
}

void TransIsoHyperImplicit::initializeCMData(const Patch* patch,
                                        const MPMMaterial* matl,
                                        DataWarehouse* new_dw)
// _____________________STRESS FREE REFERENCE CONFIG
{
   // Initialize the variables shared by all constitutive models
   // This method is defined in the ConstitutiveModel base class.

   // Put stuff in here to initialize each particle's
   // constitutive model parameters and deformationMeasure

   Matrix3 Identity, zero(0.);
   Identity.Identity();

   ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);
   ParticleVariable<Matrix3> deformationGradient, pstress;
   ParticleVariable<double> stretch,fail;

   new_dw->allocateAndPut(deformationGradient,
                                             lb->pDeformationMeasureLabel,pset);
   new_dw->allocateAndPut(pstress,           lb->pStressLabel,            pset);
   new_dw->allocateAndPut(stretch,           pStretchLabel,               pset);
   new_dw->allocateAndPut(fail,              pFailureLabel,               pset);

   for(ParticleSubset::iterator iter = pset->begin();
       iter != pset->end(); iter++){
       deformationGradient[*iter] = Identity;
       fail[*iter] = 0.0;
       pstress[*iter] = zero;
       stretch[*iter] = 1.0;
   }

}
void TransIsoHyperImplicit::allocateCMDataAddRequires(Task* task,
                                                    const MPMMaterial* matl,
                                                    const PatchSet* ,
                                                    MPMLabel* lb) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::NewDW,lb->pDeformationMeasureLabel_preReloc, matlset, Ghost::None);
  task->requires(Task::NewDW,lb->pStressLabel_preReloc,             matlset, Ghost::None);

  // Add requires local to this model
  task->requires(Task::NewDW,pFailureLabel_preReloc,      matlset, Ghost::None);
  task->requires(Task::NewDW,pStretchLabel_preReloc,      matlset, Ghost::None);
}

void TransIsoHyperImplicit::allocateCMDataAdd(DataWarehouse* new_dw,
                                            ParticleSubset* addset,
                                            map<const VarLabel*, ParticleVariableBase*>* newState,
                                            ParticleSubset* delset,
                                            DataWarehouse* )
{
  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  Matrix3 zero(0.);

  ParticleVariable<Matrix3> deformationGradient, pstress;
  constParticleVariable<Matrix3> o_defGrad, o_stress;
  ParticleVariable<double> stretch,fail;
  constParticleVariable<double> o_stretch,o_fail;

  new_dw->allocateTemporary(deformationGradient,addset);
  new_dw->allocateTemporary(pstress,            addset);
  new_dw->allocateTemporary(stretch,            addset);
  new_dw->allocateTemporary(fail,               addset);

  new_dw->get(o_stretch,     pStretchLabel_preReloc,                  delset);
  new_dw->get(o_fail,        pFailureLabel_preReloc,                  delset);
  new_dw->get(o_defGrad,     lb->pDeformationMeasureLabel_preReloc,   delset);
  new_dw->get(o_stress,      lb->pStressLabel_preReloc,               delset);

  ParticleSubset::iterator o,n = addset->begin();
  for (o=delset->begin(); o != delset->end(); o++, n++) {
    deformationGradient[*n] = o_defGrad[*o];
    pstress[*n] = o_stress[*o];
    stretch[*n] = o_stretch[*o];
    fail[*n] = o_fail[*o];
  }
  (*newState)[lb->pDeformationMeasureLabel]=deformationGradient.clone();
  (*newState)[lb->pStressLabel]=pstress.clone();
  (*newState)[pStretchLabel]=stretch.clone();
  (*newState)[pFailureLabel]=fail.clone();
}

void TransIsoHyperImplicit::addParticleState(std::vector<const VarLabel*>& from,
                                   std::vector<const VarLabel*>& to)
//____________________KEEPS TRACK OF THE PARTICLES AND THE RELATED VARIABLES
//____________________(EACH CM ADD ITS OWN STATE VARS)
//____________________AS PARTICLES MOVE FROM PATCH TO PATCH
{
   // Add the local particle state data for this constitutive model.
   from.push_back(lb->pFiberDirLabel);
   from.push_back(pStretchLabel);
   from.push_back(pFailureLabel);

   to.push_back(lb->pFiberDirLabel_preReloc);
   to.push_back(pStretchLabel_preReloc);
   to.push_back(pFailureLabel_preReloc);
}

void TransIsoHyperImplicit::computeStableTimestep(const Patch*,
                                                  const MPMMaterial*,
                                                  DataWarehouse*)
{
  // Not used for the implicit models.
}

Vector TransIsoHyperImplicit::getInitialFiberDir()
{
  return d_initialData.a0;
}

void
TransIsoHyperImplicit::computeStressTensorImplicit(const PatchSubset* patches,
                                                   const MPMMaterial* matl,
                                                   DataWarehouse* old_dw,
                                                   DataWarehouse* new_dw,
                                                   Solver* solver,
                                                   const bool )
//COMPUTES THE STRESS ON ALL THE PARTICLES IN A GIVEN PATCH FOR A GIVEN MATERIAL
//CALLED ONCE PER TIME STEP CONTAINS A COPY OF computeStableTimestep
{
  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);

    IntVector lowIndex = patch->getNodeLowIndex();
    IntVector highIndex = patch->getNodeHighIndex()+IntVector(1,1,1);
    Array3<int> l2g(lowIndex,highIndex);
    solver->copyL2G(l2g,patch);

    Matrix3 rightCauchyGreentilde_new, leftCauchyGreentilde_new;
    Matrix3 pressure, deviatoric_stress, fiber_stress;
    double I1tilde,I2tilde,I4tilde,lambda_tilde;
    double dWdI4tilde,d2WdI4tilde2;
    Matrix3 Shear;
    Vector deformed_fiber_vector;

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<Vector> d_S(interpolator->size());

    Matrix3 Identity;
    Identity.Identity();
    
    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};
    
    int dwi = matl->getDWIndex();

    ParticleSubset* pset;
    constParticleVariable<Point> px;
    constParticleVariable<Matrix3> psize;
    ParticleVariable<Matrix3> deformationGradient_new;
    constParticleVariable<Matrix3> deformationGradient;
    ParticleVariable<Matrix3> pstress;
    constParticleVariable<double> pvolumeold,pmass;
    ParticleVariable<double> pvolume_deformed;
    ParticleVariable<double> stretch;
    ParticleVariable<double> fail;
    constParticleVariable<double> fail_old;
    constParticleVariable<Vector> pvelocity;
    constParticleVariable<Vector> pfiberdir;
    constNCVariable<Vector> dispNew;
    delt_vartype delT;

    DataWarehouse* parent_old_dw =
      new_dw->getOtherDataWarehouse(Task::ParentOldDW);

    pset = parent_old_dw->getParticleSubset(dwi, patch);
    parent_old_dw->get(px,                 lb->pXLabel,                  pset);
    parent_old_dw->get(pmass,              lb->pMassLabel,               pset);
    parent_old_dw->get(psize,              lb->pSizeLabel,               pset);
    parent_old_dw->get(pvolumeold,         lb->pVolumeLabel,             pset);
    parent_old_dw->get(deformationGradient,lb->pDeformationMeasureLabel, pset);
    parent_old_dw->get(pfiberdir,          lb->pFiberDirLabel,           pset);
    parent_old_dw->get(fail_old,           pFailureLabel,                pset);

    new_dw->allocateAndPut(pstress,         lb->pStressLabel_preReloc, pset);
    new_dw->allocateAndPut(pvolume_deformed,lb->pVolumeDeformedLabel,  pset);
    new_dw->allocateTemporary(deformationGradient_new,pset);
    new_dw->allocateAndPut(stretch,         pStretchLabel_preReloc,     pset);
    new_dw->allocateAndPut(fail,            pFailureLabel_preReloc,     pset);

   //_____________________________________________material parameters
    double Bulk  = d_initialData.Bulk;
    double c1 = d_initialData.c1;
    double c2 = d_initialData.c2;
    double c3 = d_initialData.c3;
    double c4 = d_initialData.c4;
    double c5 = d_initialData.c5;
    double lambda_star = d_initialData.lambda_star;
    double c6 = c3*(exp(c4*(lambda_star-1.))-1.)-c5*lambda_star;//c6 = y-intercept
    double failure = d_initialData.failure;
    double crit_shear = d_initialData.crit_shear;
    double crit_stretch = d_initialData.crit_stretch;
    double rho_orig = matl->getInitialDensity();

    double B[6][24];
    double Bnl[3][24];
    double v[576];

    if(matl->getIsRigid()){
      for(ParticleSubset::iterator iter = pset->begin();
                                   iter != pset->end(); iter++){
        particleIndex idx = *iter;
        pstress[idx] = Matrix3(0.0);
        pvolume_deformed[idx] = pvolumeold[idx];
      }
    }
    else{
      Ghost::GhostType  gac   = Ghost::AroundCells;
      if(flag->d_doGridReset){
        constNCVariable<Vector> dispNew;
        old_dw->get(dispNew,lb->dispNewLabel,dwi,patch, gac, 1);
        computeDeformationGradientFromIncrementalDisplacement(
                                                      dispNew, pset, px,
                                                      deformationGradient,
                                                      deformationGradient_new,
                                                      dx, psize, interpolator);
      }
      else if(!flag->d_doGridReset){
        constNCVariable<Vector> gdisplacement;
        old_dw->get(gdisplacement, lb->gDisplacementLabel,dwi,patch,gac,1);
        computeDeformationGradientFromTotalDisplacement(gdisplacement,
                                                        pset, px,
                                                        deformationGradient_new,deformationGradient,                                                        dx, psize,interpolator);
      }
      for(ParticleSubset::iterator iter = pset->begin();
                                   iter != pset->end(); iter++){
        particleIndex idx = *iter;
        // Get the node indices that surround the cell
        interpolator->findCellAndShapeDerivatives(px[idx], ni, d_S, psize[idx],deformationGradient[idx]);
        int dof[24];
        loadBMats(l2g,dof,B,Bnl,d_S,ni,oodx);

        // get the volumetric part of the deformation
        double J = deformationGradient_new[idx].Determinant();
        deformed_fiber_vector =pfiberdir[idx]; // not actually deformed yet
        //________________UNCOUPLE DEVIATORIC AND DILATIONAL PARTS of DEF GRAD
        //________________Ftilde=J^(-1/3)*F and Fvol=J^1/3*Identity
        //________________right Cauchy Green (C) tilde and invariants
        rightCauchyGreentilde_new = deformationGradient_new[idx].Transpose()
                                * deformationGradient_new[idx]*pow(J,-(2./3.));
        I1tilde = rightCauchyGreentilde_new.Trace();
        I2tilde = .5*(I1tilde*I1tilde -
                (rightCauchyGreentilde_new*rightCauchyGreentilde_new).Trace());
        I4tilde = Dot(deformed_fiber_vector,
                   (rightCauchyGreentilde_new*deformed_fiber_vector));
        lambda_tilde = sqrt(I4tilde);
        double I4 = I4tilde*pow(J,(2./3.));// For diagnostics only
        stretch[idx] = sqrt(I4);
        deformed_fiber_vector = deformationGradient_new[idx]*deformed_fiber_vector
                                *(1./lambda_tilde*pow(J,-(1./3.)));
        Matrix3 DY(deformed_fiber_vector,deformed_fiber_vector);

        //________________________________left Cauchy Green (B) tilde
        leftCauchyGreentilde_new = deformationGradient_new[idx]
                     * deformationGradient_new[idx].Transpose()*pow(J,-(2./3.));

        //________________________________strain energy derivatives
        if (lambda_tilde < 1.){
          dWdI4tilde = 0.;
          d2WdI4tilde2 = 0.;
        }
        else if (lambda_tilde < lambda_star) {
          dWdI4tilde = 0.5*c3*(exp(c4*(lambda_tilde-1.))-1.)/(lambda_tilde*lambda_tilde);
          d2WdI4tilde2 = 0.25*c3*(c4*exp(c4*(lambda_tilde-1.))
                        -1./lambda_tilde*(exp(c4*(lambda_tilde-1.))-1.))
                          /(lambda_tilde*lambda_tilde*lambda_tilde);
        }
        else {
          dWdI4tilde = 0.5*(c5+c6/lambda_tilde)/lambda_tilde;
          d2WdI4tilde2 = -0.25*c6
                         /(lambda_tilde*lambda_tilde*lambda_tilde*lambda_tilde);
        }

        //_________________________________stiffness and stress vars.
        double matrix_failed,fiber_failed;

        double cc1 = d2WdI4tilde2*I4tilde*I4tilde;
        double cc2MR = (4./3.)*(1./J)*(c1*I1tilde+2.*c2*I2tilde);
        double cc2FC = (4./3.)*(1./J)*dWdI4tilde*I4tilde;

        Matrix3 RB = leftCauchyGreentilde_new;
        Matrix3 RB2 = leftCauchyGreentilde_new*leftCauchyGreentilde_new;
        Matrix3 I = Identity;
        Matrix3 devsMR = (RB*(c1+c2*I1tilde)+RB2*(-c2)+I*(c1*I1tilde+2*c2*I2tilde))*(2./J)*(-2./3.);
        Matrix3 devsFC = (DY-I*(1./3.))*dWdI4tilde*I4tilde*(2./J)*(-2./3.);
        Matrix3 termMR = RB*(1./J)*c2*I1tilde-RB2*(1./J)*c2;

        //________________________________Failure+Stress+Stiffness
        fail[idx] = 0.;
        double D[6][6]; //stiffness matrix
        if (failure == 1){
        matrix_failed = 0.;
        fiber_failed = 0.;

        //_______________Mooney Rivlin deviatoric term +failure of matrix
        Matrix3 RCG;
        RCG = deformationGradient_new[idx].Transpose()
             *deformationGradient_new[idx];
        double e1,e2,e3;//eigenvalues of C=symm.+pos.def.->Dis<=0
        double Q,R,Dis;
        double pi = 3.1415926535897932384;
        double I1 = RCG.Trace();
        double I2 = .5*(I1*I1 -(RCG*RCG).Trace());
        double I3 = RCG.Determinant();
        Q = (1./9.)*(3.*I2-pow(I1,2));
        R = (1./54.)*(-9.*I1*I2+27.*I3+2.*pow(I1,3));
        Dis = pow(Q,3)+pow(R,2);
        if (Dis <= 1.e-5 && Dis >= 0.)
          {if (R >= -1.e-5 && R<= 1.e-5)
            e1 = e2 = e3 = I1/3.;
          else
            {
              e1 = 2.*pow(R,1./3.)+I1/3.;
              e3 = -pow(R,1./3.)+I1/3.;
              if (e1 < e3) swap(e1,e3);
              e2=e3;
            }
          }
          else
          {double theta = acos(R/pow(-Q,3./2.));
          e1 = 2.*pow(-Q,1./2.)*cos(theta/3.)+I1/3.;
          e2 = 2.*pow(-Q,1./2.)*cos(theta/3.+2.*pi/3.)+I1/3.;
          e3 = 2.*pow(-Q,1./2.)*cos(theta/3.+4.*pi/3.)+I1/3.;
          if (e1 < e2) swap(e1,e2);
          if (e1 < e3) swap(e1,e3);
          if (e2 < e3) swap(e2,e3);
          };
        double max_shear_strain = (e1-e3)/2.;
        if (max_shear_strain > crit_shear || fail_old[idx] == 1. 
                                          || fail_old[idx] == 3.)
          {deviatoric_stress = Identity*0.;
          fail[idx] = 1.;
          matrix_failed = 1.;
          }
        else
         {deviatoric_stress = (leftCauchyGreentilde_new*(c1+c2*I1tilde)
               - leftCauchyGreentilde_new*leftCauchyGreentilde_new*c2
               - Identity*(1./3.)*(c1*I1tilde+2.*c2*I2tilde))*2./J;
          }
        //________________________________fiber stress term + failure of fibers
        if (stretch[idx] > crit_stretch || fail_old[idx] == 2.
                                        || fail_old[idx] == 3.)
          {fiber_stress = Identity*0.;
          fail[idx] = 2.;
          fiber_failed =1.;
          }
        else
          {fiber_stress = (DY*dWdI4tilde*I4tilde
                           - Identity*(1./3.)*dWdI4tilde*I4tilde)*2./J;
          }
        if ( (matrix_failed + fiber_failed) == 2. || fail_old[idx] == 3.)
          fail[idx] = 3.;
        //________________________________hydrostatic pressure term
        double p=0;
        if (fail[idx] == 1.0 ||fail[idx] == 3.0){
          pressure = Identity*0.;
          p=0;
        }
        else {
            p = Bulk*log(J)/J; // p -= qVisco;
            if (p >= -1.e-5 && p <= 1.e-5)
              p = 0.;
            pressure = Identity*p;
        }
        //_______________________________Cauchy stress
        pstress[idx] = pressure + deviatoric_stress + fiber_stress;

        //________________________________STIFFNESS
        //________________________________________________________vol. term
        double cvol[6][6];//failure already recorded into p
        for(int i=0;i<6;i++){
         for(int j=0;j<6;j++){
          cvol[i][j]=0.0;
         }
        }
        double K = Bulk;
        cvol[0][0] = K*(1./J)-2*p;
        cvol[0][1] = K*(1./J);
        cvol[0][2] = K*(1./J);
        cvol[1][1] = K*(1./J)-2*p;
        cvol[1][2] = K*(1./J);
        cvol[2][2] = K*(1./J)-2*p;
        cvol[3][3] = -p;
        cvol[4][4] = -p;
        cvol[5][5] = -p;

        //________________________________________________________Mooney-Rivlin term
        double cMR[6][6];
        for(int i=0;i<6;i++){
         for(int j=0;j<6;j++){
          cMR[i][j]=0.0;
         }
        }
        if (fail[idx] == 1.0 || fail[idx] == 3.0) {
          // Do nothing, cMR already initialized to 0.0
        }
        else {
         cMR[0][0] = (4./J)*c2*RB(0,0)*RB(0,0)-(4./J)*c2*(RB(0,0)*RB(0,0)
                   +RB(0,0)*RB(0,0))
                   +(2./3.)*cc2MR+(4./9.)*(1./J)*2*c2*I2tilde+devsMR(0,0)
                   +devsMR(0,0)
                   +(-4./3.)*(termMR(0,0)+termMR(0,0));
         cMR[0][1] = (4./J)*c2*RB(0,0)*RB(1,1)-(4./J)*c2*(RB(0,1)*RB(0,1)
                   +RB(0,1)*RB(0,1))
                   +(-1./3.)*cc2MR+devsMR(0,0)+devsMR(1,1)
                   +(-4./3.)*(termMR(0,0)+termMR(1,1));
         cMR[0][2] = (4./J)*c2*RB(0,0)*RB(2,2)-(4./J)*c2*(RB(0,2)*RB(0,2)+RB(0,2)*RB(0,2))
                        +(-1./3.)*cc2MR+devsMR(0,0)+devsMR(2,2)
                        +(-4./3.)*(termMR(0,0)+termMR(2,2));
         cMR[1][1] = (4./J)*c2*RB(1,1)*RB(1,1)-(4./J)*c2*(RB(1,1)*RB(1,1)+RB(1,1)*RB(1,1))
                        +(2./3.)*cc2MR+(4./9.)*(1./J)*2*c2*I2tilde+devsMR(1,1)+devsMR(1,1)
                        +(-4./3.)*(termMR(1,1)+termMR(1,1));
         cMR[1][2] = (4./J)*c2*RB(1,1)*RB(2,2)-(4./J)*c2*(RB(1,2)*RB(1,2)+RB(1,2)*RB(1,2))
                        +(-1./3.)*cc2MR+devsMR(1,1)+devsMR(2,2)
                        +(-4./3.)*(termMR(1,1)+termMR(2,2));
         cMR[2][2] = (4./J)*c2*RB(2,2)*RB(2,2)-(4./J)*c2*(RB(2,2)*RB(2,2)+RB(2,2)*RB(2,2))
                        +(2./3.)*cc2MR+(4./9.)*(1./J)*2*c2*I2tilde+devsMR(2,2)+devsMR(2,2)
                        +(-4./3.)*(termMR(2,2)+termMR(2,2));
         cMR[0][3] = (4./J)*c2*RB(0,0)*RB(0,1)-(4./J)*c2*(RB(0,0)*RB(0,1)+RB(0,1)*RB(0,0))
                        +devsMR(0,1)+(-4./3.)*termMR(0,1);
         cMR[0][4] = (4./J)*c2*RB(0,0)*RB(1,2)-(4./J)*c2*(RB(0,1)*RB(0,2)+RB(0,2)*RB(0,1))
                        +devsMR(1,2)+(-4./3.)*termMR(1,2);
         cMR[0][5] = (4./J)*c2*RB(0,0)*RB(2,0)-(4./J)*c2*(RB(0,2)*RB(0,0)+RB(0,0)*RB(0,2))
                        +devsMR(2,0)+(-4./3.)*termMR(2,0);
         cMR[1][3] = (4./J)*c2*RB(1,1)*RB(0,1)-(4./J)*c2*(RB(1,0)*RB(1,1)+RB(1,1)*RB(1,0))
                        +devsMR(0,1)+(-4./3.)*termMR(0,1);
         cMR[1][4] = (4./J)*c2*RB(1,1)*RB(1,2)-(4./J)*c2*(RB(1,1)*RB(1,2)+RB(1,2)*RB(1,1))
                        +devsMR(1,2)+(-4./3.)*termMR(1,2);
         cMR[1][5] = (4./J)*c2*RB(1,1)*RB(2,0)-(4./J)*c2*(RB(1,2)*RB(1,0)+RB(1,0)*RB(1,2))
                        +devsMR(2,0)+(-4./3.)*termMR(2,0);
         cMR[2][3] = (4./J)*c2*RB(2,2)*RB(0,1)-(4./J)*c2*(RB(2,0)*RB(2,1)+RB(2,1)*RB(2,0))
                        +devsMR(0,1)+(-4./3.)*termMR(0,1);
         cMR[2][4] = (4./J)*c2*RB(2,2)*RB(1,2)-(4./J)*c2*(RB(2,1)*RB(2,2)+RB(2,2)*RB(2,1))
                        +devsMR(1,2)+(-4./3.)*termMR(1,2);
         cMR[2][5] = (4./J)*c2*RB(2,2)*RB(2,0)-(4./J)*c2*(RB(2,2)*RB(2,0)+RB(2,0)*RB(2,2))
                        +devsMR(2,0)+(-4./3.)*termMR(2,0);
         cMR[3][3] = (4./J)*c2*RB(0,1)*RB(0,1)-(4./J)*c2*(RB(0,0)*RB(1,1)+RB(0,1)*RB(1,0))
                        +(1./2.)*cc2MR+(4./9.)*(1./J)*2*c2*I2tilde;
         cMR[3][4] = (4./J)*c2*RB(0,1)*RB(1,2)-(4./J)*c2*(RB(0,1)*RB(1,2)+RB(0,2)*RB(1,1));

         cMR[3][5] = (4./J)*c2*RB(0,1)*RB(2,0)-(4./J)*c2*(RB(0,2)*RB(1,0)+RB(0,0)*RB(1,2));

         cMR[4][4] = (4./J)*c2*RB(1,2)*RB(1,2)-(4./J)*c2*(RB(1,1)*RB(2,2)+RB(1,2)*RB(2,1))
                        +(1./2.)*cc2MR+(4./9.)*(1./J)*2*c2*I2tilde;
         cMR[4][5] = (4./J)*c2*RB(1,2)*RB(2,0)-(4./J)*c2*(RB(1,2)*RB(2,0)+RB(1,0)*RB(2,2));

         cMR[5][5] = (4./J)*c2*RB(2,0)*RB(2,0)-(4./J)*c2*(RB(2,2)*RB(0,0)+RB(2,0)*RB(0,2))
                        +(1./2.)*cc2MR+(4./9.)*(1./J)*2*c2*I2tilde;
        };
        //_______________________________________fiber contribution term

        double cFC[6][6];
        for(int i=0;i<6;i++){
         for(int j=0;j<6;j++){
          cFC[i][j]=0.0;
         }
        }
        if (fail[idx] == 2.0 || fail[idx] == 3.0) {
         cFC[0][0] = 0;
         cFC[0][1] = 0;
         cFC[0][2] = 0;
         cFC[1][1] = 0;
         cFC[1][2] = 0;
         cFC[2][2] = 0;
         cFC[0][3] = 0;
         cFC[0][4] = 0;
         cFC[0][5] = 0;
         cFC[1][3] = 0;
         cFC[1][4] = 0;
         cFC[1][5] = 0;
         cFC[2][3] = 0;
         cFC[2][4] = 0;
         cFC[2][5] = 0;
         cFC[3][3] = 0;
         cFC[3][4] = 0;
         cFC[3][5] = 0;
         cFC[4][4] = 0;
         cFC[4][5] = 0;
         cFC[5][5] = 0;
        }
        else {
         cFC[0][0] = (2./3.)*cc2FC+(4./9.)*(1./J)*cc1+devsFC(0,0)+devsFC(0,0)
                        +(-4./3.)*(1./J)*cc1*(DY(0,0)+DY(0,0))+(4./J)*cc1*DY(0,0)*DY(0,0);
         cFC[0][1] = (-1./3.)*cc2FC+devsFC(0,0)+devsFC(1,1)
                        +(-4./3.)*(1./J)*cc1*(DY(0,0)+DY(1,1))+(4./J)*cc1*DY(0,0)*DY(1,1);
         cFC[0][2] = (-1./3.)*cc2FC+devsFC(0,0)+devsFC(2,2)
                        +(-4./3.)*(1./J)*cc1*(DY(0,0)+DY(2,2))+(4./J)*cc1*DY(0,0)*DY(2,2);
         cFC[1][1] = (2./3.)*cc2FC+(4./9.)*(1./J)*cc1+devsFC(1,1)+devsFC(1,1)
                        +(-4./3.)*(1./J)*cc1*(DY(1,1)+DY(1,1))+(4./J)*cc1*DY(1,1)*DY(1,1);
         cFC[1][2] = (-1./3.)*cc2FC+devsFC(1,1)+devsFC(2,2)
                        +(-4./3.)*(1./J)*cc1*(DY(1,1)+DY(2,2))+(4./J)*cc1*DY(1,1)*DY(2,2);
         cFC[2][2] = (2./3.)*cc2FC+(4./9.)*(1./J)*cc1+devsFC(2,2)+devsFC(2,2)
                        +(-4./3.)*(1./J)*cc1*(DY(2,2)+DY(2,2))+(4./J)*cc1*DY(2,2)*DY(2,2);
         cFC[0][3] = devsFC(0,1)+(-4./3.)*(1./J)*cc1*DY(0,1)+(4./J)*cc1*DY(0,0)*DY(0,1);
         cFC[0][4] = devsFC(1,2)+(-4./3.)*(1./J)*cc1*DY(1,2)+(4./J)*cc1*DY(0,0)*DY(1,2);
         cFC[0][5] = devsFC(2,0)+(-4./3.)*(1./J)*cc1*DY(2,0)+(4./J)*cc1*DY(0,0)*DY(2,0);
         cFC[1][3] = devsFC(0,1)+(-4./3.)*(1./J)*cc1*DY(0,1)+(4./J)*cc1*DY(1,1)*DY(0,1);
         cFC[1][4] = devsFC(1,2)+(-4./3.)*(1./J)*cc1*DY(1,2)+(4./J)*cc1*DY(1,1)*DY(1,2);
         cFC[1][5] = devsFC(2,0)+(-4./3.)*(1./J)*cc1*DY(2,0)+(4./J)*cc1*DY(1,1)*DY(2,0);
         cFC[2][3] = devsFC(0,1)+(-4./3.)*(1./J)*cc1*DY(0,1)+(4./J)*cc1*DY(2,2)*DY(0,1);
         cFC[2][4] = devsFC(1,2)+(-4./3.)*(1./J)*cc1*DY(1,2)+(4./J)*cc1*DY(2,2)*DY(1,2);
         cFC[2][5] = devsFC(2,0)+(-4./3.)*(1./J)*cc1*DY(2,0)+(4./J)*cc1*DY(2,2)*DY(2,0);
         cFC[3][3] = (1./2.)*cc2FC+(4./9.)*(1./J)*cc1+(4./J)*cc1*DY(0,1)*DY(0,1);
         cFC[3][4] = (4./J)*cc1*DY(0,1)*DY(1,2);
         cFC[3][5] = (4./J)*cc1*DY(0,1)*DY(2,0);
         cFC[4][4] = (1./2.)*cc2FC+(4./9.)*(1./J)*cc1+(4./J)*cc1*DY(1,2)*DY(1,2);
         cFC[4][5] = (4./J)*cc1*DY(1,2)*DY(2,0);
         cFC[5][5] = (1./2.)*cc2FC+(4./9.)*(1./J)*cc1+(4./J)*cc1*DY(2,0)*DY(2,0);
        }

        //________________________________________________________the STIFFNESS
        for(int i=0;i<6;i++){
          for(int j=0;j<6;j++){
            D[i][j] =cvol[i][j]+cMR[i][j]+cFC[i][j];
          }
        }
      }
      else{
        deviatoric_stress = (leftCauchyGreentilde_new*(c1+c2*I1tilde)
             - leftCauchyGreentilde_new*leftCauchyGreentilde_new*c2
             - Identity*(1./3.)*(c1*I1tilde+2.*c2*I2tilde))*2./J;
        fiber_stress = (DY*dWdI4tilde*I4tilde
                        - Identity*(1./3.)*dWdI4tilde*I4tilde)*2./J;
        double p = Bulk*log(J)/J; // p -= qVisco;
        if (p >= -1.e-5 && p <= 1.e-5)
          p = 0.;
        pressure = Identity*p;
        //Cauchy stress
        pstress[idx] = pressure + deviatoric_stress + fiber_stress;

        //________________________________STIFFNESS
        //________________________________vol. term
        double cvol[6][6];
        for(int i=0;i<6;i++){
          for(int j=0;j<6;j++){
            cvol[i][j] =  0.;
          }
        }
        double K = Bulk;

        cvol[0][0] = K*(1./J)-2*p;
        cvol[0][1] = K*(1./J);
        cvol[0][2] = K*(1./J);
        cvol[1][1] = K*(1./J)-2*p;
        cvol[1][2] = K*(1./J);
        cvol[2][2] = K*(1./J)-2*p;
        cvol[3][3] = -p;
        cvol[4][4] = -p;
        cvol[5][5] = -p;

         //_________________________________________________Mooney-Rivlin term
         double cMR[6][6];
         for(int i=0;i<6;i++){
          for(int j=0;j<6;j++){
            cMR[i][j]=0.0;
          }
         }
         cMR[0][0] = (4./J)*c2*RB(0,0)*RB(0,0)
                    -(4./J)*c2*(RB(0,0)*RB(0,0)+RB(0,0)*RB(0,0))
                    +(2./3.)*cc2MR+(4./9.)*(1./J)*2*c2*I2tilde
                    +devsMR(0,0)+devsMR(0,0)
                    +(-4./3.)*(termMR(0,0)+termMR(0,0));
         cMR[0][1] = (4./J)*c2*RB(0,0)*RB(1,1)
                    -(4./J)*c2*(RB(0,1)*RB(0,1)+RB(0,1)*RB(0,1))
                    +(-1./3.)*cc2MR+devsMR(0,0)+devsMR(1,1)
                    +(-4./3.)*(termMR(0,0)+termMR(1,1));
         cMR[0][2] = (4./J)*c2*RB(0,0)*RB(2,2)
                   -(4./J)*c2*(RB(0,2)*RB(0,2)+RB(0,2)*RB(0,2))
                   +(-1./3.)*cc2MR+devsMR(0,0)+devsMR(2,2)
                   +(-4./3.)*(termMR(0,0)+termMR(2,2));
         cMR[1][1] = (4./J)*c2*RB(1,1)*RB(1,1)
                   -(4./J)*c2*(RB(1,1)*RB(1,1)+RB(1,1)*RB(1,1))
                   +(2./3.)*cc2MR+(4./9.)*(1./J)*2*c2*I2tilde
                   +devsMR(1,1)+devsMR(1,1)
                   +(-4./3.)*(termMR(1,1)+termMR(1,1));
         cMR[1][2] = (4./J)*c2*RB(1,1)*RB(2,2)
                   -(4./J)*c2*(RB(1,2)*RB(1,2)+RB(1,2)*RB(1,2))
                   +(-1./3.)*cc2MR+devsMR(1,1)+devsMR(2,2)
                   +(-4./3.)*(termMR(1,1)+termMR(2,2));
         cMR[2][2] = (4./J)*c2*RB(2,2)*RB(2,2)
                   -(4./J)*c2*(RB(2,2)*RB(2,2)+RB(2,2)*RB(2,2))
                   +(2./3.)*cc2MR+(4./9.)*(1./J)*2*c2*I2tilde
                   +devsMR(2,2)+devsMR(2,2)
                   +(-4./3.)*(termMR(2,2)+termMR(2,2));
         cMR[0][3] = (4./J)*c2*RB(0,0)*RB(0,1)
                   -(4./J)*c2*(RB(0,0)*RB(0,1)+RB(0,1)*RB(0,0))
                        +devsMR(0,1)+(-4./3.)*termMR(0,1);
         cMR[0][4] = (4./J)*c2*RB(0,0)*RB(1,2)
                   -(4./J)*c2*(RB(0,1)*RB(0,2)+RB(0,2)*RB(0,1))
                        +devsMR(1,2)+(-4./3.)*termMR(1,2);
         cMR[0][5] = (4./J)*c2*RB(0,0)*RB(2,0)
                   -(4./J)*c2*(RB(0,2)*RB(0,0)+RB(0,0)*RB(0,2))
                        +devsMR(2,0)+(-4./3.)*termMR(2,0);
         cMR[1][3] = (4./J)*c2*RB(1,1)*RB(0,1)
                   -(4./J)*c2*(RB(1,0)*RB(1,1)+RB(1,1)*RB(1,0))
                        +devsMR(0,1)+(-4./3.)*termMR(0,1);
         cMR[1][4] = (4./J)*c2*RB(1,1)*RB(1,2)
                   -(4./J)*c2*(RB(1,1)*RB(1,2)+RB(1,2)*RB(1,1))
                        +devsMR(1,2)+(-4./3.)*termMR(1,2);
         cMR[1][5] = (4./J)*c2*RB(1,1)*RB(2,0)
                   -(4./J)*c2*(RB(1,2)*RB(1,0)+RB(1,0)*RB(1,2))
                        +devsMR(2,0)+(-4./3.)*termMR(2,0);
         cMR[2][3] = (4./J)*c2*RB(2,2)*RB(0,1)
                   -(4./J)*c2*(RB(2,0)*RB(2,1)+RB(2,1)*RB(2,0))
                        +devsMR(0,1)+(-4./3.)*termMR(0,1);
         cMR[2][4] = (4./J)*c2*RB(2,2)*RB(1,2)
                   -(4./J)*c2*(RB(2,1)*RB(2,2)+RB(2,2)*RB(2,1))
                        +devsMR(1,2)+(-4./3.)*termMR(1,2);
         cMR[2][5] = (4./J)*c2*RB(2,2)*RB(2,0)
                   -(4./J)*c2*(RB(2,2)*RB(2,0)+RB(2,0)*RB(2,2))
                        +devsMR(2,0)+(-4./3.)*termMR(2,0);
         cMR[3][3] = (4./J)*c2*RB(0,1)*RB(0,1)
                   -(4./J)*c2*(RB(0,0)*RB(1,1)+RB(0,1)*RB(1,0))
                        +(1./2.)*cc2MR+(4./9.)*(1./J)*2*c2*I2tilde;
         cMR[3][4] = (4./J)*c2*RB(0,1)*RB(1,2)
                   -(4./J)*c2*(RB(0,1)*RB(1,2)+RB(0,2)*RB(1,1));
         cMR[3][5] = (4./J)*c2*RB(0,1)*RB(2,0)
                   -(4./J)*c2*(RB(0,2)*RB(1,0)+RB(0,0)*RB(1,2));
         cMR[4][4] = (4./J)*c2*RB(1,2)*RB(1,2)
                   -(4./J)*c2*(RB(1,1)*RB(2,2)+RB(1,2)*RB(2,1))
                        +(1./2.)*cc2MR+(4./9.)*(1./J)*2*c2*I2tilde;
         cMR[4][5] = (4./J)*c2*RB(1,2)*RB(2,0)
                   -(4./J)*c2*(RB(1,2)*RB(2,0)+RB(1,0)*RB(2,2));
         cMR[5][5] = (4./J)*c2*RB(2,0)*RB(2,0)
                   -(4./J)*c2*(RB(2,2)*RB(0,0)+RB(2,0)*RB(0,2))
                   +(1./2.)*cc2MR+(4./9.)*(1./J)*2*c2*I2tilde;

        //_________________________fiber contribution term
        double cFC[6][6];
         for(int i=0;i<6;i++){
          for(int j=0;j<6;j++){
           cFC[i][j]=0.0;
          }
         }
         cFC[0][0] = (2./3.)*cc2FC+(4./9.)*(1./J)*cc1+devsFC(0,0)+devsFC(0,0)
              +(-4./3.)*(1./J)*cc1*(DY(0,0)+DY(0,0))+(4./J)*cc1*DY(0,0)*DY(0,0);
         cFC[0][1] = (-1./3.)*cc2FC+devsFC(0,0)+devsFC(1,1)
              +(-4./3.)*(1./J)*cc1*(DY(0,0)+DY(1,1))+(4./J)*cc1*DY(0,0)*DY(1,1);
         cFC[0][2] = (-1./3.)*cc2FC+devsFC(0,0)+devsFC(2,2)
              +(-4./3.)*(1./J)*cc1*(DY(0,0)+DY(2,2))+(4./J)*cc1*DY(0,0)*DY(2,2);
         cFC[1][1] = (2./3.)*cc2FC+(4./9.)*(1./J)*cc1+devsFC(1,1)+devsFC(1,1)
              +(-4./3.)*(1./J)*cc1*(DY(1,1)+DY(1,1))+(4./J)*cc1*DY(1,1)*DY(1,1);
         cFC[1][2] = (-1./3.)*cc2FC+devsFC(1,1)+devsFC(2,2)
              +(-4./3.)*(1./J)*cc1*(DY(1,1)+DY(2,2))+(4./J)*cc1*DY(1,1)*DY(2,2);
         cFC[2][2] = (2./3.)*cc2FC+(4./9.)*(1./J)*cc1+devsFC(2,2)+devsFC(2,2)
              +(-4./3.)*(1./J)*cc1*(DY(2,2)+DY(2,2))+(4./J)*cc1*DY(2,2)*DY(2,2);
         cFC[0][3] = devsFC(0,1)+(-4./3.)*(1./J)*cc1*DY(0,1)
                   +(4./J)*cc1*DY(0,0)*DY(0,1);
         cFC[0][4] = devsFC(1,2)+(-4./3.)*(1./J)*cc1*DY(1,2)
                   +(4./J)*cc1*DY(0,0)*DY(1,2);
         cFC[0][5] = devsFC(2,0)+(-4./3.)*(1./J)*cc1*DY(2,0)
                   +(4./J)*cc1*DY(0,0)*DY(2,0);
         cFC[1][3] = devsFC(0,1)+(-4./3.)*(1./J)*cc1*DY(0,1)
                   +(4./J)*cc1*DY(1,1)*DY(0,1);
         cFC[1][4] = devsFC(1,2)+(-4./3.)*(1./J)*cc1*DY(1,2)
                   +(4./J)*cc1*DY(1,1)*DY(1,2);
         cFC[1][5] = devsFC(2,0)+(-4./3.)*(1./J)*cc1*DY(2,0)
                   +(4./J)*cc1*DY(1,1)*DY(2,0);
         cFC[2][3] = devsFC(0,1)+(-4./3.)*(1./J)*cc1*DY(0,1)
                   +(4./J)*cc1*DY(2,2)*DY(0,1);
         cFC[2][4] = devsFC(1,2)+(-4./3.)*(1./J)*cc1*DY(1,2)
                   +(4./J)*cc1*DY(2,2)*DY(1,2);
         cFC[2][5] = devsFC(2,0)+(-4./3.)*(1./J)*cc1*DY(2,0)
                   +(4./J)*cc1*DY(2,2)*DY(2,0);
         cFC[3][3] =.5*cc2FC+(4./9.)*(1./J)*cc1+(4./J)*cc1*DY(0,1)*DY(0,1);
         cFC[3][4] =(4./J)*cc1*DY(0,1)*DY(1,2);
         cFC[3][5] =(4./J)*cc1*DY(0,1)*DY(2,0);
         cFC[4][4] =.5*cc2FC+(4./9.)*(1./J)*cc1+(4./J)*cc1*DY(1,2)*DY(1,2);
         cFC[4][5] =(4./J)*cc1*DY(1,2)*DY(2,0);
         cFC[5][5] =.5*cc2FC+(4./9.)*(1./J)*cc1+(4./J)*cc1*DY(2,0)*DY(2,0);

        //________________________________________________________the STIFFNESS
        for(int i=0;i<6;i++){
          for(int j=0;j<6;j++){
            D[i][j] =cvol[i][j]+cMR[i][j]+cFC[i][j];
          }
        }
      }

      // kmat = B.transpose()*D*B*volold
      double kmat[24][24];
      BtDB(B,D,kmat);
      // kgeo = Bnl.transpose*sig*Bnl*volnew;
      double sig[3][3];
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          sig[i][j]=pstress[idx](i,j);
        }
      }
      double kgeo[24][24];
      BnltDBnl(Bnl,sig,kgeo);
      double volold = (pmass[idx]/rho_orig);
      double volnew = volold*J;
      pvolume_deformed[idx] = volnew;
      for(int ii = 0;ii<24;ii++){
        for(int jj = 0;jj<24;jj++){
          kmat[ii][jj]*=volold;
          kgeo[ii][jj]*=volnew;
        }
      }
      for (int I = 0; I < 24;I++){
        for (int J = 0; J < 24; J++){
          v[24*I+J] = kmat[I][J] + kgeo[I][J];
        }
      }
      solver->fillMatrix(24,dof,24,dof,v);
      }  // end of loop over particles
    }
  delete interpolator;
  }
}

void
TransIsoHyperImplicit::computeStressTensorImplicit(const PatchSubset* patches,
                                                   const MPMMaterial* matl,
                                                   DataWarehouse* old_dw,
                                                   DataWarehouse* new_dw)
//___________________the final one
{
   for(int pp=0;pp<patches->size();pp++){
     const Patch* patch = patches->get(pp);

     Matrix3 Identity;
     Identity.Identity();
     Matrix3 rightCauchyGreentilde_new, leftCauchyGreentilde_new;
     Matrix3 pressure, deviatoric_stress, fiber_stress;
     double I1tilde,I2tilde,I4tilde,lambda_tilde;
     double dWdI4tilde;
     Vector deformed_fiber_vector;

     ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
     vector<IntVector> ni(interpolator->size());
     vector<Vector> d_S(interpolator->size());

     Vector dx = patch->dCell();

     int dwi = matl->getDWIndex();
     ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
     constParticleVariable<Point> px;
     constParticleVariable<Matrix3> psize;
     ParticleVariable<Matrix3> deformationGradient_new;
     constParticleVariable<Matrix3> deformationGradient;
     ParticleVariable<Matrix3> pstress;
     constParticleVariable<double> pvolumeold;
     ParticleVariable<double> pvolume_deformed;

     ParticleVariable<double> stretch;
     ParticleVariable<double> fail;
     constParticleVariable<double> fail_old;
     constParticleVariable<Vector> pvelocity;
     constParticleVariable<Vector> pfiberdir;
     ParticleVariable<Vector> pfiberdir_carry;

     delt_vartype delT;
     old_dw->get(delT,lb->delTLabel, getLevel(patches));

     old_dw->get(px,                  lb->pXLabel,                  pset);
     old_dw->get(psize,               lb->pSizeLabel,               pset);
     old_dw->get(pvolumeold,          lb->pVolumeLabel,             pset);
     old_dw->get(pfiberdir,           lb->pFiberDirLabel,           pset);
     old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);
     old_dw->get(fail_old,            pFailureLabel,                pset);

     new_dw->allocateAndPut(pstress,         lb->pStressLabel_preReloc,   pset);
     new_dw->allocateAndPut(pvolume_deformed,lb->pVolumeDeformedLabel,    pset);
     new_dw->allocateAndPut(deformationGradient_new,
                                 lb->pDeformationMeasureLabel_preReloc,   pset);
     new_dw->allocateAndPut(pfiberdir_carry, lb->pFiberDirLabel_preReloc, pset);
     new_dw->allocateAndPut(stretch,         pStretchLabel_preReloc,      pset);
     new_dw->allocateAndPut(fail,            pFailureLabel_preReloc,      pset);
     //_____________________________________________material parameters
     double Bulk  = d_initialData.Bulk;
     //Vector a0 = d_initialData.a0;
     double c1 = d_initialData.c1;
     double c2 = d_initialData.c2;
     double c3 = d_initialData.c3;
     double c4 = d_initialData.c4;
     double c5 = d_initialData.c5;
     double lambda_star = d_initialData.lambda_star;
     double c6 = c3*(exp(c4*(lambda_star-1.))-1.)-c5*lambda_star;
     double failure = d_initialData.failure;
     double crit_shear = d_initialData.crit_shear;
     double crit_stretch = d_initialData.crit_stretch;

    if(matl->getIsRigid()){
      for(ParticleSubset::iterator iter = pset->begin();
                                   iter != pset->end(); iter++){
        particleIndex idx = *iter;
        pstress[idx] = Matrix3(0.0);
        deformationGradient_new[idx] = Identity;
        pvolume_deformed[idx] = pvolumeold[idx];
      }
    }
    else{
     Ghost::GhostType  gac   = Ghost::AroundCells;
     if(flag->d_doGridReset){
        constNCVariable<Vector> dispNew;
        new_dw->get(dispNew,lb->dispNewLabel,dwi,patch, gac, 1);
        computeDeformationGradientFromIncrementalDisplacement(
                                                    dispNew, pset, px,
                                                    deformationGradient,
                                                    deformationGradient_new,
                                                    dx, psize, interpolator);
     }
     else if(!flag->d_doGridReset){
        constNCVariable<Vector> gdisplacement;
        new_dw->get(gdisplacement, lb->gDisplacementLabel,dwi,patch,gac,1);
        computeDeformationGradientFromTotalDisplacement(gdisplacement,
                                                        pset, px,
                                                        deformationGradient_new,
                                                        deformationGradient,
                                                        dx, psize,interpolator);
     }
     for(ParticleSubset::iterator iter = pset->begin();
                                  iter != pset->end(); iter++){
        particleIndex idx = *iter;
        // get the volumetric part of the deformation
        double J = deformationGradient_new[idx].Determinant();
        double Jold = deformationGradient[idx].Determinant();
        double Jinc = J/Jold;
        // carry forward fiber direction
        pfiberdir_carry[idx] = pfiberdir[idx];
        deformed_fiber_vector =pfiberdir[idx]; // not actually deformed yet
        //_______________________UNCOUPLE DEVIATORIC AND DILATIONAL PARTS
        //_______________________Ftilde=J^(-1/3)*F, Fvol=J^1/3*Identity
        //_______________________right Cauchy Green (C) tilde and invariants
        rightCauchyGreentilde_new = deformationGradient_new[idx].Transpose()
                                * deformationGradient_new[idx]*pow(J,-(2./3.));
        I1tilde = rightCauchyGreentilde_new.Trace();
        I2tilde = .5*(I1tilde*I1tilde -
                (rightCauchyGreentilde_new*rightCauchyGreentilde_new).Trace());
        I4tilde = Dot(deformed_fiber_vector,
                   (rightCauchyGreentilde_new*deformed_fiber_vector));
        lambda_tilde = sqrt(I4tilde);
        // For diagnostics only
        double I4 = I4tilde*pow(J,(2./3.));
        stretch[idx] = sqrt(I4);
        deformed_fiber_vector = deformationGradient_new[idx]
                               *deformed_fiber_vector
                               *(1./lambda_tilde*pow(J,-(1./3.)));

        Matrix3 DY(deformed_fiber_vector,deformed_fiber_vector);

        //________________________________left Cauchy Green (B) tilde
        leftCauchyGreentilde_new = deformationGradient_new[idx]
                     * deformationGradient_new[idx].Transpose()*pow(J,-(2./3.));
        //________________________________hydrostatic pressure term
        double p = Bulk*log(J)/J;
        //________________________________strain energy derivatives
        if (lambda_tilde < 1.){
          dWdI4tilde = 0.;
        }
        else if (lambda_tilde < lambda_star) {
          dWdI4tilde = 0.5*c3*(exp(c4*(lambda_tilde-1.))-1.)
                           /lambda_tilde/lambda_tilde;
        }
        else {
          dWdI4tilde = 0.5*(c5+c6/lambda_tilde)/lambda_tilde;
        }
      //________________________________Failure and stress terms
      fail[idx] = 0.;
      if (failure == 1){
        double matrix_failed = 0.;
        double fiber_failed = 0.;
        //________________Mooney Rivlin deviatoric term +failure of matrix
        Matrix3 RCG;
        RCG = deformationGradient_new[idx].Transpose()
             *deformationGradient_new[idx];
        double e1,e2,e3;//eigenvalues of C=symm.+pos.def.->Dis<=0
        double Q,R,Dis;
        double pi = 3.1415926535897932384;
        double I1 = RCG.Trace();
        double I2 = .5*(I1*I1 -(RCG*RCG).Trace());
        double I3 = RCG.Determinant();
        Q = (1./9.)*(3.*I2-pow(I1,2));
        R = (1./54.)*(-9.*I1*I2+27.*I3+2.*pow(I1,3));
        Dis = pow(Q,3)+pow(R,2);
        if (Dis <= 1.e-5 && Dis >= 0.){
         if (R >= -1.e-5 && R<= 1.e-5)
            e1 = e2 = e3 = I1/3.;
         else {
              e1 = 2.*pow(R,1./3.)+I1/3.;
              e3 = -pow(R,1./3.)+I1/3.;
              if (e1 < e3) swap(e1,e3);
              e2=e3;
            }
        }
        else{
          double theta = acos(R/pow(-Q,3./2.));
          e1 = 2.*pow(-Q,1./2.)*cos(theta/3.)+I1/3.;
          e2 = 2.*pow(-Q,1./2.)*cos(theta/3.+2.*pi/3.)+I1/3.;
          e3 = 2.*pow(-Q,1./2.)*cos(theta/3.+4.*pi/3.)+I1/3.;
          if (e1 < e2) swap(e1,e2);
          if (e1 < e3) swap(e1,e3);
          if (e2 < e3) swap(e2,e3);
        }
        double max_shear_strain = (e1-e3)/2.;
        if (max_shear_strain > crit_shear || fail_old[idx] == 1.
                                          || fail_old[idx] == 3.){
          deviatoric_stress = Identity*0.;
          fail[idx] = 1.;
          matrix_failed = 1.;
        }
        else{
          deviatoric_stress = (leftCauchyGreentilde_new*(c1+c2*I1tilde)
               - leftCauchyGreentilde_new*leftCauchyGreentilde_new*c2
               - Identity*(1./3.)*(c1*I1tilde+2.*c2*I2tilde))*2./J;
        }
        //________________________________fiber stress term + failure of fibers
        if (stretch[idx] > crit_stretch || fail_old[idx] == 2. 
                                        || fail_old[idx] == 3.){
          fiber_stress = Identity*0.;
          fail[idx] = 2.;
          fiber_failed =1.;
        }
        else{
          fiber_stress = (DY*dWdI4tilde*I4tilde
                           - Identity*(1./3.)*dWdI4tilde*I4tilde)*2./J;
        }
        if ( (matrix_failed + fiber_failed) == 2.|| fail_old[idx] == 3.)
          fail[idx] = 3.;
        //________________________________hydrostatic pressure term
        if (fail[idx] == 1.0 ||fail[idx] == 3.0)
          pressure = Identity*0.;
        else {
            p = Bulk*log(J)/J; // p -= qVisco;
            if (p >= -1.e-5 && p <= 1.e-5)
              p = 0.;
            pressure = Identity*p;
        }
        //_______________________________Cauchy stress
        pstress[idx] = pressure + deviatoric_stress + fiber_stress;
      }
      else {
          deviatoric_stress = (leftCauchyGreentilde_new*(c1+c2*I1tilde)
               - leftCauchyGreentilde_new*leftCauchyGreentilde_new*c2
               - Identity*(1./3.)*(c1*I1tilde+2.*c2*I2tilde))*2./J;
          fiber_stress = (DY*dWdI4tilde*I4tilde
                          - Identity*(1./3.)*dWdI4tilde*I4tilde)*2./J;
          p = Bulk*log(J)/J; // p -= qVisco;
          if (p >= -1.e-5 && p <= 1.e-5)
            p = 0.;
          pressure = Identity*p;
          //Cauchy stress
          pstress[idx] = pressure + deviatoric_stress + fiber_stress;
      }
      pvolume_deformed[idx] = pvolumeold[idx]*Jinc;
     }  // end loop over particles
    }   // isn't rigid
    delete interpolator;
   }
}

void TransIsoHyperImplicit::addComputesAndRequires(Task* task,
                                                 const MPMMaterial* matl,
                                                 const PatchSet* ,
                                                 const bool /*recurse*/,
                                                 const bool SchedParent) const
//________________________________corresponds to the 1st ComputeStressTensor
{
  const MaterialSubset* matlset = matl->thisMaterial();
  bool reset = flag->d_doGridReset;
                                                                                
  addSharedCRForImplicit(task, matlset, reset, true,SchedParent);

  if(SchedParent){
    task->requires(Task::ParentOldDW, lb->pFiberDirLabel, matlset,Ghost::None);
    task->requires(Task::ParentOldDW, pFailureLabel,      matlset,Ghost::None);
  }else{
    task->requires(Task::OldDW, lb->pFiberDirLabel, matlset,Ghost::None);
    task->requires(Task::OldDW, pFailureLabel,      matlset,Ghost::None);
  }

  task->computes(lb->pFiberDirLabel_preReloc,           matlset);
  task->computes(pStretchLabel_preReloc,                matlset);
}

void TransIsoHyperImplicit::addComputesAndRequires(Task* task,
                                                 const MPMMaterial* matl,
                                                 const PatchSet*) const
//________________________________corresponds to the 2nd ComputeStressTensor
{
  const MaterialSubset* matlset = matl->thisMaterial();
  bool reset = flag->d_doGridReset;
                                                                                
  addSharedCRForImplicit(task, matlset, reset);

  task->requires(Task::OldDW, lb->pFiberDirLabel,       matlset,Ghost::None);
  task->requires(Task::OldDW, pFailureLabel,            matlset,Ghost::None);

  task->computes(lb->pFiberDirLabel_preReloc,           matlset);
  task->computes(pStretchLabel_preReloc,                matlset);
  task->computes(pFailureLabel_preReloc,                matlset);
}

// The "CM" versions use the pressure-volume relationship of the CNH model
double TransIsoHyperImplicit::computeRhoMicroCM(double pressure,
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
    rho_cur = rho_orig*(p_gauge/bulk + sqrt((p_gauge/bulk)*(p_gauge/bulk) +1));
  }
  return rho_cur;
}

void TransIsoHyperImplicit::computePressEOSCM(const double rho_cur,double& pressure, 
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
    pressure = A*pow(rho_cur/rho_orig,n);
    dp_drho  = (bulk/rho_orig)*pow(rho_cur/rho_orig,n-1);
    tmp      = dp_drho;         // speed of sound squared
  } else {                      // STANDARD EOS            
    double p_g = .5*bulk*(rho_cur/rho_orig - rho_orig/rho_cur);
    pressure   = p_ref + p_g;
    dp_drho    = .5*bulk*(rho_orig/(rho_cur*rho_cur) + 1./rho_orig);
    tmp        = bulk/rho_cur;  // speed of sound squared
  }
}

double TransIsoHyperImplicit::getCompressibility()
{
  return 1.0/d_initialData.Bulk;
}

namespace Uintah {
  
#if 0
  static MPI_Datatype makeMPI_CMData()
  {
    ASSERTEQ(sizeof(TransIsoHyperImplicit::StateData), sizeof(double)*0);
    MPI_Datatype mpitype;
    MPI_Type_vector(1, 0, 0, MPI_DOUBLE, &mpitype);
    MPI_Type_commit(&mpitype);
    return mpitype;
  }
  
  const TypeDescription* fun_getTypeDescription(TransIsoHyperImplicit::StateData*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
                                  "TransIsoHyperImplicit::StateData", true,
                                  &makeMPI_CMData);
    }
    return td;
  }
#endif
} // End namespace Uintah
