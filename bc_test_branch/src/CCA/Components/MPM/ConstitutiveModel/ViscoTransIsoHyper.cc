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

#include <CCA/Components/MPM/ConstitutiveModel/ViscoTransIsoHyper.h>
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

// _________________transversely isotropic hyperelastic material [Jeff Weiss's]

ViscoTransIsoHyper::ViscoTransIsoHyper(ProblemSpecP& ps,MPMFlags* Mflag)
  : ConstitutiveModel(Mflag)

{
  d_useModifiedEOS = false;

  //______________________material properties
  ps->require("bulk_modulus", d_initialData.Bulk);
  ps->require("c1", d_initialData.c1);//Mooney Rivlin constant 1
  ps->require("c2", d_initialData.c2);//Mooney Rivlin constant 2
  ps->require("c3", d_initialData.c3);//scales exponential stresses
  ps->require("c4", d_initialData.c4);//controls uncrimping of fibers
  ps->require("c5", d_initialData.c5);//straightened fibers modulus
  ps->require("fiber_stretch", d_initialData.lambda_star);//toe region limit
  ps->require("direction_of_symm", d_initialData.a0);//fiber direction(initial)
  ps->require("failure_option",d_initialData.failure);//failure flag True/False
  ps->require("max_fiber_strain",d_initialData.crit_stretch);//failure limit fibers
  ps->require("max_matrix_strain",d_initialData.crit_shear);//failure limit matrix
  ps->get("useModifiedEOS",d_useModifiedEOS);//no negative pressure for solids
  ps->require("y1", d_initialData.y1);//viscoelastic prop's
  ps->require("y2", d_initialData.y2);
  ps->require("y3", d_initialData.y3);
  ps->require("y4", d_initialData.y4);
  ps->require("y5", d_initialData.y5);
  ps->require("y6", d_initialData.y6);
  ps->require("t1", d_initialData.t1);//relaxation times
  ps->require("t2", d_initialData.t2);
  ps->require("t3", d_initialData.t3);
  ps->require("t4", d_initialData.t4);
  ps->require("t5", d_initialData.t5);
  ps->require("t6", d_initialData.t6);

  pStretchLabel = VarLabel::create("p.stretch",
     ParticleVariable<double>::getTypeDescription());
  pStretchLabel_preReloc = VarLabel::create("p.stretch+",
     ParticleVariable<double>::getTypeDescription());

  pFailureLabel = VarLabel::create("p.fail",
     ParticleVariable<double>::getTypeDescription());
  pFailureLabel_preReloc = VarLabel::create("p.fail+",
     ParticleVariable<double>::getTypeDescription());
     
  pElasticStressLabel = VarLabel::create("p.ElasticStress",
        ParticleVariable<Matrix3>::getTypeDescription());
  pElasticStressLabel_preReloc = VarLabel::create("p.ElasticStress+",
        ParticleVariable<Matrix3>::getTypeDescription());

  pHistory1Label = VarLabel::create("p.history1",
        ParticleVariable<Matrix3>::getTypeDescription());
  pHistory1Label_preReloc = VarLabel::create("p.history1+",
        ParticleVariable<Matrix3>::getTypeDescription());

  pHistory2Label = VarLabel::create("p.history2",
        ParticleVariable<Matrix3>::getTypeDescription());
  pHistory2Label_preReloc = VarLabel::create("p.history2+",
        ParticleVariable<Matrix3>::getTypeDescription());

  pHistory3Label = VarLabel::create("p.history3",
        ParticleVariable<Matrix3>::getTypeDescription());
  pHistory3Label_preReloc = VarLabel::create("p.history3+",
        ParticleVariable<Matrix3>::getTypeDescription());

  pHistory4Label = VarLabel::create("p.history4",
        ParticleVariable<Matrix3>::getTypeDescription());
  pHistory4Label_preReloc = VarLabel::create("p.history4+",
        ParticleVariable<Matrix3>::getTypeDescription());

  pHistory5Label = VarLabel::create("p.history5",
        ParticleVariable<Matrix3>::getTypeDescription());
  pHistory5Label_preReloc = VarLabel::create("p.history5+",
        ParticleVariable<Matrix3>::getTypeDescription());

  pHistory6Label = VarLabel::create("p.history6",
        ParticleVariable<Matrix3>::getTypeDescription());
  pHistory6Label_preReloc = VarLabel::create("p.history6+",
        ParticleVariable<Matrix3>::getTypeDescription());
}

ViscoTransIsoHyper::ViscoTransIsoHyper(const ViscoTransIsoHyper* cm)
  : ConstitutiveModel(cm)
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
  
  d_initialData.y1 = cm->d_initialData.y1;//visco parameters
  d_initialData.y2 = cm->d_initialData.y2;
  d_initialData.y3 = cm->d_initialData.y3;
  d_initialData.y4 = cm->d_initialData.y4;
  d_initialData.y5 = cm->d_initialData.y5;
  d_initialData.y6 = cm->d_initialData.y6;
  d_initialData.t1 = cm->d_initialData.t1;
  d_initialData.t2 = cm->d_initialData.t2;
  d_initialData.t3 = cm->d_initialData.t3;
  d_initialData.t4 = cm->d_initialData.t4;
  d_initialData.t5 = cm->d_initialData.t5;
  d_initialData.t6 = cm->d_initialData.t6;
}

ViscoTransIsoHyper::~ViscoTransIsoHyper()
  // _______________________DESTRUCTOR
{
  VarLabel::destroy(pStretchLabel);
  VarLabel::destroy(pStretchLabel_preReloc);
  VarLabel::destroy(pFailureLabel);
  VarLabel::destroy(pFailureLabel_preReloc);
  VarLabel::destroy(pElasticStressLabel);
  VarLabel::destroy(pElasticStressLabel_preReloc);//visco labels
  VarLabel::destroy(pHistory1Label);
  VarLabel::destroy(pHistory1Label_preReloc);
  VarLabel::destroy(pHistory2Label);
  VarLabel::destroy(pHistory2Label_preReloc);
  VarLabel::destroy(pHistory3Label);
  VarLabel::destroy(pHistory3Label_preReloc);
  VarLabel::destroy(pHistory4Label);
  VarLabel::destroy(pHistory4Label_preReloc);
  VarLabel::destroy(pHistory5Label);
  VarLabel::destroy(pHistory5Label_preReloc);
  VarLabel::destroy(pHistory6Label);
  VarLabel::destroy(pHistory6Label_preReloc);
}


void ViscoTransIsoHyper::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","visco_trans_iso_hyper");
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
  cm_ps->appendElement("y1", d_initialData.y1);
  cm_ps->appendElement("y2", d_initialData.y2);
  cm_ps->appendElement("y3", d_initialData.y3);
  cm_ps->appendElement("y4", d_initialData.y4);
  cm_ps->appendElement("y5", d_initialData.y5);
  cm_ps->appendElement("y6", d_initialData.y6);
  cm_ps->appendElement("t1", d_initialData.t1);
  cm_ps->appendElement("t2", d_initialData.t2);
  cm_ps->appendElement("t3", d_initialData.t3);
  cm_ps->appendElement("t4", d_initialData.t4);
  cm_ps->appendElement("t5", d_initialData.t5);
  cm_ps->appendElement("t6", d_initialData.t6);
}


ViscoTransIsoHyper* ViscoTransIsoHyper::clone()
{
  return scinew ViscoTransIsoHyper(*this);
}

void ViscoTransIsoHyper::initializeCMData(const Patch* patch,
                                     const MPMMaterial* matl,
                                     DataWarehouse* new_dw)
// _____________________STRESS FREE REFERENCE CONFIG
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  initSharedDataForExplicit(patch, matl, new_dw);

  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
  Matrix3 Identity, zero(0.);
  Identity.Identity();

  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);
  ParticleVariable<double> stretch,fail;
  ParticleVariable<Matrix3> ElasticStress;
  ParticleVariable<Matrix3> history1,history2,history3;
  ParticleVariable<Matrix3> history4,history5,history6;

  new_dw->allocateAndPut(stretch,pStretchLabel,   pset);
  new_dw->allocateAndPut(fail,   pFailureLabel,   pset);

  new_dw->allocateAndPut(ElasticStress,pElasticStressLabel,pset);
  new_dw->allocateAndPut(history1,     pHistory1Label,     pset);
  new_dw->allocateAndPut(history2,     pHistory2Label,     pset);
  new_dw->allocateAndPut(history3,     pHistory3Label,     pset);
  new_dw->allocateAndPut(history4,     pHistory4Label,     pset);
  new_dw->allocateAndPut(history5,     pHistory5Label,     pset);
  new_dw->allocateAndPut(history6,     pHistory6Label,     pset);

  ParticleSubset::iterator iter = pset->begin();
  for(;iter != pset->end(); iter++){
    fail[*iter] = 0.0 ;
    stretch[*iter] = 1.0;
    ElasticStress[*iter] = zero;// no pre-initial stress
    history1[*iter] = zero;// no initial 'relaxation'
    history2[*iter] = zero;
    history3[*iter] = zero;
    history4[*iter] = zero;
    history5[*iter] = zero;
    history6[*iter] = zero;
  }
  computeStableTimestep(patch, matl, new_dw);
}

void ViscoTransIsoHyper::addParticleState(std::vector<const VarLabel*>& from,
                                     std::vector<const VarLabel*>& to)
//____________________KEEPS TRACK OF THE PARTICLES AND THE RELATED VARIABLES
//____________________(EACH CM ADD ITS OWN STATE VARS)
//____________________AS PARTICLES MOVE FROM PATCH TO PATCH
{
  // Add the local particle state data for this constitutive model.
  from.push_back(lb->pFiberDirLabel);
  from.push_back(pStretchLabel);
  from.push_back(pFailureLabel);
  from.push_back(pElasticStressLabel);//visco_labels
  from.push_back(pHistory1Label);
  from.push_back(pHistory2Label);
  from.push_back(pHistory3Label);
  from.push_back(pHistory4Label);
  from.push_back(pHistory5Label);
  from.push_back(pHistory6Label);

  to.push_back(lb->pFiberDirLabel_preReloc);
  to.push_back(pStretchLabel_preReloc);
  to.push_back(pFailureLabel_preReloc);
  to.push_back(pElasticStressLabel_preReloc);
  to.push_back(pHistory1Label_preReloc);
  to.push_back(pHistory2Label_preReloc);
  to.push_back(pHistory3Label_preReloc);
  to.push_back(pHistory4Label_preReloc);
  to.push_back(pHistory5Label_preReloc);
  to.push_back(pHistory6Label_preReloc);
}

void ViscoTransIsoHyper::computeStableTimestep(const Patch* patch,
                                          const MPMMaterial* matl,
                                          DataWarehouse* new_dw)
//______TIME STEP DEPENDS ON:
//______CELL SPACING, VEL OF PARTICLE, MATERIAL WAVE SPEED @ EACH PARTICLE
//______REDUCTION OVER ALL dT'S FROM EVERY PATCH PERFORMED
//______(USE THE SMALLEST dT)
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

  // __Compute wave speed at each particle, store the maximum

  double Bulk = d_initialData.Bulk;
  double c1 = d_initialData.c1;

  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
    particleIndex idx = *iter;

    // this is valid only for F=Identity
    c_dil = sqrt((Bulk+2./3.*c1)*pvolume[idx]/pmass[idx]);

    WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                     Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                     Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
  }
  WaveSpeed = dx/WaveSpeed;
  double delT_new = WaveSpeed.minComponent();
  new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
}

Vector ViscoTransIsoHyper::getInitialFiberDir()
{
  return d_initialData.a0;
}

void ViscoTransIsoHyper::computeStressTensor(const PatchSubset* patches,
                                        const MPMMaterial* matl,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw)
//COMPUTES THE STRESS ON ALL THE PARTICLES IN A GIVEN PATCH FOR A GIVEN MATERIAL
//CALLED ONCE PER TIME STEP CONTAINS A COPY OF computeStableTimestep
{
  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);

    double J,p;
    Matrix3 Identity; Identity.Identity();
    Matrix3 rightCauchyGreentilde_new, leftCauchyGreentilde_new;
    Matrix3 pressure, deviatoric_stress, fiber_stress;
    double I1tilde,I2tilde,I4tilde,lambda_tilde;
    double dWdI4tilde,d2WdI4tilde2;
    double shear;
    Vector deformed_fiber_vector;
    double U,W,se=0.;
    double c_dil=0.0;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

    Vector dx = patch->dCell();

    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    constParticleVariable<Matrix3> deformationGradient_new;
    constParticleVariable<Matrix3> deformationGradient;
    constParticleVariable<double> pmass,fail_old,pvolume;
    ParticleVariable<double> stretch,fail;
    constParticleVariable<Vector> pvelocity, pfiberdir;
    ParticleVariable<Vector> pfiberdir_carry;
    
    ParticleVariable<Matrix3> pstress,ElasticStress;//visco
    ParticleVariable<Matrix3> history1,history2,history3;
    ParticleVariable<Matrix3> history4,history5,history6;
    ParticleVariable<double> pdTdt,p_q;
    constParticleVariable<Matrix3> history1_old,history2_old,history3_old;
    constParticleVariable<Matrix3> history4_old,history5_old,history6_old;
    constParticleVariable<Matrix3> ElasticStress_old;
    constParticleVariable<Matrix3> velGrad;

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));
    
    old_dw->get(pmass,               lb->pMassLabel,               pset);
    old_dw->get(pvelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(pfiberdir,           lb->pFiberDirLabel,           pset);
    old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);
    old_dw->get(fail_old,            pFailureLabel,                pset);
    old_dw->get(ElasticStress_old,   pElasticStressLabel,          pset);
    old_dw->get(history1_old,        pHistory1Label,               pset);
    old_dw->get(history2_old,        pHistory2Label,               pset);
    old_dw->get(history3_old,        pHistory3Label,               pset);
    old_dw->get(history4_old,        pHistory4Label,               pset);
    old_dw->get(history5_old,        pHistory5Label,               pset);
    old_dw->get(history6_old,        pHistory6Label,               pset);

    new_dw->allocateAndPut(pstress,          lb->pStressLabel_preReloc,   pset);
    new_dw->allocateAndPut(pfiberdir_carry,  lb->pFiberDirLabel_preReloc, pset);
    new_dw->allocateAndPut(stretch,         pStretchLabel_preReloc,       pset);
    new_dw->allocateAndPut(fail,            pFailureLabel_preReloc,       pset);
    new_dw->allocateAndPut(ElasticStress,   pElasticStressLabel_preReloc, pset);
    new_dw->allocateAndPut(history1,        pHistory1Label_preReloc,      pset);
    new_dw->allocateAndPut(history2,        pHistory2Label_preReloc,      pset);
    new_dw->allocateAndPut(history3,        pHistory3Label_preReloc,      pset);
    new_dw->allocateAndPut(history4,        pHistory4Label_preReloc,      pset);
    new_dw->allocateAndPut(history5,        pHistory5Label_preReloc,      pset);
    new_dw->allocateAndPut(history6,        pHistory6Label_preReloc,      pset);
    new_dw->allocateAndPut(pdTdt,           lb->pdTdtLabel_preReloc,      pset);
    new_dw->allocateAndPut(p_q,             lb->p_qLabel_preReloc,        pset);
    new_dw->get(pvolume,          lb->pVolumeLabel_preReloc,              pset);
    new_dw->get(deformationGradient_new,
                                  lb->pDeformationMeasureLabel_preReloc,  pset);
    new_dw->get(velGrad,          lb->pVelGradLabel_preReloc,             pset);

    //_____________________________________________material parameters
    double Bulk  = d_initialData.Bulk;
    double c1 = d_initialData.c1;
    double c2 = d_initialData.c2;
    double c3 = d_initialData.c3;
    double c4 = d_initialData.c4;
    double c5 = d_initialData.c5;
    double lambda_star = d_initialData.lambda_star;
    double c6 = c3*(exp(c4*(lambda_star-1.))-1.)-c5*lambda_star;//c6 = y-intercept
    double rho_orig = matl->getInitialDensity();
    double failure = d_initialData.failure;
    double crit_shear = d_initialData.crit_shear;
    double crit_stretch = d_initialData.crit_stretch;
    double y1 = d_initialData.y1;// visco
    double y2 = d_initialData.y2;
    double y3 = d_initialData.y3;
    double y4 = d_initialData.y4;
    double y5 = d_initialData.y5;
    double y6 = d_initialData.y6;
    double t1 = d_initialData.t1;
    double t2 = d_initialData.t2;
    double t3 = d_initialData.t3;
    double t4 = d_initialData.t4;
    double t5 = d_initialData.t5;
    double t6 = d_initialData.t6;

    for(ParticleSubset::iterator iter = pset->begin();
        iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Assign zero internal heating by default - modify if necessary.
      pdTdt[idx] = 0.0;

      // get the volumetric part of the deformation
      J = deformationGradient_new[idx].Determinant();

      // carry forward fiber direction
      pfiberdir_carry[idx] = pfiberdir[idx];
      deformed_fiber_vector = pfiberdir[idx]; // not actually deformed yet

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
        shear = 2.*c1+c2;
      }
      else if (lambda_tilde < lambda_star){
        dWdI4tilde = 0.5*c3*(exp(c4*(lambda_tilde-1.))-1.)
                    /lambda_tilde/lambda_tilde;
        d2WdI4tilde2 = 0.25*c3*(c4*exp(c4*(lambda_tilde-1.))
                      -1./lambda_tilde*(exp(c4*(lambda_tilde-1.))-1.))
                         /(lambda_tilde*lambda_tilde*lambda_tilde);

        shear = 2.*c1+c2+I4tilde*(4.*d2WdI4tilde2*lambda_tilde*lambda_tilde
                                      -2.*dWdI4tilde*lambda_tilde);
      }
      else{
        dWdI4tilde = 0.5*(c5+c6/lambda_tilde)/lambda_tilde;
        d2WdI4tilde2 = -0.25*c6
                         /(lambda_tilde*lambda_tilde*lambda_tilde*lambda_tilde);
        shear = 2.*c1+c2+I4tilde*(4.*d2WdI4tilde2*lambda_tilde*lambda_tilde
                                      -2.*dWdI4tilde*lambda_tilde);
      }

      // Compute deformed volume and local wave speed
      double rho_cur = rho_orig/J;
      c_dil = sqrt((Bulk+1./3.*shear)/rho_cur);

      //________________________________Failure and stress terms
      fail[idx] = 0.;
      if (failure == 1){
        double matrix_failed = 0.;
        double fiber_failed = 0.;
        //______________Mooney Rivlin deviatoric term +failure of matrix
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
        if (max_shear_strain > crit_shear || fail_old[idx]== 1.0 
                                          || fail_old[idx] == 3.0)
          {deviatoric_stress = Identity*0.;
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
        if ( (matrix_failed + fiber_failed) == 2. || fail_old[idx] == 3.)
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
        ElasticStress[idx] = pressure + deviatoric_stress + fiber_stress;
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
          ElasticStress[idx] = pressure + deviatoric_stress + fiber_stress;
      }
      //_______________________________Viscoelastic stress
      double fac1,fac2,fac3,fac4,fac5,fac6;
      double exp1=0.,exp2=0.,exp3=0.,exp4=0.,exp5=0.,exp6=0.;
      Matrix3 Zero(0.);

      if(t1>0.){    // If t1>0, enter this section, otherwise skip it
        exp1 = exp(-delT/t1);
        fac1 = (1. - exp1)*t1/delT;
        history1[idx] = history1_old[idx]*exp1+
                       (ElasticStress[idx]-ElasticStress_old[idx])*fac1;
        if (t2 > 0.){
         exp2 = exp(-delT/t2);
         fac2 = (1. - exp2)*t2/delT;
         history2[idx] = history2_old[idx]*exp2+
                       (ElasticStress[idx]-ElasticStress_old[idx])*fac2;
        }
        else{
         history2[idx]= Zero;
        }
        if (t3 > 0.){
         exp3 = exp(-delT/t3);
         fac3 = (1. - exp3)*t3/delT;
         history3[idx] = history3_old[idx]*exp3+
                      (ElasticStress[idx]-ElasticStress_old[idx])*fac3;
        }
        else{
         history3[idx]= Zero;
        }
        if (t4 > 0.){
         exp4 = exp(-delT/t4);
         fac4 = (1. - exp4)*t4/delT;
         history4[idx] = history4_old[idx]*exp4+
                       (ElasticStress[idx]-ElasticStress_old[idx])*fac4;
        }
        else{
         history4[idx]= Zero;
        }
        if (t5 > 0.){
         exp5 = exp(-delT/t5);
         fac5 = (1. - exp5)*t5/delT;
         history5[idx] = history5_old[idx]*exp5+
                       (ElasticStress[idx]-ElasticStress_old[idx])*fac5;
        }
        else{
         history5[idx]= Zero;
        }
        if (t6 > 0.){
         exp6 = exp(-delT/t6);
         fac6 = (1. - exp6)*t6/delT;
         history6[idx] = history6_old[idx]*exp6+
                       (ElasticStress[idx]-ElasticStress_old[idx])*fac6;
        }
        else{
         history6[idx]= Zero;
       }
     }
     else{
        history1[idx]= Zero;
        history2[idx]= Zero;
        history3[idx]= Zero;
        history4[idx]= Zero;
        history5[idx]= Zero;
        history6[idx]= Zero;
     }

      pstress[idx] = history1[idx]*y1+history2[idx]*y2+history3[idx]*y3
                   + history4[idx]*y4+history5[idx]*y5+history6[idx]*y6
                   + ElasticStress[idx];
      //________________________________end stress

      // Compute the strain energy for all the particles
      U = .5*log(J)*log(J)*Bulk;
      if (lambda_tilde < lambda_star){
        W = c1*(I1tilde-3.)+c2*(I2tilde-3.)+(exp(c4*(lambda_tilde-1.)-1.))*c3;
      }
      else{
        W = c1*(I1tilde-3.)+c2*(I2tilde-3.)+
                                         c5*lambda_tilde+c6*log(lambda_tilde);
      }

      double e = (U + W)*pvolume[idx]/J;
      se += e;

      Vector pvelocity_idx = pvelocity[idx];

      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity_idx.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pvelocity_idx.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pvelocity_idx.z()),WaveSpeed.z()));

      // Compute artificial viscosity term
      if (flag->d_artificial_viscosity) {
        double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;
        double c_bulk = sqrt(Bulk/rho_cur);
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
  }
}


void ViscoTransIsoHyper::carryForward(const PatchSubset* patches,
                                 const MPMMaterial* matl,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw)
  //_____________________________used with RigidMPM
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    // Carry forward the data common to all constitutive models 
    // when using RigidMPM.visco_one_cell_expl_parallel.ups
    // This method is defined in the ConstitutiveModel base class.
    carryForwardSharedData(pset, old_dw, new_dw, matl);

    // Carry forward the data local to this constitutive model 
    constParticleVariable<Vector> pfibdir;  
    ParticleVariable<Vector> pfibdir_new;
    ParticleVariable<double> pstretch;
    constParticleVariable<Vector> pfail_old;
    ParticleVariable<double> pfail;
    
    ParticleVariable<Matrix3> ElasticStress_new;//visco_label
    constParticleVariable<Matrix3> ElasticStress;
    ParticleVariable<Matrix3> history1,history2,history3;
    ParticleVariable<Matrix3> history4,history5,history6;
    constParticleVariable<Matrix3> history1_old,history2_old,history3_old;
    constParticleVariable<Matrix3> history4_old,history5_old,history6_old;

    old_dw->get(pfibdir,         lb->pFiberDirLabel,                   pset);
    old_dw->get(pfail_old,       pFailureLabel,                        pset);
    old_dw->get(ElasticStress,    pElasticStressLabel,                 pset);
    old_dw->get(history1_old,     pHistory1Label,                      pset);
    old_dw->get(history2_old,     pHistory2Label,                      pset);
    old_dw->get(history3_old,     pHistory3Label,                      pset);
    old_dw->get(history4_old,     pHistory4Label,                      pset);
    old_dw->get(history5_old,     pHistory5Label,                      pset);
    old_dw->get(history6_old,     pHistory6Label,                      pset);
    
    new_dw->allocateAndPut(ElasticStress_new,pElasticStressLabel_preReloc,pset);
    new_dw->allocateAndPut(history1,         pHistory1Label_preReloc,     pset);
    new_dw->allocateAndPut(history2,         pHistory2Label_preReloc,     pset);
    new_dw->allocateAndPut(history3,         pHistory3Label_preReloc,     pset);
    new_dw->allocateAndPut(history4,         pHistory4Label_preReloc,     pset);
    new_dw->allocateAndPut(history5,         pHistory5Label_preReloc,     pset);
    new_dw->allocateAndPut(history6,         pHistory6Label_preReloc,     pset);
    new_dw->allocateAndPut(pfibdir_new,      lb->pFiberDirLabel_preReloc, pset);
    new_dw->allocateAndPut(pstretch,         pStretchLabel_preReloc,      pset);
    new_dw->allocateAndPut(pfail,            pFailureLabel_preReloc,      pset);

    for(ParticleSubset::iterator iter = pset->begin();iter!=pset->end();iter++){
      particleIndex idx = *iter;
      pfibdir_new[idx] = pfibdir[idx];
      pstretch[idx] = 1.0;
      pfail[idx] = 0.0;
      
      ElasticStress_new[idx] = ElasticStress[idx];//visco_label
      history1[idx] = history1_old[idx];
      history2[idx] = history2_old[idx];
      history3[idx] = history3_old[idx];
      history4[idx] = history4_old[idx];
      history5[idx] = history5_old[idx];
      history6[idx] = history6_old[idx];

    }
    new_dw->put(delt_vartype(1.e10), lb->delTLabel, patch->getLevel());
    
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(0.),     lb->StrainEnergyLabel);
    }
  }
}

void ViscoTransIsoHyper::addInitialComputesAndRequires(Task* task,
                                                    const MPMMaterial* matl,
                                                    const PatchSet*) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(pStretchLabel,              matlset);
  task->computes(pFailureLabel,              matlset);
  task->computes(pElasticStressLabel,        matlset);//visco_label
  task->computes(pHistory1Label,             matlset);
  task->computes(pHistory2Label,             matlset);
  task->computes(pHistory3Label,             matlset);
  task->computes(pHistory4Label,             matlset);
  task->computes(pHistory5Label,             matlset);
  task->computes(pHistory6Label,             matlset);
}

void ViscoTransIsoHyper::addComputesAndRequires(Task* task,
                                           const MPMMaterial* matl,
                                           const PatchSet* patches) const
  //___________TELLS THE SCHEDULER WHAT DATA
  //___________NEEDS TO BE AVAILABLE AT THE TIME computeStressTensor IS CALLED
{
  // Add the computes and requires that are common to all explicit 
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.

  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForExplicit(task, matlset, patches);

  // Other constitutive model and input dependent computes and requires
  Ghost::GhostType  gnone = Ghost::None;

  task->requires(Task::OldDW, lb->pFiberDirLabel, matlset,gnone);
  task->requires(Task::OldDW, pFailureLabel,      matlset,gnone);

  task->requires(Task::OldDW, pElasticStressLabel,matlset,gnone);
  task->requires(Task::OldDW, pHistory1Label,     matlset,gnone);
  task->requires(Task::OldDW, pHistory2Label,     matlset,gnone);
  task->requires(Task::OldDW, pHistory3Label,     matlset,gnone);
  task->requires(Task::OldDW, pHistory4Label,     matlset,gnone);
  task->requires(Task::OldDW, pHistory5Label,     matlset,gnone);
  task->requires(Task::OldDW, pHistory6Label,     matlset,gnone);

  task->computes(lb->pFiberDirLabel_preReloc, matlset);
  task->computes(pStretchLabel_preReloc,      matlset);
  task->computes(pFailureLabel_preReloc,      matlset);
  task->computes(pElasticStressLabel_preReloc,          matlset);//visco_label
  task->computes(pHistory1Label_preReloc,               matlset);
  task->computes(pHistory2Label_preReloc,               matlset);
  task->computes(pHistory3Label_preReloc,               matlset);
  task->computes(pHistory4Label_preReloc,               matlset);
  task->computes(pHistory5Label_preReloc,               matlset);
  task->computes(pHistory6Label_preReloc,               matlset);
}

void ViscoTransIsoHyper::addComputesAndRequires(Task* ,
                                           const MPMMaterial* ,
                                           const PatchSet* ,
                                           const bool ) const
  //_________________________________________here this one's empty
{
}

// The "CM" versions use the pressure-volume relationship of the CNH model
double ViscoTransIsoHyper::computeRhoMicroCM(double pressure, 
                                        const double p_ref,
                                        const MPMMaterial* matl,
                                        double temperature,
                                        double rho_guess)
{
  double rho_orig = matl->getInitialDensity();
  double Bulk = d_initialData.Bulk;
  
  double p_gauge = pressure - p_ref;
  double rho_cur;
 
  if(d_useModifiedEOS && p_gauge < 0.0) {
    double A = p_ref;           // MODIFIED EOS
    double n = p_ref/Bulk;
    rho_cur = rho_orig*pow(pressure/A,n);
  } else {                      // STANDARD EOS
    rho_cur = rho_orig*(p_gauge/Bulk + sqrt((p_gauge/Bulk)*(p_gauge/Bulk) +1));
  }
  return rho_cur;
}

void ViscoTransIsoHyper::computePressEOSCM(const double rho_cur,double& pressure, 
                                      const double p_ref,
                                      double& dp_drho, double& tmp,
                                      const MPMMaterial* matl,
                                      double temperature)
{
  double Bulk = d_initialData.Bulk;
  double rho_orig = matl->getInitialDensity();

  if(d_useModifiedEOS && rho_cur < rho_orig){
    double A = p_ref;           // MODIFIED EOS
    double n = Bulk/p_ref;
    pressure = A*pow(rho_cur/rho_orig,n);
    dp_drho  = (Bulk/rho_orig)*pow(rho_cur/rho_orig,n-1);
    tmp      = dp_drho;         // speed of sound squared
  } else {                      // STANDARD EOS            
    double p_g = .5*Bulk*(rho_cur/rho_orig - rho_orig/rho_cur);
    pressure   = p_ref + p_g;
    dp_drho    = .5*Bulk*(rho_orig/(rho_cur*rho_cur) + 1./rho_orig);
    tmp        = Bulk/rho_cur;  // speed of sound squared
  }
}

double ViscoTransIsoHyper::getCompressibility()
{
  return 1.0/d_initialData.Bulk;
}


namespace Uintah {
  
#if 0
  static MPI_Datatype makeMPI_CMData()
  {
    ASSERTEQ(sizeof(ViscoTransIsoHyper::StateData), sizeof(double)*0);
    MPI_Datatype mpitype;
    MPI_Type_vector(1, 0, 0, MPI_DOUBLE, &mpitype);
    MPI_Type_commit(&mpitype);
    return mpitype;
  }
  
  const TypeDescription* fun_getTypeDescription(ViscoTransIsoHyper::StateData*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
                                  "ViscoTransIsoHyper::StateData", true, &makeMPI_CMData);
    }
    return td;
  }
#endif
} // End namespace Uintah

