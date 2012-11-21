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

#include <CCA/Components/MPM/ConstitutiveModel/ViscoTransIsoHyperImplicit.h>
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

using std::cerr;
using namespace Uintah;

// _________________this is a transversely isotropic hyperelastic material [JW]
//__________________see Material 18 in LSDYNA manual
//__________________with strain-based failure criteria
//__________________implicit MPM

ViscoTransIsoHyperImplicit::ViscoTransIsoHyperImplicit(ProblemSpecP& ps,
                                                       MPMFlags* Mflag) 
  : ConstitutiveModel(Mflag), ImplicitCM()
{  
  d_useModifiedEOS = false;
  d_StrainEnergy = "MooneyRivlin";
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
  ps->get("active",d_active);
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
  ps->get("StrainEnergy",d_StrainEnergy); // MooneyRivlin or VerondaWestmann

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

ViscoTransIsoHyperImplicit::ViscoTransIsoHyperImplicit(const ViscoTransIsoHyperImplicit* cm) : ConstitutiveModel(cm), ImplicitCM(cm)
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

ViscoTransIsoHyperImplicit::~ViscoTransIsoHyperImplicit()
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

void ViscoTransIsoHyperImplicit::outputProblemSpec(ProblemSpecP& ps,
                                                   bool output_cm_tag)
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
  cm_ps->appendElement("StrainEnergy",d_StrainEnergy); // MooneyRivlin or VerondaWestmann
  cm_ps->appendElement("active",d_active);
}

ViscoTransIsoHyperImplicit* ViscoTransIsoHyperImplicit::clone()
{
  return scinew ViscoTransIsoHyperImplicit(*this);
}

void ViscoTransIsoHyperImplicit::initializeCMData(const Patch* patch,
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
   ParticleVariable<Matrix3> ElasticStress;
   ParticleVariable<Matrix3> history1,history2,history3;
   ParticleVariable<Matrix3> history4,history5,history6;

   new_dw->allocateAndPut(deformationGradient,
                                             lb->pDeformationMeasureLabel,pset);
   new_dw->allocateAndPut(pstress,           lb->pStressLabel,            pset);
   new_dw->allocateAndPut(stretch,           pStretchLabel,               pset);
   new_dw->allocateAndPut(fail,              pFailureLabel,               pset);

   new_dw->allocateAndPut(ElasticStress,pElasticStressLabel,pset);
   new_dw->allocateAndPut(history1,     pHistory1Label,     pset);
   new_dw->allocateAndPut(history2,     pHistory2Label,     pset);
   new_dw->allocateAndPut(history3,     pHistory3Label,     pset);
   new_dw->allocateAndPut(history4,     pHistory4Label,     pset);
   new_dw->allocateAndPut(history5,     pHistory5Label,     pset);
   new_dw->allocateAndPut(history6,     pHistory6Label,     pset);
 
   for(ParticleSubset::iterator iter = pset->begin();
       iter != pset->end(); iter++){
       deformationGradient[*iter] = Identity;
       fail[*iter] = 0.0;
       pstress[*iter] = zero;
       stretch[*iter] = 1.0;
       ElasticStress[*iter] = zero;// no pre-initial stress
       history1[*iter] = zero;// no initial 'relaxation'
       history2[*iter] = zero;
       history3[*iter] = zero;
       history4[*iter] = zero;
       history5[*iter] = zero;
       history6[*iter] = zero;
  }
}
void ViscoTransIsoHyperImplicit::allocateCMDataAddRequires(Task* task,
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
  task->requires(Task::NewDW,pElasticStressLabel_preReloc,matlset, Ghost::None);
  task->requires(Task::NewDW,pHistory1Label_preReloc,     matlset, Ghost::None);
  task->requires(Task::NewDW,pHistory2Label_preReloc,     matlset, Ghost::None);
  task->requires(Task::NewDW,pHistory3Label_preReloc,     matlset, Ghost::None);
  task->requires(Task::NewDW,pHistory4Label_preReloc,     matlset, Ghost::None);
  task->requires(Task::NewDW,pHistory5Label_preReloc,     matlset, Ghost::None);
  task->requires(Task::NewDW,pHistory6Label_preReloc,     matlset, Ghost::None);
}

void ViscoTransIsoHyperImplicit::allocateCMDataAdd(DataWarehouse* new_dw,
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

void ViscoTransIsoHyperImplicit::addParticleState(
                                             std::vector<const VarLabel*>& from,
                                             std::vector<const VarLabel*>& to)
//____________________KEEPS TRACK OF THE PARTICLES AND THE RELATED VARIABLES
//____________________(EACH CM ADD ITS OWN STATE VARS)
//____________________AS PARTICLES MOVE FROM PATCH TO PATCH
{
   // Add the local particle state data for this constitutive model.
   from.push_back(lb->pFiberDirLabel);
   from.push_back(pStretchLabel);
   from.push_back(pFailureLabel);
   from.push_back(pElasticStressLabel);
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

void ViscoTransIsoHyperImplicit::computeStableTimestep(const Patch*,
                                                  const MPMMaterial*,
                                                  DataWarehouse*)
{
  // Not used for the implicit models.
}

Vector ViscoTransIsoHyperImplicit::getInitialFiberDir()
{
  return d_initialData.a0;
}

void
ViscoTransIsoHyperImplicit::computeStressTensorImplicit(const PatchSubset* patches,
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
    Matrix3 shear;
    Vector deformed_fiber_vector;

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<Vector> d_S(interpolator->size());

    Matrix3 Identity,Zero(0.);
    Identity.Identity();

    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    int dwi = matl->getDWIndex();

    ParticleSubset* pset;
    constParticleVariable<Point> px;
    constParticleVariable<Matrix3> psize;
    ParticleVariable<Matrix3> deformationGradient_new;
    constParticleVariable<Matrix3> deformationGradient;
    constParticleVariable<double> pmass,pvolumeold;
    ParticleVariable<double> pvolume_deformed;
    ParticleVariable<double> stretch;
    ParticleVariable<double> fail;
    constParticleVariable<double> fail_old;
    constParticleVariable<Vector> pvelocity;
    constParticleVariable<Vector> pfiberdir;

    ParticleVariable<Matrix3> pstress,ElasticStress;//visco
    constParticleVariable<Matrix3> ElasticStress_old;
    ParticleVariable<Matrix3> history1,history2,history3;
    ParticleVariable<Matrix3> history4,history5,history6;
    constParticleVariable<Matrix3> history1_old,history2_old,history3_old;
    constParticleVariable<Matrix3> history4_old,history5_old,history6_old;

    DataWarehouse* parent_old_dw =
      new_dw->getOtherDataWarehouse(Task::ParentOldDW);

    delt_vartype delT;
    parent_old_dw->get(delT, lb->delTLabel, getLevel(patches));

    pset = parent_old_dw->getParticleSubset(dwi, patch);
    parent_old_dw->get(px,                 lb->pXLabel,                  pset);
    parent_old_dw->get(pmass,              lb->pMassLabel,               pset);
    parent_old_dw->get(psize,              lb->pSizeLabel,               pset);
    parent_old_dw->get(pvolumeold,         lb->pVolumeLabel,             pset);
    parent_old_dw->get(deformationGradient,lb->pDeformationMeasureLabel, pset);
    parent_old_dw->get(pfiberdir,          lb->pFiberDirLabel,           pset);
    parent_old_dw->get(fail_old,           pFailureLabel,                pset);
    parent_old_dw->get(ElasticStress_old,   pElasticStressLabel,         pset);
    parent_old_dw->get(history1_old,        pHistory1Label,              pset);
    parent_old_dw->get(history2_old,        pHistory2Label,              pset);
    parent_old_dw->get(history3_old,        pHistory3Label,              pset);
    parent_old_dw->get(history4_old,        pHistory4Label,              pset);
    parent_old_dw->get(history5_old,        pHistory5Label,              pset);
    parent_old_dw->get(history6_old,        pHistory6Label,              pset);

    new_dw->allocateAndPut(pstress,         lb->pStressLabel_preReloc,   pset);
    new_dw->allocateAndPut(pvolume_deformed,lb->pVolumeDeformedLabel,    pset);
    new_dw->allocateTemporary(deformationGradient_new,pset);
    new_dw->allocateAndPut(stretch,         pStretchLabel_preReloc,      pset);
    new_dw->allocateAndPut(fail,            pFailureLabel_preReloc,      pset);
    new_dw->allocateAndPut(ElasticStress,   pElasticStressLabel_preReloc,pset);
    new_dw->allocateAndPut(history1,        pHistory1Label_preReloc,     pset);
    new_dw->allocateAndPut(history2,        pHistory2Label_preReloc,     pset);
    new_dw->allocateAndPut(history3,        pHistory3Label_preReloc,     pset);
    new_dw->allocateAndPut(history4,        pHistory4Label_preReloc,     pset);
    new_dw->allocateAndPut(history5,        pHistory5Label_preReloc,     pset);
    new_dw->allocateAndPut(history6,        pHistory6Label_preReloc,     pset);

   //_____________________________________________material parameters
    double Bulk  = d_initialData.Bulk;
    double c1 = d_initialData.c1;
    double c2 = d_initialData.c2;
    double c3 = d_initialData.c3;
    double c4 = d_initialData.c4;
    double c5 = d_initialData.c5;
    double lambda_star = d_initialData.lambda_star;
    double c6 = c3*(exp(c4*(lambda_star-1.))-1.)-c5*lambda_star;
    double third = -1./3.;
    double cc2 = 2.*third;
    double rho_orig = matl->getInitialDensity();

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
                                                        deformationGradient_new,
                                                        deformationGradient,
                                                        dx, psize,interpolator);
      }

      double time = d_sharedState->getElapsedTime();

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

        double dWdI1=0., dWdI2=0., d2WdI1dI1=0.;
        if(d_StrainEnergy=="MooneyRivlin"){
          dWdI1 = c1;
          dWdI2 = c2;
          d2WdI1dI1 = 0.;
        }
        else if(d_StrainEnergy=="VerondaWestmann"){
          dWdI1 = c1*c2*exp(c2*(I1tilde-3.));
          dWdI2 = -.5*c1*c2;
          d2WdI1dI1 = c1*c2*c2*exp(c2*(I1tilde-3.));
        }

        double d2WdI1dI2 = 0.;
        double d2WdI2dI2 = 0.;

        //________________________________strain energy derivatives
        if (lambda_tilde < 1.){
          dWdI4tilde = 0.;
          d2WdI4tilde2 = 0.;
          shear = 2.*dWdI1+dWdI2;
        }
        else if (lambda_tilde < lambda_star) {
          dWdI4tilde = 0.5*c3*(exp(c4*(lambda_tilde-1.))-1.)/(lambda_tilde*lambda_tilde);
          d2WdI4tilde2 = 0.25*c3*(c4*exp(c4*(lambda_tilde-1.))
                        -1./lambda_tilde*(exp(c4*(lambda_tilde-1.))-1.))
                          /(lambda_tilde*lambda_tilde*lambda_tilde);

          shear = 2.*dWdI1+dWdI2+I4tilde*
                         (4.*d2WdI4tilde2*lambda_tilde*lambda_tilde
                                        -2.*dWdI4tilde*lambda_tilde);
        }
        else {
          dWdI4tilde = 0.5*(c5+c6/lambda_tilde)/lambda_tilde;
          d2WdI4tilde2 = -0.25*c6
                         /(lambda_tilde*lambda_tilde*lambda_tilde*lambda_tilde);

          shear = 2.*dWdI1+dWdI2+I4tilde*
                         (4.*d2WdI4tilde2*lambda_tilde*lambda_tilde
                                        -2.*dWdI4tilde*lambda_tilde);
        }

        //_________________________________stiffness and stress vars.
        Matrix3 RB = leftCauchyGreentilde_new;
        Matrix3 RB2 = leftCauchyGreentilde_new*leftCauchyGreentilde_new;
        Matrix3 I = Identity;
        Matrix3 devsMR = (RB*(dWdI1+dWdI2*I1tilde)
                       + RB2*(-dWdI2)+I*(dWdI1*I1tilde+2*dWdI2*I2tilde))*(2./J)*(-2./3.);
        Matrix3 devsFC = (DY-I*(1./3.))*dWdI4tilde*I4tilde*(2./J)*(-2./3.);
        Matrix3 termMR = RB*(1./J)*dWdI2*I1tilde-RB2*(1./J)*dWdI2;
        
        deviatoric_stress = (leftCauchyGreentilde_new*(dWdI1+dWdI2*I1tilde)
             - leftCauchyGreentilde_new*leftCauchyGreentilde_new*dWdI2
             - Identity*(1./3.)*(dWdI1*I1tilde+2.*dWdI2*I2tilde))*2./J;
        fiber_stress = (DY*dWdI4tilde*I4tilde
                        - Identity*(1./3.)*dWdI4tilde*I4tilde)*2./J;
        double p = Bulk*log(J)/J; // p -= qVisco;
        double active_stress = d_active*(time+delT);
        if (p >= -1.e-5 && p <= 1.e-5)
          p = 0.;
        pressure = Identity*(p + active_stress);
        
        ElasticStress[idx] = pressure + deviatoric_stress + fiber_stress;

        // All of the following, through the calculation of dsave
        // comes from NIKE3D, material model 18
        double pd44 = dWdI4tilde;
        double xinv1=I1tilde;
        double xinv2=I2tilde;
        double xinv4=I4tilde;
        double bts1 = RB2(0,0);
        double bts2 = RB2(1,1);
        double bts3 = RB2(2,2);
        double bts4 = RB2(0,1);
        double bts5 = RB2(1,2);
        double bts6 = RB2(0,2);
        double bt1 = RB(0,0);
        double bt2 = RB(1,1);
        double bt3 = RB(2,2);
        double bt4 = RB(0,1);
        double bt5 = RB(1,2);
        double bt6 = RB(0,2);
        double pd1 = dWdI1;
        double pd2 = dWdI2;
        double pd4 = dWdI4tilde;
        double pd11 = d2WdI1dI1;
        double pd12 = d2WdI1dI2;
        double pd22 = d2WdI2dI2;
        double tangc = (1./J)*pd44*xinv4*xinv4;
        double trbts = bts1 + bts2 + bts3;
        double ch1 = pd2 + pd11 + xinv1 * ( 2.*pd12 + pd22 );
        double ch2 = -xinv1*pd22 - pd12;
        double ch3 = pd22;
        double ch4 = -pd2;
        double ch14 = ch1 + ch4;
        double cc4 = 4./J;
        double cc3 = cc4*third;

//   ... the cc1 term is of the form ...
//   ... 4/(3J) (dWtilde/dCtilde:Ctilde)[I - 1/3(1 X 1)] ...
        double cc1 = -third*cc3* ( pd1*xinv1 + 2.*pd2*xinv2 + pd4*xinv4 );

//   ... the cc5 term is of the form ...
//   ... (4/9J) (Ctilde : d2Wtilde/dCtilde dCtilde : Ctilde) (1 X 1) ...
        double cc5 = cc3*third*(   xinv1*( xinv1*ch1 + 6.*ch2 )
                       + trbts*( ch3*trbts + ch4 ) + pd44*xinv4*xinv4  );

        double caci1 = ch2*xinv1;
        double caci2 = xinv1*ch1 + 3.*ch2;
        double caci3 = ch3*trbts + ch4;

        double ax=deformed_fiber_vector.x();
        double ay=deformed_fiber_vector.y();
        double az=deformed_fiber_vector.z();

//   ... A:II ...
//   ... note addition here of contribution to ...
//   ... d2Wtilde / dCtilde dCtilde : Ctilde from I4 ...

        double aci1 = caci1 + bt1*caci2 + bts1*caci3 + tangc*ax*ax;
        double aci2 = caci1 + bt2*caci2 + bts2*caci3 + tangc*ay*ay;
        double aci3 = caci1 + bt3*caci2 + bts3*caci3 + tangc*az*az;
        double aci4 =         bt4*caci2 + bts4*caci3 + tangc*ax*ay;
        double aci5 =         bt5*caci2 + bts5*caci3 + tangc*ay*az;
        double aci6 =         bt6*caci2 + bts6*caci3 + tangc*ax*az;

//   ... deviatoric stress ...
        double trsig = third*ElasticStress[idx].Trace();
        double sigd1 = ElasticStress[idx](0,0) + trsig;
        double sigd2 = ElasticStress[idx](1,1) + trsig;
        double sigd3 = ElasticStress[idx](2,2) + trsig;
        double sigd4 = ElasticStress[idx](0,1);
        double sigd5 = ElasticStress[idx](1,2);
        double sigd6 = ElasticStress[idx](0,2);

//   ... cc2*(sig') + cc3*(A:II) ...
//
//   ... this term is of the form
//       -2/3*(dev(sig) X 1 + 1 X dev(sig))
//       -4/(3J)*(d2Wtilde/dCtilde dCtilde : 1
//       + 1 : d2Wtilde/dCtilde dCtilde) ...
        aci1 = cc2*sigd1 + cc3*aci1;
        aci2 = cc2*sigd2 + cc3*aci2;
        aci3 = cc2*sigd3 + cc3*aci3;
        aci4 = cc2*sigd4 + cc3*aci4;
        aci5 = cc2*sigd5 + cc3*aci5;
        aci6 = cc2*sigd6 + cc3*aci6;

//   ... material tangent stiffness ...
//
//   ... NOTE - the last two (and now three) lines of each
//       component of the spatial elasticity tensor computed
//       below are derived from the term
//       (4/J)d2Wtilde / dCtilde dCtilde ...
//
//  !!!!!!!!!!!!!!!!!!!!!!!!
        tangc = tangc*J;
        double dsave[6][6]; //stiffness matrix

        double K=Bulk;

        dsave[0][0] =   K*(1./J)-2*p - 2.*cc1 + 2.*aci1 + cc5
                    + cc4*( bt1*( bt1*ch14 + 2.*ch2 )
                    + ch3*bts1*bts1 + tangc*ax*ax*ax*ax  );

        dsave[0][1] =    K*(1./J) + cc1 +  aci1 + aci2  + cc5
                    + cc4*( ch1*bt1*bt2 + ch2*( bt1+bt2 )
                    + ch3*bts1*bts2 + ch4*bt4*bt4 + tangc*ax*ax*ay*ay  );
        dsave[0][2] =    K*(1./J) + cc1 + aci1 + aci3 + cc5
                    + cc4*( ch1*bt1*bt3 + ch2*( bt1+bt3 )
                    + ch3*bts1*bts3 + ch4*bt6*bt6 + tangc*ax*ax*az*az  );
        dsave[1][1] =   K*(1./J)-2*p - 2.*cc1 + 2.*aci2 + cc5
                    + cc4*( bt2*( bt2*ch14 + 2.*ch2 )
                    + ch3*bts2*bts2 + tangc*ay*ay*ay*ay  );
        dsave[1][2] =    K*(1./J) + cc1 + aci2 + aci3 + cc5
                    + cc4*( ch1*bt2*bt3 + ch2*( bt2+bt3 )
                    + ch3*bts2*bts3 + ch4*bt5*bt5 + tangc*ay*ay*az*az  );
        dsave[2][2] =   K*(1./J)-2*p - 2.*cc1 + 2.*aci3 + cc5
                    + cc4*( bt3*( bt3*ch14 + 2.*ch2 )
                    + ch3*bts3*bts3 + tangc*az*az*az*az  );

        dsave[0][3] = aci4 + cc4*( bt4*( ch2 + bt1*ch14 )
                    + ch3*bts1*bts4 + tangc*ax*ax*ax*ay  );
        dsave[0][4] = aci5 + cc4*( bt5*( ch2 + ch1*bt1 )
                    + ch3*bts1*bts5 + ch4*bt4*bt6 + tangc*ax*ax*ay*az  );
        dsave[0][5] = aci6 + cc4*( bt6*( ch2 + bt1*ch14 )
                    + ch3*bts1*bts6 + tangc*ax*ax*ax*az  );
        dsave[1][3] = aci4 + cc4*( bt4*( ch2 + bt2*ch14 )
                    + ch3*bts2*bts4 + tangc*ay*ay*ax*ay  );
        dsave[1][4] = aci5 + cc4*( bt5*( ch2 + bt2*ch14 )
                    + ch3*bts2*bts5 + tangc*ay*ay*ay*az  );
        dsave[1][5] = aci6 + cc4*( bt6*( ch2 + bt2*ch1 )
                    + ch3*bts2*bts6 + ch4*bt4*bt5 + tangc*ay*ay*ax*az  );
        dsave[2][3] = aci4 + cc4*( bt4*( ch2 + bt3*ch1 )
                    + ch3*bts3*bts4 + ch4*bt5*bt6 + tangc*az*az*ax*ay  );
        dsave[2][4] = aci5 + cc4*( bt5*( ch2 + bt3*ch14 )
                    + ch3*bts3*bts5 + tangc*az*az*ay*az  );
        dsave[2][5] = aci6 + cc4*( bt6*( ch2 + bt3*ch14 )
                    + ch3*bts3*bts6 + tangc*az*az*ax*az  );

        cc1  = -1.5*cc1;
        ch4  = 0.5*ch4;
        ch14 = ch1 + ch4;

        dsave[3][3] = -p + cc1 + cc4*( bt4*bt4*ch14 +
                     ch3*bts4*bts4 + ch4*bt1*bt2 + tangc*ax*ay*ax*ay  );
        dsave[3][4] =  cc4*( bt4*bt5*ch14 +
                     ch3*bts4*bts5 + ch4*bt6*bt2 + tangc*ax*ay*ay*az  );
        dsave[3][5] =  cc4*( bt4*bt6*ch14 +
                     ch3*bts4*bts6 + ch4*bt1*bt5 + tangc*ax*ay*ax*az  );
        dsave[4][4] = -p + cc1 + cc4*( bt5*bt5*ch14 +
                     ch3*bts5*bts5 + ch4*bt3*bt2 + tangc*ay*az*ay*az  );
        dsave[4][5] =  cc4*( bt5*bt6*ch14 +
                     ch3*bts5*bts6 + ch4*bt4*bt3 + tangc*ay*az*ax*az  );
        dsave[5][5] = -p + cc1 + cc4*( bt6*bt6*ch14 +
                     ch3*bts6*bts6 + ch4*bt3*bt1 + tangc*ax*az*ax*az  );


        //_________________________________viscoelastic terms
        double fac1=0.,fac2=0.,fac3=0.,fac4=0.,fac5=0.,fac6=0.,fac=1.;
        double exp1=0.,exp2=0.,exp3=0.,exp4=0.,exp5=0.,exp6=0.;
        if (t1 > 0.){
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

        fac = fac1*y1 + fac2*y2 + fac3*y3 + fac4*y4 + fac5*y5 + fac6*y6 + 1.;

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

        double kmat[24][24];
        BtDB(B,dsave,kmat);

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
ViscoTransIsoHyperImplicit::computeStressTensorImplicit(const PatchSubset* patches,
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

     ParticleVariable<Matrix3> ElasticStress;//visco
     constParticleVariable<Matrix3> ElasticStress_old;
     ParticleVariable<Matrix3> history1,history2,history3;
     ParticleVariable<Matrix3> history4,history5,history6;
     constParticleVariable<Matrix3> history1_old,history2_old,history3_old;
     constParticleVariable<Matrix3> history4_old,history5_old,history6_old;

     delt_vartype delT;
     old_dw->get(delT,lb->delTLabel, getLevel(patches));

     old_dw->get(px,                  lb->pXLabel,                  pset);
     old_dw->get(psize,               lb->pSizeLabel,               pset);
     old_dw->get(pvolumeold,          lb->pVolumeLabel,             pset);
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

     new_dw->allocateAndPut(pstress,         lb->pStressLabel_preReloc,   pset);
     new_dw->allocateAndPut(pvolume_deformed,lb->pVolumeDeformedLabel,    pset);
     new_dw->allocateAndPut(deformationGradient_new,
                                 lb->pDeformationMeasureLabel_preReloc,   pset);
     new_dw->allocateAndPut(pfiberdir_carry, lb->pFiberDirLabel_preReloc, pset);
     new_dw->allocateAndPut(stretch,         pStretchLabel_preReloc,      pset);
     new_dw->allocateAndPut(fail,            pFailureLabel_preReloc,      pset);

     new_dw->allocateAndPut(ElasticStress,  pElasticStressLabel_preReloc, pset);
     new_dw->allocateAndPut(history1,       pHistory1Label_preReloc,      pset);
     new_dw->allocateAndPut(history2,       pHistory2Label_preReloc,      pset);
     new_dw->allocateAndPut(history3,       pHistory3Label_preReloc,      pset);
     new_dw->allocateAndPut(history4,       pHistory4Label_preReloc,      pset);
     new_dw->allocateAndPut(history5,       pHistory5Label_preReloc,      pset);
     new_dw->allocateAndPut(history6,       pHistory6Label_preReloc,      pset);

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
     double time = d_sharedState->getElapsedTime();
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

        double dWdI1=0.,dWdI2=0.;
        if(d_StrainEnergy=="MooneyRivlin"){
          dWdI1 = c1;
          dWdI2 = c2;
        }
        else if(d_StrainEnergy=="VerondaWestmann"){
          dWdI1 = c1*c2*exp(c2*(I1tilde-3.));
          dWdI2 = -.5*c1*c2;
        }

        //________________________________left Cauchy Green (B) tilde
        leftCauchyGreentilde_new = deformationGradient_new[idx]
                     * deformationGradient_new[idx].Transpose()*pow(J,-(2./3.));
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
        deviatoric_stress = (leftCauchyGreentilde_new*(dWdI1+dWdI2*I1tilde)
             - leftCauchyGreentilde_new*leftCauchyGreentilde_new*dWdI2
             - Identity*(1./3.)*(dWdI1*I1tilde+2.*dWdI2*I2tilde))*2./J;
        fiber_stress = (DY*dWdI4tilde*I4tilde
                        - Identity*(1./3.)*dWdI4tilde*I4tilde)*2./J;
        double p = Bulk*log(J)/J; // p -= qVisco;
        if (p >= -1.e-5 && p <= 1.e-5)
          p = 0.;
        double active_stress = d_active*(time+delT);
        pressure = Identity*(p + active_stress);
        //Cauchy stress
        ElasticStress[idx] = pressure + deviatoric_stress + fiber_stress;

        //_________________________________viscoelastic terms
        double fac1=0.,fac2=0.,fac3=0.,fac4=0.,fac5=0.,fac6=0.;
        double exp1=0.,exp2=0.,exp3=0.,exp4=0.,exp5=0.,exp6=0.;
        if (t1 > 0.){  // if t1 is zero, assume t2-t5 are zero also.
          Matrix3 Zero(0.);
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
         Matrix3 Zero(0.);
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

        pvolume_deformed[idx] = pvolumeold[idx]*Jinc;
      }  // end loop over particles
    }   // isn't rigid
    delete interpolator;
   }
}

void ViscoTransIsoHyperImplicit::addComputesAndRequires(Task* task,
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
    task->requires(Task::ParentOldDW, pElasticStressLabel,matlset,Ghost::None);
    task->requires(Task::ParentOldDW, pHistory1Label,     matlset,Ghost::None);
    task->requires(Task::ParentOldDW, pHistory2Label,     matlset,Ghost::None);
    task->requires(Task::ParentOldDW, pHistory3Label,     matlset,Ghost::None);
    task->requires(Task::ParentOldDW, pHistory4Label,     matlset,Ghost::None);
    task->requires(Task::ParentOldDW, pHistory5Label,     matlset,Ghost::None);
    task->requires(Task::ParentOldDW, pHistory6Label,     matlset,Ghost::None);
  }else{
    task->requires(Task::OldDW, lb->pFiberDirLabel, matlset,Ghost::None);
    task->requires(Task::OldDW, pFailureLabel,      matlset,Ghost::None);
    task->requires(Task::OldDW, pElasticStressLabel,matlset,Ghost::None);
    task->requires(Task::OldDW, pHistory1Label,     matlset,Ghost::None);
    task->requires(Task::OldDW, pHistory2Label,     matlset,Ghost::None);
    task->requires(Task::OldDW, pHistory3Label,     matlset,Ghost::None);
    task->requires(Task::OldDW, pHistory4Label,     matlset,Ghost::None);
    task->requires(Task::OldDW, pHistory5Label,     matlset,Ghost::None);
    task->requires(Task::OldDW, pHistory6Label,     matlset,Ghost::None);
  }
  //
  task->computes(lb->pFiberDirLabel_preReloc,matlset);
  task->computes(pStretchLabel_preReloc,     matlset);
}

void ViscoTransIsoHyperImplicit::addComputesAndRequires(Task* task,
                                                 const MPMMaterial* matl,
                                                 const PatchSet*) const
//________________________________corresponds to the 2nd ComputeStressTensor
{
  const MaterialSubset* matlset = matl->thisMaterial();
  bool reset = flag->d_doGridReset;
                                                                                
  addSharedCRForImplicit(task, matlset, reset);

  task->requires(Task::OldDW, lb->pFiberDirLabel, matlset,Ghost::None);
  task->requires(Task::OldDW, pFailureLabel,      matlset,Ghost::None);
  task->requires(Task::OldDW, pElasticStressLabel,matlset,Ghost::None);//visco
  task->requires(Task::OldDW, pHistory1Label,     matlset,Ghost::None);
  task->requires(Task::OldDW, pHistory2Label,     matlset,Ghost::None);
  task->requires(Task::OldDW, pHistory3Label,     matlset,Ghost::None);
  task->requires(Task::OldDW, pHistory4Label,     matlset,Ghost::None);
  task->requires(Task::OldDW, pHistory5Label,     matlset,Ghost::None);
  task->requires(Task::OldDW, pHistory6Label,     matlset,Ghost::None);
  //
  task->computes(lb->pFiberDirLabel_preReloc,           matlset);
  task->computes(pStretchLabel_preReloc,                matlset);
  task->computes(pFailureLabel_preReloc,                matlset);
  
  task->computes(pElasticStressLabel_preReloc,          matlset);//visco
  task->computes(pHistory1Label_preReloc,               matlset);
  task->computes(pHistory2Label_preReloc,               matlset);
  task->computes(pHistory3Label_preReloc,               matlset);
  task->computes(pHistory4Label_preReloc,               matlset);
  task->computes(pHistory5Label_preReloc,               matlset);
  task->computes(pHistory6Label_preReloc,               matlset);
}

// The "CM" versions use the pressure-volume relationship of the CNH model
double ViscoTransIsoHyperImplicit::computeRhoMicroCM(double pressure,
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

void ViscoTransIsoHyperImplicit::computePressEOSCM(const double rho_cur,double& pressure, 
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

double ViscoTransIsoHyperImplicit::getCompressibility()
{
  return 1.0/d_initialData.Bulk;
}


namespace Uintah {
  
#if 0
  static MPI_Datatype makeMPI_CMData()
  {
    ASSERTEQ(sizeof(ViscoTransIsoHyperImplicit::StateData), sizeof(double)*0);
    MPI_Datatype mpitype;
    MPI_Type_vector(1, 0, 0, MPI_DOUBLE, &mpitype);
    MPI_Type_commit(&mpitype);
    return mpitype;
  }
  
  const TypeDescription* fun_getTypeDescription(ViscoTransIsoHyperImplicit::StateData*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
                                  "ViscoTransIsoHyperImplicit::StateData", true,
                                  &makeMPI_CMData);
    }
    return td;
  }
#endif
} // End namespace Uintah
