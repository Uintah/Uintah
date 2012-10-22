/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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
//#include </usr/include/valgrind/callgrind.h>
#include <CCA/Components/MPM/ConstitutiveModel/simplifiedGeoModel.h>
#include <Core/Grid/Patch.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Level.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/Matrix3.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Malloc/Allocator.h>

#include <sci_values.h>
#include <iostream>

using std::cerr;

using namespace Uintah;
using namespace std;

simplifiedGeoModel::simplifiedGeoModel(ProblemSpecP& ps, MPMFlags* Mflag)
  : ConstitutiveModel(Mflag)
{

  ps->require("FSLOPE",d_cm.FSLOPE);
  ps->require("FSLOPE_p",d_cm.FSLOPE_p);
  ps->require("hardening_modulus",d_cm.hardening_modulus);
  ps->require("CR",d_cm.CR);
  ps->require("p0_crush_curve",d_cm.p0_crush_curve);
  ps->require("p1_crush_curve",d_cm.p1_crush_curve);
  ps->require("p3_crush_curve",d_cm.p3_crush_curve);
  ps->require("p4_fluid_effect",d_cm.p4_fluid_effect);
  ps->require("fluid_B0",d_cm.fluid_B0);
  ps->require("fluid_pressur_initial",d_cm.fluid_pressur_initial);
  ps->require("kinematic_hardening_constant",d_cm.kinematic_hardening_constant);
  ps->require("PEAKI1",d_cm.PEAKI1);
  ps->require("B0",d_cm.B0);
  ps->require("G0",d_cm.G0);
  initializeLocalMPMLabels();
}

simplifiedGeoModel::simplifiedGeoModel(const simplifiedGeoModel* cm)
  : ConstitutiveModel(cm)
{
  d_cm.FSLOPE = cm->d_cm.FSLOPE;
  d_cm.FSLOPE_p = cm->d_cm.FSLOPE_p;
  d_cm.hardening_modulus = cm->d_cm.hardening_modulus;
  d_cm.CR = cm->d_cm.CR;
  d_cm.p0_crush_curve = cm->d_cm.p0_crush_curve;
  d_cm.p1_crush_curve = cm->d_cm.p1_crush_curve;
  d_cm.p3_crush_curve = cm->d_cm.p3_crush_curve;
  d_cm.p4_fluid_effect = cm->d_cm.p4_fluid_effect;
  d_cm.fluid_B0 = cm->d_cm.fluid_B0;
  d_cm.fluid_pressur_initial = cm->d_cm.fluid_pressur_initial;
  d_cm.kinematic_hardening_constant = cm->d_cm.kinematic_hardening_constant;
  d_cm.PEAKI1 = cm->d_cm.PEAKI1;
  d_cm.B0 = cm->d_cm.B0;
  d_cm.G0 = cm->d_cm.G0;
  initializeLocalMPMLabels();
}

simplifiedGeoModel::~simplifiedGeoModel()
{

  VarLabel::destroy(pPlasticStrainLabel);
  VarLabel::destroy(pPlasticStrainLabel_preReloc);
  VarLabel::destroy(pPlasticStrainVolLabel);
  VarLabel::destroy(pPlasticStrainVolLabel_preReloc);
  VarLabel::destroy(pElasticStrainVolLabel);
  VarLabel::destroy(pElasticStrainVolLabel_preReloc);
  VarLabel::destroy(pKappaLabel);
  VarLabel::destroy(pKappaLabel_preReloc);
  VarLabel::destroy(pBackStressLabel);
  VarLabel::destroy(pBackStressLabel_preReloc);
  VarLabel::destroy(pBackStressIsoLabel);
  VarLabel::destroy(pBackStressIsoLabel_preReloc);

}

void simplifiedGeoModel::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","simplified_geo_model");
  }

  cm_ps->appendElement("FSLOPE",d_cm.FSLOPE);
  cm_ps->appendElement("FSLOPE_p",d_cm.FSLOPE_p);
  cm_ps->appendElement("hardening_modulus",d_cm.hardening_modulus);
  cm_ps->appendElement("CR",d_cm.CR);
  cm_ps->appendElement("p0_crush_curve",d_cm.p0_crush_curve);
  cm_ps->appendElement("p1_crush_curve",d_cm.p1_crush_curve);
  cm_ps->appendElement("p3_crush_curve",d_cm.p3_crush_curve);
  cm_ps->appendElement("p4_fluid_effect",d_cm.p4_fluid_effect);
  cm_ps->appendElement("fluid_B0",d_cm.fluid_B0);
  cm_ps->appendElement("fluid_pressur_initial",d_cm.fluid_pressur_initial);
  cm_ps->appendElement("kinematic_hardening_constant",d_cm.kinematic_hardening_constant);
  cm_ps->appendElement("PEAKI1",d_cm.PEAKI1);
  cm_ps->appendElement("B0",d_cm.B0);
  cm_ps->appendElement("G0",d_cm.G0);

}

simplifiedGeoModel* simplifiedGeoModel::clone()
{
  return scinew simplifiedGeoModel(*this);
}

void simplifiedGeoModel::initializeCMData(const Patch* patch,
                                   const MPMMaterial* matl,
                                   DataWarehouse* new_dw)
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  initSharedDataForExplicit(patch, matl, new_dw);
  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(),patch);
  ParticleVariable<double> pPlasticStrain;
  ParticleVariable<double> pPlasticStrainVol;
  ParticleVariable<double> pElasticStrainVol;
  ParticleVariable<double> pKappa;
  ParticleVariable<Matrix3> pBackStress;
  ParticleVariable<Matrix3> pBackStressIso;
  new_dw->allocateAndPut(pPlasticStrain,     pPlasticStrainLabel, pset);
  new_dw->allocateAndPut(pPlasticStrainVol,     pPlasticStrainVolLabel, pset);
  new_dw->allocateAndPut(pElasticStrainVol,     pElasticStrainVolLabel, pset);
  new_dw->allocateAndPut(pKappa,     pKappaLabel, pset);
  new_dw->allocateAndPut(pBackStress,     pBackStressLabel, pset);
  new_dw->allocateAndPut(pBackStressIso,  pBackStressIsoLabel, pset);

  ParticleSubset::iterator iter = pset->begin();
  Matrix3 Identity;
  Identity.Identity();
  for(;iter != pset->end();iter++){
    pPlasticStrain[*iter] = 0.0;
    pPlasticStrainVol[*iter] = 0.0;
    pElasticStrainVol[*iter] = 0.0;
    pKappa[*iter] = (d_cm.p0_crush_curve +
      d_cm.CR*d_cm.FSLOPE*d_cm.PEAKI1 )/
      (d_cm.CR*d_cm.FSLOPE+1.0);
    pBackStress[*iter].set(0.0);
    pBackStressIso[*iter] = Identity*d_cm.fluid_pressur_initial;
  }
  computeStableTimestep(patch, matl, new_dw);
}

void
simplifiedGeoModel::allocateCMDataAddRequires(Task* task,
                                            const MPMMaterial* matl,
                                            const PatchSet* patches ,
                                            MPMLabel* lb) const
{
  const MaterialSubset* matlset = matl->thisMaterial();

  // Allocate the variables shared by all constitutive models
  // for the particle convert operation
  // This method is defined in the ConstitutiveModel base class.
  addSharedRForConvertExplicit(task, matlset, patches);

}

void simplifiedGeoModel::allocateCMDataAdd(DataWarehouse* new_dw,
                                         ParticleSubset* addset,
          map<const VarLabel*, ParticleVariableBase*>* newState,
                                         ParticleSubset* delset,
                                         DataWarehouse* )
{

}

void simplifiedGeoModel::computeStableTimestep(const Patch* patch,
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
  new_dw->get(pmass,     lb->pMassLabel,     pset);
  new_dw->get(pvolume,   lb->pVolumeLabel,   pset);
  new_dw->get(pvelocity, lb->pVelocityLabel, pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
  double bulk = d_cm.B0;
  double shear= d_cm.G0;
  for(ParticleSubset::iterator iter = pset->begin();
      iter != pset->end(); iter++){
     particleIndex idx = *iter;
     // Compute wave speed + particle velocity at each particle,
     // store the maximum
     c_dil = sqrt((bulk+4.0*shear/3.0)*pvolume[idx]/pmass[idx]);
     WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                      Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                      Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
    }
    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    if(delT_new < 1.e-12)
      new_dw->put(delt_vartype(DBL_MAX), lb->delTLabel, patch->getLevel());
    else
      new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
}

#if 0
void simplifiedGeoModel::computeStressTensor(const PatchSubset* patches,
                                           const MPMMaterial* matl,
                                           DataWarehouse* old_dw,
                                           DataWarehouse* new_dw)
{

  // Define some constants
  double one_sixth = 1.0/(6.0);
  double one_third = 1.0/(3.0);
  double two_third = 2.0/(3.0);
  double four_third = 4.0/(3.0);

  double sqrt_three = sqrt(3.0);
  double one_sqrt_three = 1.0/sqrt_three;

  for(int p=0;p<patches->size();p++){

    const Patch* patch = patches->get(p);
    Matrix3 Identity,D;
    double J;
    Identity.Identity();
    double c_dil = 0.0,se=0.0;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<Vector> d_S(interpolator->size());
    vector<double> S(interpolator->size());

    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};
    int dwi = matl->getDWIndex();

    // Create array for the particle position
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    ParticleVariable<Matrix3> defGrad_new;
    constParticleVariable<Matrix3> defGrad;
    constParticleVariable<Matrix3> stress_old;
    ParticleVariable<Matrix3> stress_new;
    constParticleVariable<Point> px;
    constParticleVariable<double> pmass;
    ParticleVariable<double> pvolume,p_q;
    constParticleVariable<Vector> pvelocity,psize;
    ParticleVariable<double> pdTdt;
    constParticleVariable<double> pPlasticStrain;
    ParticleVariable<double>  pPlasticStrain_new;
    constParticleVariable<double> pPlasticStrainVol;
    ParticleVariable<double>  pPlasticStrainVol_new;
    constParticleVariable<double> pElasticStrainVol;
    ParticleVariable<double>  pElasticStrainVol_new;
    constParticleVariable<double> pKappa;
    ParticleVariable<double>  pKappa_new;
    constParticleVariable<Matrix3> pBackStress;
    ParticleVariable<Matrix3>  pBackStress_new;
    constParticleVariable<Matrix3> pBackStressIso;
    ParticleVariable<Matrix3>  pBackStressIso_new;
    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));
    old_dw->get(pPlasticStrain, pPlasticStrainLabel, pset);
    old_dw->get(pPlasticStrainVol, pPlasticStrainVolLabel, pset);
    old_dw->get(pElasticStrainVol, pElasticStrainVolLabel, pset);
    old_dw->get(pKappa, pKappaLabel, pset);
    old_dw->get(pBackStress, pBackStressLabel, pset);
    old_dw->get(pBackStressIso, pBackStressIsoLabel, pset);
    new_dw->allocateAndPut(pPlasticStrain_new,pPlasticStrainLabel_preReloc,pset);
    new_dw->allocateAndPut(pPlasticStrainVol_new,pPlasticStrainVolLabel_preReloc,pset);
    new_dw->allocateAndPut(pElasticStrainVol_new,pElasticStrainVolLabel_preReloc,pset);
    new_dw->allocateAndPut(pKappa_new,pKappaLabel_preReloc,pset);
    new_dw->allocateAndPut(pBackStress_new,pBackStressLabel_preReloc,pset);
    new_dw->allocateAndPut(pBackStressIso_new,pBackStressIsoLabel_preReloc,pset);
    Ghost::GhostType  gac   = Ghost::AroundCells;
    old_dw->get(px,                  lb->pXLabel,                        pset);
    old_dw->get(pmass,               lb->pMassLabel,                     pset);
    old_dw->get(psize,               lb->pSizeLabel,                     pset);
    old_dw->get(pvelocity,           lb->pVelocityLabel,                 pset);
    old_dw->get(defGrad, lb->pDeformationMeasureLabel,       pset);
    old_dw->get(stress_old,             lb->pStressLabel,                pset);
    new_dw->allocateAndPut(stress_new,  lb->pStressLabel_preReloc,       pset);
    new_dw->allocateAndPut(pvolume,  lb->pVolumeLabel_preReloc,          pset);
    new_dw->allocateAndPut(pdTdt,    lb->pdTdtLabel_preReloc,            pset);
    new_dw->allocateAndPut(defGrad_new,
                                  lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->allocateAndPut(p_q,      lb->p_qLabel_preReloc,              pset);

    /*
    ParticleVariable<Matrix3> velGrad,rotation,trial_stress;
    ParticleVariable<double> f_trial,rho_cur;
    new_dw->allocateTemporary(velGrad,      pset);
    new_dw->allocateTemporary(rotation,     pset);
    new_dw->allocateTemporary(trial_stress, pset);
    new_dw->allocateTemporary(f_trial, pset);
    new_dw->allocateTemporary(rho_cur,pset);
    */

    Matrix3 velGrad,rotation,trial_stress;
    double f_trial,rho_cur;

    const double FSLOPE = d_cm.FSLOPE;
    const double FSLOPE_p = d_cm.FSLOPE_p;
    const double hardening_modulus = d_cm.hardening_modulus;
    const double CR = d_cm.CR;
    const double p0_crush_curve = d_cm.p0_crush_curve;
    const double p1_crush_curve = d_cm.p1_crush_curve;
    const double PEAKI1 = d_cm.PEAKI1;
    const double p3_crush_curve = d_cm.p3_crush_curve;
    const double p4_fluid_effect = d_cm.p4_fluid_effect;
    const double fluid_B0 = d_cm.fluid_B0;
    const double kinematic_hardening_constant = d_cm.kinematic_hardening_constant;
    double bulk = d_cm.B0;
    const double shear= d_cm.G0;

    // create node data for the plastic multiplier field
    double rho_orig = matl->getInitialDensity();
    Matrix3 L_new(0.0);

    // Get the deformation gradients first.  This is done differently
    // depending on whether or not the grid is reset.  (Should it be??? -JG)
    constNCVariable<Vector> gvelocity;
    new_dw->get(gvelocity, lb->gVelocityStarLabel,dwi,patch,gac,NGN);
    for(ParticleSubset::iterator iter=pset->begin();iter!=pset->end();iter++){
      particleIndex idx = *iter;

      //re-zero the velocity gradient:
      L_new.set(0.0);
      if(!flag->d_axisymmetric){
        // Get the node indices that surround the cell
        interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],
        defGrad[idx]);

        computeVelocityGradient(L_new,ni,d_S, oodx, gvelocity);
      } else {  // axi-symmetric kinematics
        // Get the node indices that surround the cell
        interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,
                                  psize[idx],defGrad[idx]);
        // x -> r, y -> z, z -> theta
        computeAxiSymVelocityGradient(L_new,ni,d_S,S,oodx,gvelocity,px[idx]);
      }

      // Update vel grad, def grad, J
      velGrad=L_new;
      defGrad_new[idx]=(L_new*delT+Identity)*defGrad[idx];
      J = defGrad_new[idx].Determinant();
      if (J <= 0){
        cout<< "ERROR, negative J! in particle "<< idx << endl;
        cout<<"J= "<<J<<endl;
        cout<<"L= "<<L_new<<endl;
        exit(1);
      }

      // Update particle volumes
      pvolume[idx]=(pmass[idx]/rho_orig)*J;
      rho_cur = rho_orig/J;

      // Initialize dT/dt to zero
      pdTdt[idx] = 0.0;

      // Compute incremental bulk modulus based on EOS (see EOS functions at the end)
      double Gamma = 0.11;  // HARDCODED for soil
      double c0 = sqrt(bulk/rho_orig);
      double s = 1.5;       // HARDCODED for soil

      double eta = rho_cur/rho_orig -1.0;
      double dp_drho = c0*c0*(1 - eta*(Gamma - 1-s))/pow((1 + eta*(1-s)), 3);

      bulk = dp_drho*rho_cur;
      if (idx == 3120) cerr << "bulk modulus = " << bulk << endl;

      // modify the bulk modulus based on the fluid effects
      double vol_strain = pPlasticStrainVol[idx]+pElasticStrainVol[idx];
      double bulk_temp = exp(p3_crush_curve+p4_fluid_effect+vol_strain);
      bulk = bulk + fluid_B0*
           ( exp(p3_crush_curve+p4_fluid_effect)-1.0 ) * bulk_temp
           / ( (bulk_temp-1.0)*(bulk_temp-1.0) );

      double lame = bulk - two_third*shear;
      double lame_inverse = one_sixth/(bulk*shear) * ( two_third*shear - bulk );

      double PEAKI1_hardening = PEAKI1*FSLOPE + hardening_modulus*pPlasticStrain[idx];
      //const double cap_radius=-CR*FSLOPE*(pKappa[idx]-PEAKI1);
      double cap_radius=-CR*(FSLOPE*pKappa[idx]-PEAKI1_hardening);

      // Compute the rate of deformation tensor
      Matrix3 D = (velGrad + velGrad.Transpose())*.5;
      Matrix3 tensorR, tensorU;
      defGrad_new[idx].polarDecompositionRMB(tensorU, tensorR);
      rotation=tensorR;
      D = (tensorR.Transpose())*(D*tensorR);

      // update the actual stress:
      Matrix3 unrotated_stress = (tensorR.Transpose())*(stress_old[idx]*tensorR);
      trial_stress = unrotated_stress + (Identity*lame*(D.Trace()*delT) + D*delT*2.0*shear);

      // compute shifted stress
      trial_stress -= pBackStress[idx];

      // compute the value of the yield function for the trial stress
      f_trial = YieldFunction(trial_stress,FSLOPE,pKappa[idx],cap_radius,PEAKI1_hardening);

      // initial assignment for the plastic strains and the position of the cap function
      pPlasticStrain_new[idx] = pPlasticStrain[idx];
      pPlasticStrainVol_new[idx] = pPlasticStrainVol[idx];
      pElasticStrainVol_new[idx] = pElasticStrainVol[idx] + D.Trace()*delT;
      pKappa_new[idx] = pKappa[idx];

      // compute stress invariants for the trial stress
      double I1_trial,J2_trial;
      Matrix3 S_trial;
      computeInvariants(trial_stress, S_trial, I1_trial, J2_trial);

      // check if the stress is elastic or plastic: If it is elastic the new stres is equal
      // to trial stress otherwise, the plasticity return algrithm would be used.
      Matrix3 deltaBackStress;
      Matrix3 deltaBackStressIso;
      if (f_trial < 0.0){
        stress_new[idx] = trial_stress;
        deltaBackStress.set(0.0);
        deltaBackStressIso.set(0.0);
      } else {
        // plasticity vertex treatment begins
        int condition_return_to_vertex=0;
        if (I1_trial > PEAKI1_hardening/FSLOPE) {
          if (J2_trial < 0.00000001){
            stress_new[idx] = Identity*PEAKI1_hardening/FSLOPE*one_third;
            condition_return_to_vertex = 1;
          } else {
            int counter_1_fix=0;
            int counter_2_fix=0;
            double P_component_1,P_component_2;
            double relative_stress_to_vertex_1,relative_stress_to_vertex_2;
            Matrix3 relative_stress_to_vertex,relative_stress_to_vertex_deviatoric;
            Matrix3 unit_tensor_vertex_1;
            Matrix3 unit_tensor_vertex_2;
            Matrix3 P,M,P_deviatoric;

            // compute the relative trial stress in respect with the vertex
            relative_stress_to_vertex = trial_stress - 
                                          Identity*PEAKI1_hardening/FSLOPE*one_third;
            // compute two unit tensors of the stress space
            unit_tensor_vertex_1 = Identity/sqrt_three;
            unit_tensor_vertex_2 = S_trial/sqrt(2.0*J2_trial);
            // compute the unit tensor in the direction of the plastic strain
            M = ( Identity*FSLOPE_p + S_trial*(1.0/(2.0*sqrt(J2_trial))) )/sqrt(3.0*FSLOPE_p*FSLOPE_p + 0.5);
            // compute the projection direction tensor
            P = (Identity*lame*(M.Trace()) + M*2.0*shear);
            // compute the components of P tensor in respect with two unit_tensor_vertex
            P_component_1 = P.Trace()/sqrt_three;
            P_deviatoric = P - unit_tensor_vertex_1*P_component_1;
            for (int counter_1=0 ; counter_1<=2 ; counter_1++){
              for (int counter_2=0 ; counter_2<=2 ; counter_2++){
                if (fabs(unit_tensor_vertex_2(counter_1,counter_2))>
                    fabs(unit_tensor_vertex_2(counter_1_fix,counter_2_fix))){
                  counter_1_fix = counter_1;
                  counter_2_fix = counter_2;
                }
              }
            }
            P_component_2 = P_deviatoric(counter_1_fix,counter_2_fix)/
                            unit_tensor_vertex_2(counter_1_fix,counter_2_fix);
            // calculation of the components of relative_stress_to_vertex
            // in respect with two unit_tensor_vertex
            relative_stress_to_vertex_1 = relative_stress_to_vertex.Trace()*one_sqrt_three;
            relative_stress_to_vertex_deviatoric = relative_stress_to_vertex -
                                                   unit_tensor_vertex_1*relative_stress_to_vertex_1;
            relative_stress_to_vertex_2 = relative_stress_to_vertex_deviatoric(counter_1_fix,counter_2_fix)/
                                          unit_tensor_vertex_2(counter_1_fix,counter_2_fix);
            // condition to determine if the stress_trial is in the vertex zone or not?
            double ratio_plus  = (relative_stress_to_vertex_1*P_component_2 + 
                                  relative_stress_to_vertex_2*P_component_1)/
                                 (P_component_1*P_component_1);
            double ratio_minus = (relative_stress_to_vertex_1*P_component_2 - 
                                  relative_stress_to_vertex_2*P_component_1)/
                                 (P_component_1*P_component_1);
            if ( ratio_plus >= 0.0 && ratio_minus >= 0.0) {
              stress_new[idx] = Identity*PEAKI1_hardening*one_third/FSLOPE;
              condition_return_to_vertex = 1;
            }
          }
        }
        // plasticity vertex treatment ends
        // nested return algorithm begins
        if (condition_return_to_vertex == 0){
          double gamma_tolerance = 0.01;   // **HARDCODED**
          double del_gamma = 100.;   // **HARDCODED**
          double gamma = 0.0;
          double gamma_old;
          double I1_iteration,J2_iteration;
          double beta_cap,FSLOPE_cap;
          int max_number_of_iterations = 500;   // **HARDCODED**
          int counter = 1;
          Matrix3 P,M,G;
          Matrix3 stress_iteration=trial_stress;
          Matrix3 S_iteration;
          // Multi-stage return loop begins
          while(abs(del_gamma)>gamma_tolerance && counter<=max_number_of_iterations){
            //cout << "Particle = " << idx << " return algo counter = " << counter 
            //     << " del_gamma = " << del_gamma << endl;
            counter=counter+1;
            // fast return algorithm to the yield surface
            // compute the invariants of the trial stres in the loop
            computeInvariants(stress_iteration, S_iteration, I1_iteration, J2_iteration);
            if (I1_iteration>PEAKI1_hardening/FSLOPE){
              stress_iteration += Identity*I1_iteration*one_third*
                  ((PEAKI1_hardening-sqrt(J2_iteration))/(FSLOPE*I1_iteration)-1);
            } else if (I1_iteration<pKappa[idx]-0.9*cap_radius){
              Matrix3 stress_iteration_temp;
              double I1_iteration1;
              double I1_iteration2;
              double I1_iteration3;
              double f_iteration2;
              int counter_temp=0;
              computeInvariants(stress_iteration, S_iteration, I1_iteration, J2_iteration);
              I1_iteration1=I1_iteration;
              I1_iteration2=pKappa[idx];
              I1_iteration3=(I1_iteration1+I1_iteration2)*0.5;
              stress_iteration_temp = stress_iteration + 
                Identity*I1_iteration*one_third*(I1_iteration3/I1_iteration-1.0);
              f_iteration2 = 
                YieldFunction(stress_iteration_temp,FSLOPE,pKappa[idx],cap_radius,PEAKI1_hardening);
              while ((abs(f_iteration2)>0.0000001*sqrt(J2_iteration+I1_iteration*I1_iteration) 
                      && counter_temp<2000) || counter_temp==0) {
                counter_temp = counter_temp + 1;
                if (f_iteration2<0.0){
                  I1_iteration2=I1_iteration3;
                  I1_iteration3=(I1_iteration1+I1_iteration2)*0.5;
                } else {
                  I1_iteration1=I1_iteration3;
                  I1_iteration3=(I1_iteration1+I1_iteration2)*0.5;
                }
                stress_iteration_temp = stress_iteration + 
                  Identity*I1_iteration*one_third*(I1_iteration3/I1_iteration-1.0);
                f_iteration2 = 
                 YieldFunction(stress_iteration_temp,FSLOPE,pKappa[idx],cap_radius,PEAKI1_hardening);
              }
              stress_iteration = stress_iteration_temp;
            } else if (I1_iteration<pKappa[idx]) {
              beta_cap = sqrt( 1.0 - (pKappa[idx]-I1_iteration)*(pKappa[idx]-I1_iteration)/
                         ( (cap_radius)*(cap_radius) ) );
              stress_iteration += S_iteration*((PEAKI1_hardening-FSLOPE*I1_iteration)*
                                 beta_cap/sqrt(J2_iteration)-1);
            } else {
              stress_iteration += S_iteration*((PEAKI1_hardening-FSLOPE*I1_iteration)/
                                sqrt(J2_iteration)-1);
            }
            // compute the invariants of the trial stres in the loop returned to the yield surface
            computeInvariants(stress_iteration, S_iteration, I1_iteration, J2_iteration);
            // check if the stress state is in the cap zone or not?
            if (I1_iteration>=pKappa[idx]){
              // compute the gradient of the plastic potential
              G = Identity*FSLOPE + S_iteration*(1.0/(2.0*sqrt(J2_iteration)));
              // compute the unit tensor in the direction of the plastic strain
              M = ( Identity*FSLOPE_p + S_iteration*(1.0/(2.0*sqrt(J2_iteration))) )/
                  sqrt(3*FSLOPE_p*FSLOPE_p + 0.5);
            }else{
              if (J2_iteration<0.00000001){
                // compute the gradient of the plastic potential
                G = Identity*(-1.0);
                // compute the unit tensor in the direction of the plastic strain
                M = Identity*(-one_sqrt_three);
              } else {
                beta_cap = sqrt( 1.0 - (pKappa[idx]-I1_iteration)*(pKappa[idx]-I1_iteration)/
                          ( (cap_radius)*(cap_radius) ) );
                FSLOPE_cap = (FSLOPE*I1_iteration-PEAKI1_hardening)*(pKappa[idx]-I1_iteration)/
                            ( cap_radius*cap_radius*beta_cap ) + FSLOPE*beta_cap;
                // compute the gradient of the plastic potential
                G = Identity*FSLOPE_cap + S_iteration*(1.0/(2.0*sqrt(J2_iteration)));
                // compute the unit tensor in the direction of the plastic strain
                M = ( Identity*FSLOPE_cap + S_iteration*(1.0/(2.0*sqrt(J2_iteration))) )/
                    sqrt(3.0*FSLOPE_cap*FSLOPE_cap + 0.5);
              }
            }
            // compute the projection direction tensor
            P = (Identity*lame*(M.Trace()) + M*2.0*shear);
            // store the last value of gamma for calculation of the changes in gamma
            gamma_old = gamma;
            // compute the new value for gamma
            gamma = ( G.Contract(trial_stress-stress_iteration) )/( G.Contract(P) );
            // compute new trial stress in the loop
            stress_iteration = trial_stress - P*gamma;
            // compute the changes of gamma in order to control converging
            del_gamma = (gamma-gamma_old)/gamma;
          }
          // Multi-stage return loop ends
          // compute the new stress state
          double hardeningEns;
          if (I1_iteration>=pKappa[idx]){
            hardeningEns = hardening_modulus/G.Norm();
          } else{
            if (J2_iteration<0.00000001){
              beta_cap = 0.0;
            }else{
              beta_cap = 1.0 - (pKappa[idx]-I1_iteration)*(pKappa[idx]-I1_iteration)/
                        ( (cap_radius)*(cap_radius) );
            }
            hardeningEns = sqrt(beta_cap)*hardening_modulus/G.Norm()
                         +2.0*CR*(FSLOPE*I1_iteration-PEAKI1_hardening) *
                         (pKappa[idx]-I1_iteration)
                         /( cap_radius*cap_radius*cap_radius*(1.0+FSLOPE*CR)*p3_crush_curve*p3_crush_curve )*
                         exp(-p1_crush_curve*(pKappa[idx]-cap_radius-p0_crush_curve))
                         *M.Trace()/G.Norm();
          }
          Matrix3 G_unit = G/G.Norm();
          gamma=G_unit.Contract(P)/( G_unit.Contract(P)+hardeningEns )*gamma;
             stress_new[idx] = trial_stress - P*gamma;
        }

        double shear_inverse = 0.5/shear;
        double lame_inverse = one_sixth/(bulk*shear) * ( two_third*shear - bulk );
        Matrix3 diff_stress_iteration = trial_stress - stress_new[idx];
        Matrix3 strain_iteration = (Identity*lame_inverse*(diff_stress_iteration.Trace()) +
                                  diff_stress_iteration*2.0*shear_inverse);
        // update total plastic strain magnitude
        pPlasticStrain_new[idx] = pPlasticStrain[idx] + strain_iteration.Norm();
        // update volumetric part of the plastic strain magnitude
        pPlasticStrainVol_new[idx] = pPlasticStrainVol[idx] + strain_iteration.Trace();
        // update volumetric part of the elastic strain magnitude
        pElasticStrainVol_new[idx] = pElasticStrainVol_new[idx] - strain_iteration.Trace();
        // update the position of the cap
        double pKappa_temp = exp(p3_crush_curve+p4_fluid_effect+pPlasticStrainVol[idx]);
        double pKappa_temp1 = exp(p3_crush_curve+pPlasticStrainVol[idx]);
        pKappa_new[idx] = pKappa[idx] + ( exp(-p1_crush_curve*(pKappa[idx]-cap_radius-p0_crush_curve))
                       /( p3_crush_curve*p1_crush_curve ) -
                       3.0*fluid_B0*(exp(p3_crush_curve+p4_fluid_effect)-1.0)*pKappa_temp
                       /( (pKappa_temp-1.0)*(pKappa_temp-1.0) ) +
                       3.0*fluid_B0*(exp(p3_crush_curve+p4_fluid_effect)-1.0)*pKappa_temp1
                       /( (pKappa_temp1-1.0)*(pKappa_temp1-1.0) ) )
                       *strain_iteration.Trace()/(1.0+FSLOPE*CR);

        PEAKI1_hardening = PEAKI1*FSLOPE + hardening_modulus*pPlasticStrain_new[idx];
        cap_radius=-CR*(FSLOPE*pKappa_new[idx]-PEAKI1_hardening);
        // compute the value of the yield function for the new stress
        double f_new =
          YieldFunction(stress_new[idx],FSLOPE,pKappa_new[idx],cap_radius,PEAKI1_hardening);
        double J2_new,I1_new;
        Matrix3 S_new;
        computeInvariants(stress_new[idx], S_new, I1_new, J2_new);

        // send an error message to the host code if the new stress is not on the yield surface
        if (abs(f_new) > 0.01*sqrt(J2_new+I1_new*I1_new)) {
          cerr<<"ERROR!  Particle " << idx << 
                " did not return to yield surface (simplifiedGeomodel.cc)"<<endl;
          cerr << "f_new= " << f_new << " sqrt(J2_new)= " << sqrt(J2_new) 
               <<" I1_new= " << I1_new << endl;
          cerr << "f_trial= " << f_trial << " sqrt(J2_trial)= " << sqrt(J2_trial) 
               <<" I1_trial= " << I1_trial << endl;
          cerr << "sig_n = " << unrotated_stress << "\n D = " << D  
               << "\n sig_trial = " << trial_stress << "\n sig_n+1 = " << stress_new[idx] << endl; 
          //exit(1);
        }

        // compute invariants of the plastic strain rate
        double I1_plasStrain,J2_plasStrain;
        Matrix3 S_plasStrain;
        computeInvariants(strain_iteration, S_plasStrain, I1_plasStrain, J2_plasStrain);
        // update the backstress
        double deltaBackStressIso_temp = exp(p3_crush_curve+pPlasticStrainVol[idx]);
        deltaBackStress = stress_new[idx]*kinematic_hardening_constant*sqrt(J2_plasStrain);
        deltaBackStressIso = Identity*( 3.0*fluid_B0*
                            (exp(p3_crush_curve+p4_fluid_effect)-1.0) * deltaBackStressIso_temp
                            /( (deltaBackStressIso_temp-1.0)*(deltaBackStressIso_temp-1.0) ) )*
                            strain_iteration.Trace();

      } // nested return algorithm ends

      // compute stress from the shifted stress
      stress_new[idx] += pBackStress[idx];
      pBackStress_new[idx] = pBackStress[idx] + deltaBackStress + deltaBackStressIso;
      pBackStressIso_new[idx] = pBackStressIso[idx] + deltaBackStressIso;

      stress_new[idx] = (rotation*stress_new[idx])*(rotation.Transpose());
      // Compute wave speed + particle velocity at each particle,
      // store the maximum
      c_dil = sqrt((bulk+four_third*shear)/(rho_cur));
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
      // Compute artificial viscosity term
      if (flag->d_artificial_viscosity) {
        double dx_ave = (dx.x() + dx.y() + dx.z())*one_third;
        double c_bulk = sqrt(bulk/rho_cur);
        p_q[idx] = artificialBulkViscosity(D.Trace(), c_bulk, rho_cur, dx_ave);
      } else {
        p_q[idx] = 0.;
      }
      Matrix3 AvgStress = (stress_new[idx] + stress_old[idx])*0.5;

      double e = (D(0,0)*AvgStress(0,0) +
                  D(1,1)*AvgStress(1,1) +
                  D(2,2)*AvgStress(2,2) +
              2.*(D(0,1)*AvgStress(0,1) +
                  D(0,2)*AvgStress(0,2) +
                  D(1,2)*AvgStress(1,2))) * pvolume[idx]*delT;
      se += e;

    }  // end loop over particles

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());

    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(se),        lb->StrainEnergyLabel);
    }

    delete interpolator;

  }// end loop over patches

}
#endif

void simplifiedGeoModel::computeStressTensor(const PatchSubset* patches,
                                           const MPMMaterial* matl,
                                           DataWarehouse* old_dw,
                                           DataWarehouse* new_dw)
{

  // Define some constants
  double one_sixth = 1.0/(6.0);
  double one_third = 1.0/(3.0);
  double two_third = 2.0/(3.0);
  double four_third = 4.0/(3.0);

  for(int p=0;p<patches->size();p++){

    const Patch* patch = patches->get(p);
    Matrix3 Identity,D;
    double J;
    Identity.Identity();
    double c_dil = 0.0,se=0.0;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<Vector> d_S(interpolator->size());
    vector<double> S(interpolator->size());

    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};
    int dwi = matl->getDWIndex();

    // Create array for the particle position
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    ParticleVariable<Matrix3> defGrad_new;
    constParticleVariable<Matrix3> defGrad;
    constParticleVariable<Matrix3> stress_old;
    ParticleVariable<Matrix3> stress_new;
    constParticleVariable<Point> px;
    constParticleVariable<double> pmass;
    ParticleVariable<double> pvolume,p_q;
    constParticleVariable<Vector> pvelocity,psize;
    ParticleVariable<double> pdTdt;
    constParticleVariable<double> pPlasticStrain;
    ParticleVariable<double>  pPlasticStrain_new;
    constParticleVariable<double> pPlasticStrainVol;
    ParticleVariable<double>  pPlasticStrainVol_new;
    constParticleVariable<double> pElasticStrainVol;
    ParticleVariable<double>  pElasticStrainVol_new;
    constParticleVariable<double> pKappa;
    ParticleVariable<double>  pKappa_new;
    constParticleVariable<Matrix3> pBackStress;
    ParticleVariable<Matrix3>  pBackStress_new;
    constParticleVariable<Matrix3> pBackStressIso;
    ParticleVariable<Matrix3>  pBackStressIso_new;
    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));
    old_dw->get(pPlasticStrain, pPlasticStrainLabel, pset);
    old_dw->get(pPlasticStrainVol, pPlasticStrainVolLabel, pset);
    old_dw->get(pElasticStrainVol, pElasticStrainVolLabel, pset);
    old_dw->get(pKappa, pKappaLabel, pset);
    old_dw->get(pBackStress, pBackStressLabel, pset);
    old_dw->get(pBackStressIso, pBackStressIsoLabel, pset);
    new_dw->allocateAndPut(pPlasticStrain_new,pPlasticStrainLabel_preReloc,pset);
    new_dw->allocateAndPut(pPlasticStrainVol_new,pPlasticStrainVolLabel_preReloc,pset);
    new_dw->allocateAndPut(pElasticStrainVol_new,pElasticStrainVolLabel_preReloc,pset);
    new_dw->allocateAndPut(pKappa_new,pKappaLabel_preReloc,pset);
    new_dw->allocateAndPut(pBackStress_new,pBackStressLabel_preReloc,pset);
    new_dw->allocateAndPut(pBackStressIso_new,pBackStressIsoLabel_preReloc,pset);
    Ghost::GhostType  gac   = Ghost::AroundCells;
    old_dw->get(px,                  lb->pXLabel,                        pset);
    old_dw->get(pmass,               lb->pMassLabel,                     pset);
    old_dw->get(psize,               lb->pSizeLabel,                     pset);
    old_dw->get(pvelocity,           lb->pVelocityLabel,                 pset);
    old_dw->get(defGrad, lb->pDeformationMeasureLabel,       pset);
    old_dw->get(stress_old,             lb->pStressLabel,                pset);
    new_dw->allocateAndPut(stress_new,  lb->pStressLabel_preReloc,       pset);
    new_dw->allocateAndPut(pvolume,  lb->pVolumeLabel_preReloc,          pset);
    new_dw->allocateAndPut(pdTdt,    lb->pdTdtLabel_preReloc,            pset);
    new_dw->allocateAndPut(defGrad_new,
                                  lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->allocateAndPut(p_q,      lb->p_qLabel_preReloc,              pset);

    Matrix3 velGrad(0.0), rotation(0.0), trial_stress(0.0);
    double rho_cur = 0.0;

    const double p3_crush_curve = d_cm.p3_crush_curve;
    const double p4_fluid_effect = d_cm.p4_fluid_effect;
    const double fluid_B0 = d_cm.fluid_B0;
    const double kinematic_hardening_constant = d_cm.kinematic_hardening_constant;
    double bulk = d_cm.B0;
    const double shear= d_cm.G0;

    // create node data for the plastic multiplier field
    double rho_orig = matl->getInitialDensity();
    Matrix3 L_new(0.0);

    // Get the deformation gradients first.  This is done differently
    // depending on whether or not the grid is reset.  (Should it be??? -JG)
    constNCVariable<Vector> gvelocity;
    new_dw->get(gvelocity, lb->gVelocityStarLabel,dwi,patch,gac,NGN);
    for(ParticleSubset::iterator iter=pset->begin();iter!=pset->end();iter++){
      particleIndex idx = *iter;

      //re-zero the velocity gradient:
      L_new.set(0.0);
      if(!flag->d_axisymmetric){
        // Get the node indices that surround the cell
        interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],
        defGrad[idx]);

        computeVelocityGradient(L_new,ni,d_S, oodx, gvelocity);
      } else {  // axi-symmetric kinematics
        // Get the node indices that surround the cell
        interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,
                                  psize[idx],defGrad[idx]);
        // x -> r, y -> z, z -> theta
        computeAxiSymVelocityGradient(L_new,ni,d_S,S,oodx,gvelocity,px[idx]);
      }

      // Update vel grad, def grad, J
      velGrad=L_new;
      defGrad_new[idx]=(L_new*delT+Identity)*defGrad[idx];
      J = defGrad_new[idx].Determinant();
      if (J <= 0){
        cout<< "ERROR, negative J! in particle "<< idx << endl;
        cout<<"J= "<<J<<endl;
        cout<<"L= "<<L_new<<endl;
        exit(1);
      }

      // Update particle volumes
      pvolume[idx]=(pmass[idx]/rho_orig)*J;
      rho_cur = rho_orig/J;

      // Initialize dT/dt to zero
      pdTdt[idx] = 0.0;

      // Compute incremental bulk modulus based on EOS (see EOS functions at the end)
      double Gamma = 0.11;  // HARDCODED for soil
      double c0 = sqrt(bulk/rho_orig);
      double s = 1.5;       // HARDCODED for soil

      double eta = rho_cur/rho_orig -1.0;
      if (fabs(eta) > 0.05) {
        double dp_drho = c0*c0*(1 - eta*(Gamma - 1-s))/pow((1 + eta*(1-s)), 3);
        //bulk = dp_drho*rho_cur;
        if (idx == 3120) cerr << "bulk modulus = " << bulk << endl;
      }

      // modify the bulk modulus based on the fluid effects
      double vol_strain = pPlasticStrainVol[idx]+pElasticStrainVol[idx];
      double bulk_temp = exp(p3_crush_curve+p4_fluid_effect+vol_strain);
      bulk = bulk + fluid_B0*
           ( exp(p3_crush_curve+p4_fluid_effect)-1.0 ) * bulk_temp
           / ( (bulk_temp-1.0)*(bulk_temp-1.0) );

      double lame = bulk - two_third*shear;
      double lame_inverse = one_sixth/(bulk*shear) * ( two_third*shear - bulk );

      // Compute stress using the return algorithm      
      double eps_p = pPlasticStrain[idx];
      double epsv_p = pPlasticStrainVol[idx];
      double epsv_e = pElasticStrainVol[idx];
      double kappa = pKappa[idx];

      Matrix3 strain_inc(0.0);
      int lvl = 0;
      computeStress(idx, lvl, delT, lame, lame_inverse, 
                    L_new, defGrad[idx], stress_old[idx], pBackStress[idx],
                    eps_p, epsv_e, epsv_p, kappa, strain_inc, defGrad_new[idx], rotation,
                    stress_new[idx]);
      // update total plastic strain magnitude
      pPlasticStrain_new[idx] = eps_p;
      // update volumetric part of the plastic strain magnitude
      pPlasticStrainVol_new[idx] = epsv_p;
      // update volumetric part of the elastic strain magnitude
      pElasticStrainVol_new[idx] = epsv_e;
      // update the position of the cap
      pKappa_new[idx] = kappa;

      Matrix3 deltaBackStress(0.0);
      Matrix3 deltaBackStressIso(0.0);
      if (strain_inc.Norm() != 0.0) {

        // compute invariants of the plastic strain rate
        double I1_plasStrain,J2_plasStrain;
        Matrix3 S_plasStrain;
        computeInvariants(strain_inc, S_plasStrain, I1_plasStrain, J2_plasStrain);
        // update the backstress
        double deltaBackStressIso_temp = exp(p3_crush_curve+pPlasticStrainVol[idx]);
        deltaBackStress = stress_new[idx]*kinematic_hardening_constant*sqrt(J2_plasStrain);
        deltaBackStressIso = Identity*( 3.0*fluid_B0*
                            (exp(p3_crush_curve+p4_fluid_effect)-1.0) * deltaBackStressIso_temp
                            /( (deltaBackStressIso_temp-1.0)*(deltaBackStressIso_temp-1.0) ) )*
                            strain_inc.Trace();
      }

      // compute stress from the shifted stress
      stress_new[idx] += pBackStress[idx];
      pBackStress_new[idx] = pBackStress[idx] + deltaBackStress + deltaBackStressIso;
      pBackStressIso_new[idx] = pBackStressIso[idx] + deltaBackStressIso;

      stress_new[idx] = (rotation*stress_new[idx])*(rotation.Transpose());
      // Compute wave speed + particle velocity at each particle,
      // store the maximum
      c_dil = sqrt((bulk+four_third*shear)/(rho_cur));
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
      // Compute artificial viscosity term
      if (flag->d_artificial_viscosity) {
        double dx_ave = (dx.x() + dx.y() + dx.z())*one_third;
        double c_bulk = sqrt(bulk/rho_cur);
        p_q[idx] = artificialBulkViscosity(D.Trace(), c_bulk, rho_cur, dx_ave);
      } else {
        p_q[idx] = 0.;
      }
      Matrix3 AvgStress = (stress_new[idx] + stress_old[idx])*0.5;

      double e = (D(0,0)*AvgStress(0,0) +
                  D(1,1)*AvgStress(1,1) +
                  D(2,2)*AvgStress(2,2) +
              2.*(D(0,1)*AvgStress(0,1) +
                  D(0,2)*AvgStress(0,2) +
                  D(1,2)*AvgStress(1,2))) * pvolume[idx]*delT;
      se += e;

    }  // end loop over particles

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());

    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(se),        lb->StrainEnergyLabel);
    }

    delete interpolator;

  }// end loop over patches

}


void simplifiedGeoModel::computeStress(const particleIndex idx, int& lvl, const double delT, 
                                       const double lame, const double lame_inv, 
                                       const Matrix3& L_new, const Matrix3& F_old,
                                       const Matrix3& Sig_old, const Matrix3& Alpha_old,
                                       double& eps_p, double& epsv_e, double& epsv_p, double& kappa, 
                                       Matrix3& Eps_inc, Matrix3& F_new, Matrix3& R_new,
                                       Matrix3& Sig_new)
{
  if (lvl > 5) return;

  // Constants
  /*
  double one_sixth = 1.0/(6.0);
  double two_third = 2.0/(3.0);
  double four_third = 4.0/(3.0);
  */
  double one_third = 1.0/(3.0);
  double sqrt_two = sqrt(2.0);
  double sqrt_three = sqrt(3.0);
  double one_sqrt_two = 1.0/sqrt_two;
  double one_sqrt_three = 1.0/sqrt_three;
  Matrix3 One, Zero(0.0);
  One.Identity();
  
  // Read constants again
  const double fSlope = d_cm.FSLOPE;
  const double fSlope_p = d_cm.FSLOPE_p;
  const double iso_hard = d_cm.hardening_modulus;
  const double cap_ratio = d_cm.CR;
  const double p0 = d_cm.p0_crush_curve;
  const double p1 = d_cm.p1_crush_curve;
  const double p3 = d_cm.p3_crush_curve;
  const double p4 = d_cm.p4_fluid_effect;
  const double fluid_B0 = d_cm.fluid_B0;
  const double i1_peak = d_cm.PEAKI1;
  const double shear= d_cm.G0;

  // Calculate derived quantities
  double shear_inv = 0.5/shear;
  double i1_peak_hard = i1_peak*fSlope + iso_hard*eps_p;
  double cap_rad = -cap_ratio*(fSlope*kappa - i1_peak_hard);

  // Compute deformation gradient and rate of deformation tensor
  F_new = (L_new*delT+One)*F_old;
  if (isnan(F_new(0,0))) {
    cerr << "F_new = " << F_new << endl;
    cerr << "F_old = " << F_old << endl;
    cerr << "L_new = " << L_new << endl;
    cerr << "delT = " << delT << endl;
  }
  //double j_new = F_new.Determinant();
  Matrix3 U_new;
  F_new.polarDecompositionRMB(U_new, R_new);
  Matrix3 D_new = (L_new + L_new.Transpose())*.5;
  D_new = (R_new.Transpose())*(D_new*R_new);

  // modify the bulk modulus based on the fluid effects  (** WILL have to add back later **)
  // double vol_strain = epsv_p+pElasticStrainVol[idx];

  // update the actual stress:
  Matrix3 Sig_unrot = (R_new.Transpose())*(Sig_old*R_new);
  Matrix3 Sig_trial = Sig_unrot + One*lame*(D_new.Trace()*delT) + D_new*(2.0*shear*delT);

  // compute shifted stress
  Sig_trial -= Alpha_old;
  Sig_new = Sig_unrot;

  // compute stress invariants for the trial stress
  double i1_trial,j2_trial;
  Matrix3 S_trial;
  computeInvariants(Sig_trial, S_trial, i1_trial, j2_trial);

  // compute the value of the yield function for the trial stress
  double f_trial = evalYieldFunction(j2_trial, i1_trial, fSlope, kappa, cap_rad, 
                                     i1_peak_hard);

  // save old values
  double eps_p_old = eps_p;
  double epsv_e_old = epsv_p;
  double epsv_p_old = epsv_p;
  double kappa_old = kappa;
  Matrix3 Eps_inc_old = Eps_inc;

  // initial assignment for the plastic strains and the position of the cap function
  double eps_p_new = eps_p;
  double epsv_p_new = epsv_p;
  epsv_e = epsv_e + D_new.Trace()*delT;
  /* double kappa_new = kappa; */

  // check if the stress is elastic or plastic: If it is elastic the new stres is equal
  // to trial stress otherwise, the plasticity return algrithm would be used.
  if (f_trial < 0.0) {
    Sig_new = Sig_trial;
  } else {
    // plasticity vertex treatment begins
    double i1_hard_fSlope = i1_peak_hard/fSlope;
    bool return_to_vertex = true;
    if (i1_trial > i1_hard_fSlope) {
      if (j2_trial < 0.00000001){
        Sig_new = One*i1_hard_fSlope*one_third;
        return_to_vertex = false;
      } else {
        int count_1_fix=0;
        int count_2_fix=0;

        // compute the relative trial stress in respect with the vertex
        Matrix3 Sig_rel = Sig_trial - One*(i1_hard_fSlope*one_third);

        // compute two unit tensors of the stress space
        Matrix3 One_scaled = One/sqrt_three;
        double sqrt_J2_trial = sqrt(j2_trial);
        double one_sqrt_J2_trial = 1.0/sqrt_J2_trial;
        Matrix3 S_trial_scaled = S_trial*(one_sqrt_J2_trial*one_sqrt_two);

        // compute the unit tensor in the direction of the plastic strain
        Matrix3 M = (One*fSlope_p + S_trial*(0.5*one_sqrt_J2_trial))/
                       sqrt(3.0*fSlope_p*fSlope_p + 0.5);

        // compute the projection direction tensor
        Matrix3 P = One*(lame*M.Trace()) + M*(2.0*shear);

        // compute the components of P tensor in respect with two scaled tensors
        double p_vol = P.Trace()*one_sqrt_three;
        Matrix3 P_dev = P - One_scaled*p_vol;
        for (int count_1=0 ; count_1<=2 ; count_1++){
          for (int count_2=0 ; count_2<=2 ; count_2++){
            if (fabs(S_trial_scaled(count_1,count_2))>
                fabs(S_trial_scaled(count_1_fix,count_2_fix))){
              count_1_fix = count_1;
              count_2_fix = count_2;
            }
          }
        }
        double p_dev_scaled_ij = P_dev(count_1_fix,count_2_fix)/
                                 S_trial_scaled(count_1_fix,count_2_fix);

        // calculation of the components of Sig_rel with respect to two scaled unit tensors
        double sig_rel_vol_scaled = Sig_rel.Trace()*one_sqrt_three;
        Matrix3 S_rel = Sig_rel - One_scaled*sig_rel_vol_scaled;
        double s_rel_scaled_ij = S_rel(count_1_fix,count_2_fix)/
                                 S_trial_scaled(count_1_fix,count_2_fix);

        // condition to determine if the stress_trial is in the vertex zone or not?
        double ratio_plus  = (sig_rel_vol_scaled*p_dev_scaled_ij + s_rel_scaled_ij*p_vol)/ 
                             (p_vol*p_vol);
        double ratio_minus = (sig_rel_vol_scaled*p_dev_scaled_ij - s_rel_scaled_ij*p_vol)/ 
                             (p_vol*p_vol);
        if ( ratio_plus >= 0.0 && ratio_minus >= 0.0) {
          Sig_new = One*(i1_hard_fSlope*one_third);
          return_to_vertex = false;
        }
      }
    }
    // plasticity vertex treatment ends
    // nested return algorithm begins
    if (return_to_vertex){
      double gamma_tol = 0.01;   // **HARDCODED**
      double del_gamma = 100.;   // **HARDCODED**
      int max_iter = 500;        // **HARDCODED**
      double gamma = 0.0;
      double gamma_old = 0.0;
      double i1_upd = 0.0, j2_upd = 0.0;
      double beta_cap = 0.0, fSlope_cap = 0.0;
      Matrix3 G(0.0);
      Matrix3 M(0.0);
      Matrix3 P(0.0);
      Matrix3 Sig_upd = Sig_trial;
      Matrix3 S_upd = Zero;

      // Multi-stage return loop begins
      int count = 1;
      while(abs(del_gamma)>gamma_tol && count<=max_iter){
        //cout << "Particle = " << idx << " return algo count = " << count 
        //     << " del_gamma = " << del_gamma << endl;
        count=count+1;

        // compute the invariants of the trial stres in the loop
        computeInvariants(Sig_upd, S_upd, i1_upd, j2_upd);
        double sqrt_J2_upd = sqrt(j2_upd);
        double one_sqrt_J2_upd = 1.0/sqrt_J2_upd;
        Matrix3 S_upd_temp_Ali;
        double i1_upd_Ali,j2_upd_Ali;

        // fast return algorithm to the yield surface
        if (i1_upd > i1_hard_fSlope){
          Sig_upd += One*(i1_upd*one_third)*((i1_peak_hard-sqrt_J2_upd)/(fSlope*i1_upd)-1);

        } else if (i1_upd < kappa-0.9*cap_rad) {
          //double i1_upd_old = i1_upd;
          double i1_upd1 = i1_upd;
          double i1_upd2 = kappa;
          double i1_upd3 = (i1_upd1+i1_upd2)*0.5;
          Matrix3 Sig_upd_temp = Sig_upd + One*(i1_upd*one_third)*(i1_upd3/i1_upd-1.0);
          //Matrix3 Sig_upd_temp = Sig_upd + One*(i1_upd1*one_third)*(i1_upd3/i1_upd1-1.0);
          Matrix3 S_upd_temp(0.0);
          computeInvariants(Sig_upd_temp, S_upd_temp_Ali, i1_upd_Ali, j2_upd_Ali);
          double f_upd = evalYieldFunction(j2_upd_Ali, i1_upd_Ali, fSlope, kappa, cap_rad, 
                                           i1_peak_hard);
          int count_temp=0;
          while ((abs(f_upd)>0.0000001*sqrt(j2_upd+i1_upd*i1_upd) 
                      && count_temp<2000) || count_temp==0) {
            count_temp = count_temp + 1;
            if (f_upd < 0.0){
              i1_upd2 = i1_upd3;
              i1_upd3 = (i1_upd1+i1_upd2)*0.5;
            } else {
              i1_upd1 = i1_upd3;
              i1_upd3 = (i1_upd1+i1_upd2)*0.5;
            }
            Sig_upd_temp = Sig_upd + One*i1_upd*one_third*(i1_upd3/i1_upd-1.0);
            computeInvariants(Sig_upd_temp, S_upd_temp_Ali, i1_upd_Ali, j2_upd_Ali);
            f_upd = evalYieldFunction(j2_upd_Ali, i1_upd_Ali, fSlope, kappa, cap_rad, 
                                       i1_peak_hard);
          }
          Sig_upd = Sig_upd_temp;
        } else if (i1_upd < kappa) {
          beta_cap = sqrt(1.0 - (kappa-i1_upd)*(kappa-i1_upd)/(cap_rad*cap_rad));
          Sig_upd += S_upd*((i1_peak_hard-fSlope*i1_upd)*beta_cap*one_sqrt_J2_upd - 1);
        } else {

          Sig_upd += S_upd*((i1_peak_hard-fSlope*i1_upd)*one_sqrt_J2_upd - 1);
        }

        // compute the invariants of the trial stres in the loop returned to the yield surface
        computeInvariants(Sig_upd, S_upd, i1_upd, j2_upd);
        sqrt_J2_upd = sqrt(j2_upd);
        one_sqrt_J2_upd = 1.0/sqrt_J2_upd;

        // check if the stress state is in the cap zone or not?
        if (i1_upd >= kappa){
          // compute the gradient of the plastic potential
          G = One*fSlope + S_upd*(0.5*one_sqrt_J2_upd);
          // compute the unit tensor in the direction of the plastic strain
          M = ( One*fSlope_p + S_upd*(0.5*one_sqrt_J2_upd) )/ sqrt(3.0*fSlope_p*fSlope_p + 0.5);
        } else {
          if (j2_upd<0.00000001){
            // compute the gradient of the plastic potential
            G = One*(-1.0);
            // compute the unit tensor in the direction of the plastic strain
            M = One*(-one_sqrt_three);
          } else {
            beta_cap = sqrt( 1.0 - (kappa-i1_upd)*(kappa-i1_upd)/(cap_rad*cap_rad) );
            fSlope_cap = (fSlope*i1_upd-i1_peak_hard)*(kappa-i1_upd)/
                            (cap_rad*cap_rad*beta_cap) + fSlope*beta_cap;
            // compute the gradient of the plastic potential
            G = One*fSlope_cap + S_upd*(0.5*one_sqrt_J2_upd);
            // compute the unit tensor in the direction of the plastic strain
            M = G*(1.0/sqrt(3.0*fSlope_cap*fSlope_cap + 0.5));
          }
        }
        // compute the projection direction tensor
        P = One*(lame*M.Trace()) + M*(2.0*shear);
        // store the last value of gamma for calculation of the changes in gamma
        gamma_old = gamma;
        // compute the new value for gamma
        gamma = (G.Contract(Sig_trial-Sig_upd))/(G.Contract(P));
        // compute new trial stress in the loop
        Sig_upd = Sig_trial - P*gamma;
        // compute the changes of gamma in order to control converging
        del_gamma = (gamma-gamma_old)/gamma;
      }
      // Multi-stage return loop ends

      // compute the new stress state
      double hard_scaled = 0.0;
      double G_mag = G.Norm();
      double one_G_mag = 1/G_mag;
      double kappa_i1_upd = kappa - i1_upd;
      double cap_rad_sq = cap_rad*cap_rad;
      if (kappa_i1_upd <= 0.0){
        hard_scaled = iso_hard*one_G_mag;
      } else {
        if (j2_upd < 0.00000001) {
          beta_cap = 0.0;
        } else {
          beta_cap = 1.0 - (kappa_i1_upd*kappa_i1_upd)/cap_rad_sq;
        }
        hard_scaled = (sqrt(beta_cap)*iso_hard +
                      2.0*cap_ratio*(fSlope*i1_upd-i1_peak_hard)*kappa_i1_upd
                         /(cap_rad_sq*cap_rad*(1.0+fSlope*cap_ratio)*p3*p3)*
                         exp(-p1*(kappa-cap_rad-p0))*M.Trace())*one_G_mag;
      }
      Matrix3 G_unit = G*one_G_mag;
      double GP = G_unit.Contract(P);
      gamma *= GP/(GP+hard_scaled);
      Sig_new = Sig_trial - P*gamma;

    } // End return to vertex if

    Matrix3 Sig_diff = Sig_trial - Sig_new;
    Matrix3 Eps_diff = One*(lame_inv*Sig_diff.Trace()) + Sig_diff*(2.0*shear_inv);
    Eps_inc += Eps_diff;

    // update total plastic strain magnitude
    eps_p_new = eps_p + Eps_diff.Norm();
    eps_p = eps_p_new;
    // update volumetric part of the plastic strain magnitude
    epsv_p_new = epsv_p + Eps_diff.Trace();
    epsv_p = epsv_p_new;
    // update volumetric part of the elastic strain magnitude
    epsv_e = epsv_e - Eps_diff.Trace();

    // update the position of the cap
    double var1 = exp(p3+p4+epsv_p_new);
    double var2 = exp(p3+epsv_p_new);
    kappa += ( exp(-p1*(kappa-cap_rad-p0))/(p3*p1) -
               3.0*fluid_B0*(exp(p3+p4)-1.0)*(var1/((var1-1.0)*(var1-1.0)) +
                                              var2/((var2-1.0)*(var2-1.0))) )
               *Eps_diff.Trace()/(1.0+fSlope*cap_ratio);

    i1_peak_hard = i1_peak*fSlope + iso_hard*eps_p_new;
    cap_rad = -cap_ratio*(fSlope*kappa-i1_peak_hard);

    // compute the value of the yield function for the new stress
    Matrix3 S_new(0.0);
    double j2_new = 0.0, i1_new = 0.0;
    computeInvariants(Sig_new, S_new, i1_new, j2_new);
    double f_new = evalYieldFunction(j2_new, i1_new, fSlope, kappa, cap_rad, 
                                       i1_peak_hard);
    if (abs(f_new) > 0.01*sqrt(j2_new+i1_new*i1_new)) {
      // Error message
      lvl += 1;
      cerr << "ERROR!  Particle " << idx << " Recursion level = " << lvl  
           << " did not return to yield surface (simplifiedGeomodel.cc)"
           << " with delT = " << delT << endl;
      cerr << "  f_new= " << f_new << " sqrt(J2_new)= " << sqrt(j2_new) 
           << " I1_new= " << i1_new << endl;
      cerr << "  f_trial= " << f_trial << " sqrt(J2_trial)= " << sqrt(j2_trial) 
           << " I1_trial= " << i1_trial << endl;
      //exit(1);

      // Split again and see it is gets to the yield surface
      double delT_new = delT*0.5;
      Matrix3 Sig_tmp(0.0);
      Matrix3 F_tmp(0.0);
      Matrix3 R_tmp(0.0);
      computeStress(idx, lvl, delT_new, lame, lame_inv, 
                    L_new, F_old, Sig_old, Alpha_old,
                    eps_p_old, epsv_e_old, epsv_p_old, kappa_old, Eps_inc_old, F_tmp, R_tmp,
                    Sig_tmp);
      computeStress(idx, lvl, delT_new, lame, lame_inv, 
                    L_new, F_tmp, Sig_tmp, Alpha_old,
                    eps_p, epsv_e, epsv_p, kappa, Eps_inc, F_new, R_new,
                    Sig_new);
      lvl -= 1;
    }
  }
  return;
}

void simplifiedGeoModel::computeInvariants(Matrix3& stress, Matrix3& S,  double& I1, double& J2){

  Matrix3 Identity;
  Identity.Identity();
  I1 = stress.Trace();
  S = stress - Identity*(1.0/3.0)*I1;
  J2 = 0.5*S.Contract(S);

}

void simplifiedGeoModel::computeInvariants(const Matrix3& stress, Matrix3& S,  double& I1, double& J2){

  Matrix3 Identity;
  Identity.Identity();
  I1 = stress.Trace();
  S = stress - Identity*(1.0/3.0)*I1;
  J2 = 0.5*S.Contract(S);

}

 double simplifiedGeoModel::YieldFunction(const Matrix3& stress, const double& FSLOPE, const double& kappa, const double& cap_radius, const double&PEAKI1){

  Matrix3 S;
  double I1,J2,b;
  computeInvariants(stress,S,I1,J2);
  if (I1>kappa){
    return sqrt(J2) + FSLOPE*I1 - PEAKI1;
  }else{
    b = 1.0 - (kappa-I1)/(cap_radius)*(kappa-I1)/(cap_radius);
    if (b>0.0){
      return sqrt(J2) + FSLOPE*I1*sqrt(b) - PEAKI1*sqrt(b);
    }else{
      return sqrt(J2)+kappa-cap_radius-I1;
    }
  }

 }

 double simplifiedGeoModel::YieldFunction(Matrix3& stress, const double& FSLOPE, const double& kappa, const double& cap_radius, const double&PEAKI1){

  Matrix3 S;
  double I1,J2,b;
  computeInvariants(stress,S,I1,J2);
  if (I1>kappa){
    return sqrt(J2) + FSLOPE*I1 - PEAKI1;
  }else{
    b = 1.0 - (kappa-I1)/(cap_radius)*(kappa-I1)/(cap_radius);
    if (b>0.0){
      return sqrt(J2) + FSLOPE*I1*sqrt(b) - PEAKI1*sqrt(b);
    }else{
      return sqrt(J2)+kappa-cap_radius-I1;
    }
  }

 }

double simplifiedGeoModel::evalYieldFunction(const double& J2, const double& I1, 
                                             const double& f_slope, 
                                             const double& kappa, 
                                             const double& cap_radius, 
                                             const double& i1_peak_hard) 
{
  double kappa_I1 = kappa - I1;
  if (kappa_I1 < 0) {
    return sqrt(J2) + f_slope*I1 - i1_peak_hard;
  } else {
    double b = 1.0 - kappa_I1*kappa_I1/(cap_radius*cap_radius);
    if (b > 0.0){
      return sqrt(J2) + (f_slope*I1 - i1_peak_hard)*sqrt(b);
    }else{
      return sqrt(J2) + kappa_I1 - cap_radius;
    }
  }
}

void simplifiedGeoModel::carryForward(const PatchSubset* patches,
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

void simplifiedGeoModel::addParticleState(std::vector<const VarLabel*>& from,
                                        std::vector<const VarLabel*>& to)
{


  from.push_back(pPlasticStrainLabel);
  from.push_back(pPlasticStrainVolLabel);
  from.push_back(pElasticStrainVolLabel);
  from.push_back(pKappaLabel);
  from.push_back(pBackStressLabel);
  from.push_back(pBackStressIsoLabel);
  to.push_back(pPlasticStrainLabel_preReloc);
  to.push_back(pPlasticStrainVolLabel_preReloc);
  to.push_back(pElasticStrainVolLabel_preReloc);
  to.push_back(pKappaLabel_preReloc);
  to.push_back(pBackStressLabel_preReloc);
  to.push_back(pBackStressIsoLabel_preReloc);

}

void simplifiedGeoModel::addInitialComputesAndRequires(Task* task,
                                            const MPMMaterial* matl,
                                            const PatchSet* ) const
{
  // Add the computes and requires that are common to all explicit
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();

  // Other constitutive model and input dependent computes and requires

  task->computes(pPlasticStrainLabel, matlset);
  task->computes(pPlasticStrainVolLabel, matlset);
  task->computes(pElasticStrainVolLabel, matlset);
  task->computes(pKappaLabel, matlset);
  task->computes(pBackStressLabel, matlset);
  task->computes(pBackStressIsoLabel, matlset);

}

void simplifiedGeoModel::addComputesAndRequires(Task* task,
                                              const MPMMaterial* matl,
                                              const PatchSet* patches ) const
{

  // Add the computes and requires that are common to all explicit
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForHypoExplicit(task, matlset, patches);
  task->requires(Task::OldDW, pPlasticStrainLabel,    matlset, Ghost::None);
  task->requires(Task::OldDW, pPlasticStrainVolLabel,    matlset, Ghost::None);
  task->requires(Task::OldDW, pElasticStrainVolLabel,    matlset, Ghost::None);
  task->requires(Task::OldDW, pKappaLabel,    matlset, Ghost::None);
  task->requires(Task::OldDW, pBackStressLabel,    matlset, Ghost::None);
  task->requires(Task::OldDW, pBackStressIsoLabel,    matlset, Ghost::None);
  task->computes(pPlasticStrainLabel_preReloc,  matlset);
  task->computes(pPlasticStrainVolLabel_preReloc,  matlset);
  task->computes(pElasticStrainVolLabel_preReloc,  matlset);
  task->computes(pKappaLabel_preReloc,  matlset);
  task->computes(pBackStressLabel_preReloc,  matlset);
  task->computes(pBackStressIsoLabel_preReloc,  matlset);

}

void
simplifiedGeoModel::addComputesAndRequires(Task* ,
                                   const MPMMaterial* ,
                                   const PatchSet* ,
                                   const bool ) const
{
}

//*******************************************************************************
// Assume EOS of the form (note that this is an Eulerian measure of pressure)
// Under linear conditions:
//   p = K(rho-rho0)/rho = K*eta/(1 + eta)
//   eta = rho/rho0 - 1  (0 = reference state, > 0 = compression, < 0 = tension)
//       = p/(K-p)
//   rho = rho0*(1+eta)
// Under high pressures:
//  (from "Craters produced by underground explosions" by Luccioni et al., 2009, 
//         Computers and Structures, vol. 87, no. 21-22, pp. 1366-1373.)
//   p = p_h + Gamma rho0 (E - E_h)
//     where p_h = rho0*c0^2*eta*(1+eta)/[1 - (s-1)eta]^2
//           E = rho0*cv*(T-T0)
//           E_h = 1/2 p_h/rho0 * eta/(1+eta)
//     =  rho0 Gamma E + rho0 c0^2 eta (2 (1+eta)- Gamma eta)/(2 (1 + (1-s)eta)^2)
//   eta = rho/rho0 -1
//   rho = rho0*(1+eta)
//
// ** Will need to be changed to something consistent with the constitutive model.**
//*******************************************************************************
double simplifiedGeoModel::computeRhoMicroCM(double pressure,
                                             const double p_ref,
                                             const MPMMaterial* matl,
                                             double temperature,
                                             double rho_guess)
{
  double rho0 = matl->getInitialDensity();
  double K = d_cm.B0;
  double p_gauge = pressure - p_ref;

  double eta = 1.0;
  double rho = rho0;
  bool d_linear_eos = false;

  if (d_linear_eos) {
     eta = p_gauge/(K - p_gauge);
     rho = rho0*(1+eta);
  } else {
    double Cv = matl->getSpecificHeat();
    double T0 = matl->getRoomTemperature();
    double E = rho0*Cv*(temperature-T0);
    E = 0.0; // HARDCODED so that sp. internal energy change is zero
             // Not tension cutoff yet
    double Gamma = 0.11;  // HARDCODED for soil
    double c0 = sqrt(K/rho0);
    double s = 1.5;       // HARDCODED for soil

    // Newton iterations to find eta
    eta = 1.9;
    double f = 0.0;
    double fPrime = 0.0;
    do {
      double p = rho0*(Gamma*E + 0.5*c0*c0*eta*(2*(1+eta) - Gamma*eta)/pow((1 + (1-s)*eta),2));
      f = p - p_gauge;
      double dp_drho = c0*c0*(1 - eta*(Gamma - 1-s))/pow((1 + eta*(1-s)), 3);
      fPrime = rho0*dp_drho; 
      eta -= f/fPrime;
    } while (fabs(f) > 1.0e-5);
    //cerr << "eta = " << eta << " rho = " << rho << " p = " << p_gauge << endl;
    rho = rho0*(1+eta);
  }

  if (!(rho > 0.0)) {
    cerr << "**Warning**Negative mass density in SimplifiedGeoModel::computeRhoMicroCM " << rho
         << " rho0 = " << rho0 << " eta = " << eta << " p = " << p_gauge << endl;
  }

  // eta cannot be less than -1 or greater than 2
  // if (eta <= -1.0) eta = -0.999999999999; 
  // if (eta >= 2.0) eta = 1.999999999999;
  // rho = rho0*(1+eta);
  return rho;
}

//*******************************************************************************
// Assume EOS of the form (note that this is an Eulerian measure of pressure)
// Under normal circumstances:
//   p = K(rho-rho0)/rho = K*eta/(1 + eta)
//   eta = rho/rho0 - 1  (0 = reference state, > 0 = compression, < 0 = tension)
//   dp/drho = dp/deta*deta/drho = K/(rho0*(1+eta)^2) = K*rho0/rho^2
//   c = sqrt(K/rho)
// Under high pressures:
//  (from "Craters produced by underground explosions" by Luccioni et al., 2009, 
//         Computers and Structures, vol. 87, no. 21-22, pp. 1366-1373.)
//   p = p_h + Gamma rho (E - E_h)
//     where p_h = rho0*c0^2*eta*(1+eta)/[1 - (s-1)eta]^2
//           E = rho0*cv*(T-T0)
//           E_h = 1/2 p_h/rho0 * eta/(1+eta)
//     =  rho0 Gamma E + rho0 c0^2 eta (2 (1+eta)- Gamma eta)/(2 (1 + (1-s)eta)^2)
//   eta = rho/rho0 -1
//   dp_drho = c0*c0*(1 - eta*(Gamma - 1-s))/(1 + eta*(1-s))^3;
//   c^2 = (dp/drho)^2
//
// ** Will need to be changed to something consistent with the constitutive model.**
//*******************************************************************************
void simplifiedGeoModel::computePressEOSCM(double rho, double& pressure,
                                           double p_ref,
                                           double& dp_drho, double& csquared,
                                           const MPMMaterial* matl,
                                           double temperature)
{

  double K = d_cm.B0;
  double rho0 = matl->getInitialDensity();
  double eta = rho/rho0 - 1;

  // eta cannot be less than -1 or greater than 2
  // if (eta <= -1.0) eta = -0.999999999999;
  // if (eta >= 2.0) eta = 1.999999999999;
  bool d_linear_eos = false;
  if (d_linear_eos) {
    pressure = K*eta/(1+eta) + p_ref;
    dp_drho  = K*rho0/(rho*rho);
    csquared = K/rho;  // speed of sound squared
  } else {
    double Cv = matl->getSpecificHeat();
    double T0 = matl->getRoomTemperature();
    double E = rho0*Cv*(temperature-T0);
    E = 0.0; // HARDCODED so that sp. internal energy change is zero
             // Not tension cutoff yet
    double Gamma = 0.11;  // HARDCODED for soil
    double c0 = sqrt(K/rho0);
    double s = 1.5;       // HARDCODED for soil

    pressure = rho0*(Gamma*E + 0.5*c0*c0*eta*(2*(1+eta) - Gamma*eta)/pow((1 + (1-s)*eta),2));
    pressure += p_ref;
    dp_drho = c0*c0*(1 - eta*(Gamma - 1-s))/pow((1 + eta*(1-s)), 3);
    csquared = dp_drho*dp_drho;
    if (csquared < 0.0) {
      cerr << "**ERROR**simplifiedGeoMode:computePressEOSCM:impaginary sound speed: c^2 = " 
           << csquared << " p = " << (pressure - p_ref) << " eta = " << eta << endl;
    }
    //if (pressure < p_ref) {
    //  cerr << "simplifiedGeoMode:computePressEOSCM:eta = " << eta 
    //       << " p = " << (pressure - p_ref) << " dp_drho = " << dp_drho << endl;
    //}
  }
  //cout << "NO VERSION OF computePressEOSCM EXISTS YET FOR simplifiedGeoModel"
  //     << endl;
}

//***************************************************************************
// Inverse of the bulk modulus
//***************************************************************************
double simplifiedGeoModel::getCompressibility()
{
  cout << "NO VERSION OF getCompressibility EXISTS YET FOR simplifiedGeoModel"
       << endl;
  return 1.0;
}

void
simplifiedGeoModel::initializeLocalMPMLabels()
{

  pPlasticStrainLabel = VarLabel::create("p.plasticStrain",
    ParticleVariable<double>::getTypeDescription());
  pPlasticStrainLabel_preReloc = VarLabel::create("p.plasticStrain+",
    ParticleVariable<double>::getTypeDescription());
  pPlasticStrainVolLabel = VarLabel::create("p.plasticStrainVol",
    ParticleVariable<double>::getTypeDescription());
  pPlasticStrainVolLabel_preReloc = VarLabel::create("p.plasticStrainVol+",
    ParticleVariable<double>::getTypeDescription());
  pElasticStrainVolLabel = VarLabel::create("p.elasticStrainVol",
    ParticleVariable<double>::getTypeDescription());
  pElasticStrainVolLabel_preReloc = VarLabel::create("p.elasticStrainVol+",
    ParticleVariable<double>::getTypeDescription());
  pKappaLabel = VarLabel::create("p.kappa",
    ParticleVariable<double>::getTypeDescription());
  pKappaLabel_preReloc = VarLabel::create("p.kappa+",
    ParticleVariable<double>::getTypeDescription());
  pBackStressLabel = VarLabel::create("p.BackStress",
    ParticleVariable<Matrix3>::getTypeDescription());
  pBackStressLabel_preReloc = VarLabel::create("p.BackStress+",
    ParticleVariable<Matrix3>::getTypeDescription());
  pBackStressIsoLabel = VarLabel::create("p.BackStressIso",
    ParticleVariable<Matrix3>::getTypeDescription());
  pBackStressIsoLabel_preReloc = VarLabel::create("p.BackStressIso+",
    ParticleVariable<Matrix3>::getTypeDescription());

}
