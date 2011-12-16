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

//#include </usr/include/valgrind/callgrind.h>
#include <CCA/Components/MPM/ConstitutiveModel/Arenisca.h>
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

Arenisca::Arenisca(ProblemSpecP& ps, MPMFlags* Mflag)
  : ConstitutiveModel(Mflag)
{

  ps->require("FSLOPE",d_initialData.FSLOPE);
  ps->require("FSLOPE_p",d_initialData.FSLOPE_p);
  ps->require("hardening_modulus",d_initialData.hardening_modulus);
  ps->require("CR",d_initialData.CR);
  ps->require("p0_crush_curve",d_initialData.p0_crush_curve);
  ps->require("p1_crush_curve",d_initialData.p1_crush_curve);
  ps->require("p3_crush_curve",d_initialData.p3_crush_curve);
  ps->require("p4_fluid_effect",d_initialData.p4_fluid_effect);
  ps->require("fluid_B0",d_initialData.fluid_B0);
  ps->require("fluid_pressure_initial",d_initialData.fluid_pressure_initial);
  ps->require("kinematic_hardening_constant",d_initialData.kinematic_hardening_constant);
  ps->require("PEAKI1",d_initialData.PEAKI1);
  ps->require("B0",d_initialData.B0);
  ps->require("G0",d_initialData.G0);
  initializeLocalMPMLabels();
}

Arenisca::Arenisca(const Arenisca* cm)
  : ConstitutiveModel(cm)
{
  d_initialData.FSLOPE = cm->d_initialData.FSLOPE;
  d_initialData.FSLOPE_p = cm->d_initialData.FSLOPE_p;
  d_initialData.hardening_modulus = cm->d_initialData.hardening_modulus;
  d_initialData.CR = cm->d_initialData.CR;
  d_initialData.p0_crush_curve = cm->d_initialData.p0_crush_curve;
  d_initialData.p1_crush_curve = cm->d_initialData.p1_crush_curve;
  d_initialData.p3_crush_curve = cm->d_initialData.p3_crush_curve;
  d_initialData.p4_fluid_effect = cm->d_initialData.p4_fluid_effect;
  d_initialData.fluid_B0 = cm->d_initialData.fluid_B0;
  d_initialData.fluid_pressure_initial = cm->d_initialData.fluid_pressure_initial;
  d_initialData.kinematic_hardening_constant = cm->d_initialData.kinematic_hardening_constant;
  d_initialData.PEAKI1 = cm->d_initialData.PEAKI1;
  d_initialData.B0 = cm->d_initialData.B0;
  d_initialData.G0 = cm->d_initialData.G0;
  initializeLocalMPMLabels();
}

Arenisca::~Arenisca()
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

void Arenisca::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","Arenisca");
  }

  cm_ps->appendElement("FSLOPE",d_initialData.FSLOPE);
  cm_ps->appendElement("FSLOPE_p",d_initialData.FSLOPE_p);
  cm_ps->appendElement("hardening_modulus",d_initialData.hardening_modulus);
  cm_ps->appendElement("CR",d_initialData.CR);
  cm_ps->appendElement("p0_crush_curve",d_initialData.p0_crush_curve);
  cm_ps->appendElement("p1_crush_curve",d_initialData.p1_crush_curve);
  cm_ps->appendElement("p3_crush_curve",d_initialData.p3_crush_curve);
  cm_ps->appendElement("p4_fluid_effect",d_initialData.p4_fluid_effect);
  cm_ps->appendElement("fluid_B0",d_initialData.fluid_B0);
  cm_ps->appendElement("fluid_pressure_initial",d_initialData.fluid_pressure_initial);
  cm_ps->appendElement("kinematic_hardening_constant",d_initialData.kinematic_hardening_constant);
  cm_ps->appendElement("PEAKI1",d_initialData.PEAKI1);
  cm_ps->appendElement("B0",d_initialData.B0);
  cm_ps->appendElement("G0",d_initialData.G0);

}

Arenisca* Arenisca::clone()
{
  return scinew Arenisca(*this);
}

void Arenisca::initializeCMData(const Patch* patch,
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
    pKappa[*iter] = ( d_initialData.p0_crush_curve +
      d_initialData.CR*d_initialData.FSLOPE*d_initialData.PEAKI1 )/
      (d_initialData.CR*d_initialData.FSLOPE+1.0);
    pBackStress[*iter].set(0.0);
    pBackStressIso[*iter] = Identity*d_initialData.fluid_pressure_initial;
  }
  computeStableTimestep(patch, matl, new_dw);
}

void
Arenisca::allocateCMDataAddRequires(Task* task,
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

void Arenisca::allocateCMDataAdd(DataWarehouse* new_dw,
                                         ParticleSubset* addset,
          map<const VarLabel*, ParticleVariableBase*>* newState,
                                         ParticleSubset* delset,
                                         DataWarehouse* )
{

}

void Arenisca::computeStableTimestep(const Patch* patch,
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
  double bulk = d_initialData.B0;
  double shear= d_initialData.G0;
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

void Arenisca::computeStressTensor(const PatchSubset* patches,
                                           const MPMMaterial* matl,
                                           DataWarehouse* old_dw,
                                           DataWarehouse* new_dw)
{

  // Define some constants
  //double one_sixth = 1.0/(6.0);
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
    ParticleVariable<Matrix3> deformationGradient_new;
    constParticleVariable<Matrix3> deformationGradient;
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
    old_dw->get(deformationGradient, lb->pDeformationMeasureLabel,       pset);
    old_dw->get(stress_old,             lb->pStressLabel,                pset);
    new_dw->allocateAndPut(stress_new,  lb->pStressLabel_preReloc,       pset);
    new_dw->allocateAndPut(pvolume,  lb->pVolumeLabel_preReloc,          pset);
    new_dw->allocateAndPut(pdTdt,    lb->pdTdtLabel_preReloc,            pset);
    new_dw->allocateAndPut(deformationGradient_new,
                                  lb->pDeformationMeasureLabel_preReloc, pset);
    new_dw->allocateAndPut(p_q,      lb->p_qLabel_preReloc,              pset);

    ParticleVariable<Matrix3> velGrad,rotation,trial_stress;
    ParticleVariable<double> f_trial,rho_cur;
    new_dw->allocateTemporary(velGrad,      pset);
    new_dw->allocateTemporary(rotation,     pset);
    new_dw->allocateTemporary(trial_stress, pset);
    new_dw->allocateTemporary(f_trial, pset);
    new_dw->allocateTemporary(rho_cur,pset);

    const double FSLOPE = d_initialData.FSLOPE;
    const double FSLOPE_p = d_initialData.FSLOPE_p;
    const double hardening_modulus = d_initialData.hardening_modulus;
    const double CR = d_initialData.CR;
    const double p0_crush_curve = d_initialData.p0_crush_curve;
    const double p1_crush_curve = d_initialData.p1_crush_curve;
    const double p3_crush_curve = d_initialData.p3_crush_curve;
    const double p4_fluid_effect = d_initialData.p4_fluid_effect;
    const double fluid_B0 = d_initialData.fluid_B0;
    const double kinematic_hardening_constant = d_initialData.kinematic_hardening_constant;
    const double PEAKI1 = d_initialData.PEAKI1;
    double bulk = d_initialData.B0;
    const double shear= d_initialData.G0;

    // create node data for the plastic multiplier field
    double rho_orig = matl->getInitialDensity();
    Matrix3 tensorL(0.0);

    // Get the deformation gradients first.  This is done differently
    // depending on whether or not the grid is reset.  (Should it be??? -JG)

    constNCVariable<Vector> gvelocity;
    new_dw->get(gvelocity, lb->gVelocityStarLabel,dwi,patch,gac,NGN);
    for(ParticleSubset::iterator iter=pset->begin();iter!=pset->end();iter++){
      particleIndex idx = *iter;

	     //re-zero the velocity gradient:
	     tensorL.set(0.0);
      if(!flag->d_axisymmetric){
	       // Get the node indices that surround the cell
	       interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],
                                                   deformationGradient[idx]);
	       computeVelocityGradient(tensorL,ni,d_S, oodx, gvelocity);
      } else {  // axi-symmetric kinematics
	       // Get the node indices that surround the cell
	       interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,
							                                   psize[idx],deformationGradient[idx]);
	       // x -> r, y -> z, z -> theta
	       computeAxiSymVelocityGradient(tensorL,ni,d_S,S,oodx,gvelocity,px[idx]);
      }
      velGrad[idx]=tensorL;

      deformationGradient_new[idx]=(tensorL*delT+Identity)*deformationGradient[idx];

	     J = deformationGradient_new[idx].Determinant();
	     if (J<=0){
	       cout<< "ERROR, negative J! "<<endl;
	       cout<<"J= "<<J<<endl;
	       cout<<"L= "<<tensorL<<endl;
        exit(1);
	     }
	     // Update particle volumes
	     pvolume[idx]=(pmass[idx]/rho_orig)*J;
	     rho_cur[idx] = rho_orig/J;
    }

    for(ParticleSubset::iterator iter = pset->begin();iter!=pset->end();iter++){
      particleIndex idx = *iter;

      pdTdt[idx] = 0.0;
      double PEAKI1_hardening = PEAKI1*FSLOPE + hardening_modulus*pPlasticStrain[idx];
      //const double cap_radius=-CR*FSLOPE*(pKappa[idx]-PEAKI1);
      double cap_radius=-CR*(FSLOPE*pKappa[idx]-PEAKI1_hardening);

      // Compute the rate of deformation tensor
      Matrix3 D = (velGrad[idx] + velGrad[idx].Transpose())*.5;
      Matrix3 tensorR, tensorU;
      deformationGradient_new[idx].polarDecompositionRMB(tensorU, tensorR);
      rotation[idx]=tensorR;
      D = (tensorR.Transpose())*(D*tensorR);

      // modify the bulk modulus based on the fluid effects
      double bulk_temp = exp(p3_crush_curve+p4_fluid_effect+pPlasticStrainVol[idx]+pElasticStrainVol[idx]);
      bulk = bulk + fluid_B0*
           ( exp(p3_crush_curve+p4_fluid_effect)-1.0 ) * bulk_temp
           / ( (bulk_temp-1.0)*(bulk_temp-1.0) );
      double lame = bulk - two_third*shear;

      // update the actual stress:
      Matrix3 unrotated_stress = (tensorR.Transpose())*(stress_old[idx]*tensorR);
      Matrix3 stress_diff = (Identity*lame*(D.Trace()*delT) + D*delT*2.0*shear);
      trial_stress[idx] = unrotated_stress + stress_diff;

      // compute shifted stress
      trial_stress[idx] = trial_stress[idx] - pBackStress[idx];

      // compute the value of the yield function for the trial stress
      f_trial[idx] = YieldFunction(trial_stress[idx],FSLOPE,pKappa[idx],cap_radius,PEAKI1_hardening);

      // initial assignment for the plastic strains, the position of the cap function,
      // and the backstress
      pPlasticStrain_new[idx] = pPlasticStrain[idx];
      pPlasticStrainVol_new[idx] = pPlasticStrainVol[idx];
      pElasticStrainVol_new[idx] = pElasticStrainVol[idx] + D.Trace()*delT;
      pKappa_new[idx] = pKappa[idx];
      pBackStress_new[idx] = pBackStress[idx];
      pBackStressIso_new[idx] = pBackStressIso[idx];

      // compute stress invariants for the trial stress
      double I1_trial,J2_trial;
      Matrix3 S_trial;
      computeInvariants(trial_stress[idx], S_trial, I1_trial, J2_trial);

      // check if the stress is elastic or plastic: If it is elastic the new stres is equal
      // to trial stress otherwise, the plasticity return algrithm would be used.
      Matrix3 deltaBackStress;
      Matrix3 deltaBackStressIso;
      deltaBackStress.set(0.0);
      deltaBackStressIso.set(0.0);

	     if (f_trial[idx]<0){ // ###1 (BEGIN: condition for elastic or plastic)

        // elastic step
	       stress_new[idx] = trial_stress[idx];

	     }else{ // ###1 (ELSE: condition for elastic or plastic)

        // Determine a characteristic length of the elastic zone
        double char_length_yield_surface;
        if (pKappa[idx]<-1.0e90){
          char_length_yield_surface = abs(2.0*(PEAKI1_hardening-FSLOPE*I1_trial));
        } else {
          if (PEAKI1_hardening-(pKappa[idx]-cap_radius) 
              < -2.0*(FSLOPE*(pKappa_new[idx]-cap_radius)-PEAKI1_hardening)){
              char_length_yield_surface = PEAKI1_hardening-(pKappa[idx]-cap_radius);
          } else {
              char_length_yield_surface = -2.0*(FSLOPE*(pKappa_new[idx]-cap_radius)-PEAKI1_hardening);
          }
        }
	       int condition_return_to_vertex=0;

	       if (I1_trial>PEAKI1_hardening/FSLOPE){ // ###2 (BEGIN: plasticity vertex treatment)

	         if (J2_trial<1.0e-10*char_length_yield_surface){
            // hydrostatic loading
            stress_new[idx] = Identity*PEAKI1_hardening/FSLOPE*one_third;
            condition_return_to_vertex = 1;
	         }else{
	           int counter_1_fix=0;
	           int counter_2_fix=0;
	           double P_component_1,P_component_2;
	           double relative_stress_to_vertex_1,relative_stress_to_vertex_2;
	           Matrix3 relative_stress_to_vertex,relative_stress_to_vertex_deviatoric;
	           Matrix3 unit_tensor_vertex_1;
	           Matrix3 unit_tensor_vertex_2;
	           Matrix3 P,M,P_deviatoric;
            // compute the relative trial stress in respect with the vertex
	           relative_stress_to_vertex = trial_stress[idx] - Identity*PEAKI1_hardening/FSLOPE*one_third;
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
            if ( ((relative_stress_to_vertex_1*P_component_2 + relative_stress_to_vertex_2*P_component_1)/
               (P_component_1*P_component_1) >=0 ) && ((relative_stress_to_vertex_1*P_component_2 +
               relative_stress_to_vertex_2*(-1.0)*P_component_1)/(P_component_1*P_component_1) >=0 ) ){
              stress_new[idx] = Identity*PEAKI1_hardening*one_third/FSLOPE;
              condition_return_to_vertex = 1;
            }
	         }
          double shear_inverse = 0.5/shear;
	         double lame_inverse = (-1.0)*lame/(2.0*shear*(2.0*shear+3.0*lame));
          Matrix3 diff_stress_iteration = trial_stress[idx] - stress_new[idx];
	         Matrix3 strain_iteration = (Identity*lame_inverse*(diff_stress_iteration.Trace()) +
                                     diff_stress_iteration*shear_inverse);
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

        } // ###2 (END: plasticity vertex treatment)

        if (condition_return_to_vertex == 0){ // ###3 (BEGIN CONDITION: nested return algorithm)

        Matrix3 trial_stress_loop;
        int num_subcycles = floor (sqrt(f_trial[idx])/(char_length_yield_surface/100) + 1);
        trial_stress[idx] = trial_stress[idx] + pBackStress[idx];
        trial_stress[idx] = trial_stress[idx] - stress_diff;
        stress_diff = stress_diff/num_subcycles;
        pKappa_new[idx] = pKappa[idx];
        pPlasticStrain_new[idx] = pPlasticStrain[idx];
        pPlasticStrainVol_new[idx] = pPlasticStrainVol[idx];
        pBackStress_new[idx] = pBackStress[idx];
        stress_new[idx] = trial_stress[idx];

        for (int subcycle_counter=0 ; subcycle_counter<=num_subcycles-1 ; subcycle_counter++){ // ###SUBCYCLING LOOP

          trial_stress[idx] = stress_new[idx];
          trial_stress[idx] = trial_stress[idx] + stress_diff;
          trial_stress[idx] = trial_stress[idx] - pBackStress_new[idx];
          trial_stress_loop = trial_stress[idx];

	         double gamma = 0.0;;
	         double gammaOld;
	         double I1_iteration,J2_iteration;
	         double beta_cap,FSLOPE_cap;
          double f_new_loop = 1e99;
          Matrix3 pBackStress_loop = pBackStress_new[idx];
          double pKappa_loop = pKappa_new[idx];
	         int max_number_of_iterations = 100;
	         int counter = 1;
	         Matrix3 P,M,G;
	         Matrix3 stress_iteration=trial_stress[idx];
	         Matrix3 S_iteration;
          Matrix3 plasStrain_loop;
          plasStrain_loop.set(0.0);

	         while( abs(f_new_loop)>1.0e-3*char_length_yield_surface
                 && counter<=max_number_of_iterations ){ // ###4 (BEGIN LOOP: nested return algorithm)

	           counter=counter+1;

            // compute the invariants of the trial stres in the loop
	           computeInvariants(stress_iteration, S_iteration, I1_iteration, J2_iteration);
	           if (I1_iteration>PEAKI1_hardening/FSLOPE){ // ###5 (BEGIN: fast return algorithm)

              stress_iteration = stress_iteration + Identity*I1_iteration*one_third*((PEAKI1_hardening-sqrt(J2_iteration))/
                                 (FSLOPE*I1_iteration)-1);

            } else if ( (I1_iteration<pKappa_loop-0.9*cap_radius)
                       || (I1_iteration<pKappa_loop && J2_iteration<0.01) ){ // ###5 (ELSE: fast return algorithm)

              Matrix3 stress_iteration_temp;
              double I1_iteration1;
              double I1_iteration2;
              double I1_iteration3;
              double f_iteration2;
              int counter_temp=0;
              computeInvariants(stress_iteration, S_iteration, I1_iteration, J2_iteration);
              f_iteration2=YieldFunction(stress_iteration,FSLOPE,pKappa_loop,cap_radius,PEAKI1_hardening);
              if(f_iteration2<0.0){
                I1_iteration1=pKappa_loop-2.0*cap_radius;
                I1_iteration2=I1_iteration;
              }else{
                I1_iteration1=I1_iteration;
                I1_iteration2=pKappa_loop;
                stress_iteration_temp = stress_iteration + Identity*I1_iteration*one_third*(I1_iteration2/I1_iteration-1.0);
                f_new_loop=YieldFunction(stress_iteration_temp,FSLOPE,pKappa_loop,cap_radius,PEAKI1_hardening);
                if (f_new_loop>=0.0){
                  Matrix3 S_iteration_temp;
                  double I1_iteration_temp;
                  double J2_iteration_temp;
                  double var1=1.0;
                  computeInvariants(stress_iteration_temp, S_iteration_temp, I1_iteration_temp, J2_iteration_temp);
                  while (f_new_loop>=0.0){
                    beta_cap = sqrt( 1.0 - (pKappa_loop-I1_iteration_temp)*(pKappa_loop-I1_iteration_temp)/
                               ( (cap_radius)*(cap_radius) ) );
                    var1=var1*0.5;
                    stress_iteration_temp = stress_iteration_temp + S_iteration_temp*(sqrt(var1)-1);
                    f_new_loop=YieldFunction(stress_iteration_temp,FSLOPE,pKappa_loop,cap_radius,PEAKI1_hardening);
                  }
                  beta_cap = sqrt( 1.0 - (pKappa_loop-I1_iteration)*(pKappa_loop-I1_iteration)/
                             ( (cap_radius)*(cap_radius) ) );
                  stress_iteration = stress_iteration + S_iteration*(sqrt(var1)-1);
                  computeInvariants(stress_iteration, S_iteration, I1_iteration, J2_iteration);
                  f_new_loop=YieldFunction(stress_iteration,FSLOPE,pKappa_loop,cap_radius,PEAKI1_hardening);
                  if (f_new_loop<=0.0){
                    I1_iteration1=pKappa_loop-2.0*cap_radius;
                    I1_iteration2=I1_iteration;
                  }
                }
              }
              I1_iteration3=(I1_iteration1+I1_iteration2)*0.5;
              stress_iteration_temp = stress_iteration + Identity*I1_iteration*one_third*(I1_iteration3/I1_iteration-1.0);
              f_iteration2=YieldFunction(stress_iteration_temp,FSLOPE,pKappa_loop,cap_radius,PEAKI1_hardening);

              while ((abs(f_iteration2)>1.0e-11*char_length_yield_surface && counter_temp<2000) || counter_temp==0) {
                counter_temp = counter_temp + 1;
                if (f_iteration2<0.0){
                  I1_iteration2=I1_iteration3;
                  I1_iteration3=(I1_iteration1+I1_iteration2)*0.5;
                } else {
                  I1_iteration1=I1_iteration3;
                  I1_iteration3=(I1_iteration1+I1_iteration2)*0.5;
                }
                stress_iteration_temp = stress_iteration + Identity*I1_iteration*one_third*(I1_iteration3/I1_iteration-1.0);
                f_iteration2=YieldFunction(stress_iteration_temp,FSLOPE,pKappa_loop,cap_radius,PEAKI1_hardening);
              }
              stress_iteration = stress_iteration_temp;

            }else if (I1_iteration<pKappa_new[idx]){ // ###5 (ELSE: fast return algorithm)

              beta_cap = sqrt( 1.0 - (pKappa_loop-I1_iteration)*(pKappa_loop-I1_iteration)/
                         ( (cap_radius)*(cap_radius) ) );
              stress_iteration = stress_iteration + S_iteration*((PEAKI1_hardening-FSLOPE*I1_iteration)*
                                 beta_cap/sqrt(J2_iteration)-1);

            }else{ // ###5 (ELSE: fast return algorithm)

	             stress_iteration = stress_iteration + S_iteration*((PEAKI1_hardening-FSLOPE*I1_iteration)/
                                sqrt(J2_iteration)-1);

            } // ###5 (END: fast return algorithm)

	           // compute the invariants of the trial stres in the loop returned to the yield surface
	           computeInvariants(stress_iteration, S_iteration, I1_iteration, J2_iteration);

	           if (I1_iteration>=pKappa_loop){ // ###6 (BEGIN: calculation of G and M tensors)

              // compute the gradient of the plastic potential
              G = Identity*(-2.0)*FSLOPE*(FSLOPE*I1_iteration-PEAKI1_hardening) + S_iteration;
              // compute the unit tensor in the direction of the plastic strain
              M = Identity*(-2.0)*FSLOPE_p*(FSLOPE_p*I1_iteration-PEAKI1_hardening) + S_iteration;
              M = M/M.Norm();

	           }else{ // ###6 (ELSE: calculation of G and M tensors)

              beta_cap = 1.0 - (pKappa_loop-I1_iteration)*(pKappa_loop-I1_iteration)/
                        ( (cap_radius)*(cap_radius) );
              FSLOPE_cap = -2.0*(FSLOPE*I1_iteration-PEAKI1_hardening)
                               *(FSLOPE*I1_iteration-PEAKI1_hardening)
                               *(pKappa_loop-I1_iteration)/( cap_radius*cap_radius ) 
                           -2.0*FSLOPE*beta_cap*(FSLOPE*I1_iteration-PEAKI1_hardening);
              // compute the gradient of the plastic potential
              G = Identity*FSLOPE_cap + S_iteration;
              // compute the unit tensor in the direction of the plastic strain
              M = G/G.Norm();

	           } // ###6 (END: calculation of G and M tensors)

            // compute the projection direction tensor
            double deltaBackStressIso_temp = exp(p3_crush_curve+pPlasticStrainVol_new[idx]);
            deltaBackStress = stress_iteration*kinematic_hardening_constant;
            deltaBackStressIso = Identity*( 3.0*fluid_B0*
                                (exp(p3_crush_curve+p4_fluid_effect)-1.0) * deltaBackStressIso_temp
                                /( (deltaBackStressIso_temp-1.0)*(deltaBackStressIso_temp-1.0) ) );
            double I1_M,J2_M;
            Matrix3 S_M;
            computeInvariants(M, S_M, I1_M, J2_M);
	           P = (Identity*lame*(M.Trace()) + M*2.0*shear)
                -deltaBackStressIso*M.Trace()-deltaBackStress*sqrt(J2_M);

            // compute the new value for gamma
            gammaOld = gamma;
	           gamma = ( G.Contract(trial_stress_loop-stress_iteration) )/( G.Contract(P) );

            // compute new trial stress in the loop
	           stress_iteration = trial_stress_loop - P*gamma;
            stress_iteration = stress_iteration + pBackStress_loop;
            trial_stress[idx] = trial_stress[idx] + pBackStress_loop;
            double I1_plasStrain,J2_plasStrain;
            Matrix3 S_plasStrain;
            plasStrain_loop = M*gamma;

            computeInvariants(plasStrain_loop, S_plasStrain, I1_plasStrain, J2_plasStrain);
            deltaBackStressIso_temp = exp(p3_crush_curve+pPlasticStrainVol_new[idx]);
            deltaBackStress = stress_iteration*kinematic_hardening_constant*sqrt(J2_plasStrain);
            deltaBackStressIso = Identity*( 3.0*fluid_B0*
                                (exp(p3_crush_curve+p4_fluid_effect)-1.0) * deltaBackStressIso_temp
                                /( (deltaBackStressIso_temp-1.0)*(deltaBackStressIso_temp-1.0) ) )*
                                plasStrain_loop.Trace();
            pBackStress_loop = pBackStress_new[idx] + deltaBackStress + deltaBackStressIso;
            stress_iteration = stress_iteration - pBackStress_loop;
            trial_stress[idx] = trial_stress[idx] - pBackStress_loop;
            trial_stress_loop = trial_stress[idx];
            double hardeningEns;
            if (I1_iteration>=pKappa_loop){
              hardeningEns = -2.0*hardening_modulus*(FSLOPE*I1_iteration-PEAKI1_hardening)/G.Norm();
            }else{
              beta_cap = 1.0 - (pKappa_loop-I1_iteration)*(pKappa_loop-I1_iteration)/
                         ( (cap_radius)*(cap_radius) );
              double pKappa_tempA = exp(p3_crush_curve+p4_fluid_effect+pPlasticStrainVol_new[idx]+(M*gamma).Trace());
              double pKappa_tempA1 = exp(p3_crush_curve+pPlasticStrainVol_new[idx]+(M*gamma).Trace());
              hardeningEns = -2.0*beta_cap*(FSLOPE*I1_iteration-PEAKI1_hardening)
                                 *hardening_modulus/G.Norm()
                             +2.0/G.Norm()*CR*(FSLOPE*I1_iteration-PEAKI1_hardening)
                                 *(FSLOPE*I1_iteration-PEAKI1_hardening)
                                 *(FSLOPE*I1_iteration-PEAKI1_hardening)
                                 *(pKappa_loop-I1_iteration)
                                 /( cap_radius*cap_radius*cap_radius*(1.0+FSLOPE*CR) )
                                 *( exp(-p1_crush_curve*(pKappa_loop-cap_radius-p0_crush_curve))
                                    /( p3_crush_curve*p1_crush_curve ) -
                                    3.0*fluid_B0*(exp(p3_crush_curve+p4_fluid_effect)-1.0)*pKappa_tempA
                                    /( (pKappa_tempA-1.0)*(pKappa_tempA-1.0) ) +
                                    3.0*fluid_B0*(exp(p3_crush_curve+p4_fluid_effect)-1.0)*pKappa_tempA1
                                    /( (pKappa_tempA1-1.0)*(pKappa_tempA1-1.0) ) )
                                 *M.Trace();
            }
            Matrix3 G_unit = G/G.Norm();
            gamma=gammaOld+(G_unit.Contract(P)/( G_unit.Contract(P)+hardeningEns ))*(gamma-gammaOld);
	           stress_iteration = trial_stress_loop - P*gamma;

            // update the position of the cap
            double pKappa_temp = exp(p3_crush_curve+p4_fluid_effect+pPlasticStrainVol_new[idx]+(M*gamma).Trace());
            double pKappa_temp1 = exp(p3_crush_curve+pPlasticStrainVol_new[idx]+(M*gamma).Trace());
            pKappa_loop = pKappa_loop + ( exp(-p1_crush_curve*(pKappa_loop-cap_radius-p0_crush_curve))
                             /( p3_crush_curve*p1_crush_curve ) -
                             3.0*fluid_B0*(exp(p3_crush_curve+p4_fluid_effect)-1.0)*pKappa_temp
                             /( (pKappa_temp-1.0)*(pKappa_temp-1.0) ) +
                             3.0*fluid_B0*(exp(p3_crush_curve+p4_fluid_effect)-1.0)*pKappa_temp1
                             /( (pKappa_temp1-1.0)*(pKappa_temp1-1.0) ) )
                             *(M*(gamma-gammaOld)).Trace()/(1.0+FSLOPE*CR);
            PEAKI1_hardening = PEAKI1*FSLOPE + hardening_modulus*(pPlasticStrain_new[idx]+(M*gamma).Norm());
            cap_radius=-CR*(FSLOPE*pKappa_loop-PEAKI1_hardening);

            f_new_loop=YieldFunction(stress_iteration,FSLOPE,pKappa_loop,cap_radius,PEAKI1_hardening);
            f_new_loop=sqrt(abs(f_new_loop));

	        } // ###4 (END LOOP: nested return algorithm)

         pBackStress_new[idx] = pBackStress_loop;
         pKappa_new[idx] = pKappa_loop;
         stress_new[idx] = stress_iteration;

         double shear_inverse = 0.5/shear;
	        double lame_inverse = (-1.0)*lame/(2.0*shear*(2.0*shear+3.0*lame));
         Matrix3 diff_stress_iteration = trial_stress_loop - stress_new[idx];
	        Matrix3 strain_iteration = (Identity*lame_inverse*(diff_stress_iteration.Trace()) +
                                    diff_stress_iteration*shear_inverse);
         // update total plastic strain magnitude
	        pPlasticStrain_new[idx] = pPlasticStrain_new[idx] + strain_iteration.Norm();
         // update volumetric part of the plastic strain magnitude
         pPlasticStrainVol_new[idx] = pPlasticStrainVol_new[idx] + strain_iteration.Trace();
         // update volumetric part of the elastic strain magnitude
         pElasticStrainVol_new[idx] = pElasticStrainVol_new[idx] - strain_iteration.Trace();
         stress_new[idx] = stress_new[idx] + pBackStress_new[idx];

       } // ###SUBCYCLING LOOP

       stress_new[idx] = stress_new[idx] - pBackStress_new[idx];

       } // ###3 (END CONDITION: nested return algorithm)

       double f_new;
	      f_new=YieldFunction(stress_new[idx],FSLOPE,pKappa_new[idx],cap_radius,PEAKI1_hardening);
       f_new=sqrt(abs(f_new));
       double J2_new,I1_new;
       Matrix3 S_new;
       computeInvariants(stress_new[idx], S_new, I1_new, J2_new);

       // send an error message to the host code if the new stress is not on the yield surface
	      if (abs(f_new)<1.0e-2*char_length_yield_surface) {}
	      else {
	        cerr<<"ERROR!  did not return to yield surface (Arenisca.cc)"<<endl;
	        cerr<<"sqrt(J2_new)= "<<sqrt(J2_new)<<endl;
	        cerr<<"I1_new= "<<I1_new<<endl;
	        cerr<<"pKappa_new[idx]= "<<pKappa_new[idx]<<endl;
         cerr<<"f_new= "<<f_new<<endl;
	        exit(1);
	      }

     } // ###1 (condition for elastic or plastic)

     // compute stress from the shifted stress
     stress_new[idx] = stress_new[idx] + pBackStress_new[idx];

    }//end loop over particles
    
    // final loop over all particles
    for(ParticleSubset::iterator iter = pset->begin();iter!=pset->end();iter++){

      particleIndex idx = *iter;
      stress_new[idx] = (rotation[idx]*stress_new[idx])*(rotation[idx].Transpose());
      // Compute wave speed + particle velocity at each particle,
      // store the maximum
      c_dil = sqrt((bulk+four_third*shear)/(rho_cur[idx]));
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
      // Compute artificial viscosity term
      if (flag->d_artificial_viscosity) {
        double dx_ave = (dx.x() + dx.y() + dx.z())*one_third;
        double c_bulk = sqrt(bulk/rho_cur[idx]);
        p_q[idx] = artificialBulkViscosity(D.Trace(), c_bulk, rho_cur[idx], dx_ave);
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

void Arenisca::computeInvariants(Matrix3& stress, Matrix3& S,  double& I1, double& J2){

  Matrix3 Identity;
  Identity.Identity();
  I1 = stress.Trace();
  S = stress - Identity*(1.0/3.0)*I1;
  J2 = 0.5*S.Contract(S);

}

void Arenisca::computeInvariants(const Matrix3& stress, Matrix3& S,  double& I1, double& J2){

  Matrix3 Identity;
  Identity.Identity();
  I1 = stress.Trace();
  S = stress - Identity*(1.0/3.0)*I1;
  J2 = 0.5*S.Contract(S);

}

 double Arenisca::YieldFunction(const Matrix3& stress, const double& FSLOPE, const double& kappa, const double& cap_radius, const double&PEAKI1){

  Matrix3 S;
  double I1,J2,b;
  computeInvariants(stress,S,I1,J2);
  if (I1>kappa){
    return J2 - ( FSLOPE*I1 - PEAKI1 ) * ( FSLOPE*I1 - PEAKI1 );
  }else{
    b = 1.0 - (kappa-I1)/(cap_radius)*(kappa-I1)/(cap_radius);
    return J2 - b * ( FSLOPE*I1 - PEAKI1 ) * ( FSLOPE*I1 - PEAKI1 );
  }

 }

 double Arenisca::YieldFunction(Matrix3& stress, const double& FSLOPE, const double& kappa, const double& cap_radius, const double&PEAKI1){

  Matrix3 S;
  double I1,J2,b;
  computeInvariants(stress,S,I1,J2);
  if (I1>kappa){
    return J2 - ( FSLOPE*I1 - PEAKI1 ) * ( FSLOPE*I1 - PEAKI1 );
  }else{
    b = 1.0 - (kappa-I1)/(cap_radius)*(kappa-I1)/(cap_radius);
    return J2 - b * ( FSLOPE*I1 - PEAKI1 ) * ( FSLOPE*I1 - PEAKI1 );
  }

 }

void Arenisca::carryForward(const PatchSubset* patches,
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

void Arenisca::addParticleState(std::vector<const VarLabel*>& from,
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

void Arenisca::addInitialComputesAndRequires(Task* task,
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

void Arenisca::addComputesAndRequires(Task* task,
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
Arenisca::addComputesAndRequires(Task* ,
                                   const MPMMaterial* ,
                                   const PatchSet* ,
                                   const bool ) const
{
}

double Arenisca::computeRhoMicroCM(double pressure,
                                      const double p_ref,
                                           const MPMMaterial* matl,
                                           double temperature,
                                           double rho_guess)
{
  double rho_orig = matl->getInitialDensity();
  double p_gauge = pressure - p_ref;
  double rho_cur;
  double bulk = d_initialData.B0;

  rho_cur = rho_orig/(1-p_gauge/bulk);

  return rho_cur;

#if 1
  cout << "NO VERSION OF computeRhoMicroCM EXISTS YET FOR Arenisca"<<endl;
#endif

}

void Arenisca::computePressEOSCM(double rho_cur,double& pressure,
                                         double p_ref,
                                         double& dp_drho, double& tmp,
                                         const MPMMaterial* matl,
                                         double temperature)
{

  double bulk = d_initialData.B0;
  double shear = d_initialData.G0;
  double rho_orig = matl->getInitialDensity();

  double p_g = .5*bulk*(rho_cur/rho_orig - rho_orig/rho_cur);
  pressure = p_ref + p_g;
  dp_drho  = .5*bulk*(rho_orig/(rho_cur*rho_cur) + 1./rho_orig);
  tmp = (bulk + 4.*shear/3.)/rho_cur;  // speed of sound squared


  cout << "NO VERSION OF computePressEOSCM EXISTS YET FOR Arenisca"
       << endl;
}

double Arenisca::getCompressibility()
{
  cout << "NO VERSION OF computePressEOSCM EXISTS YET FOR Arenisca"
       << endl;
  return 1.0;
}

void
Arenisca::initializeLocalMPMLabels()
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
