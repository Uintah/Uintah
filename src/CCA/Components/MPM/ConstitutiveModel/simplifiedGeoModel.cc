/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and
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

simplifiedGeoModel::simplifiedGeoModel(ProblemSpecP& ps, MPMFlags* Mflag)
  : ConstitutiveModel(Mflag)
{

  ps->require("alpha",d_initialData.alpha);
  ps->require("alpha_p",d_initialData.alpha_p);
  ps->require("hardening_modulus",d_initialData.hardening_modulus);
  ps->require("kappa_initial",d_initialData.kappa_initial);
  ps->require("cap_radius",d_initialData.cap_radius);
  ps->require("p0_crush_curve",d_initialData.p0_crush_curve);
  ps->require("p1_crush_curve",d_initialData.p1_crush_curve);
  ps->require("p3_crush_curve",d_initialData.p3_crush_curve);
  ps->require("k_o",d_initialData.k_o);
  ps->require("bulk_modulus",d_initialData.bulk_modulus);
  ps->require("shear_modulus",d_initialData.shear_modulus);
  initializeLocalMPMLabels();
}

simplifiedGeoModel::simplifiedGeoModel(const simplifiedGeoModel* cm)
  : ConstitutiveModel(cm)
{
  d_initialData.alpha = cm->d_initialData.alpha;
  d_initialData.alpha_p = cm->d_initialData.alpha_p;
  d_initialData.hardening_modulus = cm->d_initialData.hardening_modulus;
  d_initialData.kappa_initial = cm->d_initialData.kappa_initial;
  d_initialData.cap_radius = cm->d_initialData.cap_radius;
  d_initialData.p0_crush_curve = cm->d_initialData.p0_crush_curve;
  d_initialData.p1_crush_curve = cm->d_initialData.p1_crush_curve;
  d_initialData.p3_crush_curve = cm->d_initialData.p3_crush_curve;
  d_initialData.k_o = cm->d_initialData.k_o;
  d_initialData.bulk_modulus = cm->d_initialData.bulk_modulus;
  d_initialData.shear_modulus = cm->d_initialData.shear_modulus;
  initializeLocalMPMLabels();
}

simplifiedGeoModel::~simplifiedGeoModel()
{

  VarLabel::destroy(pPlasticStrainLabel);
  VarLabel::destroy(pPlasticStrainLabel_preReloc);
  VarLabel::destroy(pPlasticStrainVolLabel);
  VarLabel::destroy(pPlasticStrainVolLabel_preReloc);
  VarLabel::destroy(pKappaLabel);
  VarLabel::destroy(pKappaLabel_preReloc);

}

void simplifiedGeoModel::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","simplified_geo_model");
  }

  cm_ps->appendElement("alpha",d_initialData.alpha);
  cm_ps->appendElement("alpha_p",d_initialData.alpha_p);
  cm_ps->appendElement("hardening_modulus",d_initialData.hardening_modulus);
  cm_ps->appendElement("kappa_initial",d_initialData.kappa_initial);
  cm_ps->appendElement("cap_radius",d_initialData.cap_radius);
  cm_ps->appendElement("p0_crush_curve",d_initialData.p0_crush_curve);
  cm_ps->appendElement("p1_crush_curve",d_initialData.p1_crush_curve);
  cm_ps->appendElement("p3_crush_curve",d_initialData.p3_crush_curve);
  cm_ps->appendElement("k_o",d_initialData.k_o);
  cm_ps->appendElement("bulk_modulus",d_initialData.bulk_modulus);
  cm_ps->appendElement("shear_modulus",d_initialData.shear_modulus);

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
  ParticleVariable<double> pKappa;
  new_dw->allocateAndPut(pPlasticStrain,     pPlasticStrainLabel, pset);
  new_dw->allocateAndPut(pPlasticStrainVol,     pPlasticStrainVolLabel, pset);
  new_dw->allocateAndPut(pKappa,     pKappaLabel, pset);

  ParticleSubset::iterator iter = pset->begin();
  for(;iter != pset->end();iter++){
    pPlasticStrain[*iter] = 0.0;
    pPlasticStrainVol[*iter] = 0.0;
    pKappa[*iter] = d_initialData.kappa_initial;
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
  double bulk = d_initialData.bulk_modulus;
  double shear= d_initialData.shear_modulus;
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

void simplifiedGeoModel::computeStressTensor(const PatchSubset* patches,
                                           const MPMMaterial* matl,
                                           DataWarehouse* old_dw,
                                           DataWarehouse* new_dw)
{

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
    constParticleVariable<double> pKappa;
    ParticleVariable<double>  pKappa_new;
    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));
    old_dw->get(pPlasticStrain, pPlasticStrainLabel, pset);
    old_dw->get(pPlasticStrainVol, pPlasticStrainVolLabel, pset);
    old_dw->get(pKappa, pKappaLabel, pset);
    new_dw->allocateAndPut(pPlasticStrain_new,pPlasticStrainLabel_preReloc,pset);
    new_dw->allocateAndPut(pPlasticStrainVol_new,pPlasticStrainVolLabel_preReloc,pset);
    new_dw->allocateAndPut(pKappa_new,pKappaLabel_preReloc,pset);
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

    const double alpha = d_initialData.alpha;
    const double alpha_p = d_initialData.alpha_p;
    const double hardening_modulus = d_initialData.hardening_modulus;
    const double cap_radius = d_initialData.cap_radius;
    const double p0_crush_curve = d_initialData.p0_crush_curve;
    const double p1_crush_curve = d_initialData.p1_crush_curve;
    const double p3_crush_curve = d_initialData.p3_crush_curve;
    const double k_o = d_initialData.k_o;
    const double bulk = d_initialData.bulk_modulus;
    const double shear= d_initialData.shear_modulus;

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

      deformationGradient_new[idx]=(tensorL*delT+Identity)
	    *deformationGradient[idx];

	    J = deformationGradient_new[idx].Determinant();
	    if (J<=0){
	      cout<< "ERROR, negative J! "<<endl;
	      cout<<"J= "<<J<<endl;
	      cout<<"L= "<<tensorL<<endl;
	    }
	    // Update particle volumes
	    pvolume[idx]=(pmass[idx]/rho_orig)*J;
	    rho_cur[idx] = rho_orig/J;
    }

    for(ParticleSubset::iterator iter = pset->begin();iter!=pset->end();iter++){
      particleIndex idx = *iter;

      pdTdt[idx] = 0.0;
      const double k_o_hardening = k_o + hardening_modulus*pPlasticStrain[idx];

      // Compute the rate of deformation tensor
      Matrix3 D = (velGrad[idx] + velGrad[idx].Transpose())*.5;
      Matrix3 tensorR, tensorU;
      deformationGradient_new[idx].polarDecompositionRMB(tensorU, tensorR);
      rotation[idx]=tensorR;
      D = (tensorR.Transpose())*(D*tensorR);
      double lame = bulk - 2.0/3.0*shear;

      // update the actual stress:
      Matrix3 unrotated_stress = (tensorR.Transpose())*(stress_old[idx]*tensorR);
      trial_stress[idx] = unrotated_stress + (Identity*lame*(D.Trace()*delT) + D*delT*2.0*shear);

      // compute the value of the yield function for the trial stress
      f_trial[idx] = YieldFunction(trial_stress[idx],alpha,pKappa[idx],cap_radius,k_o_hardening);

      // initial assignment for the plastic strains and the position of the cap function
      pPlasticStrain_new[idx] = pPlasticStrain[idx];
      pPlasticStrainVol_new[idx] = pPlasticStrainVol[idx];
      pKappa_new[idx] = pKappa[idx];

      // compute stress invariants for the trial stress
      double I1_trial,J2_trial;
      Matrix3 S_trial;
      computeInvariants(trial_stress[idx], S_trial, I1_trial, J2_trial);

      // check if the stress is elastic or plastic: If it is elastic the new stres is equal
      // to trial stress otherwise, the plasticity return algrithm would be used.
	     if (f_trial[idx]<0){
	       stress_new[idx] = trial_stress[idx];
	     }else{
        // plasticity vertex treatment begins
	       int condition_return_to_vertex=0;
	       if (I1_trial>k_o_hardening/alpha){
	         if (J2_trial<0.00000001){
            stress_new[idx] = Identity*k_o_hardening/alpha/3.0;
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
	           relative_stress_to_vertex = trial_stress[idx] - Identity*k_o_hardening/alpha/3.0;
            // compute two unit tensors of the stress space
	           unit_tensor_vertex_1 = Identity/sqrt(3.0);
	           unit_tensor_vertex_2 = S_trial/sqrt(2.0*J2_trial);
            // compute the unit tensor in the direction of the plastic strain
            M = ( Identity*alpha_p + S_trial*(1.0/(2*sqrt(J2_trial))) )/sqrt(3*alpha_p*alpha_p + 0.5);
            // compute the projection direction tensor
            P = (Identity*lame*(M.Trace()) + M*2.0*shear);
            // compute the components of P tensor in respect with two unit_tensor_vertex
            P_component_1 = P.Trace()/sqrt(3.0);
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
            relative_stress_to_vertex_1 = relative_stress_to_vertex.Trace()/sqrt(3.0);
            relative_stress_to_vertex_deviatoric = relative_stress_to_vertex - 
                                                   unit_tensor_vertex_1*relative_stress_to_vertex_1;
            relative_stress_to_vertex_2 = relative_stress_to_vertex_deviatoric(counter_1_fix,counter_2_fix)/
                                          unit_tensor_vertex_2(counter_1_fix,counter_2_fix);
            // condition to determine if the stress_trial is in the vertex zone or not?
            if ( ((relative_stress_to_vertex_1*P_component_2 + relative_stress_to_vertex_2*P_component_1)/
               (P_component_1*P_component_1) >=0 ) && ((relative_stress_to_vertex_1*P_component_2 + 
               relative_stress_to_vertex_2*(-1.0)*P_component_1)/(P_component_1*P_component_1) >=0 ) ){
              stress_new[idx] = Identity*k_o_hardening/alpha/3.0;
              condition_return_to_vertex = 1;
            }
	         }
        }
        // plasticity vertex treatment ends
        // nested return algorithm begins
        if (condition_return_to_vertex == 0){
	         double gamma_tolerance = 0.0001;
	         double del_gamma = 100.;
	         double gamma = 0.0;;
	         double gamma_old;
	         double I1_iteration,J2_iteration;
	         double beta_cap,alpha_cap;
	         int max_number_of_iterations = 1000;
	         int counter = 0;
	         Matrix3 P,M,G;
	         Matrix3 stress_iteration=trial_stress[idx];
	         Matrix3 S_iteration;
	         // Multi-stage return loop begins
	         while(abs(del_gamma)>gamma_tolerance && counter<=max_number_of_iterations){
	           counter=counter+1;
	           // fast return algorithm to the yield surface
            // compute the invariants of the trial stres in the loop
	           computeInvariants(stress_iteration, S_iteration, I1_iteration, J2_iteration);
	           if (I1_iteration>k_o_hardening/alpha){
              stress_iteration = stress_iteration + Identity*I1_iteration/3.0*((k_o_hardening-sqrt(J2_iteration))/
                                 (alpha*I1_iteration)-1);
            }else if (I1_iteration<pKappa[idx]-0.9*cap_radius){
              stress_iteration = stress_iteration + Identity*I1_iteration/3.0*((pKappa[idx]-
                                 sqrt(cap_radius*cap_radius-J2_iteration))/I1_iteration-1);
              computeInvariants(stress_iteration, S_iteration, I1_iteration, J2_iteration);
              beta_cap = sqrt( 1.0 - (pKappa[idx]-I1_iteration)/(cap_radius)*
                         (pKappa[idx]-I1_iteration)/(cap_radius) );
              if (J2_iteration>0.00000001){
                stress_iteration = stress_iteration + S_iteration*((k_o_hardening-alpha*I1_iteration)*
                                   beta_cap/sqrt(J2_iteration)-1);
              }
            }else if (I1_iteration<pKappa[idx]){
              beta_cap = sqrt( 1.0 - (pKappa[idx]-I1_iteration)/(cap_radius)*(pKappa[idx]-I1_iteration)/
                         (cap_radius) );
              stress_iteration = stress_iteration + S_iteration*((k_o_hardening-alpha*I1_iteration)*
                                 beta_cap/sqrt(J2_iteration)-1);
            }else{
	            stress_iteration = stress_iteration + S_iteration*((k_o_hardening-alpha*I1_iteration)/
                                sqrt(J2_iteration)-1);
            }
	           // compute the invariants of the trial stres in the loop returned to the yield surface
	           computeInvariants(stress_iteration, S_iteration, I1_iteration, J2_iteration);
            // check if the stress state is in the cap zone or not?
	           if (I1_iteration>=pKappa[idx]){
              // compute the gradient of the plastic potential
	             G = Identity*alpha + S_iteration*(1.0/(2*sqrt(J2_iteration)));
              // compute the unit tensor in the direction of the plastic strain
	             M = ( Identity*alpha_p + S_iteration*(1.0/(2*sqrt(J2_iteration))) )/
                  sqrt(3*alpha_p*alpha_p + 0.5);
	           }else{
              if (J2_iteration<0.00000001){
                // compute the gradient of the plastic potential
                G = Identity*(-1.0);
                // compute the unit tensor in the direction of the plastic strain
                M = Identity*(-1.0)/sqrt(3);
              }else{
                beta_cap = sqrt( 1.0 - (pKappa[idx]-I1_iteration)/(cap_radius)*
                          (pKappa[idx]-I1_iteration)/(cap_radius) );
                alpha_cap = (alpha*I1_iteration-k_o_hardening)*(pKappa[idx]-I1_iteration)/
                            cap_radius/cap_radius/beta_cap + alpha*beta_cap;
                // compute the gradient of the plastic potential
                G = Identity*alpha_cap + S_iteration*(1.0/(2*sqrt(J2_iteration)));
                // compute the unit tensor in the direction of the plastic strain
                M = ( Identity*alpha_cap + S_iteration*(1.0/(2*sqrt(J2_iteration))) )/
                    sqrt(3*alpha_cap*alpha_cap + 0.5);
	             }
	           }
            // compute the projection direction tensor
	           P = (Identity*lame*(M.Trace()) + M*2.0*shear);
            // store the last value of gamma for calculation of the changes in gamma
	           gamma_old = gamma;
            // compute the new value for gamma
	           gamma = ( G.Contract(trial_stress[idx]-stress_iteration) )/( G.Contract(P) );
            // compute new trial stress in the loop
	           stress_iteration = trial_stress[idx] - P*gamma;
            // compute the changes of gamma in order to control converging 
            del_gamma = (gamma-gamma_old)/gamma;
	        }
         // Multi-stage return loop ends
         // compute the new stress state
	        stress_new[idx] = trial_stress[idx] - P*gamma;
       }
	      double f_new;
       // compute the value of the yield function for the new stress
	      f_new=YieldFunction(stress_new[idx],alpha,pKappa[idx],cap_radius,k_o_hardening);
       // send an error message to the host code if the new stress is not on the yield surface
	      if (abs(f_new)>10.0){
	        cerr<<"ERROR!  did not return to yield surface"<<endl;
	        double J2_new,I1_new;
	        Matrix3 S_new;
	        computeInvariants(stress_new[idx], S_new, I1_new, J2_new);
	        cerr<<"sqrt(J2_new)= "<<sqrt(J2_new)<<endl;
	        cerr<<"I1_new= "<<I1_new<<endl;
	      }
	      double shear_inverse = 1.0/2.0/shear;
	      double lame_inverse = 1.0/6.0/bulk/shear * ( 2.0/3.0*shear - bulk );
       Matrix3 diff_stress_iteration = trial_stress[idx] - stress_new[idx];
	      Matrix3 strain_iteration = (Identity*lame_inverse*(diff_stress_iteration.Trace()) + 
                                  diff_stress_iteration*2.0*shear_inverse);
       // update total plastic strain magnitude
	      pPlasticStrain_new[idx] = pPlasticStrain[idx] + strain_iteration.Norm();
       // update volumetric part of the plastic strain magnitude
       pPlasticStrainVol_new[idx] = pPlasticStrainVol[idx] + strain_iteration.Trace();
       // update the position of the cap
       pKappa_new[idx] = pKappa[idx] + 1.0/p3_crush_curve/p1_crush_curve * 
                         exp(-p1_crush_curve*(pKappa[idx]-cap_radius-p0_crush_curve))*
                         strain_iteration.Trace();
     } // nested return algorithm ends

    }//end loop over particles

    // final loop over all particles
    for(ParticleSubset::iterator iter = pset->begin();iter!=pset->end();iter++){

      particleIndex idx = *iter;
      stress_new[idx] = (rotation[idx]*stress_new[idx])*(rotation[idx].Transpose());
      // Compute wave speed + particle velocity at each particle,
      // store the maximum
      c_dil = sqrt((bulk+(4.0/3.0)*shear)/(rho_cur[idx]));
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
      // Compute artificial viscosity term
      if (flag->d_artificial_viscosity) {
        double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;
        double c_bulk = sqrt(bulk/rho_cur[idx]);
        p_q[idx] = artificialBulkViscosity(D.Trace(), c_bulk, rho_cur[idx], dx_ave);
      } else {
        p_q[idx] = 0.;
      }
      Matrix3 AvgStress = (stress_new[idx] + stress_old[idx])*.5;

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

 double simplifiedGeoModel::YieldFunction(const Matrix3& stress, const double& alpha, const double& kappa, const double& cap_radius, const double&k_o){

  Matrix3 S;
  double I1,J2,b;
  computeInvariants(stress,S,I1,J2);
  if (I1>kappa){
    return sqrt(J2) + alpha*I1 - k_o;
  }else{
    b = 1.0 - (kappa-I1)/(cap_radius)*(kappa-I1)/(cap_radius);
    if (b>0.0){
      return sqrt(J2) + alpha*I1*sqrt(b) - k_o*sqrt(b);
    }else{
      return 1.0;
    }
  }

 }

 double simplifiedGeoModel::YieldFunction(Matrix3& stress, const double& alpha, const double& kappa, const double& cap_radius, const double&k_o){

  Matrix3 S;
  double I1,J2,b;
  computeInvariants(stress,S,I1,J2);
  if (I1>kappa){
    return sqrt(J2) + alpha*I1 - k_o;
  }else{
    b = 1.0 - (kappa-I1)/(cap_radius)*(kappa-I1)/(cap_radius);
    if (b>0.0){
      return sqrt(J2) + alpha*I1*sqrt(b) - k_o*sqrt(b);
    }else{
      return 1.0;
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
  from.push_back(pKappaLabel);
  to.push_back(pPlasticStrainLabel_preReloc);
  to.push_back(pPlasticStrainVolLabel_preReloc);
  to.push_back(pKappaLabel_preReloc);

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
  task->computes(pKappaLabel, matlset);

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
  task->requires(Task::OldDW, pKappaLabel,    matlset, Ghost::None);
  task->computes(pPlasticStrainLabel_preReloc,  matlset);
  task->computes(pPlasticStrainVolLabel_preReloc,  matlset);
  task->computes(pKappaLabel_preReloc,  matlset);

}

void
simplifiedGeoModel::addComputesAndRequires(Task* ,
                                   const MPMMaterial* ,
                                   const PatchSet* ,
                                   const bool ) const
{
}

double simplifiedGeoModel::computeRhoMicroCM(double pressure,
                                      const double p_ref,
                                           const MPMMaterial* matl,
                                           double temperature,
                                           double rho_guess)
{
  double rho_orig = matl->getInitialDensity();
  double p_gauge = pressure - p_ref;
  double rho_cur;
  double bulk = d_initialData.bulk_modulus;

  rho_cur = rho_orig/(1-p_gauge/bulk);

  return rho_cur;

#if 1
  cout << "NO VERSION OF computeRhoMicroCM EXISTS YET FOR simplifiedGeoModel"<<endl;
#endif

}

void simplifiedGeoModel::computePressEOSCM(double rho_cur,double& pressure,
                                         double p_ref,
                                         double& dp_drho, double& tmp,
                                         const MPMMaterial* matl,
                                         double temperature)
{

  double bulk = d_initialData.bulk_modulus;
  double shear = d_initialData.shear_modulus;
  double rho_orig = matl->getInitialDensity();

  double p_g = .5*bulk*(rho_cur/rho_orig - rho_orig/rho_cur);
  pressure = p_ref + p_g;
  dp_drho  = .5*bulk*(rho_orig/(rho_cur*rho_cur) + 1./rho_orig);
  tmp = (bulk + 4.*shear/3.)/rho_cur;  // speed of sound squared


  cout << "NO VERSION OF computePressEOSCM EXISTS YET FOR simplifiedGeoModel"
       << endl;
}

double simplifiedGeoModel::getCompressibility()
{
  cout << "NO VERSION OF computePressEOSCM EXISTS YET FOR simplifiedGeoModel"
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
  pKappaLabel = VarLabel::create("p.kappa",
    ParticleVariable<double>::getTypeDescription());
  pKappaLabel_preReloc = VarLabel::create("p.kappa+",
    ParticleVariable<double>::getTypeDescription());

}
