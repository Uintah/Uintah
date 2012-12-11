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

#include <CCA/Components/MPM/ConstitutiveModel/HypoElastic.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Grid/Patch.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Math/MinMax.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Math/Matrix3.h>
#include <Core/Math/Short27.h>
#include <Core/Grid/Variables/NodeIterator.h> 
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Malloc/Allocator.h>
#include <fstream>
#include <iostream>

using namespace std;
using namespace Uintah;

HypoElastic::HypoElastic(ProblemSpecP& ps,MPMFlags* Mflag)
  : ConstitutiveModel(Mflag)
{
  ps->require("G",d_initialData.G);
  ps->require("K",d_initialData.K);

  // Thermal expansion coefficient 
  d_initialData.alpha=0.0;
  ps->get("alpha",d_initialData.alpha); // for thermal stress 

  if (flag->d_fracture) {
    // Read in fracture criterion and the toughness curve
    ProblemSpecP curve_ps = ps->findBlock("fracture_toughness_curve");
    if(curve_ps!=0) {
      crackPropagationCriterion="max_hoop_stress";        
      curve_ps->get("crack_propagation_criterion",crackPropagationCriterion);     

      if(crackPropagationCriterion!="max_hoop_stress" && 
         crackPropagationCriterion!="max_principal_stress" &&
         crackPropagationCriterion!="max_energy_release_rate" &&
         crackPropagationCriterion!="strain_energy_density" &&
         crackPropagationCriterion!="empirical_criterion") {
        cout << "Error: undefinded crack propagation criterion: "
             << crackPropagationCriterion 
             << " for hypoelastic materials. Program terminated."
             << endl;
        exit(1);       
      }     

      if(crackPropagationCriterion=="empirical_criterion") {
        // Get parameters p & q in the fracture locus equation
        // (KI/Ic)^p+(KII/KIIc)^q=1 and KIIc=r*KIc       
        p=q=2.0;  // Default elliptical fracture locus  
        r=-1.;
        curve_ps->get("p",p);
        curve_ps->get("q",q);
        curve_ps->get("r",r);
      }
            
      for(ProblemSpecP child_ps=curve_ps->findBlock("point"); child_ps!=0; 
          child_ps=child_ps->findNextBlock("point")) {
        double Vc,KIc,KIIc;
        child_ps->require("Vc",Vc);
        child_ps->require("KIc",KIc);
        if(r<0.) { // Input KIIc manually
          child_ps->get("KIIc",KIIc);
        }
        else { // The ratio of KIIc to KIc is a constant (r) 
          KIIc=r*KIc;
        }       
        d_initialData.Kc.push_back(Vector(Vc,KIc,KIIc));
      }           
    }
  }
}

HypoElastic::HypoElastic(const HypoElastic* cm) : ConstitutiveModel(cm)
{
  d_initialData.G = cm->d_initialData.G;
  d_initialData.K = cm->d_initialData.K;
  d_initialData.alpha = cm->d_initialData.alpha; // for thermal stress
  if (flag->d_fracture)
    d_initialData.Kc = cm->d_initialData.Kc;
}

HypoElastic::~HypoElastic()
{
}


void HypoElastic::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","hypo_elastic");
  }

  cm_ps->appendElement("G",d_initialData.G);
  cm_ps->appendElement("K",d_initialData.K);
  cm_ps->appendElement("alpha",d_initialData.alpha);

  // Still need to do the FRACTURE thing
}



HypoElastic* HypoElastic::clone()
{
  return scinew HypoElastic(*this);
}

void HypoElastic::initializeCMData(const Patch* patch,
                                   const MPMMaterial* matl,
                                   DataWarehouse* new_dw)
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  initSharedDataForExplicit(patch, matl, new_dw);

  if (flag->d_fracture) {
    // Put stuff in here to initialize each particle's
    // constitutive model parameters and deformationMeasure
    Matrix3 Identity, zero(0.);
    Identity.Identity();

    ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

    // for J-Integral
    ParticleVariable<Matrix3> pdispGrads;
    ParticleVariable<double>  pstrainEnergyDensity;
    new_dw->allocateAndPut(pdispGrads, lb->pDispGradsLabel, pset);
    new_dw->allocateAndPut(pstrainEnergyDensity, lb->pStrainEnergyDensityLabel, 
                           pset);

    ParticleSubset::iterator iter = pset->begin();
    for(;iter != pset->end();iter++){
      pdispGrads[*iter] = zero;
      pstrainEnergyDensity[*iter] = 0.0;
    }
  }

  computeStableTimestep(patch, matl, new_dw);
}

void HypoElastic::addParticleState(std::vector<const VarLabel*>& from,
                                   std::vector<const VarLabel*>& to)
{
  // Add the local particle state data for this constitutive model.
  if (flag->d_fracture) {
    from.push_back(lb->pDispGradsLabel);
    from.push_back(lb->pStrainEnergyDensityLabel);
    to.push_back(lb->pDispGradsLabel_preReloc);
    to.push_back(lb->pStrainEnergyDensityLabel_preReloc);
  }
}

void HypoElastic::computeStableTimestep(const Patch* patch,
                                           const MPMMaterial* matl,
                                           DataWarehouse* new_dw)
{
   // This is only called for the initial timestep - all other timesteps
   // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int dwi = matl->getDWIndex();
  ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
  constParticleVariable<double> pmass, pvolume;
  constParticleVariable<Vector> pvelocity;

  new_dw->get(pmass,     lb->pMassLabel,     pset);
  new_dw->get(pvolume,   lb->pVolumeLabel,   pset);
  new_dw->get(pvelocity, lb->pVelocityLabel, pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double G = d_initialData.G;
  double bulk = d_initialData.K;
  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
     particleIndex idx = *iter;

     // Compute wave speed at each particle, store the maximum
     c_dil = sqrt((bulk + 4.*G/3.)*pvolume[idx]/pmass[idx]);
     WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                      Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                      Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
    }
    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
}

void HypoElastic::computeStressTensor(const PatchSubset* patches,
                                        const MPMMaterial* matl,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw)
{
  double rho_orig = matl->getInitialDensity();
  for(int p=0;p<patches->size();p++){
    double se = 0.0;
    const Patch* patch = patches->get(p);
    //
    //  FIX  To do:  Read in table for vres
    //               Obtain and modify particle temperature (deg K)
    //

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<Vector> d_S(interpolator->size());
    vector<double> S(interpolator->size());

    Matrix3 velGrad,deformationGradientInc,Identity,zero(0.),One(1.);
    double c_dil=0.0,Jinc;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
    double onethird = (1.0/3.0);

    Identity.Identity();

    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};
    //double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;

    int dwi = matl->getDWIndex();
    // Create array for the particle position
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    constParticleVariable<Point> px;
    constParticleVariable<Matrix3> deformationGradient, pstress;
    ParticleVariable<Matrix3> pstress_new;
    ParticleVariable<Matrix3> deformationGradient_new;
    constParticleVariable<double> pmass, pvolume, ptemperature, pTempPrevious;
    ParticleVariable<double> pvolume_new;
    constParticleVariable<Vector> pvelocity;
    constParticleVariable<Matrix3> psize;
    constNCVariable<Vector> gvelocity;
    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    Ghost::GhostType  gac   = Ghost::AroundCells;
    
    old_dw->get(px,                  lb->pXLabel,                  pset);
    old_dw->get(pstress,             lb->pStressLabel,             pset);
    old_dw->get(psize,               lb->pSizeLabel,               pset);
    old_dw->get(pmass,               lb->pMassLabel,               pset);
    old_dw->get(pvolume,             lb->pVolumeLabel,             pset);
    old_dw->get(pvelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(ptemperature,        lb->pTemperatureLabel,        pset);
    old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);
    // for thermal stress
    old_dw->get(pTempPrevious,       lb->pTempPreviousLabel,       pset); 

    new_dw->get(gvelocity,lb->gVelocityStarLabel, dwi,patch, gac, NGN);

    constNCVariable<Vector> Gvelocity;
    constParticleVariable<Short27> pgCode;
    constParticleVariable<Matrix3> pdispGrads;
    constParticleVariable<double>  pstrainEnergyDensity;
    ParticleVariable<Matrix3> pdispGrads_new,pvelGrads;
    ParticleVariable<double> pstrainEnergyDensity_new;
    ParticleVariable<double> pdTdt,p_q;
    if (flag->d_fracture) {
      new_dw->get(Gvelocity,lb->GVelocityStarLabel, dwi, patch, gac, NGN);
      new_dw->get(pgCode,              lb->pgCodeLabel,              pset);
      old_dw->get(pdispGrads,          lb->pDispGradsLabel,          pset);
      old_dw->get(pstrainEnergyDensity,lb->pStrainEnergyDensityLabel,pset);
      new_dw->allocateAndPut(pvelGrads,lb->pVelGradsLabel,           pset);
      new_dw->allocateAndPut(pdispGrads_new, lb->pDispGradsLabel_preReloc,pset);
      new_dw->allocateAndPut(pstrainEnergyDensity_new,
                             lb->pStrainEnergyDensityLabel_preReloc, pset);
    }

    new_dw->allocateAndPut(pstress_new,     lb->pStressLabel_preReloc,   pset);
    new_dw->allocateAndPut(pvolume_new,     lb->pVolumeLabel_preReloc,   pset);
    new_dw->allocateAndPut(pdTdt,           lb->pdTdtLabel_preReloc,     pset);
    new_dw->allocateAndPut(p_q,             lb->p_qLabel_preReloc,       pset);
    new_dw->allocateAndPut(deformationGradient_new,
                           lb->pDeformationMeasureLabel_preReloc,        pset);
    ParticleVariable<Matrix3> vG;
    new_dw->allocateTemporary(vG, pset);

    CCVariable<double> vol_0_CC,dvol_CC;
    CCVariable<int> PPC;
    new_dw->allocateTemporary(vol_0_CC,  patch);
    new_dw->allocateTemporary(dvol_CC,  patch);
    new_dw->allocateTemporary(PPC,  patch);

    vol_0_CC.initialize(0.);
    dvol_CC.initialize(0.);
    PPC.initialize(0);

    double G    = d_initialData.G;
    double bulk = d_initialData.K;
    double alpha = d_initialData.alpha;   // for thermal stress    

    for(ParticleSubset::iterator iter = pset->begin();
                                        iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Assign zero internal heating by default - modify if necessary.
      pdTdt[idx] = 0.0;
      // Initialize velocity gradient
      velGrad.set(0.0);

      if(!flag->d_axisymmetric){
        // Get the node indices that surround the cell
        interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],deformationGradient[idx]);

        short pgFld[27];
        if (flag->d_fracture) {
         for(int k=0; k<27; k++){
           pgFld[k]=pgCode[idx][k];
         }
         computeVelocityGradient(velGrad,ni,d_S,oodx,pgFld,gvelocity,Gvelocity);
        } else {
         computeVelocityGradient(velGrad,ni,d_S,oodx,gvelocity);

        }
      } else {  // axi-symmetric kinematics
        // Get the node indices that surround the cell
        interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,
                                                                    psize[idx],deformationGradient[idx]);
        // x -> r, y -> z, z -> theta
        computeAxiSymVelocityGradient(velGrad,ni,d_S,S,oodx,gvelocity,px[idx]);
      }

      // Rate of particle temperature change for thermal stress
      double ptempRate=(ptemperature[idx]-pTempPrevious[idx])/delT; 
      // Calculate rate of deformation D, and deviatoric rate DPrime,
      // including effect of thermal strain
      Matrix3 D = (velGrad + velGrad.Transpose())*.5-Identity*alpha*ptempRate;

      vG[idx]=velGrad;

      IntVector cell_index;
      patch->findCell(px[idx],cell_index);

      vol_0_CC[cell_index]+=pvolume[idx];
      dvol_CC[cell_index]+=D.Trace()*pvolume[idx];
      PPC[cell_index]++;
    }

    double press_stab=0.;
    if(flag->d_doPressureStabilization) {
      press_stab=1.;
      for(CellIterator iter=patch->getCellIterator();!iter.done();iter++){
         IntVector c = *iter;
         dvol_CC[c]/=vol_0_CC[c];
      }
    }

    for(ParticleSubset::iterator iter = pset->begin();
                                        iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Rate of particle temperature change for thermal stress
      double ptempRate=(ptemperature[idx]-pTempPrevious[idx])/delT;
      // Calculate rate of deformation D, and deviatoric rate DPrime,
      // including effect of thermal strain
      IntVector cell_index;
      patch->findCell(px[idx],cell_index);

      Matrix3 D = (vG[idx] + vG[idx].Transpose())*.5-Identity*alpha*ptempRate;
      double DTrace = D.Trace();
      // Alter D to stabilize the pressure in each cell
      D = D + Identity*onethird*press_stab*(dvol_CC[cell_index] - DTrace);
      DTrace = D.Trace();
      Matrix3 DPrime = D - Identity*onethird*DTrace;

      // Compute the deformation gradient increment using the time_step
      // velocity gradient
      // F_n^np1 = dudx * dt + Identity
      deformationGradientInc = vG[idx] * delT + Identity;

      Jinc = deformationGradientInc.Determinant();

      // Update the deformation gradient tensor to its time n+1 value.
      deformationGradient_new[idx] = deformationGradientInc *
                                     deformationGradient[idx];

      // get the volumetric part of the deformation
      double J = deformationGradient[idx].Determinant();
      pvolume_new[idx]=Jinc*pvolume[idx];

      // Compute the local sound speed
      double rho_cur = rho_orig/J;
      c_dil = sqrt((bulk + 4.*G/3.)/rho_cur);
       
      // This is the (updated) Cauchy stress
      pstress_new[idx] = Matrix3(0.0);
//      pstress_new[idx] = pstress[idx] + 
//                         (DPrime*2.*G + Identity*bulk*DTrace)*delT;

      // Compute the strain energy for all the particles
      Matrix3 AvgStress = (pstress_new[idx] + pstress[idx])*.5;

      double e = (D(0,0)*AvgStress(0,0) +
                  D(1,1)*AvgStress(1,1) +
                  D(2,2)*AvgStress(2,2) +
              2.*(D(0,1)*AvgStress(0,1) +
                  D(0,2)*AvgStress(0,2) +
                  D(1,2)*AvgStress(1,2))) * pvolume_new[idx]*delT;

      se += e;

      if (flag->d_fracture) {
        pvelGrads[idx]=vG[idx];
        // Update particle displacement gradients
        pdispGrads_new[idx] = pdispGrads[idx] + vG[idx] * delT;
        // Update particle strain energy density 
        pstrainEnergyDensity_new[idx] = pstrainEnergyDensity[idx] + 
                                         e/pvolume_new[idx];
      }

      // Compute wave speed at each particle, store the maximum
      Vector pvelocity_idx = pvelocity[idx];
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity_idx.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pvelocity_idx.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pvelocity_idx.z()),WaveSpeed.z()));

      // Compute artificial viscosity term
      if (flag->d_artificial_viscosity) {
        double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;
        double c_bulk = sqrt(bulk/rho_cur);
        p_q[idx] = artificialBulkViscosity(DTrace, c_bulk, rho_cur, dx_ave);
      } else {
        p_q[idx] = 0.;
      }
    }  // end loop over particles

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
    
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(se),     lb->StrainEnergyLabel);
    }
    delete interpolator;
  }
}

void HypoElastic::carryForward(const PatchSubset* patches,
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
      new_dw->put(sum_vartype(0.),   lb->StrainEnergyLabel);
    }
  }
}

// Convert J-integral into stress intensity factors
// for Fracture
void 
HypoElastic::ConvertJToK(const MPMMaterial* matl,const string& stressState,
const Vector& J,const double& C,const Vector& D,Vector& SIF)
{                    
  // J--J-integral vector, 
  // C--Crack velocity, 
  // D--COD near crack tip in local coordinates.  

  double GT,CC,D1,D2,D3;
  double KI,KII,KIII,
         K1=0,K2=0,K3=0;

  GT=fabs(J.x());                     // total energy release rate
  CC=C*C;                             // square of crack propagating velocity
  D1=D.y(); D2=D.x(); D3=D.z();       // D1,D2,D3: opening, sliding, tearing COD

  // Material properties
  double rho,G,K,v,k;
  rho=matl->getInitialDensity();      // mass density
  G=d_initialData.G;                  // shear modulus
  K=d_initialData.K;                  // bulk modulus
  v=0.5*(3.*K-2.*G)/(3*K+G);          // Poisson ratio
  k = (stressState=="planeStress")? (3.-v)/(1.+v) : (3.-4.*v);

  // Calculate stress intensity
  if(D1==0. && D2==0. && D3==0.) {    // COD is zero
    KI=KII=KIII=0.;
  } 
  else { // COD is not zero
    // Parameters (A1,A2,A3) related to crack velocity
    double A1,A2,A3;
    if(sqrt(CC)<1.e-16) { // for stationary crack
      A1=(k+1.)/4.;
      A2=(k+1.)/4.;
      A3=1.;
    } 
    else { // for dynamic crack
      double Cs2=G/rho;
      double Cd2=(k+1.)/(k-1.)*Cs2;
      if(CC>Cs2) CC=Cs2;

      double B1=sqrt(1.-CC/Cd2);
      double B2=sqrt(1.-CC/Cs2);
      double DC=4.*B1*B2-(1.+B2*B2)*(1.+B2*B2);
      A1=B1*(1.-B2*B2)/DC;
      A2=B2*(1.-B2*B2)/DC;
      A3=1./B2;
    }

    // Solve stress intensity factors (absolute values)
    short  CASE=1;
    if(fabs(D2)>fabs(D1) && fabs(D2)>fabs(D3)) CASE=2;
    if(fabs(D3)>fabs(D1) && fabs(D3)>fabs(D2)) CASE=3;

    if(CASE==1) { // Mode I COD is dominated
      double g21=D2/D1;
      double g31=(1.-v)*D3/D1;
      K1=sqrt(2.*G*GT/(A1+A2*g21*g21+A3*g31*g31));
      K2=fabs(g21*K1);
      K3=fabs(g31*K1);
    }

    if(CASE==2) { // Mode II COD is dominated
      double g12=D1/D2;
      double g32=(1.-v)*D3/D2;
      K2=sqrt(2.*G*GT/(A1*g12*g12+A2+A3*g32*g32));
      K1=fabs(g12*K2);
      K3=fabs(g32*K2);
    }

    if(CASE==3) { // Mode III COD is dominated
      double g13=D1/D3/(1.-v);
      double g23=D2/D3/(1.-v);
      K3=sqrt(2.*G*GT/(A1*g13*g13+A2*g23*g23+A3));
      K1=fabs(g13*K3);
      K2=fabs(g23*K3);
    }

    // The signs of stress intensity are determined by the signs of the CODs
    double sign1 = D1>0.? 1.:-1.;
    double sign2 = D2>0.? 1.:-1.;
    double sign3 = D3>0.? 1.:-1.;
    KI   = D1==0. ? 0. : sign1*K1;
    KII  = D2==0. ? 0. : sign2*K2;
    KIII = D3==0. ? 0. : sign3*K3;
  }

  // Stress intensity vector
  SIF=Vector(KI,KII,KIII);
}


// Detect if crack propagates and the propagation direction
// for FRACTURE
short
HypoElastic::CrackPropagates(const double& Vc,const double& KI,
                             const double& KII, double& theta)
{
  /* Task 1: Determine fracture toughness KIc at velocity Vc
  */
  // Dynamic fracture toughness Kc(Vc,KIc,KIIC) 
  vector<Vector> Kc = d_initialData.Kc;
  int num = (int) Kc.size();
  double KIc=-1., KIIc=-1.0;
  if(Vc<=Kc[0].x()) { // Beyond the left bound
    KIc=Kc[0].y();
    KIIc=Kc[0].z();
  }
  else if(Vc>=Kc[num-1].x()) { // Beyond the right bound
    KIc=Kc[num-1].y();
    KIIc=Kc[num-1].z();
  }
  else { // In between 
    for(int i=0; i<num-1;i++) {
      double Vi=Kc[i].x();
      double Vj=Kc[i+1].x();
      if(Vc>=Vi && Vc<Vj) {
        double KIi=Kc[i].y();
        double KIj=Kc[i+1].y();
        double KIIi=Kc[i].z();
        double KIIj=Kc[i+1].z();
        KIc=KIi+(KIj-KIi)*(Vc-Vi)/(Vj-Vi);
        KIIc=KIIi+(KIIj-KIIi)*(Vc-Vi)/(Vj-Vi);  
        break;
      }
    } // End of loop over i
  }

  /* Task 2: Determine crack propagation direction (theta) and 
             the equivalent stress intensity factor (Kq)
  */         
  double Kq=-9e32;
  if(crackPropagationCriterion=="max_hoop_stress" ||
     crackPropagationCriterion=="max_energy_release_rate") {
    // Crack propagation direction        
    double sinTheta,cosTheta,value;
    if(KI==0.0 || (KI!=0. && fabs(KII/KI)>1000.)) { // Pure mode II
      cosTheta = 1./3.;
      sinTheta = (KII>=0.) ? -sqrt(8./9.) : sqrt(8./9.);
    }
    else {  // Mixed mode or pure mode I 
      double R=KII/KI;      
      cosTheta=(3.*R*R+sqrt(1.+8.*R*R))/(1.+9.*R*R);
      value=fabs(R*(3.*cosTheta-1.));
      sinTheta = (KII>=0.)? -value : value;
    }
    theta=asin(sinTheta);
      
    // Equivalent stress intensity
    double ct=cos(theta/2.);
    double st=sin(theta/2.);
    Kq=KI*pow(ct,3)-3*KII*ct*ct*st;
  } // End of max_hoop_stress criterion
 
  if(crackPropagationCriterion=="max_principal_stress") {
    if(KII==0. || (KII!=0. && fabs(KI/KII)>1000.)) { // Pure mode I
      theta=0.;
      Kq=KI; 
    }       
    else { // Mixed mode or pure mode II
      double R=KI/KII;      
      int sign = (KII>0.)? -1 : 1;
      double tanTheta2=(R+sign*sqrt(R*R+8.))/4.;
      double theta2=atan(tanTheta2);
      double ct=cos(theta2);
      double st=sin(theta2);
      Kq=KI*pow(ct,3)-3*KII*ct*ct*st;
      theta=2*theta2;
    }                             
  } // End of max_principal_stress criterion
  
  if(crackPropagationCriterion=="strain_energy_density") {
    // Calculate parameter k
    double k;
    string stressState="planeStress";     // Plane stress
    double G=d_initialData.G;             // Shear modulus
    double K=d_initialData.K;             // Bulk modulus
    double v=0.5*(3*K-2*G)/(3*K+G);     // Poisson ratio
    k = (stressState=="planeStress")? (3.-v)/(1.+v) : (3.-4.*v);
    
    // Crack propagation direction
    if(KII==0.0 || (KII!=0. && fabs(KI/KII)>1000.)) { // Pure mode I
      theta=0.0;
      Kq=KI;
    }
    else { // Mixed mode or pure mode II
      theta=CrackPropagationAngleFromStrainEnergyDensityCriterion(k,KI,KII);
      // Equivalent stress intensity
      double ct=cos(theta),st=sin(theta);
      double a11=(1+ct)*(k-ct);
      double a12=st*(2*ct-k+1);
      double a22=(k+1)*(1-ct)+(1+ct)*(3*ct-1);
      Kq=sqrt((a11*KI*KI+2*a12*KI*KII+a22*KII*KII)/2/(k-1)); 
    }  
  }  // End of strain_energy_density criterion       

  if(crackPropagationCriterion=="empirical_criterion") {
    if(KII==0. || (KII!=0. && fabs(KI/KII)>1000.)) { // Pure mode I
      theta=0.;
      Kq=KI;
    }
    else { // For mixed mode or pure mode II, use maximum pricipal criterion to
           // determine the crack propagation direction and the emprical
           // criterion to determine if crack will propagate.
      double R=KI/KII;
      int sign = (KII>0.)? -1 : 1;
      double tanTheta2=(R+sign*sqrt(R*R+8.))/4.;
      theta=2*atan(tanTheta2);
      Kq=(pow(KI/KIc,p)+pow(KII/KIIc,q))*KIc;
    }
  } // End of empirical_criterion

  if(Kq>=KIc)
    return 1;
  else
    return 0;
}

// Obtain crack propagation angle numerically from strain energy density criterion
// for FRACTURE
double
HypoElastic::CrackPropagationAngleFromStrainEnergyDensityCriterion(const double& k,
                const double& KI, const double& KII)
{
  double errF=1.e-6,errV=1.e-2,PI=3.141592654;
  double a,b,c,fa,fb,fc;
  
  double A=-PI, B=PI;   // The region of the roots 
  int n=36;             // Divide [A,B] into n intervals
  double h=(B-A)/n;     // Subinterval length
  
  double theta=0.0;
  double theta0=atan(KI/KII);
  vector<double> root;  // Store the solutions of the equation
  // Solve the equation numerically
  for(int i=0; i<n; i++) { // Loop over the whole interval [A,B]
    a=A+i*h;
    b=A+(i+1)*h;
    fa=(k-1)*sin(a-2*theta0)-2*sin(2*(a-theta0))-sin(2*a);
    fb=(k-1)*sin(b-2*theta0)-2*sin(2*(b-theta0))-sin(2*b);
    
    // Find the root in [a,b)
    if(fabs(fa)<errF) { // Where f(a)=0
      root.push_back(a);
    }
    else if(fa*fb<0.) { // There is a root in (a,b)
      double cp=2*B;    // Set the value beyond [A,B]      
      for(int j=0; j<32768; j++) { // 32768=2^15 (a big int) 
        c=b-(a-b)*fb/(fa-fb);
        fc=(k-1)*sin(c-2*theta0)-2*sin(2*(c-theta0))-sin(2*c);
        if(fabs(fc)<errF || fabs(c-cp)<errV) { // c is the root
          root.push_back(c);
          break;
        }
        else { // Record the cross point with axis x
          cp=c;
        }  
          
        // Narrow the region of the root
        if(fc*fa<0.) { // The root is in (a,c)
          fb=fc; b=c;
        }
        else if(fc*fb<0.) { // The root is in (c,b)
          fa=fc; a=c;
        }
      } // End of loop over j
    } // End of if(fa*fb<0.)
  } // End of loop over i
    
  // Select the direction from the solutions 
  // along which there exists the minimum strain energy density   
  int count=0;
  double S0=0.0;
  for(int i=0;i<(int)root.size();i++) {
    double r=root[i];       

    // The signs of propagation angle and KII must be opposite 
    if(KII*r>0.) continue;

    // Calculate the second derivative of the strain energy density
    double sr=sin(r),cr=cos(r),sr2=sin(2*r),cr2=cos(2*r);
    double dsdr2=KI*KI*((1-k)*cr+2*cr2)-2*KI*KII*(4*sr2+(1-k)*sr)+
                 KII*KII*((k-1)*cr-6*cr2); 
    if(dsdr2>0.) { 
      // Determine propagation angle by comparison of strain energy density. 
      // Along the angle there exists the minimum strain energy density. 
      double S=(1+cr)*(k-cr)*KI*KI+2*sr*(2*cr-k+1)*KI*KII+
               ((k+1)*(1-cr)+(1+cr)*(3*cr-1))*KII*KII; 
      if(count==0 || (count>0 && S<S0)) {
        theta=r;  
        S0=S;
        count++;
      } 
    }
  } // Enf of loop over i
  root.clear();

  return theta;
}

void HypoElastic::addComputesAndRequires(Task* task,
                                         const MPMMaterial* matl,
                                         const PatchSet* patches) const
{
  // Add the computes and requires that are common to all explicit 
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForHypoExplicit(task, matlset, patches);
  
  Ghost::GhostType gnone = Ghost::None;
  // for thermal stress
  task->requires(Task::OldDW, lb->pTempPreviousLabel, matlset, gnone); 

  // Other constitutive model and input dependent computes and requires
  if (flag->d_fracture) {
    Ghost::GhostType  gnone = Ghost::None;
    task->requires(Task::OldDW, lb->pDispGradsLabel,           matlset, gnone);
    task->requires(Task::OldDW, lb->pStrainEnergyDensityLabel, matlset, gnone);
    
    task->computes(lb->pDispGradsLabel_preReloc,             matlset);
    task->computes(lb->pVelGradsLabel,                       matlset);
    task->computes(lb->pStrainEnergyDensityLabel_preReloc,   matlset);
  }
}

void 
HypoElastic::addComputesAndRequires(Task*,const MPMMaterial*, const PatchSet*,
                                    const bool ) const
{
}

double HypoElastic::computeRhoMicroCM(double pressure,
                                      const double p_ref,
                                      const MPMMaterial* matl, 
                                      double temperature,
                                      double rho_guess)
{
  double rho_orig = matl->getInitialDensity();
  //double p_ref=101325.0;
  double p_gauge = pressure - p_ref;
  double rho_cur;
  //double G = d_initialData.G;
  double bulk = d_initialData.K;

  rho_cur = rho_orig/(1-p_gauge/bulk);

  return rho_cur;

#if 0
  cout << "NO VERSION OF computeRhoMicroCM EXISTS YET FOR HypoElastic"
       << endl;
#endif
}

void HypoElastic::computePressEOSCM(double rho_cur, double& pressure,
                                    double p_ref,
                                    double& dp_drho,      double& tmp,
                                    const MPMMaterial* matl, 
                                    double temperature)
{

  //double G = d_initialData.G;
  double bulk = d_initialData.K;
  double rho_orig = matl->getInitialDensity();

  double p_g = bulk*(1.0 - rho_orig/rho_cur);
  pressure = p_ref + p_g;
  dp_drho  = bulk*rho_orig/(rho_cur*rho_cur);
  tmp = bulk/rho_cur;  // speed of sound squared

#if 0
  cout << "NO VERSION OF computePressEOSCM EXISTS YET FOR HypoElastic"
       << endl;
#endif
}

double HypoElastic::getCompressibility()
{
  return 1.0/d_initialData.K;
}


namespace Uintah {

} // End namespace Uintah
