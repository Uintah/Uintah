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


#include "SuvicI.h"        
#include <Core/Math/FastMatrix.h>       
#include <Core/Exceptions/ProblemSetupException.h>
#include <math.h>

using namespace std;
using namespace Uintah;

SuvicI::SuvicI(ProblemSpecP& ps)
{

  d_SV.ba1=75e6; /*< coefficient of backstress evol [Pa]  */
  d_SV.bq=1.0;      /*< exponent of backstress evol NOTUSED */
  d_SV.bc=1.0e6;     /*< normalizing back stress [Pa] NOTUSED*/
  d_SV.b0=0.1e6;     /*< coeff of saturation backstress [Pa] */
  d_SV.de0=7.794e-08;     /*< ref strain rate [1/sec] */
  d_SV.bn=4.0;     /*< exponent of backstress */

  d_SV.a=5e+09;       /*< normalizing inelastic strain rate [1/sec] */
  d_SV.q=67500.0;      /*< activation energy [J/Mole] */
  d_SV.t=269.15;      /*< temperature [K]; TODO- where defined elsewhere*/
  d_SV.xn=4.0;     /*< exponent of inelastic strain rate [1/sec] */

  d_SV.r0=0.8e6;     /*< coef of yield stress saturation [Pa] */
  d_SV.rm=4.0;     /*< exponent of yield stress */
  d_SV.rai=1600e6;    /*< A3 [Pa] */

  d_SV.xmn=4.0;    /*< exponent in K */
  d_SV.xai=95.0e6;    /*< A5 [Pa] */
  d_SV.rr=8.3144;      /*< R Universal Gas Constant [8.3144J/(mole . K)] */

  d_SV.s0=1.0e6;      /*< coeff of saturation of stress [Pa]*/
  d_SV.initial_yield=0.0; /* initial yield stress [Pa] */
  d_SV.initial_drag=0.05e6;  /*initial drag stress - never zero! [Pa]*/
  
  d_SV.theta=0.0;  /*0 (explict) to 1(implicit)*/

  ps->get("coeff_backstress_evol",              d_SV.ba1);
  ps->get("exponent_backstress_evol",           d_SV.bq);
  ps->get("normalizing_backstress",             d_SV.bc);
  ps->get("coeff_saturation_backstress",        d_SV.b0);
  ps->get("exponent_backstress",                d_SV.bn);
  
  ps->get("ref_strainrate",                     d_SV.de0);
  ps->get("normalizing_inelastic_strainrate",   d_SV.a);
  ps->get("activation_energy",                  d_SV.q);
  ps->get("universal_gas_constant",             d_SV.rr);
  ps->get("temperature",                        d_SV.t);
  ps->get("exponent_inelastic_strainrate",      d_SV.xn);
  
  ps->get("coeff_yieldstress_saturation",       d_SV.r0);
  ps->get("exponent_yieldstress",               d_SV.rm);
  ps->get("coeff_yieldstress_evol",             d_SV.rai);
  
  ps->get("exponent_dragstress",                d_SV.xmn);
  ps->get("coeff_dragstress_evol",              d_SV.xai);
  ps->get("coeff_stress_saturation",            d_SV.s0);
  
  ps->get("intial_drag",            d_SV.initial_drag);
  ps->get("initial_yield",          d_SV.initial_yield);
  ps->get("integration_parameter_theta",          d_SV.theta);
  

  // Initialize internal variable labels for evolution
  pYieldLabel = VarLabel::create("p.yield",
        ParticleVariable<double>::getTypeDescription());
  pYieldLabel_preReloc = VarLabel::create("p.yield+",
        ParticleVariable<double>::getTypeDescription());
  pDragLabel = VarLabel::create("p.drag",
        ParticleVariable<double>::getTypeDescription());
  pDragLabel_preReloc = VarLabel::create("p.drag+",
        ParticleVariable<double>::getTypeDescription());
  pBackStressLabel = VarLabel::create("p.backstress",
        ParticleVariable<Matrix3>::getTypeDescription());
  pBackStressLabel_preReloc = VarLabel::create("p.backstress+",
        ParticleVariable<Matrix3>::getTypeDescription());
}
         
SuvicI::SuvicI(const SuvicI* cm)
{
  
  d_SV.ba1 = cm->d_SV.ba1;
  d_SV.bq = cm->d_SV.bq;
  d_SV.bc = cm->d_SV.bc;
  d_SV.b0 = cm->d_SV.b0;
  d_SV.de0 = cm->d_SV.de0;
  d_SV.bn = cm->d_SV.bn;

  d_SV.a = cm->d_SV.a;
  d_SV.q = cm->d_SV.q;
  d_SV.t = cm->d_SV.t;
  d_SV.xn = cm->d_SV.xn;

  d_SV.r0 = cm->d_SV.r0;
  d_SV.rm = cm->d_SV.rm;
  d_SV.rai = cm->d_SV.rai;

  d_SV.xmn = cm->d_SV.xmn;
  d_SV.xai = cm->d_SV.xai;
  d_SV.rr = cm->d_SV.rr;

  d_SV.s0 = cm->d_SV.s0;
  d_SV.initial_yield = cm->d_SV.initial_yield;
  d_SV.initial_drag = cm->d_SV.initial_drag;
  d_SV.theta = cm->d_SV.theta;


  // Initialize internal variable labels for evolution
  pYieldLabel = VarLabel::create("p.yield",
        ParticleVariable<double>::getTypeDescription());
  pYieldLabel_preReloc = VarLabel::create("p.yield+",
        ParticleVariable<double>::getTypeDescription());
  pDragLabel = VarLabel::create("p.drag",
        ParticleVariable<double>::getTypeDescription());
  pDragLabel_preReloc = VarLabel::create("p.drag+",
        ParticleVariable<double>::getTypeDescription());
  pBackStressLabel = VarLabel::create("p.backstress",
        ParticleVariable<Matrix3>::getTypeDescription());
  pBackStressLabel_preReloc = VarLabel::create("p.backstress+",
        ParticleVariable<Matrix3>::getTypeDescription());

}
         
SuvicI::~SuvicI()
{
  VarLabel::destroy(pYieldLabel);
  VarLabel::destroy(pYieldLabel_preReloc);
  VarLabel::destroy(pDragLabel);
  VarLabel::destroy(pDragLabel_preReloc);
  VarLabel::destroy(pBackStressLabel);
  VarLabel::destroy(pBackStressLabel_preReloc);

}

void SuvicI::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP plastic_ps = ps->appendChild("visco_plasticity_model");
  plastic_ps->setAttribute("type","suvic_i");
  
  plastic_ps->appendElement("coeff_backstress_evol",         d_SV.ba1);
  plastic_ps->appendElement("exponent_backstress_evol",      d_SV.bq);
  plastic_ps->appendElement("normalizing_backstress",        d_SV.bc);
  plastic_ps->appendElement("coeff_saturation_backstress",   d_SV.b0);
  plastic_ps->appendElement("exponent_backstress",           d_SV.bn);
  
  plastic_ps->appendElement("ref_strainrate",                d_SV.de0);
  plastic_ps->appendElement("normalizing_inelastic_strainrate",   d_SV.a);
  plastic_ps->appendElement("activation_energy",             d_SV.q);
  plastic_ps->appendElement("universal_gas_constant",        d_SV.rr);
  plastic_ps->appendElement("temperature",                   d_SV.t);
  plastic_ps->appendElement("exponent_inelastic_strainrate", d_SV.xn);

  plastic_ps->appendElement("coeff_yieldstress_saturation",   d_SV.r0);
  plastic_ps->appendElement("exponent_yieldstress",          d_SV.rm);
  plastic_ps->appendElement("coeff_yieldstress_evol",        d_SV.rai);
  
  plastic_ps->appendElement("exponent_dragstress",           d_SV.xmn);
  plastic_ps->appendElement("coeff_dragstress_evol",         d_SV.xai);
  
  plastic_ps->appendElement("coeff_stress_saturation",       d_SV.s0);
  plastic_ps->appendElement("initial_yield",            d_SV.initial_yield);
  plastic_ps->appendElement("initial_drag",             d_SV.initial_drag);
  plastic_ps->appendElement("integration_parameter_theta",              d_SV.theta);

}
         
void 
SuvicI::addInitialComputesAndRequires(Task* task,
                                                   const MPMMaterial* matl,
                                                   const PatchSet*) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes(pYieldLabel, matlset);
  task->computes(pDragLabel, matlset);
  task->computes(pBackStressLabel, matlset);
}

void 
SuvicI::addComputesAndRequires(Task* task,
                                            const MPMMaterial* matl,
                                            const PatchSet*) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW, pYieldLabel, matlset,Ghost::None);
  task->computes(pYieldLabel_preReloc, matlset);
  task->requires(Task::OldDW, pDragLabel, matlset,Ghost::None);
  task->computes(pDragLabel_preReloc, matlset);
  task->requires(Task::OldDW, pBackStressLabel, matlset,Ghost::None);
  task->computes(pBackStressLabel_preReloc, matlset);

}

void 
SuvicI::addComputesAndRequires(Task* task,
                                   const MPMMaterial* matl,
                                   const PatchSet*,
                                   bool ) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::ParentOldDW, pYieldLabel, matlset,Ghost::None);
  task->requires(Task::ParentOldDW, pDragLabel, matlset,Ghost::None);
  task->requires(Task::ParentOldDW, pBackStressLabel, matlset,Ghost::None);
}

void 
SuvicI::addParticleState(std::vector<const VarLabel*>& from,
                                      std::vector<const VarLabel*>& to)
{
  from.push_back(pYieldLabel);
  to.push_back(pYieldLabel_preReloc);
  from.push_back(pDragLabel);
  to.push_back(pDragLabel_preReloc);
  from.push_back(pBackStressLabel);
  to.push_back(pBackStressLabel_preReloc);
}

void 
SuvicI::allocateCMDataAddRequires(Task* task,
                                               const MPMMaterial* matl,
                                               const PatchSet* ,
                                               MPMLabel* ) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::NewDW, pYieldLabel_preReloc, matlset, Ghost::None);
  task->requires(Task::NewDW, pDragLabel_preReloc, matlset, Ghost::None);
  task->requires(Task::NewDW, pBackStressLabel_preReloc, matlset, Ghost::None);
  //task->requires(Task::OldDW, pAlphaLabel, matlset, Ghost::None);
}

void SuvicI::allocateCMDataAdd(DataWarehouse* new_dw,
                                            ParticleSubset* addset,
                                            map<const VarLabel*, ParticleVariableBase*>* newState,
                                            ParticleSubset* delset,
                                            DataWarehouse* )
{
  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
 
  ParticleVariable<double> pYield;
  constParticleVariable<double> o_Yield;
  ParticleVariable<double> pDrag;
  constParticleVariable<double> o_Drag;
  ParticleVariable<Matrix3> pBackStress;
  constParticleVariable<Matrix3> o_BackStress;


  new_dw->allocateTemporary(pYield,addset);
  new_dw->allocateTemporary(pDrag,addset);
  new_dw->allocateTemporary(pBackStress,addset);

  new_dw->get(o_Yield,pYieldLabel_preReloc,delset);
  new_dw->get(o_Drag,pDragLabel_preReloc,delset);
  new_dw->get(o_BackStress,pBackStressLabel_preReloc,delset);
  //old_dw->get(o_Alpha,pAlphaLabel,delset);

  ParticleSubset::iterator o,n = addset->begin();
  for(o = delset->begin(); o != delset->end(); o++, n++) {
    pYield[*n] = o_Yield[*o];
    pDrag[*n] = o_Drag[*o];
    pBackStress[*n] = o_BackStress[*o];
  }

  (*newState)[pYieldLabel]=pYield.clone();
  (*newState)[pDragLabel]=pDrag.clone();
  (*newState)[pBackStressLabel]=pBackStress.clone();
}

// initial values may not be equal to zero!!!
void 
SuvicI::initializeInternalVars(ParticleSubset* pset,
                                            DataWarehouse* new_dw)
{
  new_dw->allocateAndPut(pYield_new, pYieldLabel, pset);
  new_dw->allocateAndPut(pDrag_new, pDragLabel, pset);
  new_dw->allocateAndPut(pBackStress_new, pBackStressLabel, pset);
  ParticleSubset::iterator iter = pset->begin();
  for(;iter != pset->end(); iter++) {
    pYield_new[*iter] = d_SV.initial_yield;
    pDrag_new[*iter] = d_SV.initial_drag;
    pBackStress_new[*iter] = 0.0;
  }
}

void 
SuvicI::getInternalVars(ParticleSubset* pset,
                                     DataWarehouse* old_dw) 
{
  old_dw->get(pYield, pYieldLabel, pset);
  old_dw->get(pDrag, pDragLabel, pset);
  old_dw->get(pBackStress, pBackStressLabel, pset);
}

void 
SuvicI::allocateAndPutInternalVars(ParticleSubset* pset,
                                                DataWarehouse* new_dw) 
{
  new_dw->allocateAndPut(pYield_new, pYieldLabel_preReloc, pset);
  new_dw->allocateAndPut(pDrag_new, pDragLabel_preReloc, pset);
  new_dw->allocateAndPut(pBackStress_new, pBackStressLabel_preReloc, pset);
}

//should be ok....
void
SuvicI::allocateAndPutRigid(ParticleSubset* pset,
                                         DataWarehouse* new_dw)
{
  new_dw->allocateAndPut(pYield_new, pYieldLabel_preReloc, pset);
  new_dw->allocateAndPut(pDrag_new, pDragLabel_preReloc, pset);
  new_dw->allocateAndPut(pBackStress_new, pBackStressLabel_preReloc, pset);
  
  // Initializing to zero for the sake of RigidMPM's carryForward
  ParticleSubset::iterator iter = pset->begin();
  for(;iter != pset->end(); iter++){
     pYield_new[*iter] = 0.0;
     pDrag_new[*iter] = 0.0;
     pBackStress_new[*iter] = 0.0;
  }
}

void
SuvicI::updateElastic(const particleIndex idx)
{
  pYield_new[idx] = pYield[idx];
  pDrag_new[idx] = pDrag[idx];
  pBackStress_new[idx] = pBackStress[idx];
}


//calculate deviatoric stress and the direction of the reduced stress 
void
SuvicI::computeNij(Matrix3& nij, 
                        Matrix3& reducedEta, 
                        double& xae, 
                        const particleIndex idx, 
                        const Matrix3 pStress, 
                        const Matrix3 tensorR,
                        const int implicitFlag)
{
  Matrix3 one; one.Identity();
  Matrix3 tensorEta, rotatedBack;
  Matrix3 tmp1;
  double tmp2=1.5;  //no *operator for scalar * Matrix3.....
  
  if (implicitFlag==0)
      rotatedBack = (tensorR*pBackStress[idx])*(tensorR.Transpose());
  else
      rotatedBack = pBackStress_new[idx];
      
  tensorEta = pStress - one*(pStress.Trace()/3.0);  
  reducedEta = tensorEta - rotatedBack;
  xae = sqrt(reducedEta.NormSquared()*1.5);
//   cout << "xae=" << xae << "\n";
  nij = (one*tmp2)*reducedEta/xae;
}

double 
SuvicI::computeFlowStress(const particleIndex idx,
                          const Matrix3 pStress,
                          const Matrix3 tensorR,
                          const int implicitFlag)
{
  Matrix3 one; one.Identity();
  Matrix3 rotatedBack;
  
  if (implicitFlag ==0)
      rotatedBack = (tensorR*pBackStress[idx])*(tensorR.Transpose());
  else
      rotatedBack = pBackStress[idx];
      
  Matrix3 tensorEta = pStress - one*(pStress.Trace()/3.0)-rotatedBack;  
  double xae = sqrt(tensorEta.NormSquared()*1.5);
  double flowStress = xae - pYield[idx]; 
  return flowStress;
}

//calculate components of the tangent elastic-plastic moduli
//update internal state variables
void
SuvicI::computeStressIncTangent(double& epdot,
                                   Matrix3& stressRate,
                                   TangentModulusTensor& Cep,
                                   const double delT,
                                   const particleIndex idx,
                                   const TangentModulusTensor Ce,
                                   const Matrix3 tensorD,
                                   const Matrix3 pStress,
                                   const int implicitFlag,
                                   const Matrix3 tensorR)
{
  
  //implicitFlag=1, tensorD is increment, not rate
  Matrix3 pij, qij, nij, reducedEta, plasticStrainRateTensor, one;
  double xae, diedt;
  one.Identity();

  computeNij(nij, reducedEta, xae, idx, pStress, tensorR, implicitFlag);
  Ce.contract(nij, pij);

  //prefix is 1/sec
  double prefix = d_SV.a*exp(-d_SV.q/d_SV.rr/d_SV.t);
//   cout << "prefix= " << prefix << " \n";
  double xn = d_SV.xn;
  double powTmp = pow( (xae-pYield[idx])/pDrag[idx], xn-1.0); 
  
  //tmp1 is 1/sec
  double tmp1 = prefix*xn*powTmp;

  // these are 1/sec
  double dedxae = 1.0/pDrag[idx]*tmp1;
  double dedxr = -1.0/pDrag[idx]*tmp1;
  double dedxk = -1.0/pow(pDrag[idx],2)*tmp1;

// plastic strain rate (1/sec) - not increment
  diedt = prefix* pow( (xae-pYield[idx])/pDrag[idx], xn); 
  
// saturated yield stress, total stress, backstress and drag stress
  double rpr = d_SV.r0* pow( (diedt/d_SV.de0), 1/d_SV.rm);
  double sep = d_SV.s0* pow( (diedt/d_SV.de0), 1/xn);
  double bep = d_SV.b0* pow( (diedt/d_SV.de0), 1/d_SV.bn);
  double xkpr = (sep-bep-rpr)/pow(diedt/prefix, 1.0/xn);

//   cout << "rpr= " << rpr << "sep= " << sep << "bep= " << bep << "xpr= " << xkpr << "\n";
  
  double gamma1 = -d_SV.ba1;
  gamma1+= d_SV.ba1/bep*\
     nij.Contract(pBackStress[idx])-nij.Contract(pij);

  double gamma2 = d_SV.rai*(1.0 - pYield[idx]/rpr);
  double gamma3 = d_SV.xai*(1.0 - pDrag[idx]/xkpr);

//note hbar is a rate term
  double hbar = -dedxae*gamma1-dedxr*gamma2-dedxk*gamma3;

//note xsi has no time
  double xsi = d_SV.theta*delT*hbar;

  Matrix3 qijTmp;
  Ce.contract(nij,qijTmp);
  
  //qij is a rate term
  qij = (one*dedxae)*qijTmp;
  
  //qmde is a rate^2 term
  double qmde = qij.Contract(tensorD);
  if (implicitFlag==1)
      qmde/=delT;

//inelastic strain rate (scalar) 
  epdot = diedt/(1.0+xsi)+delT*d_SV.theta/(1.0+xsi)*qmde;
  plasticStrainRateTensor = (one*epdot)*nij;

  //no method yet to form from two 2nd order tensor to a 4th order on
  //maybe should have one 
  //update jacobian

  for (int ii=0; ii<3; ++ii) {
    for (int jj=0; jj<3; ++jj) {
      for (int kk=0; kk<3; ++kk) {
        for (int ll=0; ll<3; ++ll) {
            Cep(ii,jj,kk,ll)=Ce(ii,jj,kk,ll)-xsi/(1.0+xsi)/hbar* pij(ii,jj)*qij(kk,ll);
        }
      }
    }
  }
 
  if (implicitFlag==1)
     Ce.contract(tensorD/delT-plasticStrainRateTensor, stressRate);
  else
    Ce.contract(tensorD-plasticStrainRateTensor, stressRate);
  
  //update internal state variables
  
//   pPlasticStrain_new[idx]=pPlasticStrain[idx]+diedt;

        pBackStress_new[idx]=pBackStress[idx]+\
        one*(delT*2.0/3.0*d_SV.ba1)*plasticStrainRateTensor-\
        one*(delT*(d_SV.ba1/bep*epdot))*pBackStress[idx];
        
 if (implicitFlag ==0)
        pBackStress_new[idx] = (tensorR*pBackStress_new[idx])*(tensorR.Transpose());
  
  //update yield stress - can't be larger than saturation value
  if (pYield[idx] <rpr) {
     pYield_new[idx]=pYield[idx]+delT*d_SV.rai*epdot*(1.0-pYield[idx]/rpr);
     }
     else 
     {
     pYield_new[idx]=pYield[idx];     
     }
  
  //update drag stress - can't be larger than saturation value
  if (pDrag[idx] < xkpr) {
     pDrag_new[idx]=pDrag[idx]+delT*d_SV.xai*epdot*(1.0-pDrag[idx]/xkpr);
     }
     else 
     {
     pDrag_new[idx]=pDrag[idx];
     }  
     
//   cout << "diedt=" << diedt << "\n";   
//   cout << "pYield_new=" << pYield_new[idx] << "\n";  
//   cout << "pDrag_new=" << pDrag_new[idx] << "\n";   
// //   cout << "tensorD=" << tensorD << "\n"; 
//   cout << "plasticStrainRateTensor=" << plasticStrainRateTensor << "\n"; 
}

bool 
SuvicI::checkFailureMaxTensileStress(const Matrix3 pStress)
{
double s1, s2, s3;

pStress.getEigenValues(s1, s2, s3);
double max1 = max(s1, s2);
double max2 = max(max1,s3);

//from johnson and hopkins, 2005, J. Glaciology. 
double maxStress = 0.99648e+06 - 0.045904e+6*(d_SV.t - 273.0);

cout << "s1= " << s1 << " s2= " << s2 << " s3= " << s3 << endl;
cout << "max2= " << max2 << " maxStress= " << maxStress << endl;

if (max2 <= 0.0 || max2 < maxStress)
   return false;
else 
   return true;
}
///////////////////////////////////////////////////////////////////////////
/*  Compute the shear modulus. */
///////////////////////////////////////////////////////////////////////////
double
SuvicI::computeShearModulus(const PlasticityState* state)
{
  return state->shearModulus;
}

///////////////////////////////////////////////////////////////////////////
/* Compute the melting temperature */
///////////////////////////////////////////////////////////////////////////
double
SuvicI::computeMeltingTemp(const PlasticityState* state)
{
  return state->meltingTemp;
}







