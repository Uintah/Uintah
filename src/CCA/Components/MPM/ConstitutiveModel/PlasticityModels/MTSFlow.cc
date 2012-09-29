/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "MTSFlow.h"
#include <cmath>
#include <iostream>
#include <Core/Exceptions/InvalidValue.h>

using namespace Uintah;
using namespace std;


MTSFlow::MTSFlow(ProblemSpecP& ps)
{
  ps->require("sigma_a",d_CM.sigma_a);
  ps->require("mu_0",d_CM.mu_0); //b1
  ps->require("D",d_CM.D); //b2
  ps->require("T_0",d_CM.T_0); //b3
  ps->require("koverbcubed",d_CM.koverbcubed);
  ps->require("g_0i",d_CM.g_0i);
  ps->require("g_0e",d_CM.g_0e); // g0
  ps->require("edot_0i",d_CM.edot_0i);
  ps->require("edot_0e",d_CM.edot_0e); //edot
  ps->require("p_i",d_CM.p_i);
  ps->require("q_i",d_CM.q_i);
  ps->require("p_e",d_CM.p_e); //p
  ps->require("q_e",d_CM.q_e); //q
  ps->require("sigma_i",d_CM.sigma_i);
  ps->require("a_0",d_CM.a_0);
  ps->require("a_1",d_CM.a_1);
  ps->require("a_2",d_CM.a_2);
  ps->require("a_3",d_CM.a_3);
  ps->require("theta_IV",d_CM.theta_IV);
  ps->require("alpha",d_CM.alpha);
  ps->require("edot_es0",d_CM.edot_es0);
  ps->require("g_0es",d_CM.g_0es); //A
  ps->require("sigma_es0",d_CM.sigma_es0);

  // Above phase transition temperature (only for steels)
  //d_CM.Tc = 1040.0;
  d_CM.Tc = 10400.0;  // For the other materials not to get
                      // screwed up
  ps->get("T_c", d_CM.Tc);   
  d_CM.g_0i_c = 0.57582;
  ps->get("g_0i_c", d_CM.g_0i_c);
  d_CM.sigma_i_c = 896.14e6;
  ps->get("sigma_i_c", d_CM.sigma_i_c);
  d_CM.g_0es_c = 0.294;
  ps->get("g_0es_c", d_CM.g_0es_c);
  d_CM.sigma_es0_c = 478.36e6;
  ps->get("sigma_es0_c", d_CM.sigma_es0_c);
  d_CM.a_0_c = 7.5159e9;
  ps->get("a_0_c", d_CM.a_0_c);
  d_CM.a_3_c = 3.7796e6;
  ps->get("a_3_c", d_CM.a_3_c);

  // Initialize internal variable labels for evolution
  //pMTSLabel = VarLabel::create("p.mtStress",
  //      ParticleVariable<double>::getTypeDescription());
  //pMTSLabel_preReloc = VarLabel::create("p.mtStress+",
  //      ParticleVariable<double>::getTypeDescription());
}
         
MTSFlow::MTSFlow(const MTSFlow* cm)
{
  d_CM.sigma_a = cm->d_CM.sigma_a;
  d_CM.mu_0 = cm->d_CM.mu_0;
  d_CM.D = cm->d_CM.D;
  d_CM.T_0 = cm->d_CM.T_0;
  d_CM.koverbcubed = cm->d_CM.koverbcubed;
  d_CM.g_0i = cm->d_CM.g_0i;
  d_CM.g_0e = cm->d_CM.g_0e;
  d_CM.edot_0i = cm->d_CM.edot_0i;
  d_CM.edot_0e = cm->d_CM.edot_0e;
  d_CM.p_i = cm->d_CM.p_i;
  d_CM.q_i = cm->d_CM.q_i;
  d_CM.p_e = cm->d_CM.p_e;
  d_CM.q_e = cm->d_CM.q_e;
  d_CM.sigma_i = cm->d_CM.sigma_i;
  d_CM.a_0 = cm->d_CM.a_0;
  d_CM.a_1 = cm->d_CM.a_1;
  d_CM.a_2 = cm->d_CM.a_2;
  d_CM.a_3 = cm->d_CM.a_3;
  d_CM.theta_IV = cm->d_CM.theta_IV;
  d_CM.alpha = cm->d_CM.alpha;
  d_CM.edot_es0 = cm->d_CM.edot_es0;
  d_CM.g_0es = cm->d_CM.g_0es;
  d_CM.sigma_es0 = cm->d_CM.sigma_es0;

  d_CM.Tc = cm->d_CM.Tc;
  d_CM.g_0i_c = cm->d_CM.g_0i_c;
  d_CM.sigma_i_c = cm->d_CM.sigma_i_c;
  d_CM.g_0es_c = cm->d_CM.g_0es_c;
  d_CM.sigma_es0_c = cm->d_CM.sigma_es0_c;
  d_CM.a_0_c = cm->d_CM.a_0_c;
  d_CM.a_3_c = cm->d_CM.a_3_c;

  // Initialize internal variable labels for evolution
  //pMTSLabel = VarLabel::create("p.mtStress",
  //      ParticleVariable<double>::getTypeDescription());
  //pMTSLabel_preReloc = VarLabel::create("p.mtStress+",
  //      ParticleVariable<double>::getTypeDescription());
}
         
MTSFlow::~MTSFlow()
{
  //VarLabel::destroy(pMTSLabel);
  //VarLabel::destroy(pMTSLabel_preReloc);
}


void MTSFlow::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP flow_ps = ps->appendChild("flow_model");
  flow_ps->setAttribute("type","mts_model");

  flow_ps->appendElement("sigma_a",d_CM.sigma_a);
  flow_ps->appendElement("mu_0",d_CM.mu_0); //b1
  flow_ps->appendElement("D",d_CM.D); //b2
  flow_ps->appendElement("T_0",d_CM.T_0); //b3
  flow_ps->appendElement("koverbcubed",d_CM.koverbcubed);
  flow_ps->appendElement("g_0i",d_CM.g_0i);
  flow_ps->appendElement("g_0e",d_CM.g_0e); // g0
  flow_ps->appendElement("edot_0i",d_CM.edot_0i);
  flow_ps->appendElement("edot_0e",d_CM.edot_0e); //edot
  flow_ps->appendElement("p_i",d_CM.p_i);
  flow_ps->appendElement("q_i",d_CM.q_i);
  flow_ps->appendElement("p_e",d_CM.p_e); //p
  flow_ps->appendElement("q_e",d_CM.q_e); //q
  flow_ps->appendElement("sigma_i",d_CM.sigma_i);
  flow_ps->appendElement("a_0",d_CM.a_0);
  flow_ps->appendElement("a_1",d_CM.a_1);
  flow_ps->appendElement("a_2",d_CM.a_2);
  flow_ps->appendElement("a_3",d_CM.a_3);
  flow_ps->appendElement("theta_IV",d_CM.theta_IV);
  flow_ps->appendElement("alpha",d_CM.alpha);
  flow_ps->appendElement("edot_es0",d_CM.edot_es0);
  flow_ps->appendElement("g_0es",d_CM.g_0es); //A
  flow_ps->appendElement("sigma_es0",d_CM.sigma_es0);

  flow_ps->appendElement("T_c", d_CM.Tc);   
  flow_ps->appendElement("g_0i_c", d_CM.g_0i_c);
  flow_ps->appendElement("sigma_i_c", d_CM.sigma_i_c);
  flow_ps->appendElement("g_0es_c", d_CM.g_0es_c);
  flow_ps->appendElement("sigma_es0_c", d_CM.sigma_es0_c);
  flow_ps->appendElement("a_0_c", d_CM.a_0_c);
  flow_ps->appendElement("a_3_c", d_CM.a_3_c);
}

         
void 
MTSFlow::addInitialComputesAndRequires(Task* ,
                                          const MPMMaterial* ,
                                          const PatchSet*)
{
  //const MaterialSubset* matlset = matl->thisMaterial();
  //task->computes(pMTSLabel, matlset);
}

void 
MTSFlow::addComputesAndRequires(Task* ,
                                   const MPMMaterial* ,
                                   const PatchSet*)
{
  //const MaterialSubset* matlset = matl->thisMaterial();
  //task->requires(Task::OldDW, pMTSLabel, matlset,Ghost::None);
  //task->computes(pMTSLabel_preReloc, matlset);
}

void 
MTSFlow::addComputesAndRequires(Task* ,
                                   const MPMMaterial* ,
                                   const PatchSet*,
                                   bool ,
                                   bool )
{
  //const MaterialSubset* matlset = matl->thisMaterial();
  //task->requires(Task::ParentOldDW, pMTSLabel, matlset,Ghost::None);
}

void 
MTSFlow::addParticleState(std::vector<const VarLabel*>& ,
                             std::vector<const VarLabel*>& )
{
  //from.push_back(pMTSLabel);
  //to.push_back(pMTSLabel_preReloc);
}

void 
MTSFlow::allocateCMDataAddRequires(Task* ,
                                      const MPMMaterial* ,
                                      const PatchSet* ,
                                      MPMLabel* )
{
  //const MaterialSubset* matlset = matl->thisMaterial();
  //task->requires(Task::NewDW, pMTSLabel_preReloc, matlset, Ghost::None);
}

void 
MTSFlow::allocateCMDataAdd(DataWarehouse* ,
                              ParticleSubset* ,
                              map<const VarLabel*, 
                                ParticleVariableBase*>* ,
                              ParticleSubset* ,
                              DataWarehouse* )
{
  // Put stuff in here to initialize each particle's
  // constitutive model parameters and deformationMeasure
 
  //ParticleVariable<double> pMTS;
  //constParticleVariable<double> o_MTS;

  //new_dw->allocateTemporary(pMTS,addset);

  //new_dw->get(o_MTS,pMTSLabel_preReloc,delset);

  //ParticleSubset::iterator o,n = addset->begin();
  //for(o = delset->begin(); o != delset->end(); o++, n++) {
  //  pMTS[*n] = o_MTS[*o];
  //}

  //(*newState)[pMTSLabel]=pMTS.clone();

}



void 
MTSFlow::initializeInternalVars(ParticleSubset* ,
                                   DataWarehouse* )
{
  //new_dw->allocateAndPut(pMTS_new, pMTSLabel, pset);
  //ParticleSubset::iterator iter = pset->begin();
  //for(;iter != pset->end(); iter++) {
  //  pMTS_new[*iter] = 0.0;
  //}
}

void 
MTSFlow::getInternalVars(ParticleSubset* ,
                            DataWarehouse* ) 
{
  //old_dw->get(pMTS, pMTSLabel, pset);
}

void 
MTSFlow::allocateAndPutInternalVars(ParticleSubset* ,
                                       DataWarehouse* ) 
{
  //new_dw->allocateAndPut(pMTS_new, pMTSLabel_preReloc, pset);
}

void
MTSFlow::allocateAndPutRigid(ParticleSubset* ,
                                DataWarehouse* )
{
  //new_dw->allocateAndPut(pMTS_new, pMTSLabel_preReloc, pset);
  //ParticleSubset::iterator iter = pset->begin();
  //for(;iter != pset->end(); iter++){
  //   pMTS_new[*iter] = 0.0;
  //}
}

void
MTSFlow::updateElastic(const particleIndex )
{
  //pMTS_new[idx] = pMTS[idx];
}

void
MTSFlow::updatePlastic(const particleIndex , const double& )
{
  //pMTS_new[idx] = pMTS_new[idx];
}

double 
MTSFlow::computeFlowStress(const PlasticityState* state,
                              const double& ,
                              const double& ,
                              const MPMMaterial* ,
                              const particleIndex )
{
  // Calculate strain rate and incremental strain
  //double edot = state->plasticStrainRate;
  double edot = state->strainRate;
  if (edot == 0.0) edot = 1.0e-7;

  // Check if temperature is correct
  // Check if shear modulus is correct
  double T = state->temperature;
  double mu = state->shearModulus;
  if ((mu <= 0.0) || (T <= 0.0) ) {
    cerr << "**ERROR** MTSFlow::computeFlowStress: mu = " << mu 
         << " T = " << T  << endl;
  }
  double mu_mu_0 = mu/d_CM.mu_0;

  // If temperature is greater than phase transition temperature
  // then update the constants
  double g_0i = 0.0;
  double sigma_i = 0.0;
  double g_0es = 0.0;
  double sigma_es0 = 0.0;
  double a_0 = 0.0;
  double a_3 = 0.0;
  if (T > d_CM.Tc) {
    g_0i = d_CM.g_0i_c;
    sigma_i = d_CM.sigma_i_c;
    g_0es = d_CM.g_0es_c;
    sigma_es0 = d_CM.sigma_es0_c;
    a_0 = d_CM.a_0_c;
    a_3 = d_CM.a_3_c;
  } else {
    g_0i = d_CM.g_0i;
    sigma_i = d_CM.sigma_i;
    g_0es = d_CM.g_0es;
    sigma_es0 = d_CM.sigma_es0;
    a_0 = d_CM.a_0;
    a_3 = d_CM.a_3;
  }

  // Calculate S_i
  double CC = d_CM.koverbcubed*T/mu;
  double S_i = 0.0;
  if (d_CM.p_i > 0.0) {
    double CCi = CC/g_0i;
    double logei = log(d_CM.edot_0i/edot);
    double logei_q = 1.0 - pow((CCi*logei),(1.0/d_CM.q_i));
    if (logei_q > 0.0) S_i = pow(logei_q,(1.0/d_CM.p_i));
  }

  // Calculate S_e
  double CCe = CC/d_CM.g_0e;
  double logee = log(d_CM.edot_0e/edot);
  double logee_q = 1.0 - pow((CCe*logee),(1.0/d_CM.q_e));
  double S_e = 0.0;
  if (logee_q > 0.0) S_e = pow(logee_q,(1.0/d_CM.p_e));

  // Calculate theta_0
  double theta_0 = a_0 + d_CM.a_1*log(edot) + d_CM.a_2*sqrt(edot)
    - a_3*T;
  
  if(theta_0 < 0.0){
    ostringstream desc;
    desc << " **ERROR** MTS Plasticity Model: Negative initial hardening rate! " << endl;
    desc << "     edot = " << edot << " ep = " << state->plasticStrain << endl;
    desc << "     Tm = " << state->meltingTemp << " T = " << T << endl;
    desc << "     mu = " << mu << " mu_0 = " << d_CM.mu_0 << endl;
    desc << "     theta_0 = " << theta_0 << " epdot = " << state->plasticStrainRate << endl;
    throw InvalidValue(desc.str(), __FILE__, __LINE__);
  }

  // Calculate sigma_es
  double CCes = CC/g_0es;
  double powees = pow(edot/d_CM.edot_es0, CCes);
  double sigma_es = sigma_es0*powees;

  // Compute sigma_e (total)
  double sigma_e_old = 0.0;
  double delEps = state->plasticStrain;
  double sigma_e = computeSigma_e(theta_0, sigma_es, sigma_e_old, delEps, 4);

  // Compute sigma_e (incremental)
  //double sigma_e_old = pMTS[idx];
  //double delEps = edot*delT;

  // Calculate X and FX
  //double X = pMTS[idx]/sigma_es;
  //double FX = tanh(d_CM.alpha*X)/tanh(d_CM.alpha);

  // Calculate theta
  //double theta = theta_0*(1.0 - FX) + d_CM.theta_IV*FX;

  //double sigma_e = pMTS[idx] + delEps*theta; 
  //sigma_e = (sigma_e > sigma_es) ? sigma_es : sigma_e;
  //pMTS_new[idx] = sigma_e;

  // Calculate the flow stress
  double sigma = d_CM.sigma_a + (S_i*sigma_i + S_e*sigma_e)*mu_mu_0;

  //cout << "MTS: edot = " << edot << " T = " << T << " ep = " << delEps << endl;
  //cout << "     mu = " << mu << " mu_0 = " << d_CM.mu_0 << endl;
  //cout << "     S_i = " << S_i << " sigma_i = " << sigma_i << endl;
  //cout << "     S_e = " << S_e << " sigma_e = " << sigma_e << endl;
  //cout << "     theta_0 = " << theta_0 << endl;
  return sigma;
}

double 
MTSFlow::computeSigma_e(const double& theta_0, 
                           const double& sigma_es,
                           const double& ,
                           const double& ep,
                           const int& numSubcycles)
{
  // Midpoint rule integration
  double sigma_e = 0.0;
  if (ep == 0.0) return sigma_e; 

  double tol = 1.0;
  double talpha = tanh(d_CM.alpha);
  int nep = numSubcycles;
  double dep = ep/(double) nep;
  double beta = dep*theta_0;
  double delta = beta/talpha;
  double gamma = 0.5*d_CM.alpha/sigma_es;
  for (int ii = 0; ii < nep; ++ii) {
    double phi = sigma_e + beta;
    double psi = sigma_e*gamma;
    double f = 1.0e6;
    while (fabs(f) > tol) {
      double gsig = gamma*sigma_e;
      double tpgsig = tanh(psi + gsig);
      f = sigma_e + delta*tpgsig - phi;
      double fp = 1.0 + delta*gamma*(1.0 - tpgsig*tpgsig);
      sigma_e = fabs(sigma_e - f/fp);
    }
    if (sigma_e > sigma_es) {
      sigma_e = sigma_es;
      break;
    }
  }

  // Old Forward Euler integration
  /*
  double delEps = deltaEps/(double) numSubcycles;
  double sigma_e = sigma_e_old; 
  for (int ii = 0; ii < numSubcycles; ++ii) {

    // Calculate X and FX
    double X = sigma_e/sigma_es;
    double FX = tanh(d_CM.alpha*X)/tanh(d_CM.alpha);

    // Calculate theta
    double theta = theta_0*(1.0 - FX) + d_CM.theta_IV*FX;

    // Calculate sigma_e
    sigma_e += delEps*theta; 
    if (sigma_e > sigma_es) {
      sigma_e = sigma_es;
      break;
    }
  }
  */

  return sigma_e;
}

double 
MTSFlow::computeEpdot(const PlasticityState* state,
                         const double& delT,
                         const double& ,
                         const MPMMaterial* ,
                         const particleIndex )
{
  // Get the needed data
  double tau = state->yieldStress;
  double T = state->temperature;
  double mu = state->shearModulus;

  // If temperature is greater than phase transition temperature
  // then update the constants
  double g_0es = 0.0;
  double sigma_es0 = 0.0;
  double a_0 = 0.0;
  double a_3 = 0.0;
  if (T > d_CM.Tc) {
    g_0es = d_CM.g_0es_c;
    sigma_es0 = d_CM.sigma_es0_c;
    a_0 = d_CM.a_0_c;
    a_3 = d_CM.a_3_c;
  } else {
    g_0es = d_CM.g_0es;
    sigma_es0 = d_CM.sigma_es0;
    a_0 = d_CM.a_0;
    a_3 = d_CM.a_3;
  }

  // Calculate theta_0
  //double edot = state->plasticStrainRate;
  double edot = state->strainRate;
  if (edot == 0.0) edot = 1.0e-7;
  double theta_0 = a_0 + d_CM.a_1*log(edot) + d_CM.a_2*sqrt(edot) - a_3*T;

  // Calculate sigma_es
  double CC = d_CM.koverbcubed*T/mu;
  double CCes = CC/g_0es;
  double powees = pow(edot/d_CM.edot_es0, CCes);
  double sigma_es = sigma_es0*powees;

  // Compute sigma_e (total)
  double sigma_e_old = 0.0;
  double delEps = state->plasticStrain;
  double sigma_e = computeSigma_e(theta_0, sigma_es, sigma_e_old, delEps, 4);
  //double sigma_e = pMTS[idx];

  // Do Newton iteration
  double epdot = 1.0;
  double f = 0.0;
  double fPrime = 0.0;
  do {
    evalFAndFPrime(tau, epdot, T, mu, sigma_e, delT, f, fPrime);
    epdot -= f/fPrime;
  } while (fabs(f) > 1.0e-6);

  return epdot;
}

void 
MTSFlow::evalFAndFPrime(const double& tau,
                           const double& epdot,
                           const double& T,
                           const double& mu,
                           const double& sigma_e,
                           const double& ,
                           double& f,
                           double& fPrime)
{
  // If temperature is greater than phase transition temperature
  // then update the constants
  double g_0i = 0.0;
  double sigma_i = 0.0;
  if (T > d_CM.Tc) {
    g_0i = d_CM.g_0i_c;
    sigma_i = d_CM.sigma_i_c;
  } else {
    g_0i = d_CM.g_0i;
    sigma_i = d_CM.sigma_i;
  }

  // Compute mu/mu0
  double mu_mu_0 = mu/d_CM.mu_0;

  // Calculate S_i
  double CC = d_CM.koverbcubed*T/mu;
  double S_i = 0.0;
  double t_i = 0.0;
  if (d_CM.p_i > 0.0) {

    // f(epdot)
    double CCi = CC/g_0i;
    double logei = log(d_CM.edot_0i/epdot);
    double logei_q = 1.0 - pow((CCi*logei),(1.0/d_CM.q_i));
    if (logei_q > 0.0) S_i = pow(logei_q,(1.0/d_CM.p_i));

    // f'(epdot)
    double numer_t_i = S_i*(1.0 - logei_q);
    double denom_t_i = d_CM.p_i*d_CM.q_i*epdot*logei*logei_q;
    t_i = numer_t_i/denom_t_i;
  }

  // Calculate S_e
  double CCe = CC/d_CM.g_0e;
  double logee = log(d_CM.edot_0e/epdot);
  double logee_q = 1.0 - pow((CCe*logee),(1.0/d_CM.q_e));
  double S_e = 0.0;
  if (logee_q > 0.0) S_e = pow(logee_q,(1.0/d_CM.p_e));

  double numer_t_e = S_e*(1.0 - logee_q);
  double denom_t_e = d_CM.p_e*d_CM.q_e*epdot*logee*logee_q;
  double t_e = numer_t_e/denom_t_e;

  // Calculate f(epdot)
  f = tau - (d_CM.sigma_a + (S_i*sigma_i + S_e*sigma_e)*mu_mu_0);

  // Calculate f'(epdot)
  fPrime =  - (t_i*sigma_i + t_e*sigma_e)*mu_mu_0;
}


void 
MTSFlow::computeTangentModulus(const Matrix3& ,
                                  const PlasticityState* ,
                                  const double& ,
                                  const MPMMaterial* ,
                                  const particleIndex ,
                                  TangentModulusTensor& ,
                                  TangentModulusTensor& )
{
  throw InternalError("Empty Function: MTSFlow::computeTangentModulus", __FILE__, __LINE__);
}

void
MTSFlow::evalDerivativeWRTScalarVars(const PlasticityState* state,
                                        const particleIndex idx,
                                        Vector& derivs)
{
  derivs[0] = evalDerivativeWRTStrainRate(state, idx);
  derivs[1] = evalDerivativeWRTTemperature(state, idx);
  derivs[2] = evalDerivativeWRTPlasticStrain(state, idx);
}

double
MTSFlow::evalDerivativeWRTPlasticStrain(const PlasticityState* state,
                                           const particleIndex idx)
{
  // Get the state data
  //double edot = state->plasticStrainRate;
  double edot = state->strainRate;
  if (edot == 0.0) edot = 1.0e-7;
  double mu = state->shearModulus;
  double T = state->temperature;

  // If temperature is greater than phase transition temperature
  // then update the constants
  double g_0es = 0.0;
  double sigma_es0 = 0.0;
  double a_0 = 0.0;
  double a_3 = 0.0;
  if (T > d_CM.Tc) {
    g_0es = d_CM.g_0es_c;
    sigma_es0 = d_CM.sigma_es0_c;
    a_0 = d_CM.a_0_c;
    a_3 = d_CM.a_3_c;
  } else {
    g_0es = d_CM.g_0es;
    sigma_es0 = d_CM.sigma_es0;
    a_0 = d_CM.a_0;
    a_3 = d_CM.a_3;
  }

  //double sigma_e = pMTS_new[idx];
  double dsigY_dsig_e = evalDerivativeWRTSigmaE(state, idx);

  // Calculate theta_0
  double theta_0 = a_0 + d_CM.a_1*log(edot) + d_CM.a_2*sqrt(edot) - a_3*T;

  // Check mu
  ASSERT (mu > 0.0);

  // Calculate sigma_es
  double CC = d_CM.koverbcubed*T/mu;
  double CCes = CC/g_0es;
  double powees = pow(edot/d_CM.edot_es0, CCes);
  double sigma_es = sigma_es0*powees;

  // Compute sigma_e (total)
  double sigma_e_old = 0.0;
  double delEps = state->plasticStrain;
  double sigma_e = computeSigma_e(theta_0, sigma_es, sigma_e_old, delEps, 4);

  // Calculate X and FX
  //double X = pMTS_new[idx]/sigma_es;
  double X = sigma_e/sigma_es;
  double FX = tanh(d_CM.alpha*X)/tanh(d_CM.alpha);

  // Calculate theta
  double dsig_e_dep = theta_0*(1.0 - FX) + d_CM.theta_IV*FX;
  
  return (dsigY_dsig_e*dsig_e_dep);
}

///////////////////////////////////////////////////////////////////////////
/*  Compute the shear modulus. */
///////////////////////////////////////////////////////////////////////////
double
MTSFlow::computeShearModulus(const PlasticityState* state)
{
  double T = state->temperature;
  ASSERT(T > 0.0);
  double expT0_T = exp(d_CM.T_0/T) - 1.0;
  ASSERT(expT0_T != 0);
  double mu = d_CM.mu_0 - d_CM.D/expT0_T;
  if (!(mu > 0.0)) {
    ostringstream desc;
    desc << "**MTS Deriv Edot ERROR** Shear modulus <= 0." << endl;
    desc << "T = " << T << " mu0 = " << d_CM.mu_0 << " T0 = " << d_CM.T_0
         << " exp(To/T) = " << expT0_T << " D = " << d_CM.D << endl;
    throw InvalidValue(desc.str(), __FILE__, __LINE__);
  }
  return mu;
}

///////////////////////////////////////////////////////////////////////////
/* Compute the melting temperature */
///////////////////////////////////////////////////////////////////////////
double
MTSFlow::computeMeltingTemp(const PlasticityState* state)
{
  return state->meltingTemp;
}

double
MTSFlow::evalDerivativeWRTTemperature(const PlasticityState* state,
                                         const particleIndex )
{
  // Get the state data
  //double edot = state->plasticStrainRate;
  double edot = state->strainRate;
  if (edot == 0.0) edot = 1.0e-7;
  double T = state->temperature;
  double mu = state->shearModulus;

  // Calculate exp(T0/T)
  double expT0T = exp(d_CM.T_0/T);

  // Calculate mu/mu_0 and CC
  double mu_mu_0 = mu/d_CM.mu_0;
  double CC = d_CM.koverbcubed/mu;

  // If temperature is greater than phase transition temperature
  // then update the constants
  double g_0i = 0.0;
  double sigma_i = 0.0;
  double g_0es = 0.0;
  double sigma_es0 = 0.0;
  double a_0 = 0.0;
  double a_3 = 0.0;
  if (T > d_CM.Tc) {
    g_0i = d_CM.g_0i_c;
    sigma_i = d_CM.sigma_i_c;
    g_0es = d_CM.g_0es_c;
    sigma_es0 = d_CM.sigma_es0_c;
    a_0 = d_CM.a_0_c;
    a_3 = d_CM.a_3_c;
  } else {
    g_0i = d_CM.g_0i;
    sigma_i = d_CM.sigma_i;
    g_0es = d_CM.g_0es;
    sigma_es0 = d_CM.sigma_es0;
    a_0 = d_CM.a_0;
    a_3 = d_CM.a_3;
  }

  // Calculate theta_0
  double theta_0 = a_0 + d_CM.a_1*log(edot) + d_CM.a_2*sqrt(edot) - a_3*T;

  // Calculate sigma_es
  double CCes = CC/g_0es;
  double powees = pow(edot/d_CM.edot_es0, CCes);
  double sigma_es = sigma_es0*powees;

  // Compute sigma_e (total)
  double sigma_e_old = 0.0;
  double delEps = state->plasticStrain;
  double sigma_e = computeSigma_e(theta_0, sigma_es, sigma_e_old, delEps, 4);

  //double sigma_e = pMTS_new[idx];

  // Materials with (sigma_i != 0) are treated specially
  double numer1i = 0.0;
  double ratio2i = 0.0;
  if (d_CM.p_i > 0.0) {

    // Calculate log(edot0/edot)
    double logei = log(d_CM.edot_0i/edot);

    // Calculate E = k/(g0*mu*b^3) log(edot_0/edot)
    double E_i = (CC/g_0i)*logei;

    // Calculate (E*T)^{1/q}
    double ET_i = pow(E_i*T, 1.0/d_CM.q_i);

    // Calculate (1-(E*T)^{1/q})^{1/p)
    double ETpq_i = pow(1.0-ET_i,1.0/d_CM.p_i);

    // Calculate Ist term numer
    numer1i = ETpq_i*sigma_i;

    // Calculate second term denom
    double denom2i = T*(1.0-ET_i);

    // Calculate second term numerator
    double numer2i = ET_i*ETpq_i*sigma_i/(d_CM.p_i*d_CM.q_i);

    // Calculate the ratios
    ratio2i = numer2i/denom2i;
  }

  // Calculate log(edot0/edot)
  double logee = log(d_CM.edot_0e/edot);

  // Calculate E = k/(g0*mu*b^3) log(edot_0/edot)
  double E_e = (CC/d_CM.g_0e)*logee;

  // Calculate (E*T)^{1/q}
  double ET_e = pow(E_e*T, 1.0/d_CM.q_e);

  // Calculate (1-(E*T)^{1/q})^{1/p)
  double ETpq_e = pow(1.0-ET_e,1.0/d_CM.p_e);

  // Calculate Ist term denom
  double denom1 = (expT0T - 1.0)*(expT0T - 1.0)*T*T*d_CM.mu_0;

  // Calculate Ist term numer
  double numer1e = ETpq_e*sigma_e;

  // Calculate the total (first term)
  double numer1 = - d_CM.D*d_CM.T_0*expT0T*(numer1i + numer1e);

  // Calculate the first term
  double first = numer1/denom1;

  // Calculate second term denom
  double denom2e = T*(1.0-ET_e);

  // Calculate second term numerator
  double numer2e = ET_e*ETpq_e*sigma_e/(d_CM.p_e*d_CM.q_e);

  // Calculate the ratios
  double ratio2e = numer2e/denom2e;

  // Calculate the total (second term)
  double second = - mu_mu_0*(ratio2i + ratio2e);

  return (first+second);
}

double
MTSFlow::evalDerivativeWRTStrainRate(const PlasticityState* state,
                                        const particleIndex )
{
  // Get the state data
  //double edot = state->plasticStrainRate;
  double edot = state->strainRate;
  if (edot == 0.0) edot = 1.0e-7;
  double T = state->temperature;
  double mu = state->shearModulus;

  double mu_mu_0 = mu/d_CM.mu_0;
  double CC = d_CM.koverbcubed*T/mu;

  // If temperature is greater than phase transition temperature
  // then update the constants
  double g_0i = 0.0;
  double sigma_i = 0.0;
  double g_0es = 0.0;
  double sigma_es0 = 0.0;
  double a_0 = 0.0;
  double a_3 = 0.0;
  if (T > d_CM.Tc) {
    g_0i = d_CM.g_0i_c;
    sigma_i = d_CM.sigma_i_c;
    g_0es = d_CM.g_0es_c;
    sigma_es0 = d_CM.sigma_es0_c;
    a_0 = d_CM.a_0_c;
    a_3 = d_CM.a_3_c;
  } else {
    g_0i = d_CM.g_0i;
    sigma_i = d_CM.sigma_i;
    g_0es = d_CM.g_0es;
    sigma_es0 = d_CM.sigma_es0;
    a_0 = d_CM.a_0;
    a_3 = d_CM.a_3;
  }

  // Calculate theta_0
  double theta_0 = a_0 + d_CM.a_1*log(edot) + d_CM.a_2*sqrt(edot) - a_3*T;

  // Calculate sigma_es
  double CCes = CC/g_0es;
  double powees = pow(edot/d_CM.edot_es0, CCes);
  double sigma_es = sigma_es0*powees;

  // Compute sigma_e (total)
  double sigma_e_old = 0.0;
  double delEps = state->plasticStrain;
  double sigma_e = computeSigma_e(theta_0, sigma_es, sigma_e_old, delEps, 4);
  //double sigma_e = pMTS_new[idx];

  // Materials with (sigma_i != 0) are treated specially
  double ratioi = 0.0;
  if (d_CM.p_i > 0.0) {

    // Calculate (mu/mu0)*sigma_hat
    double M_i = mu_mu_0*sigma_i;

    // Calculate CC/g0
    double K_i = CC/g_0i;

    // Calculate log(edot0/edot)
    double logei = log(d_CM.edot_0i/edot);

    // Calculate (K*log(edot0/edot))^{1/q}
    double Klogi = pow(K_i*logei, 1.0/d_CM.q_i);
 
    // Calculate the denominator
    double denomi = edot*logei*(1.0-Klogi);

    // Calculate the numerator
    double numeri = (M_i*Klogi)/
      (d_CM.p_i*d_CM.q_i)*pow(1.0-Klogi,1.0/d_CM.p_i);

    // Calculate the ratios
    ratioi = numeri/denomi;
  }
  // Calculate (mu/mu0)*sigma_hat
  double M_e = mu_mu_0*sigma_e;

  // Calculate CC/g0
  double K_e = CC/d_CM.g_0e;

  // Calculate log(edot0/edot)
  double logee = log(d_CM.edot_0e/edot);

  // Calculate (K*log(edot0/edot))^{1/q}
  double Kloge = pow(K_e*logee, 1.0/d_CM.q_e);
 
  // Calculate the denominator
  double denome = edot*logee*(1.0-Kloge);

  // Calculate the numerator
  double numere = (M_e*Kloge)/
    (d_CM.p_e*d_CM.q_e)*pow(1.0-Kloge, 1.0/d_CM.p_e);

  // Calculate the ratios
  double ratioe = numere/denome;

  return (ratioi+ratioe);
}

double
MTSFlow::evalDerivativeWRTSigmaE(const PlasticityState* state,
                                    const particleIndex )
{
  // Get the state data
  //double edot = state->plasticStrainRate;
  double edot = state->strainRate;
  if (edot == 0.0) edot = 1.0e-7;
  double T = state->temperature;
  double mu = state->shearModulus;
  double mu_mu_0 = mu/d_CM.mu_0;

  // Calculate S_e
  double CC = d_CM.koverbcubed*T/mu;
  double CCe = CC/d_CM.g_0e;
  double logee = log(d_CM.edot_0e/edot);
  double logee_q = 1.0 - pow((CCe*logee),(1.0/d_CM.q_e));
  double S_e = 0.0;
  if (logee_q > 0.0) S_e = pow(logee_q,(1.0/d_CM.p_e));
  S_e = mu_mu_0*S_e;
  return S_e;
}
