#include "ZAPlastic.h"
#include <math.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>


using namespace Uintah;
using namespace SCIRun;
////////////////////////////////////////////////////////////////////////////////

ZAPlastic::ZAPlastic(ProblemSpecP& ps)
{
  d_CM.c_0 = 0.0;
  ps->get("c_0", d_CM.c_0);
  if (d_CM.c_0 == 0.0) {
    ps->require("sigma_g",d_CM.sigma_g);  
    ps->require("k_H",d_CM.k_H);  
    ps->require("sqrt_l_inv",d_CM.sqrt_l);  
    d_CM.c_0 = d_CM.sigma_g + d_CM.k_H*d_CM.sqrt_l;
  }
  ps->require("B",d_CM.B);  
  ps->require("beta_0",d_CM.beta_0);  
  ps->require("beta_1",d_CM.beta_1);  
  ps->require("B_0",d_CM.B_0);  
  ps->require("alpha_0",d_CM.alpha_0);  
  ps->require("alpha_1",d_CM.alpha_1);  
  ps->require("K",d_CM.K);  
  ps->require("n",d_CM.n);  
}
         
ZAPlastic::ZAPlastic(const ZAPlastic* cm)
{
  d_CM.c_0 = cm->d_CM.c_0;
  d_CM.sigma_g = cm->d_CM.sigma_g;
  d_CM.k_H = cm->d_CM.k_H;
  d_CM.sqrt_l = cm->d_CM.sqrt_l;
  d_CM.B = cm->d_CM.B;
  d_CM.beta_0 = cm->d_CM.beta_0;
  d_CM.beta_1 = cm->d_CM.beta_1;
  d_CM.B_0 = cm->d_CM.B_0;
  d_CM.alpha_0 = cm->d_CM.alpha_0;
  d_CM.alpha_1 = cm->d_CM.alpha_1;
  d_CM.K = cm->d_CM.K;
  d_CM.n = cm->d_CM.n;
}
         
ZAPlastic::~ZAPlastic()
{
}
         
void 
ZAPlastic::addInitialComputesAndRequires(Task* ,
                                         const MPMMaterial* ,
                                         const PatchSet*) const
{
}

void 
ZAPlastic::addComputesAndRequires(Task* ,
                                  const MPMMaterial* ,
                                  const PatchSet*) const
{
}

void 
ZAPlastic::addParticleState(std::vector<const VarLabel*>& ,
                            std::vector<const VarLabel*>& )
{
}

void 
ZAPlastic::allocateCMDataAddRequires(Task* ,
                                     const MPMMaterial* ,
                                     const PatchSet* ,
                                     MPMLabel* ) const
{
}

void ZAPlastic::allocateCMDataAdd(DataWarehouse* ,
                                  ParticleSubset* ,
                                  map<const VarLabel*, 
                                  ParticleVariableBase*>* ,
                                  ParticleSubset* ,
                                  DataWarehouse* )
{
}


void 
ZAPlastic::initializeInternalVars(ParticleSubset* ,
                                  DataWarehouse* )
{
}

void 
ZAPlastic::getInternalVars(ParticleSubset* ,
                           DataWarehouse* ) 
{
}

void 
ZAPlastic::allocateAndPutInternalVars(ParticleSubset* ,
                                      DataWarehouse* ) 
{
}

void 
ZAPlastic::allocateAndPutRigid(ParticleSubset* ,
                               DataWarehouse* ) 
{
}

void
ZAPlastic::updateElastic(const particleIndex )
{
}

void
ZAPlastic::updatePlastic(const particleIndex , 
                         const double& )
{
}

double 
ZAPlastic::computeFlowStress(const PlasticityState* state,
                             const double& ,
                             const double& ,
                             const MPMMaterial* ,
                             const particleIndex idx)
{
  double epdot = state->plasticStrainRate;
  double ep = state->plasticStrain;
  double T = state->temperature;
  ASSERT(epdot > 0.0);
  ASSERT(!(ep < 0.0));
  ASSERT(T > 0.0);
  epdot = (epdot == 0.0) ? 1.0e-8 : epdot;
  ep = (ep == 0.0) ? 1.0e-8 : ep;
  T = (T == 0.0) ? 1.0e-8 : T;

  double sigma_a = d_CM.c_0 + d_CM.K*pow(ep, d_CM.n);
  double alpha = d_CM.alpha_0 - d_CM.alpha_1*log(epdot);
  double beta = d_CM.beta_0 - d_CM.beta_1*log(epdot);
  double sigma_y = sigma_a + d_CM.B*exp(-beta*T) + 
                   d_CM.B_0*sqrt(ep)*exp(-alpha*T);
  if (isnan(sigma_y)) {
    cout << "ZA_Flow_Stress:: idx = " << idx << " epdot = " << epdot
         << " ep = " << ep << " T = " << T << endl;
    cout << " idx = " << idx << " sigma_a = " << sigma_a
          << " alpha = " << alpha << " beta = " << beta
          << " sigma_y = " << sigma_y << endl;
  }

  return sigma_y;
}


//////////
/*! \brief Calculate the plastic strain rate [epdot(tau,ep,T)] */
//////////
double 
ZAPlastic::computeEpdot(const PlasticityState* state ,
                        const double& ,
                        const double& tolerance,
                        const MPMMaterial* ,
                        const particleIndex )
{
  double tau = state->yieldStress;
  double ep = state->plasticStrain;
  double T = state->temperature;
  ASSERT(!(ep < 0.0));
  ASSERT(T > 0.0);
  ep = (ep == 0.0) ? 1.0e-8 : ep;
  T = (T == 0.0) ? 1.0e-8 : T;

  double sigma_a = d_CM.c_0 + d_CM.K*pow(ep, d_CM.n);
  double B0_sqrtEp = d_CM.B_0*sqrt(ep);

  // Do Newton iteration
  double epdot = 1.0;
  double f = 0.0;
  double fPrime = 0.0;
  do {
    ASSERT(epdot >= 0.0);
    double alpha = d_CM.alpha_0 - d_CM.alpha_1*log(epdot);
    double beta = d_CM.beta_0 - d_CM.beta_1*log(epdot);
    double tAlpha = B0_sqrtEp*exp(-alpha*T);
    double tBeta = d_CM.B*exp(-beta*T);
    double sigma_y = sigma_a + tBeta + tAlpha;
    double term1 = d_CM.beta_1*T*tBeta;
    double term2 = d_CM.alpha_1*T*tAlpha;
    f = tau - sigma_y;
    fPrime = -(term1+term2)/epdot;
    epdot -= f/fPrime;
  } while (fabs(f) > tolerance);

  return epdot;
}
 
void 
ZAPlastic::computeTangentModulus(const Matrix3& stress,
                                 const PlasticityState* state,
                                 const double& ,
                                 const MPMMaterial* ,
                                 const particleIndex idx,
                                 TangentModulusTensor& Ce,
                                 TangentModulusTensor& Cep)
{
  // Calculate the deviatoric stress and rate of deformation
  Matrix3 one; one.Identity();
  Matrix3 sigdev = stress - one*(stress.Trace()/3.0);

  // Calculate the equivalent stress
  double sigeqv = sqrt(sigdev.NormSquared()); 

  // Calculate the direction of plastic loading (r)
  Matrix3 rr = sigdev*(1.5/sigeqv);

  // Get f_q = dsigma/dep (h = 1, therefore f_q.h = f_q)
  double f_q = evalDerivativeWRTPlasticStrain(state, idx);

  // Form the elastic-plastic tangent modulus
  Matrix3 Cr, rC;
  double rCr = 0.0;
  for (int ii = 0; ii < 3; ++ii) {
    for (int jj = 0; jj < 3; ++jj) {
      Cr(ii,jj) = 0.0;
      rC(ii,jj) = 0.0;
      for (int kk = 0; kk < 3; ++kk) {
        for (int ll = 0; ll < 3; ++ll) {
          Cr(ii,jj) += Ce(ii,jj,kk,ll)*rr(kk,ll);
          rC(ii,jj) += rr(kk,ll)*Ce(kk,ll,ii,jj);
        }
      }
      rCr += rC(ii,jj)*rr(ii,jj);
    }
  }
  for (int ii = 0; ii < 3; ++ii) {
    for (int jj = 0; jj < 3; ++jj) {
      for (int kk = 0; kk < 3; ++kk) {
        for (int ll = 0; ll < 3; ++ll) {
          Cep(ii,jj,kk,ll) = Ce(ii,jj,kk,ll) - 
                             Cr(ii,jj)*rC(kk,ll)/(-f_q + rCr);
        }  
      }  
    }  
  }  
}

void
ZAPlastic::evalDerivativeWRTScalarVars(const PlasticityState* state,
                                       const particleIndex idx,
                                       Vector& derivs)
{
  derivs[0] = evalDerivativeWRTStrainRate(state, idx);
  derivs[1] = evalDerivativeWRTTemperature(state, idx);
  derivs[2] = evalDerivativeWRTPlasticStrain(state, idx);
}


double
ZAPlastic::evalDerivativeWRTPlasticStrain(const PlasticityState* state,
                                          const particleIndex )
{
  // Get the state data
  double ep = state->plasticStrain;
  double epdot = state->plasticStrainRate;
  double T = state->temperature;
  ASSERT(!(ep < 0.0));
  ASSERT(epdot > 0.0);
  ASSERT(T > 0.0);

  epdot = (epdot == 0.0) ? 1.0e-8 : epdot;
  ep = (ep == 0.0) ? 1.0e-8 : ep;
  T = (T == 0.0) ? 1.0e-8 : T;

  double alpha = d_CM.alpha_0 - d_CM.alpha_1*log(epdot);
  double term1 = d_CM.K*d_CM.n*pow(ep,d_CM.n-1.0);
  double term2 = 0.5*d_CM.B_0*exp(-alpha*T)/sqrt(ep);
  double deriv = term1 + term2;;  
  
  return deriv;
}

///////////////////////////////////////////////////////////////////////////
/*  Compute the shear modulus. */
///////////////////////////////////////////////////////////////////////////
double
ZAPlastic::computeShearModulus(const PlasticityState* state)
{
  return state->shearModulus;
}

///////////////////////////////////////////////////////////////////////////
/* Compute the melting temperature */
///////////////////////////////////////////////////////////////////////////
double
ZAPlastic::computeMeltingTemp(const PlasticityState* state)
{
  return state->meltingTemp;
}

double
ZAPlastic::evalDerivativeWRTTemperature(const PlasticityState* state,
                                        const particleIndex )
{
  // Get the state data
  double ep = state->plasticStrain;
  double epdot = state->plasticStrainRate;
  double T = state->temperature;
  ASSERT(!(ep < 0.0));
  ASSERT(epdot > 0.0);
  ASSERT(T > 0.0);

  epdot = (epdot == 0.0) ? 1.0e-8 : epdot;
  ep = (ep == 0.0) ? 1.0e-8 : ep;
  T = (T == 0.0) ? 1.0e-8 : T;

  double alpha = d_CM.alpha_0 - d_CM.alpha_1*log(epdot);
  double beta = d_CM.beta_0 - d_CM.beta_1*log(epdot);
  double term1 = -d_CM.B_0*alpha*sqrt(ep)*exp(-alpha*T);
  double term2 = -d_CM.B*beta*exp(-beta*T);
  double deriv = term1 + term2;
  
  return deriv;
}

double
ZAPlastic::evalDerivativeWRTStrainRate(const PlasticityState* state,
                                       const particleIndex )
{
  // Get the state data
  double ep = state->plasticStrain;
  double epdot = state->plasticStrainRate;
  double T = state->temperature;
  ASSERT(!(ep < 0.0));
  ASSERT(epdot > 0.0);
  ASSERT(T > 0.0);

  epdot = (epdot == 0.0) ? 1.0e-8 : epdot;
  ep = (ep == 0.0) ? 1.0e-8 : ep;
  T = (T == 0.0) ? 1.0e-8 : T;

  double alpha = d_CM.alpha_0 - d_CM.alpha_1*log(epdot);
  double beta = d_CM.beta_0 - d_CM.beta_1*log(epdot);
  double term1 = d_CM.B*d_CM.beta_1*T*exp(-beta*T);
  double term2 = d_CM.B_0*sqrt(ep)*d_CM.alpha_1*T*exp(-alpha*T);
  double deriv = (term1+term2)/epdot;

  return deriv;
}


