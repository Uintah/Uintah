
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/SCGPlastic.h>
#include <math.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;


SCGPlastic::SCGPlastic(ProblemSpecP& ps)
{
  ps->require("mu_0",d_CM.mu_0); 
  ps->require("A",d_CM.A); 
  ps->require("B",d_CM.B); 
  ps->require("sigma_0",d_CM.sigma_0); 
  ps->require("beta",d_CM.beta); 
  ps->require("n",d_CM.n); 
  ps->require("epsilon_p0",d_CM.epsilon_p0); 
  ps->require("Y_max",d_CM.Y_max); 
  ps->require("T_m0",d_CM.T_m0); 
  ps->require("a",d_CM.a); 
  ps->require("Gamma_0",d_CM.Gamma_0); 
}
	 
SCGPlastic::~SCGPlastic()
{
}
	 
void 
SCGPlastic::addInitialComputesAndRequires(Task* ,
					  const MPMMaterial* ,
					  const PatchSet*) const
{
}

void 
SCGPlastic::addComputesAndRequires(Task* ,
				   const MPMMaterial* ,
				   const PatchSet*) const
{
}

void 
SCGPlastic::addParticleState(std::vector<const VarLabel*>& ,
			     std::vector<const VarLabel*>& )
{
}

void 
SCGPlastic::allocateCMDataAddRequires(Task* ,
				      const MPMMaterial* ,
				      const PatchSet* ,
				      MPMLabel* ) const
{
}

void 
SCGPlastic::allocateCMDataAdd(DataWarehouse* ,
			      ParticleSubset* ,
			      map<const VarLabel*, 
                                 ParticleVariableBase*>* ,
			      ParticleSubset* ,
			      DataWarehouse* )
{
}



void 
SCGPlastic::initializeInternalVars(ParticleSubset* ,
				   DataWarehouse* )
{
}

void 
SCGPlastic::getInternalVars(ParticleSubset* ,
			    DataWarehouse* ) 
{
}

void 
SCGPlastic::allocateAndPutInternalVars(ParticleSubset* ,
				       DataWarehouse* ) 
{
}

void
SCGPlastic::allocateAndPutRigid(ParticleSubset* ,
                                DataWarehouse* )
{
}

void
SCGPlastic::updateElastic(const particleIndex )
{
}

void
SCGPlastic::updatePlastic(const particleIndex , const double& )
{
}

double 
SCGPlastic::computeFlowStress(const PlasticityState* state,
                              const double& delT,
                              const double& ,
                              const MPMMaterial* ,
                              const particleIndex )
{
  // Get the state data
  double ep = state->plasticStrain;
  double mu = state->shearModulus;
  ASSERT(mu > 0.0);

  // Calculate mu/mu_0
  double mu_mu_0 = mu/d_CM.mu_0;

  // Calculate Y <= Ymax
  double Ya = 1.0 + d_CM.beta*(ep - d_CM.epsilon_p0);
  ASSERT(Ya >= 0.0);
  double Y = Min(d_CM.sigma_0*pow(Ya, d_CM.n), d_CM.Y_max);

  double sigma = Y*mu_mu_0;
  return sigma;
}

/*! The evolving internal variable is \f$q = \epsilon_p\f$.  If the 
  evolution equation for internal variables is of the form 
  \f$ \dot q = \gamma h (\sigma, q) \f$, then 
  \f[
  \dot q = \frac{d\epsilon_p}{dt} = \dot\epsilon_p .
  \f] 
  If \f$\dot\epsilon_p = \gamma\f$, then \f$ 1 = h \f$.
  Also, \f$ f_q = \frac{\partial f}{\partial \epsilon_p} \f$.
  For the von Mises yield condition, \f$(f)\f$, 
  \f$ f_q = \frac{\partial \sigma}{\partial \epsilon_p} \f$
  where \f$\sigma\f$ is the SCG flow stress.
*/
void 
SCGPlastic::computeTangentModulus(const Matrix3& stress,
				  const PlasticityState* state,
				  const double& ,
				  const MPMMaterial* ,
				  const particleIndex idx,
				  TangentModulusTensor& Ce,
				  TangentModulusTensor& Cep)
{
  // Get f_q = dsigma/dep (h = 1, therefore f_q.h = f_q)
  double f_q = evalDerivativeWRTPlasticStrain(state, idx);

  // Calculate the deviatoric stress and rate of deformation
  Matrix3 one; one.Identity();
  Matrix3 sigdev = stress - one*(stress.Trace()/3.0);

  // Calculate the equivalent stress and strain rate
  double sigeqv = sqrt(sigdev.NormSquared()); 

  // Calculate the direction of plastic loading (r)
  Matrix3 rr = sigdev*(1.5/sigeqv);

  // Form the elastic-plastic tangent modulus
  Matrix3 Cr, rC;
  double rCr = 0.0;
  for (int ii = 0; ii < 3; ++ii) {
    int ii1 = ii+1;
    for (int jj = 0; jj < 3; ++jj) {
      int jj1 = jj+1;
      Cr(ii1,jj1) = 0.0;
      rC(ii1,jj1) = 0.0;
      for (int kk = 0; kk < 3; ++kk) {
        int kk1 = kk+1;
	for (int ll = 0; ll < 3; ++ll) {
          Cr(ii1,jj1) += Ce(ii,jj,kk,ll)*rr(kk1,ll+1);
          rC(ii1,jj1) += rr(kk1,ll+1)*Ce(kk,ll,ii,jj);
        }
      }
      rCr += rC(ii1,jj1)*rr(ii1,jj1);
    }
  }
  for (int ii = 0; ii < 3; ++ii) {
    int ii1 = ii+1;
    for (int jj = 0; jj < 3; ++jj) {
      int jj1 = jj+1;
      for (int kk = 0; kk < 3; ++kk) {
        int kk1 = kk+1;
	for (int ll = 0; ll < 3; ++ll) {
          Cep(ii,jj,kk,ll) = Ce(ii,jj,kk,ll) - 
	    Cr(ii1,jj1)*rC(kk1,ll+1)/(-f_q + rCr);
	}  
      }  
    }  
  }  
}

void
SCGPlastic::evalDerivativeWRTScalarVars(const PlasticityState* state,
                                        const particleIndex idx,
                                        Vector& derivs)
{
  derivs[0] = evalDerivativeWRTPressure(state, idx);
  derivs[1] = evalDerivativeWRTTemperature(state, idx);
  derivs[2] = evalDerivativeWRTPlasticStrain(state, idx);
}

double
SCGPlastic::evalDerivativeWRTPlasticStrain(const PlasticityState* state,
					   const particleIndex )
{
  // Get the state data
  double ep = state->plasticStrain;
  double mu = state->shearModulus;
  ASSERT(mu > 0.0);

  // Calculate mu/mu_0
  double mu_mu_0 = mu/d_CM.mu_0;

  // Calculate Y <= Ymax
  double Ya = 1.0 + d_CM.beta*(ep - d_CM.epsilon_p0);
  ASSERT(Ya >= 0.0);
  double Y = pow(Ya, d_CM.n-1.0);

  double dsigma_dep = d_CM.sigma_0*mu_mu_0*d_CM.n*d_CM.beta*Y;
  return dsigma_dep;
}

///////////////////////////////////////////////////////////////////////////
/*  Compute the shear modulus. */
///////////////////////////////////////////////////////////////////////////
double
SCGPlastic::computeShearModulus(const PlasticityState* state)
{
  double eta = state->density/state->initialDensity;
  ASSERT(eta > 0.0);
  eta = pow(eta, 1.0/3.0);
  double mu = d_CM.mu_0*(1.0 + d_CM.A*state->pressure/eta - 
              d_CM.B*(state->temperature - 300.0));
  return mu;
}

///////////////////////////////////////////////////////////////////////////
/* Compute the melting temperature */
///////////////////////////////////////////////////////////////////////////
double
SCGPlastic::computeMeltingTemp(const PlasticityState* state)
{
  double eta = state->density/state->initialDensity;
  double power = 2.0*(d_CM.Gamma_0 - d_CM.a - 1.0/3.0);
  double Tm = state->initialMeltTemp*exp(2.0*d_CM.a*(1.0 - 1.0/eta))*
              pow(eta,power);
  return Tm;
}

double
SCGPlastic::evalDerivativeWRTTemperature(const PlasticityState* state,
					 const particleIndex )
{
  // Get the state data
  double ep = state->plasticStrain;

  // Calculate Y <= Ymax
  double Ya = 1.0 + d_CM.beta*(ep - d_CM.epsilon_p0);
  ASSERT(Ya >= 0.0);
  double Y = Min(d_CM.sigma_0*pow(Ya, d_CM.n), d_CM.Y_max);

  return -Y*d_CM.B;
}

double
SCGPlastic::evalDerivativeWRTPressure(const PlasticityState* state,
				      const particleIndex )
{
  // Get the state data
  double ep = state->plasticStrain;

  // Calculate Y <= Ymax
  double Ya = 1.0 + d_CM.beta*(ep - d_CM.epsilon_p0);
  ASSERT(Ya >= 0.0);
  double Y = Min(d_CM.sigma_0*pow(Ya, d_CM.n), d_CM.Y_max);

  double eta = state->density/state->initialDensity;
  return Y*d_CM.A/pow(eta,1.0/3.0);
}

