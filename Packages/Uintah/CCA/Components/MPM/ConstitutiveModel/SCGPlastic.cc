
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
SCGPlastic::computeFlowStress(const double& plasticStrainRate,
			      const double& ,
                              const double& T,
                              const double& delT,
                              const double& ,
                              const MPMMaterial* ,
                              const particleIndex idx)
{
  // Calculate strain rate and incremental strain
  double edot = plasticStrainRate;
  double delEps = edot*delT;

  // Check if temperature is correct
  if (T <= 0.0) {
    ostringstream desc;
    desc << "**SCG ERROR** Absolute temperature <= 0." << endl;
    desc << "T = " << T << " edot = " << edot << endl;
    throw InvalidValue(desc.str());
  }

  double sigma = 0.0;
  return sigma;
}

/*! The evolving internal variable is \f$q = \hat\sigma_e\f$.  If the 
  evolution equation for internal variables is of the form 
  \f$ \dot q = \gamma h (\sigma, q) \f$, then 
  \f[
  \dot q = \frac{d\hat\sigma_e}{dt} 
  = \frac{d\hat\sigma_e}{d\epsilon} \frac{d\epsilon}{dt}
  = \theta \dot\epsilon .
  \f] 
  If \f$\dot\epsilon = \gamma\f$, then \f$ \theta = h \f$.
  Also, \f$ f_q = \frac{\partial f}{\partial \hat\sigma_e} \f$.
  For the von Mises yield condition, \f$(f)\f$, 
  \f$ f_q = \frac{\partial \sigma}{\partial \hat\sigma_e} \f$
  where \f$\sigma\f$ is the SCG flow stress.
*/
void 
SCGPlastic::computeTangentModulus(const Matrix3& sig,
				  const double& plasticStrainRate,
				  const double& plasticStrain,
				  double T,
				  double ,
				  const particleIndex idx,
				  const MPMMaterial* ,
				  TangentModulusTensor& Ce,
				  TangentModulusTensor& Cep)
{
  // Calculate the deviatoric stress and rate of deformation
  Matrix3 one; one.Identity();
  Matrix3 sigdev = sig - one*(sig.Trace()/3.0);

  // Calculate the equivalent stress and strain rate
  double sigeqv = sqrt(sigdev.NormSquared()); 
  double edot = plasticStrainRate;

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
	    Cr(ii1,jj1)*rC(kk1,ll+1)/rCr;
	}  
      }  
    }  
  }  
}


double
SCGPlastic::evalDerivativeWRTTemperature(double edot,
                                         double ep,
                                         double T,
					 const particleIndex idx)
{
}

double
SCGPlastic::evalDerivativeWRTPlasticStrain(double edot,
                                           double ep,
                                           double T,
					   const particleIndex idx)
{
}


void
SCGPlastic::evalDerivativeWRTScalarVars(double edot,
                                        double ep,
                                        double T,
                                        const particleIndex idx,
                                        Vector& derivs)
{
  derivs[0] = evalDerivativeWRTPlasticStrain(edot, ep, T, idx);
  derivs[1] = evalDerivativeWRTTemperature(edot, ep, T, idx);
  derivs[2] = derivs[0];
}
