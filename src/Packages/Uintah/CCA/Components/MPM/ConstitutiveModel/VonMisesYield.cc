#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/VonMisesYield.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <math.h>

using namespace Uintah;
using namespace SCIRun;

VonMisesYield::VonMisesYield(ProblemSpecP&)
{
}
	 
VonMisesYield::~VonMisesYield()
{
}
	 
double 
VonMisesYield::evalYieldCondition(const double sigEqv,
                                  const double sigFlow,
                                  const double,
                                  const double,
                                  double& sig)
{
  sig = sigFlow;
  return (sigEqv*sigEqv-sigFlow*sigFlow);
}

void 
VonMisesYield::evalDerivOfYieldFunction(const Matrix3& sig,
					const double ,
					const double ,
					Matrix3& derivative)
{
  Matrix3 I; I.Identity();
  double trSig = sig.Trace();
  Matrix3 sigDev = sig - I*(trSig/3.0);
  derivative = sigDev*3.0;
}

void 
VonMisesYield::evalDevDerivOfYieldFunction(const Matrix3& sig,
					   const double ,
					   const double ,
					   Matrix3& derivative)
{
  Matrix3 I; I.Identity();
  double trSig = sig.Trace();
  Matrix3 sigDev = sig - I*(trSig/3.0);
  derivative = sigDev*3.0;
}

void 
VonMisesYield::computeElasPlasTangentModulus(const TangentModulusTensor& Ce,
					     const Matrix3& sigma, 
					     double sigY,
					     double dsigYdep,
					     double porosity,
					     double voidNuclFac,
					     TangentModulusTensor& Cep)
{
  // Calculate the derivative of the yield function wrt sigma
  Matrix3 f_sigma;
  evalDerivOfYieldFunction(sigma, sigY, porosity, f_sigma);

  // Calculate derivative wrt plastic strain 
  double f_q1 = dsigYdep;

  // Compute h_q1
  double sigma_f_sigma = sigma.Contract(f_sigma);
  double h_q1 = computePlasticStrainFactor(sigma_f_sigma, sigY);

  // Calculate elastic-plastic tangent modulus
  computeTangentModulus(Ce, f_sigma, f_q1, h_q1, Cep);
}

inline double
VonMisesYield::computePlasticStrainFactor(double sigma_f_sigma,
                                          double sigma_Y)
{
  return sigma_f_sigma/sigma_Y;
}

void 
VonMisesYield::computeTangentModulus(const TangentModulusTensor& Ce,
				     const Matrix3& f_sigma, 
				     double f_q1,
				     double h_q1,
				     TangentModulusTensor& Cep)
{
  double fqhq = f_q1*h_q1;
  Matrix3 Cr(0.0), rC(0.0);
  double rCr = 0.0;
  for (int ii = 0; ii < 3; ++ii) {
    for (int jj = 0; jj < 3; ++jj) {
      Cr(ii+1,jj+1) = 0.0;
      rC(ii+1,jj+1) = 0.0;
      for (int kk = 0; kk < 3; ++kk) {
	for (int ll = 0; ll < 3; ++ll) {
          Cr(ii+1,jj+1) += Ce(ii,jj,kk,ll)*f_sigma(kk+1,ll+1);
          rC(ii+1,jj+1) += f_sigma(kk+1,ll+1)*Ce(kk,ll,ii,jj);
        }
      }
      rCr += rC(ii+1,jj+1)*f_sigma(ii+1,jj+1);
    }
  }
  for (int ii = 0; ii < 3; ++ii) {
    for (int jj = 0; jj < 3; ++jj) {
      for (int kk = 0; kk < 3; ++kk) {
	for (int ll = 0; ll < 3; ++ll) {
          Cep(ii,jj,kk,ll) = Ce(ii,jj,kk,ll) - 
	    Cr(ii+1,jj+1)*rC(kk+1,ll+1)/(-fqhq + rCr);
	}  
      }  
    }  
  }  
}


