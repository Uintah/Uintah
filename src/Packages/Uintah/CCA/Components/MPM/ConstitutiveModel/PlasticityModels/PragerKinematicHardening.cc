#ifdef __APPLE__
// This is a hack.  gcc 3.3 #undefs isnan in the cmath header, which
// make the isnan function not work.  This define makes the cmath header
// not get included since we do not need it anyway.
#  define _CPP_CMATH
#endif

#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/PlasticityModels/PragerKinematicHardening.h>
#include <cmath>

#ifdef _WIN32
#include <process.h>
#include <cfloat>
#define isnan _isnan
#endif

using namespace Uintah;
using namespace SCIRun;

PragerKinematicHardening::PragerKinematicHardening(ProblemSpecP& ps)
{
  d_cm.beta = 1.0;
  ps->get("beta", d_cm.beta);
  ps->require("hardening_modulus", d_cm.hardening_modulus);
}
         
PragerKinematicHardening::PragerKinematicHardening(const PragerKinematicHardening* cm)
{
  d_cm.beta = cm->d_cm.beta;
  d_cm.hardening_modulus = cm->d_cm.hardening_modulus;
}
         
PragerKinematicHardening::~PragerKinematicHardening()
{
}

void PragerKinematicHardening::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP plastic_ps = ps->appendChild("kinematic_hardening_model");
  plastic_ps->setAttribute("type","prager_hardening");

  plastic_ps->appendElement("beta", d_cm.beta);
  plastic_ps->appendElement("hardening_modulus", d_cm.hardening_modulus);
}

/* Assumes von Mises plasticity and an associated flow rule.  The back stress
is given by the rate equation D/Dt(beta) = 2/3~gammadot~Hprime~df/dsigma */
void 
PragerKinematicHardening::computeBackStress(const PlasticityState* state,
                                            const double& delT,
                                            const particleIndex idx,
                                            const double& delLambda,
                                            const Matrix3& df_dsigma_normal_new,
                                            const Matrix3& backStress_old,
                                            Matrix3& backStress_new) 
{
  // Get the hardening modulus (constant for Prager kinematic hardening)
  double H_prime = d_cm.beta*d_cm.hardening_modulus;
  double stt = sqrt(2.0/3.0);

  // Compute updated backstress
  backStress_new = backStress_old + df_dsigma_normal_new*(delLambda*H_prime*stt);

  return;
}

void 
PragerKinematicHardening::eval_h_beta(const Matrix3& df_dsigma,
                                      const PlasticityState* ,
                                      Matrix3& h_beta)
{
  double H_prime = d_cm.beta*d_cm.hardening_modulus;
  h_beta = df_dsigma*(2.0/3.0*H_prime);
  return;
}


