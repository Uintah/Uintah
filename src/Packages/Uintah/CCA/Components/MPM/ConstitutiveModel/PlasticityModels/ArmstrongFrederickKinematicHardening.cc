#ifdef __APPLE__
// This is a hack.  gcc 3.3 #undefs isnan in the cmath header, which
// make the isnan function not work.  This define makes the cmath header
// not get included since we do not need it anyway.
#  define _CPP_CMATH
#endif

#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/PlasticityModels/ArmstrongFrederickKinematicHardening.h>
#include <math.h>

#ifdef _WIN32
#include <process.h>
#include <float.h>
#define isnan _isnan
#endif

using namespace Uintah;
using namespace SCIRun;

ArmstrongFrederickKinematicHardening::ArmstrongFrederickKinematicHardening(ProblemSpecP& ps)
{
  d_cm.beta = 1.0;
  ps->get("beta", d_cm.beta);
  ps->require("hardening_modulus_1", d_cm.hardening_modulus_1);
  ps->require("hardening_modulus_2", d_cm.hardening_modulus_2);
}
         
ArmstrongFrederickKinematicHardening::ArmstrongFrederickKinematicHardening(const ArmstrongFrederickKinematicHardening* cm)
{
  d_cm.beta = cm->d_cm.beta;
  d_cm.hardening_modulus_1 = cm->d_cm.hardening_modulus_1;
  d_cm.hardening_modulus_2 = cm->d_cm.hardening_modulus_2;
}
         
ArmstrongFrederickKinematicHardening::~ArmstrongFrederickKinematicHardening()
{
}

void ArmstrongFrederickKinematicHardening::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP plastic_ps = ps->appendChild("kinematic_hardening_model");
  plastic_ps->setAttribute("type","armstrong_frederick_hardening");

  plastic_ps->appendElement("beta", d_cm.beta);
  plastic_ps->appendElement("hardening_modulus_1", d_cm.hardening_modulus_1);
  plastic_ps->appendElement("hardening_modulus_2", d_cm.hardening_modulus_2);
}

void 
ArmstrongFrederickKinematicHardening::computeBackStress(const PlasticityState* state,
                                                        const double& delT,
                                                        const particleIndex idx,
                                                        const double& delLambda,
                                                        const Matrix3& df_dsigma_normal_new,
                                                        const Matrix3& backStress_old,
                                                        Matrix3& backStress_new)
{
  // Get the hardening modulus 
  double H_1 = d_cm.beta*d_cm.hardening_modulus_1;
  double H_2 = d_cm.beta*d_cm.hardening_modulus_2;
  double stt = sqrt(3.0/2.0);
  double o_stt = 1.0/stt;
  double denom = 1.0/(1.0 + stt*H_2*delLambda);

  // Compute updated backstress
  backStress_new = backStress_old + df_dsigma_normal_new*(delLambda*H_1*o_stt);
  backStress_new = backStress_new*denom;

  return;
}



