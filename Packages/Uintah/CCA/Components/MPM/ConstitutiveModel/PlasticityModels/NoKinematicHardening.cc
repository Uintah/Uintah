#ifdef __APPLE__
// This is a hack.  gcc 3.3 #undefs isnan in the cmath header, which
// make the isnan function not work.  This define makes the cmath header
// not get included since we do not need it anyway.
#  define _CPP_CMATH
#endif

#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/PlasticityModels/NoKinematicHardening.h>
#include <math.h>

#ifdef _WIN32
#include <process.h>
#include <float.h>
#define isnan _isnan
#endif

using namespace Uintah;
using namespace SCIRun;

NoKinematicHardening::NoKinematicHardening()
{
}
         
NoKinematicHardening::NoKinematicHardening(ProblemSpecP& ps)
{
}
         
NoKinematicHardening::NoKinematicHardening(const NoKinematicHardening* cm)
{
}
         
NoKinematicHardening::~NoKinematicHardening()
{
}

void NoKinematicHardening::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP plastic_ps = ps->appendChild("kinematic_hardening_model");
  plastic_ps->setAttribute("type","none");
}

void 
NoKinematicHardening::computeBackStress(const PlasticityState* state,
                                        const double& delT,
                                        const particleIndex idx,
                                        const double& delLambda,
                                        const Matrix3& df_dsigma_new,
                                        const Matrix3& backStress_old,
                                        Matrix3& backStress_new)
{
  Matrix3 Zero(0.0);
  backStress_new = Zero;
  return;
}



