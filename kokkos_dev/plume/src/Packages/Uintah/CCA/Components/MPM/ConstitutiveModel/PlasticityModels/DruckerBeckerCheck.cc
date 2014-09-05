#include "DruckerBeckerCheck.h"
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Math/SymmMatrix3.h>
#include <math.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>


using namespace Uintah;
using namespace std;

DruckerBeckerCheck::DruckerBeckerCheck(ProblemSpecP& )
{
}

DruckerBeckerCheck::DruckerBeckerCheck(const DruckerBeckerCheck* )
{
}

DruckerBeckerCheck::~DruckerBeckerCheck()
{
}
	 
bool 
DruckerBeckerCheck::checkStability(const Matrix3& stress,
				   const Matrix3& deformRate ,
				   const TangentModulusTensor& M ,
				   Vector& )
{
  // Do the Drucker stability check
  Matrix3 stressRate(0.0);
  M.contract(deformRate, stressRate);
  double val = stressRate.Contract(deformRate);
  if (val <= 0.0) return true; // Bifurcation

  // Do the Becker check
  // Find the magnitudes and directions of the principal stresses
  
  SymmMatrix3 sigma(stress);
  Vector sig(0.0,0.0,0.0);
  Matrix3 evec;
  sigma.eigen(sig, evec);

  double A = M(2,0,2,0)*(-sig[2] + 2.0*M(2,2,2,2));
  double C = M(2,0,2,0)*(-sig[0] + 2.0*M(0,0,0,0));
  double B = M(2,0,2,0)*(sig[2] - 2.0*M(0,0,2,2) + sig[0] - 2.0*M(2,2,0,0)) +
             sig[0]*(M(2,2,0,0) - M(2,2,2,2)) + 
             sig[2]*(M(0,0,2,2) - M(0,0,0,0)) +
             2.0*(-M(0,0,2,2)*M(2,2,0,0)+M(0,0,0,0)*M(2,2,2,2));

  // Solve the quadric
  // Substitute x^2 by y and solve the quadratic
  double B2_4AC = B*B - 4.0*A*C;
  if (B2_4AC < 0.0) {
    // No real roots - no bifurcation
    return false;
  } else {
    ASSERT(!(A == 0));
    double yplus = (-B + sqrt(B2_4AC))/(2.0*(A+1.0e-20));
    double yminus = (-B - sqrt(B2_4AC))/(2.0*(A+1.0e-20));
    if (yplus < 0.0 && yminus < 0.0) {
      // No real roots - no bifurcation
      return false;
    } else {
      if (yplus < 0.0 || yminus < 0.0) {
        // Two real roots -  bifurcation ? (parabolic)
        return false;
      } 
    }
  }
  // Four real roots -  bifurcation 
  return true;
}

