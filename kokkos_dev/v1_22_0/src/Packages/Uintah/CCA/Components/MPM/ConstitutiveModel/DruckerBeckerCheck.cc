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
  //cout << "stress = \n";
  //cout << stress << endl;
  //cout << "Eigenvalues : " << sig << endl;
  //cout << "Eigenvectors : " << evec << endl;

  /*  OLD CODE USING MATRIX3 Routines
  // If all three principal stresses are equal, numEV = 1,
  // If two principal stresses are equal, numEV = 2, and the largest
  // is in sig[0].
  // If no principal stresses are equal, numEV = 3, and the largest
  // is in sig[0] and the smallest is in sig[2].
  double sig[3];
  int numEV = stress.getEigenValues(sig[0], sig[1], sig[2]);

  cout << "stress = \n";
  cout << stress;
  cout << "numEV = " << numEV << " s1 = " << sig[0] << " s2 = " << sig[1]
       << " s3 = " << sig[2] << endl;
  // Get the eigenvectors of the stress tensor
  vector<Vector> eigVec;
  for (int ii = 0; ii < numEV; ii++)  
     eigVec = stress.getEigenVectors(sig[ii], sig[0]);

  // Calculate the coefficients of the quadric
  if (numEV == 1) sig[2] = sig[0];
  else if (numEV == 2) sig[2] = sig[1];
  */

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
    double yplus = (-B + sqrt(B2_4AC))/(2.0*A);
    double yminus = (-B - sqrt(B2_4AC))/(2.0*A);
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

