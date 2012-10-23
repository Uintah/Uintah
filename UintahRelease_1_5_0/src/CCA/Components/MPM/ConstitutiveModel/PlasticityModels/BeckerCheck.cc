/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "BeckerCheck.h"
#include <Core/Math/TangentModulusTensor.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Math/SymmMatrix3.h>
#include <cmath>
#include <vector>


using namespace Uintah;
using namespace std;

BeckerCheck::BeckerCheck(ProblemSpecP& )
{
}

BeckerCheck::BeckerCheck(const BeckerCheck* )
{
}

BeckerCheck::~BeckerCheck()
{
}

void BeckerCheck::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP stability_ps = ps->appendChild("stability_check");
  stability_ps->setAttribute("type","becker");
}
         
bool 
BeckerCheck::checkStability(const Matrix3& stress ,
                            const Matrix3& ,
                            const TangentModulusTensor& M,
                            Vector& )
{
  // Find the magnitudes and directions of the principal stresses
        
  SymmMatrix3 sigma(stress);
  Vector sig(0.0,0.0,0.0);
  Matrix3 evec;
  sigma.eigen(sig, evec);
  //cout << "stress = \n";
  //cout << stress << endl;
  //cout << "Eigenvalues : " << sig << endl;
  //cout << "Eigenvectors : " << evec << endl;

  /* OLD CALC USING MATRIX3 methods
  // If all three principal stresses are equal, numEV = 1,
  // If two principal stresses are equal, numEV = 2, and the largest
  // is in sig[0].
  // If no principal stresses are equal, numEV = 3, and the largest
  // is in sig[0] and the smallest is in sig[2].
  double sig[3];
  int numEV = stress.getEigenValues(sig[0], sig[1], sig[2]);

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

