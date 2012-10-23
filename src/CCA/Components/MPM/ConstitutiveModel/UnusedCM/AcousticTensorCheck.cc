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

#include <CCA/Components/MPM/ConstitutiveModel/AcousticTensorCheck.h>
#include <Core/Math/TangentModulusTensor.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <cmath>
#include <vector>


using namespace Uintah;
using namespace std;

// NOTE :
// Most of this code is a silightly modified version of the 
// Tahoe AcousticTensorCheck class 

/* Local functions */

/*! Tests whether two normals are equivalent, checking to see whether they
 *   are the same within a given tolerance tol, or opposites within that same
 *   tolerance (opposite normals are considered equilvalent for this purpose).
 *   Returns true if the normals are distinct, false if not. 
 *   Both vectors should be normalized before being passed into this 
 *   function. */
bool normalCompare(const Vector& normal1, const Vector& normal2, double tol);
bool 
normalCompare(const Vector& normal1, const Vector& normal2, double tol)
{
  Vector diff = normal1 - normal2;
  Vector sum = normal1 + normal2;
  return ( diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2] < tol ||
           sum[0]*sum[0] + sum[1]*sum[1] + sum[2]*sum[2] < tol);
}

/*! As above, 10e-10 seems to be a good tolerance for this comparison */
bool normalCompare(const Vector& normal1, const Vector& normal2);
bool 
normalCompare(const Vector& normal1, const Vector& normal2)
{
  return normalCompare(normal1, normal2, 10e-10);
}

/*! Read in the sweep increment, number of theta checks, and number of phi
 *   checks from inputs file problem specification 
 *  \param ps Pointer to problem specification */
AcousticTensorCheck::AcousticTensorCheck(ProblemSpecP& ps)
{
  ps->require("d_sweepIncement",d_sweepInc);
  ps->require("numTheta",d_numTheta);
  ps->require("numPhi",d_numPhi);
}

AcousticTensorCheck::AcousticTensorCheck(const AcousticTensorCheck* cm)
{
  d_sweepInc = cm->d_sweepInc;
  d_numTheta = cm->d_numTheta;
  d_numPhi = cm->d_numPhi;
}

/*! Nothing to delete except the object itself */
AcousticTensorCheck::~AcousticTensorCheck()
{
}
         
/*! Check stability 
  \return true if unstable
  \return false if stable
*/
bool 
AcousticTensorCheck::checkStability(const Matrix3& ,
                                    const Matrix3& ,
                                    const TangentModulusTensor& tangentModulus,
                                    Vector& direction)
{
  return isLocalized(tangentModulus, direction);
}

/*! Check for localization */
//  theta =  Horizontal plane angle 
//  phi   =  Polar angle 
bool 
AcousticTensorCheck::isLocalized(const TangentModulusTensor& C,
                                 Vector& normal)
{
  // Constants
  Vector zero(0.0,0.0,0.0);

  // normalTol : Tolerance for convergence of normals, based on square of 
  //             norm of difference
  // setTol    : Tolerance to check if normals should be in normal set 
  // leastMin  : Tolerance for choosing normals w/ least determinant 
  double normalTol = 10e-20;  
  double setTol = 1.0e-7;     
  double leastMin = 2*setTol; 
  
  // detA      : Array to store the determinant of the acoustic tensor at 
  //             each increment
  // localMin  : Array pointing to local minima : 1 for local minimum, 0 if not
  double** detA =scinew double*[d_numTheta];
  int** localMin = scinew int*[d_numTheta]; 
  for (int ii = 0; ii < d_numTheta; ++ii) {
    detA[ii] =scinew double[d_numPhi];  
    localMin[ii] = scinew int[d_numPhi];
    for (int jj = 0; jj < d_numPhi; jj++) {
      detA[ii][jj] = 0.0;
      localMin[ii][jj] = 0;
    }
  }

  // Initial sweep through angles to find approximate local minima 
  findApproxLocalMins(detA, localMin, C);

  // Create a vector to store the set of normals
  vector<Vector> normalSet;

  // Newton iteration to determine minima
  normal = zero;
  for (int ii = 0; ii < d_numTheta; ii++) {
    for (int jj = 0 ; jj < d_numPhi; jj++) {
      if (localMin[ii][jj] ==1) {

        /* form starting approximate normal */
        double theta = M_PI/180.0*d_sweepInc*ii;
        double phi = M_PI/180.0*d_sweepInc*jj;
        normal[0] = cos(theta)*cos(phi);
        normal[1] = sin(theta)*cos(phi);
        normal[2] = sin(phi);

        /* Iteration to refine normal */
        Vector prevNormal(zero);
        int newtonCounter=0;      
        while (!normalCompare(normal, prevNormal, normalTol) && 
               newtonCounter <= 100 ) {
          newtonCounter++;
          ASSERT(!(newtonCounter > 100));
          prevNormal=normal;

          // Form acoustic tensor A and store determinant and inverse
          Matrix3 A(0.0);
          formAcousticTensor(normal, C, A);
          detA[ii][jj] = A.Determinant();
          Matrix3 AInv = A.Inverse();

          // Form Jmn=det(A)*Cmkln*(A^-1)lk
          Matrix3 J(0.0);
          for (int mm = 0; mm < 3; mm++) {
            for (int nn = 0; nn < 3; nn++) {
              for (int kk = 0; kk < 3; kk++) {
                for (int ll = 0; ll < 3; ll++) {
                  J(mm,nn) += detA[ii][jj]*C(mm,kk,ll,nn)*AInv(ll,kk);
                }
              }
            }
          }
          // find least eigenvector of J, the next approx normal
          normal = chooseNewNormal(prevNormal, J);        
        } //end while statement

        // Determine which normals have least value of DetA and record
        // them. Typically, there are two distinct normals which produce
        // the same minimum value. Choose between these later

        if ((fabs(detA[ii][jj]-leastMin) < setTol) || 
            (fabs((detA[ii][jj]-leastMin)/leastMin) < setTol)) {

          // add normal to auto array and reset leastMin
          normalSet.push_back(normal);
          leastMin = detA[ii][jj];
        }
      } //end if localMin  
    } //end j         
  } //end i

  if (leastMin > setTol) {
    //no bifurcation has occured
    normal = zero;
    return false;
  } else {
    //bifurcation has occured
    //choose normal from set of normals producing least detA
    normal = chooseNormalFromNormalSet(normalSet, C); 
    return true;
  }
}


// Initial sweep in d_sweepInc-degree increments to determine approximate 
// local minima of determinant of acoustic tensor A as fn of normal. Sweeps
// only over half sphere since A(n) = A(-n)
void 
AcousticTensorCheck::findApproxLocalMins(double** detA, 
                                         int** localMin, 
                                         const TangentModulusTensor& C)
{
  // First sweep :
  // 1) Set initial normal
  // 2) Form the acoustic tensor
  // 3) Find determinant of A as function of angle 
  //    (case where j = numPhi - 1, i.e. pole)
  Vector normal(0.0, 0.0, 0.0);
  normal[2] = 1.0;
  Matrix3 A(0.0);
  formAcousticTensor(normal, C, A);
  double det = A.Determinant();
  for (int ii = 0; ii < d_numTheta; ii++) detA[ii][d_numPhi-1] = det;

  // Rest of the sphere by increments
  // 1) Find current polar co-ordinates
  // 2) Calculate normal
  // 3) Form the acoustic tensor
  // 4) Find determinant of A as function of angle 
  for (int ii = 0; ii < d_numTheta; ii++) {
    for (int jj = 0; jj < d_numPhi-1; jj++) {
      double theta = M_PI/180.0*d_sweepInc*ii;
      double phi = M_PI/180.0*d_sweepInc*jj;
      normal[0] = cos(theta)*cos(phi);
      normal[1] = sin(theta)*cos(phi);
      normal[2] = sin(phi);
      formAcousticTensor(normal, C, A);
      detA[ii][jj] = A.Determinant();
    }
  }

  // Check detA against 4 values around it to see if it is local min 
  for (int ii = 0; ii < d_numTheta; ii++) {
    for (int jj = 0; jj < d_numPhi; jj++) {

      if (jj == d_numPhi-1) {
        // Pole, checks only vs. points around it
        if (ii == 0) {
          localMin[ii][jj] = 1;
          for (int kk = 0; kk < d_numTheta; kk++) {
            if (detA[ii][jj] > detA[kk][jj-1]) {
              localMin[ii][jj] = 0;
              break;
            }
          }
        }
      } else {
        int ll = (ii-1)%d_numTheta; 
        int mm = (ii+1)%d_numTheta; 
        if (detA[ii][jj] < detA[ll][jj]) {
          if (detA[ii][jj] <= detA[mm][jj]) {
            if (detA[ii][jj] < detA[ii][jj+1]) {
              if (jj == 0) {
                // check against normal across sphere
                // only need to check to 180 degrees, rest are opposite
                // of already tested, speeds up plane strain problems 
                int mid = d_numTheta/2;
                if (ii < mid) { 
                  if (detA[ii][jj] <= detA[ii+mid][jj+1]) localMin[ii][jj] = 1;
                }
              } else {
                //standard case
                if (detA[ii][jj] <= detA[ii][jj-1]) localMin[ii][jj] = 1;
              }
            }
          }
        }
      }
    } //end j, end i   
  }
} // end FindApproxLocalMins

// Form the acoustic tensor
void
AcousticTensorCheck::formAcousticTensor(const Vector& normal,
                                        const TangentModulusTensor& C,
                                        Matrix3& A)
{
  // Form the acoustic tensor  A = n*C*n
  for (int ii = 0; ii < 3; ++ii) {
    for (int kk = 0; kk < 3; ++kk) {
      A(ii,kk) = 0.0;
      for (int jj = 0; jj < 3; ++jj) {
        for (int ll = 0; ll < 3; ++ll) {
           double cc = C(ii,jj,kk,ll);
           A(ii,kk) += cc*normal[jj]*normal[ll];
        }
      }
    }
  }
}

// Finds next iteration of on a normal by by finding Eigenvalues of Matrix
// J and choosing the best one, i.e. closest in norm to previous vector 
Vector 
AcousticTensorCheck::chooseNewNormal(Vector& prevNormal, 
                                     Matrix3& J)
{
  // Find the eigenvalues of J
  double eigVal[3];
  int numEV = J.getEigenValues(eigVal[0], eigVal[1], eigVal[2]);

  // chooses eigvector by closest approx to previous normal
  double maxInner = 0.0;
  Vector normal(0.0,0.0,0.0);
  for (int ii = 0; ii < numEV; ii++) {  
    vector<Vector> eigVec = J.getEigenVectors(eigVal[ii], eigVal[0]);
    Vector trialNormal = eigVec[0]/eigVec[0].length();
    double inner = fabs(Dot(trialNormal,prevNormal));

    if (ii == 0 ) {
      maxInner = inner;
      normal = trialNormal;
    } else if (inner > maxInner) {
      maxInner = inner;
      normal = trialNormal;
    }       
  }
  return normal;
}


// Chooses normal from a set that have essentially same detA that is least
// of all the normals. Typically there are two
Vector 
AcousticTensorCheck::chooseNormalFromNormalSet(vector <Vector> &normalSet, 
                                               const TangentModulusTensor &C)
{
  // First item
  vector<Vector>::iterator iter = normalSet.begin();
  Matrix3 A(0.0);
  formAcousticTensor(*iter, C, A);
  double detA = A.Determinant();
  double leastMin = detA;
  Vector bestNormal = *iter;
  ++iter;
 
  // The rest
  for (; iter < normalSet.end(); ++iter) {
    // Currently chooses normal by least minimum value of the determinant
    // of the acoustic tensor A. Experience suggests that two unique 
    // normals are created and that this choice is based simply on 
    // numerical error. Better to choose another criterion. Wells and Sluys
    // (2001) suggest maximum plastic dissipation, which may be best
    // handled in the function calling IsLocalized_SS, after it is called.  
    formAcousticTensor(*iter, C, A);
    detA = A.Determinant();
    if (detA < leastMin) {
      leastMin = detA;
      bestNormal = *iter;
    }
  }      
  return bestNormal;
}

