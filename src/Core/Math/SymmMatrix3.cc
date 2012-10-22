/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#include "SymmMatrix3.h"
#include "./TntJama/tnt.h"
#include "./TntJama/jama_eig.h"

using namespace TNT;
using namespace JAMA;
using namespace SCIRun;

namespace Uintah{

  void
  SymmMatrix3::eigen(SCIRun::Vector& eval, Matrix3& evec)
  {
    // Convert the current matrix into a 2x2 TNT Array
    TNT::Array2D<double> A = toTNTArray2D();

    // Compute the eigenvectors using JAMA
    JAMA::Eigenvalue<double> eig(A);
    TNT::Array1D<double> d(3);
    eig.getRealEigenvalues(d);
    TNT::Array2D<double> V(3,3);
    eig.getV(V);

    // Sort in descending order
    for (int ii = 0; ii < 2; ++ii) {
      int kk = ii;
      double valk = d[ii];
      for (int jj = ii+1; jj < 3; jj++) {
        double valj = d[jj];
        if (valj > valk) 
          {
            kk = jj;
            valk = valj; 
          }
      }
      if (kk != ii) {
        double temp = d[ii];
        d[ii] = d[kk];
        d[kk] = temp;
        for (int ll = 0; ll < 3; ++ll) {
          temp = V[ll][ii];
          V[ll][ii] = V[ll][kk];
          V[ll][kk] = temp;
        }
      }
    }

    // Store in eval and evec
    for (int ii = 0; ii < 3; ++ii) {
      eval[ii] = d[ii];
      for (int jj = 0; jj < 3; ++jj) {
        evec(ii,jj) = V[ii][jj];
      }
    }

  }

  SymmMatrix3
  SymmMatrix3::Deviatoric() const
  {
    SymmMatrix3 matDev;
    double traceby3 = Trace()/3.0;
    matDev.mat3[0] = mat3[0] - traceby3;
    matDev.mat3[1] = mat3[1] - traceby3;
    matDev.mat3[2] = mat3[2] - traceby3;
    matDev.mat3[3] = mat3[3];
    matDev.mat3[4] = mat3[4];
    matDev.mat3[5] = mat3[5];
    return matDev;
  }

  double
  SymmMatrix3::Norm() const
  {
    double normSq = mat3[0]*mat3[0] + mat3[1]*mat3[1] + mat3[2]*mat3[2] +
      2.0*(mat3[3]*mat3[3] + mat3[4]*mat3[4] + mat3[5]*mat3[5]);
    return (sqrt(normSq));
  }

  void 
  SymmMatrix3::Dyad(const SymmMatrix3& V, double dyad[6][6]) const
  {
    for (int ii=0; ii < 6; ++ii) {
      double nn = mat3[ii];
      double mm = V.mat3[ii];
      dyad[ii][ii] = nn*mm;
      for (int jj=ii+1; jj < 6; ++jj) {
        dyad[ii][jj] = nn*V.mat3[jj];
        dyad[jj][ii] = dyad[ii][jj];
      }
    }
  }

  Matrix3 
  SymmMatrix3::Multiply(const SymmMatrix3& V) const
  {
    Matrix3 dot;
    double u1 = mat3[0]; double u2 = mat3[1]; double u3 = mat3[2];
    double u4 = mat3[3]; double u5 = mat3[4]; double u6 = mat3[5];
    double v1 = V.mat3[0]; double v2 = V.mat3[1]; double v3 = V.mat3[2];
    double v4 = V.mat3[3]; double v5 = V.mat3[4]; double v6 = V.mat3[5];
    dot(0,0) = u1*v1 + u6*v6 + u5*v5; 
    dot(0,1) = u1*v6 + u6*v2 + u5*v4; 
    dot(0,2) = u1*v5 + u6*v4 + u5*v3; 
    dot(1,0) = u6*v1 + u2*v6 + u4*v5; 
    dot(1,1) = u6*v6 + u2*v2 + u4*v4; 
    dot(1,2) = u6*v5 + u2*v4 + u4*v3; 
    dot(2,0) = u5*v1 + u4*v6 + u3*v5; 
    dot(2,1) = u5*v6 + u4*v2 + u3*v4; 
    dot(2,2) = u5*v5 + u4*v4 + u3*v3; 
    return dot;
  }

  SymmMatrix3
  SymmMatrix3::Square() const
  {
    SymmMatrix3 square;
    double u1 = mat3[0]; double u2 = mat3[1]; double u3 = mat3[2];
    double u4 = mat3[3]; double u5 = mat3[4]; double u6 = mat3[5];
    square.mat3[0] = u1*u1 + u6*u6 + u5*u5; 
    square.mat3[1] = u6*u6 + u2*u2 + u4*u4; 
    square.mat3[2] = u5*u5 + u4*u4 + u3*u3; 
    square.mat3[3] = u6*u5 + u2*u4 + u4*u3; 
    square.mat3[4] = u1*u5 + u6*u4 + u5*u3; 
    square.mat3[5] = u1*u6 + u6*u2 + u5*u4; 
    return square;
  }

  double 
  SymmMatrix3::Contract(const SymmMatrix3& V) const
  {
    double contract = mat3[0]*V.mat3[0] + mat3[1]*V.mat3[1] + mat3[2]*V.mat3[2] +
      2.0*(mat3[3]*V.mat3[3] + mat3[4]*V.mat3[4] + 
           mat3[5]*V.mat3[5]);
    return contract;
  }

} // namespace Uintah
