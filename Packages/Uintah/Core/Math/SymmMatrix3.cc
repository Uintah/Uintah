#include "SymmMatrix3.h"
#include "./TntJama/tnt.h"
#include "./TntJama/jama_eig.h"

using namespace TNT;
using namespace JAMA;
using namespace SCIRun;

void
SymmMatrix3::eigen(Vector& eval, Matrix3& evec)
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
