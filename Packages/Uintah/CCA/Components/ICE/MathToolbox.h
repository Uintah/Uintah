#ifndef UINTAH_MATHTOOLBOX_H
#define UINTAH_MATHTOOLBOX_H

#include <vector>


namespace SCIRun {
  class DenseMatrix;
}

namespace Uintah {

  using namespace SCIRun;
  using namespace std;

  void matrixInverse(int numMatls, DenseMatrix& a, DenseMatrix& aInverse);
  void matrixSolver(int numMatls, DenseMatrix& a,vector<double>& b, 
		    vector<double>& X);
  void multiplyMatrixAndVector(int numMatls, DenseMatrix& a, vector<double>& b,
			       vector<double>& X);
  double conditionNumber(const int numMatls,const DenseMatrix& a);

}


#endif
