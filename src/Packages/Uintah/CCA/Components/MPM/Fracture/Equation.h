#ifndef UINTAH_MPM_EQUATION
#define UINTAH_MPM_EQUATION

#include <math.h>
#include "Packages/Uintah/CCA/Components/MPM/Util/Matrix3.h"

namespace Uintah {

class Equation {
public:
  void             solve();
                   Equation();
  double           mat[4][4];
  double           vec[4];
};

template<class T>
void swap(T& a, T& b);

template<class T>
T SQR(const T& a);

#define SIGN(a,b) ((b) > 0.0 ? fabs(a) : -fabs(a))

//Computes (a^2 + b^2 ) ^ (1/2) without destructive underflow or overflow.
double pythag(double a, double b);

double getMaxEigenvalue(const Matrix3& mat, Vector& eigenVector);
void QLAlgorithm(Matrix3& z, Vector& d);
void HouseholderReduction(Matrix3& z, Vector& d, Vector& e);

} // End namespace Uintah

#endif
