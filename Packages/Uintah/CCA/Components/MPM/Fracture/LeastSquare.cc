#include "LeastSquare.h"

#include "Spline.h"
#include <Core/Geometry/Vector.h>

#include <iostream>

namespace Uintah {
LeastSquare::
LeastSquare(const Spline& spline)
{
  _spline = &spline;
}

void
LeastSquare::
clean()
{
  int n=4;
  for(int i=0;i<n;++i) {
    _equ.vec[i] = 0;
    for(int j=0;j<n;++j) {
      _equ.mat[i][j] = 0;
    }
  }
}

void
LeastSquare::
input(const Vector& x,double y)
{
  double weight = _spline->w(x);
  std::cout<<weight<<std::endl;
  std::cout<<x<<std::endl;
  std::cout<<y<<std::endl;

  int i,j;

  _equ.mat[0][0] += weight;
  for(j=0;j<3;++j) _equ.mat[0][j+1] += weight * x(j);
  _equ.vec[0] += weight * y;

  for(i=0;i<3;++i) {
    for(j=0;j<3;++j) _equ.mat[i+1][j+1] += weight * x(i) * x(j);
    _equ.vec[i+1] += weight * x(i) * y;
  }

  for(i=0;i<3;++i) _equ.mat[i+1][0] = _equ.mat[0][i+1];
  
}

void
LeastSquare::
output(double& a,Vector& b)
{
  _equ.solve();
  a = _equ.vec[0];
  for(int i=0;i<3;++i) b(i) = _equ.vec[i+1];
}


} // End namespace Uintah



