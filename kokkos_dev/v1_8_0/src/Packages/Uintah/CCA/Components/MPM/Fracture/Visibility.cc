#include "Visibility.h"

#include <Core/Exceptions/InternalError.h>
#include <iostream>

namespace Uintah {
using namespace SCIRun;

Visibility::
Visibility()
{
  d_flag = 255;
}

void
Visibility::
operator=(int v)
{
  d_flag = v;
}

bool
Visibility::
visible(const int i) const
{
  return ( ( d_flag>>i ) & 1 ) ? true : false;
}

void
Visibility::
setVisible(const int i)
{
  d_flag = d_flag | (1<<i);
}

void
Visibility::
setUnvisible(const int i)
{
  short visibility = 255 ^ d_flag;
  visibility = visibility | (1<<i);
  
  d_flag = 255 ^ visibility;
}

void
Visibility::
modifyWeights(double S[8]) const
{
  double N[8];
  
  int num = 0;
  for(int i=0;i<8;++i)
  if(visible(i)) {
    ++num;
    N[i] = 0;
  }
  
/*
  if(num == 0) {
    std::cout<<d_flag<<std::endl;
    throw InternalError("Isolated particle");
  }
*/

  for(int i=0;i<8;++i) {
    if(visible(i)) N[i] += S[i];
    else {
      double delta = S[i] / num;
      for(int j=0;j<8;j++) {
        if(visible(j)) N[j] += delta;
      }
    }
  }

  for(int i=0;i<8;++i) S[i] = N[i];
}

void
Visibility::
modifyShapeDerivatives(Vector d_S[8]) const
{
  Vector d_N[8];
  
  int num = 0;
  for(int i=0;i<8;++i)
  if(visible(i)) {
    ++num;
    d_N[i] = Vector(0.,0.,0.);
  }

/*
  if(num == 0) {
    std::cout<<d_flag<<std::endl;
    throw InternalError("Isolated particle");
  }
*/

  for(int i=0;i<8;++i) {
    if(visible(i)) d_N[i] += d_S[i];
    else {
      Vector delta = d_S[i] / num;
      for(int j=0;j<8;j++) {
        if(visible(j)) d_N[j] += delta;
      }
    }
  }

  for(int i=0;i<8;++i) d_S[i] = d_N[i];
}
} // End namespace Uintah


