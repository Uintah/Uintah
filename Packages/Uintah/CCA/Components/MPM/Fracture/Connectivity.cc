#include "Connectivity.h"

#include <Core/Exceptions/InternalError.h>
#include <iostream>

namespace Uintah {
using namespace SCIRun;

Connectivity::
Connectivity()
{
  d_flag = 6560;
}

Connectivity::
Connectivity(int v)
{
  d_flag = v;
}

Connectivity::
Connectivity(const int info[8])
{
  setInfo(info);
}

void
Connectivity::
operator=(int v)
{
  d_flag = v;
}



void
Connectivity::
getInfo(int info[8]) const
{
  int flag = d_flag;
  int a;
  for(int n=0;n<8;n++) {
    a = flag /3;
    info[n] = flag - a*3;
    flag = a;
  }
}

void
Connectivity::
setInfo(const int info[8])
{
  int a = 1;
  d_flag = 0;
  for(int n=0;n<8;n++) {
    d_flag += info[n] * a;
    a *= 3;
  }
}

void
Connectivity::
modifyWeights(const int connectivity[8],double S[8],Cond cond)
{
  double N[8];
  
  int num = 0;

  if(cond==connect) {
    for(int i=0;i<8;++i)
    if(connectivity[i] == 1) {
      ++num;
      N[i] = 0;
    }
  
    for(int i=0;i<8;++i) {
      if(connectivity[i] == 1) N[i] += S[i];
      else {
        double delta = S[i] / num;
        for(int j=0;j<8;j++) {
          if(connectivity[j] == 1) N[j] += delta;
        }
      }
    }
  }
  else if(cond==contact) {
    for(int i=0;i<8;++i)
    if(connectivity[i] >= 1) {
      ++num;
      N[i] = 0;
    }
  
    for(int i=0;i<8;++i) {
      if(connectivity[i] >= 1) N[i] += S[i];
      else {
        double delta = S[i] / num;
        for(int j=0;j<8;j++) {
          if(connectivity[j] >= 1) N[j] += delta;
        }
      }
    }
  }

  for(int i=0;i<8;++i) S[i] = N[i];
}

void
Connectivity::
modifyShapeDerivatives(const int connectivity[8],Vector d_S[8],Cond cond)
{
  Vector d_N[8];
  
  int num = 0;

  if(cond==connect) {
    for(int i=0;i<8;++i)
    if(connectivity[i] == 1) {
      ++num;
      d_N[i] = Vector(0.,0.,0.);
    }

    for(int i=0;i<8;++i) {
      if(connectivity[i] == 1) d_N[i] += d_S[i];
      else {
        Vector delta = d_S[i] / num;
        for(int j=0;j<8;j++) {
          if(connectivity[j] == 1) d_N[j] += delta;
        }
      }
    }
  }
  else if(cond==contact) {
    for(int i=0;i<8;++i)
    if(connectivity[i] >= 1) {
      ++num;
      d_N[i] = Vector(0.,0.,0.);
    }

    for(int i=0;i<8;++i) {
      if(connectivity[i] >= 1) d_N[i] += d_S[i];
      else {
        Vector delta = d_S[i] / num;
        for(int j=0;j<8;j++) {
          if(connectivity[j] >= 1) d_N[j] += delta;
        }
      }
    }
  }

  for(int i=0;i<8;++i) d_S[i] = d_N[i];
}

} // End namespace Uintah


