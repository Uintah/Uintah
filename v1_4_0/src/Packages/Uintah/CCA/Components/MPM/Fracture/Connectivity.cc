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
  for(int i=0;i<8;++i) {
    N[i] = S[i];
    S[i] = 0;
  }
  
  int num = 0;

  if(cond==connect) {
    for(int i=0;i<8;++i) {
      if(connectivity[i] == 1) ++num;
    }
    if(num==0) return;
  
    for(int i=0;i<8;++i) {
      if(connectivity[i] == 1) S[i] += N[i];
      else {
        double delta = N[i] / num;
        for(int j=0;j<8;j++) {
          if(connectivity[j] == 1) S[j] += delta;
        }
      }
    }
  }
  else if(cond==contact) {
    for(int i=0;i<8;++i) {
      if(connectivity[i] >= 1) ++num;
    }
    if(num==0) return;
  
    for(int i=0;i<8;++i) {
      if(connectivity[i] >= 1) S[i] += N[i];
      else {
        double delta = N[i] / num;
        for(int j=0;j<8;j++) {
          if(connectivity[j] >= 1) S[j] += delta;
        }
      }
    }
  }
}

void
Connectivity::
modifyShapeDerivatives(const int connectivity[8],Vector d_S[8],Cond cond)
{
  Vector d_N[8];

  for(int i=0;i<8;++i) {
    d_N[i] = d_S[i];
    d_S[i] = Vector(0.,0.,0.);
  }
  
  int num = 0;

  if(cond==connect) {
    for(int i=0;i<8;++i) {
      if(connectivity[i] == 1) ++num;
    }
    if(num==0) return;

    for(int i=0;i<8;++i) {
      if(connectivity[i] == 1) d_S[i] += d_N[i];
      else {
        Vector delta = d_N[i] / num;
        for(int j=0;j<8;j++) {
          if(connectivity[j] == 1) d_S[j] += delta;
        }
      }
    }
  }
  else if(cond==contact) {
    for(int i=0;i<8;++i) {
      if(connectivity[i] >= 1) ++num;
    }
    if(num==0) return;

    for(int i=0;i<8;++i) {
      if(connectivity[i] >= 1) d_S[i] += d_N[i];
      else {
        Vector delta = d_N[i] / num;
        for(int j=0;j<8;j++) {
          if(connectivity[j] >= 1) d_S[j] += delta;
        }
      }
    }
  }
}

} // End namespace Uintah


