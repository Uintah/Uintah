#include "Equation.h"

#include <iostream>

using namespace Uintah;
using namespace std;

int main()
{
  Equation equ;
  equ.mat[0][0] = 1;
  equ.mat[0][1] = 3;
  equ.mat[0][2] = 7;
  equ.mat[0][3] = 0;

  equ.mat[1][0] = 5;
  equ.mat[1][1] = 3;
  equ.mat[1][2] = 4;
  equ.mat[1][3] = 2;

  equ.mat[2][0] = 8;
  equ.mat[2][1] = 7;
  equ.mat[2][2] = 2;
  equ.mat[2][3] = 1;

  equ.mat[3][0] = 1;
  equ.mat[3][1] = 2;
  equ.mat[3][2] = 0;
  equ.mat[3][3] = 3;
  
  equ.vec[0] = 28;
  equ.vec[1] = 31;
  equ.vec[2] = 32;
  equ.vec[3] = 17;

  equ.solve();
  for(int i=0;i<4;++i) cout<<equ.vec[i]<<" ";
  cout<<endl;
}
