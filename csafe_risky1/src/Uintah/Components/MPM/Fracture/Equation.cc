#include "Equation.h"

#include <math.h>

namespace Uintah {
namespace MPM {

Equation::
Equation()
{
}

void
Equation::
solve()
{
  int i,j,k,n,io,jo,i1;
  n = 4;
  
  int m[4];

  for(i=0;i<n;++i) m[i] = i;

  for(k=0;k<n;++k){
    double p = 0;
    for(i=k;i<n;++i)
    for(j=k;j<n;++j) {
      double tmp = fabs( mat[i][j] );
      if( tmp > p ) {
        p = tmp;
        io = i;
        jo = j;
      }
    }
    if(p <= 1.e-13) {
      throw "Sigular Matrix";
    }
    p = mat[io][jo];

    if(jo != k) {
      for(i=0;i<n;++i) {
        swap( mat[i][jo], mat[i][k] );
      }
      swap( m[jo], m[k] );
    }

    if(io != k) {
      for(j=k;j<n;++j) swap( mat[io][j], mat[k][j] );
      swap( vec[io], vec[k] );
    }

    if(k != n-1) for(j=k;j<n-1;++j) mat[k][j+1] /= p;
    vec[k] /= p;

    if(k != n-1) {
      for(i=k;i<n-1;++i) {
        for(j=k;j<n-1;++j)
          mat[i+1][j+1] -= mat[i+1][k] * mat[k][j+1];
          vec[i+1] -= mat[i+1][k] * vec[k];
      }
    }
  }

  for(i1=1;i1<n;++i1) {
    i = n - 1 - i1;
    for(j=i;j<n-1;++j)
    vec[i] -= mat[i][j+1] * vec[j+1];
  }
  for(k=0;k<n;++k) mat[0][m[k]] = vec[k];
  for(k=0;k<n;++k) vec[k] = mat[0][k];
}

template<class T>
void swap(T& a, T& b)
{
  T tmp = a;
  a = b;
  b = tmp;
}

}} //namespace

// $Log$
// Revision 1.1  2000/07/05 23:12:41  tan
// Added equation class for least square approximation.
//
