
#include <Packages/Uintah/Core/Math/FastMatrix.h>
#include <Core/Math/MiscMath.h>
#include <math.h>
#include <iostream>
using namespace std;
using namespace SCIRun;
using namespace Uintah;

static const double tolerance = 1.e-10;

static void checkIdentity(const FastMatrix& m)
{
  int size=m.numRows();
  bool err=false;
  for(int i=0;i<size;i++){
    for(int j=0;j<size;j++){
      double want=0;
      if(i==j)
	want=1;
      if(Abs(m(i,j)-want) > tolerance){
	cerr << "Error: product(" << i << ", " << j << ")=" << m(i,j) << '\n';
	err=true;
      }
    }
  }
  if(err)
    exit(1);
}

int main(int argc, char* argv[])
{
  int max=20;
  if(argc == 2)
    max=atoi(argv[1]);
  for(int size=1;size<=max;size++){
    FastMatrix m(size, size);
    for(int i=0;i<size;i++){
      for(int j=0;j<size;j++){
	m(i,j)=drand48();
      }
    }
    FastMatrix m2(size, size);
    m2.copy(m);
    FastMatrix minv(size, size);
    minv.destructiveInvert(m2);

    FastMatrix product(size, size);
    product.multiply(minv, m);

    checkIdentity(product);
    product.multiply(m, minv);
    checkIdentity(product);

    vector<double> v(size);
    vector<double> vcopy(size);
    for(int i=0;i<size;i++)
      v[i]=vcopy[i]=drand48();
    FastMatrix m3(size, size);
    m3.copy(m);
    vector<double> x(size);
    m3.destructiveSolve(vcopy, x);
    vector<double> xx(size);
    minv.multiply(v, xx);
    bool err=false;
    for(int i=0;i<size;i++){
      if(Abs(x[i]-xx[i] > tolerance)){
	if(!err)
	  cerr << "size: " << size << '\n';
	cerr << "Error: rhs[" << i << "]=" << x[i] << " vs. " << xx[i] << '\n';
	err=true;
      }
    }
    if(err)
      exit(1);
  }
  exit(0);
}
