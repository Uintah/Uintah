
#include <Packages/Uintah/Core/Math/FastMatrix.h>
#include <Packages/Uintah/Core/Math/Rand48.h>
#include <Core/Math/MiscMath.h>
#include <Core/Thread/Time.h>
#include <math.h>
#include <iostream>
using namespace std;
using namespace SCIRun;
using namespace Uintah;

int main(int argc, char* argv[])
{
  int size = 5;
  int reps = 100;
  if(argc >= 2)
    size=atoi(argv[1]);
  if(argc >= 3)
    reps = atoi(argv[2]);
  FastMatrix m(size, size);
  for(int i=0;i<size;i++){
    for(int j=0;j<size;j++){
      m(i,j)=drand48();
    }
  }
  FastMatrix minv(size, size);
  vector<double> b(size);
  vector<double> b2(size);
  for(int i=0;i<size;i++)
    b[i] = b2[i] = drand48();
  vector<double> x(size);
  vector<double> x2(size);
  double start = Time::currentSeconds();
#if 0
  for(int i=0;i<reps;i++){
    minv.destructiveInvert(m);
    minv.multiply(b, x);
    minv.multiply(b2, x2);
  }
#else
  for(int i=0;i<reps;i++){
    minv.destructiveSolve(&b[0], &b[1]);
  }
#endif
  double dt = Time::currentSeconds()-start;
  cerr << reps << " in " << dt << " seconds, " << dt/reps*1000000 << " us/rep\n";
  exit(0);
}

