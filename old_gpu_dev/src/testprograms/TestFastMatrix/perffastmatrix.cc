/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



#include <Core/Math/FastMatrix.h>
#include <Core/Math/Rand48.h>
#include <Core/Math/MiscMath.h>
#include <Core/Thread/Time.h>
#include <cmath>
#include <cstdlib>
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

