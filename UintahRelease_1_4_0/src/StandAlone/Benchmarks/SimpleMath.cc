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


/*
 *  SimpleMath.cc: Simple math benchmark.
 *
 *  Written by:
 *   Randy Jones
 *   Department of Computer Science
 *   University of Utah
 *   July 31, 2003
 *
 *  Copyright (C) 2000 U of U
 */

#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Thread/Time.h>
#include <iostream>

using namespace SCIRun;
using namespace std;

const int SIZE_DEFAULT = 80;
const int LOOP_DEFAULT = 1;

void usage ( void )
{
  cerr << "Usage: SimpleMath <size> [<loop>]" << endl;
  cerr << endl;
  cerr << "  <size>  This benchmark will use four CCVariable<double>" << endl;
  cerr << "          variables, each with a size*size*size resolution" << endl;
  cerr << "          and perform the operation 'result = a * x + b'" << endl;
  cerr << "          on each of their elements."<< endl;
  cerr << endl;
  cerr << "  <loop>  The above operation will be repeated <loop> times." << endl;
}

void bench1(int loop, const IntVector& low, const IntVector& high,
	   CCVariable<double>& result, double a,
	   CCVariable<double>& x, CCVariable<double>& b)
{
  for ( int i = 0; i < loop; i++ )
    for ( CellIterator iter(low, high); !iter.done(); iter++ )
      result[*iter] = a * x[*iter] + b[*iter];
}

void bench2(int loop, const IntVector& low, const IntVector& high,
	   CCVariable<double>& result, double a,
	   CCVariable<double>& x, CCVariable<double>& b)
{
  for ( int i = 0; i < loop; i++ ){
    IntVector d(high-low);
    int size = d.x()*d.y()*d.z();
    double* rr = &result[low];
    const double* xx = &x[low];
    const double* bb = &b[low];
    for(int i = 0; i < size; i++)
      rr[i] = a * xx[i] + bb[i];
  }
}

int main ( int argc, char** argv )
{
  int size = SIZE_DEFAULT;
  int loop = LOOP_DEFAULT;

  /*
   * Parse arguments
   */
  if ( argc > 1 ) {
     size = atoi( argv[1] );


     if (size <= 0) {
       usage();
       return EXIT_FAILURE;
     }

     if ( argc > 2 ) {
       loop = atoi( argv[2] );

       if (loop <= 0) {
	 usage();
	 return EXIT_FAILURE;
       }
     }
  }
  else {
    usage();
    return EXIT_FAILURE;
  }

  cout << "Simple Math Benchmark: " << endl;
  cout << "Resolution (" << size << ", " << size << ", " << size << ")" << endl;
  cout << "Repeating " << loop << " time(s)." << endl;

  IntVector low ( 0,0,0 );
  IntVector high( size,size,size );

  CCVariable<double> result, x, b;
  double a = 5;

  result.allocate( low, high );
  //  a.allocate( low, high );
  x.allocate( low, high );
  b.allocate( low, high );
  
  //a.initialize( 5 );
  x.initialize( 6 );
  b.initialize( 2 );

  {
    double startTime = Time::currentSeconds();
    bench1(loop, low, high, result, a, x, b);

    double deltaTime = Time::currentSeconds() - startTime;
    double megaFlops = (loop * size * size * size * 2.0) / 1000000.0 / deltaTime;

    cout << "Completed in " << deltaTime << " seconds.";
    cout << " (" << megaFlops << " MFLOPS)" << endl;
  }
  {
    double startTime = Time::currentSeconds();
    bench2(loop, low, high, result, a, x, b);

    double deltaTime = Time::currentSeconds() - startTime;
    double megaFlops = (loop * size * size * size * 2.0) / 1000000.0 / deltaTime;

    cout << "Completed in " << deltaTime << " seconds.";
    cout << " (" << megaFlops << " MFLOPS)" << endl;
  }
 
  return EXIT_SUCCESS;
}

