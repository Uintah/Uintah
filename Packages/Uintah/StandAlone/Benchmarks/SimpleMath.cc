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

#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
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

  CCVariable<double> result, a, x, b;

  result.allocate( low, high );
  a.allocate( low, high );
  x.allocate( low, high );
  b.allocate( low, high );
  
  a.initialize( 5 );
  x.initialize( 6 );
  b.initialize( 2 );

  double startTime = Time::currentSeconds();

  for ( int i = 0; i < loop; i++ )
    for ( CellIterator iter(low, high); !iter.done(); iter++ )
      result[*iter] = a[*iter] * x[*iter] + b[*iter];

  double deltaTime = Time::currentSeconds() - startTime;
  double megaFlops = (loop * size * size * size * 2.0) / 1000000.0 / deltaTime;

  cout << "Completed in " << deltaTime << " seconds.";
  cout << " (" << megaFlops << " MFLOPS)" << endl;

  return EXIT_SUCCESS;
}
